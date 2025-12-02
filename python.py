# app.py - High-ER JAMB RAG Assistant Pro 2025 Edition
# 🚀 Next-gen JAMB/UTME AI Tutor with OCR, RAG, LLM & Vector Search
# Author: Psycho Lord Dev Team
# Description: Upload past JAMB papers, get answers, focus tips, and study guidance in Nigerian English.
# Features:
# - PDF/Image ingestion with OCR
# - Deduplication
# - Qdrant vector DB search
# - LLM powered answer generation
# - Background ingestion
# - Health checks and wipe endpoint

import uuid
import re
import logging
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Dict
from functools import lru_cache
from contextlib import asynccontextmanager

import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
from pdf2image import convert_from_bytes

# === Embedding & Vector DB ===
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, FieldCondition, MatchValue

# === LLM ===
import litellm

litellm.set_verbose = False

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jamb-rag")

# === Config ===
class Config:
    API_KEY = "jamb-secret-2025-change-in-prod"
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "jamb_questions_v3"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384
    OCR_LANG = "eng"
    DEFAULT_TOP_K = 6
    LLM_MODEL = "groq/llama3-8b-8192"

config = Config()

# === FastAPI Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load embedder and connect to Qdrant
    app.state.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    app.state.qdrant = QdrantClient(url=config.QDRANT_URL, timeout=20)

    if not app.state.qdrant.collection_exists(config.COLLECTION_NAME):
        app.state.qdrant.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {config.COLLECTION_NAME}")

    logger.info("🚀 JAMB RAG Assistant Ready")
    yield
    # Shutdown
    app.state.qdrant.close()
    logger.info("Shutdown complete")

app = FastAPI(
    title="JAMB AI Tutor Pro",
    version="2.0.0",
    description="Next-gen JAMB/UTME RAG Assistant with OCR, Vector Search & Real LLM",
    lifespan=lifespan
)

# === Security ===
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def get_api_key(key: str = Depends(api_key_header)):
    if key != config.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

# === Models ===
class QuestionPayload(BaseModel):
    text: str = Field(..., max_length=3000)
    subject: str = Field(..., max_length=50)
    year: int = Field(..., ge=1978, le=2030)
    question_type: str = "MultipleChoice"
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    difficulty: Optional[int] = Field(None, ge=1, le=5)

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20)
    subject: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    tags: Optional[List[str]] = None

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 6
    history: Optional[List[Dict]] = None

class SearchResult(BaseModel):
    id: str
    question: str
    subject: str
    year: int
    score: float
    tags: List[str]

class AskResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    usage: Dict

# === OCR & Preprocessing ===
def enhance_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img

def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(enhance_image(img), lang=config.OCR_LANG)

# === Deduplication ===
async def find_duplicates(texts: List[str]) -> set[int]:
    if len(texts) < 2:
        return set()
    embeddings = app.state.embedder.encode(texts, normalize_embeddings=True)
    duplicates = set()
    seen = {}
    for i, vec in enumerate(embeddings):
        vec_key = tuple(np.round(vec, 5))
        if vec_key in seen:
            duplicates.add(i)
        else:
            seen[vec_key] = i
    return duplicates

# === Background Ingestion ===
async def ingest_file_task(file_content: bytes, filename: str, subject: str, year: int):
    questions = []
    texts = []

    try:
        if filename.lower().endswith(".pdf"):
            images = convert_from_bytes(file_content, dpi=300)
        else:
            img = Image.open(BytesIO(file_content))
            images = [img]

        for img in images:
            text = ocr_image(img).strip()
            if not text:
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            current = ""
            for line in lines:
                if re.match(r"^\d+[\.\)]", line) and current:
                    texts.append(current)
                    current = line
                else:
                    current += " " + line
            if current:
                texts.append(current)

        # Remove duplicates
        dup_indices = await find_duplicates(texts)
        clean_texts = [t for i, t in enumerate(texts) if i not in dup_indices]

        # Create payloads
        for text in clean_texts:
            qid = str(uuid.uuid4())
            questions.append({
                "id": qid,
                "text": text[:3000],
                "subject": subject.upper(),
                "year": year,
                "tags": [subject.lower(), f"y{year}"],
                "question_type": "MultipleChoice" if any(c in text for c in "ABCD") else "Theory"
            })

        # Batch upsert to Qdrant
        if questions:
            vectors = app.state.embedder.encode([q["text"] for q in questions]).tolist()
            points = [
                models.PointStruct(
                    id=q["id"],
                    vector=vec,
                    payload={
                        "question": q["text"],
                        "subject": q["subject"],
                        "year": q["year"],
                        "tags": q["tags"],
                        "type": q["question_type"]
                    }
                )
                for q, vec in zip(questions, vectors)
            ]
            app.state.qdrant.upsert(collection_name=config.COLLECTION_NAME, points=points)
            logger.info(f"Ingested {len(questions)} questions from {filename}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

# === Routes ===
@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    subject: str = "General",
    year: int = 2024,
    background: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(get_api_key)
):
    if not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only PDF and images allowed")

    content = await file.read()
    background.add_task(ingest_file_task, content, file.filename, subject, year)
    return {"status": "queued", "file": file.filename, "questions_will_be_processed": True}

@app.post("/search", response_model=List[SearchResult])
async def search(req: SearchRequest, api_key: str = Depends(get_api_key)):
    query_vec = app.state.embedder.encode(req.query).tolist()
    filters = []
    if req.subject:
        filters.append(FieldCondition(key="subject", match=MatchValue(value=req.subject.upper())))
    if req.year_min or req.year_max:
        filters.append(FieldCondition(
            key="year",
            range=models.Range(gte=req.year_min or 1978, lte=req.year_max or 2030)
        ))

    search_result = app.state.qdrant.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vec,
        limit=req.top_k,
        query_filter=models.Filter(must=filters) if filters else None,
        with_payload=True
    )

    results = [
        SearchResult(
            id=hit.id,
            question=hit.payload["question"],
            subject=hit.payload["subject"],
            year=hit.payload["year"],
            score=round(hit.score, 4),
            tags=hit.payload.get("tags", [])
        )
        for hit in search_result
    ]
    return results

@app.post("/ask")
async def ask(req: AskRequest, api_key: str = Depends(get_api_key)):
    results = await search(SearchRequest(query=req.query, top_k=req.top_k or 6))
    if not results:
        answer = "I couldn't find any similar past questions. Try rephrasing your question or uploading more JAMB papers."
    else:
        context = "\n\n".join([
            f"[{r.subject} {r.year}] Score: {r.score:.3f}\nQ: {r.question}"
            for r in results[:5]
        ])
        prompt = f"""You are a brilliant JAMB tutor helping Nigerian students score 300+.
Use ONLY the context below to answer. If unsure, say you need more past questions.
Context (past JAMB questions):
{context}
Student Question: {req.query}
Answer conversationally in clear Nigerian English. Explain concepts. Suggest focus areas."""
        response = litellm.completion(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        answer = response.choices[0].message.content

    return AskResponse(
        answer=answer,
        sources=results,
        usage={"model": config.LLM_MODEL, "retrieved": len(results)}
    )

@app.get("/health")
async def health():
    try:
        info = app.state.qdrant.get_collection(config.COLLECTION_NAME)
        count = info.points_count
        status = "healthy"
    except:
        count = 0
        status = "degraded"
    return {
        "status": status,
        "questions_in_db": count,
        "embedding_model": config.EMBEDDING_MODEL,
        "up_since": datetime.now().isoformat()
    }

@app.delete("/wipe")
async def wipe(api_key: str = Depends(get_api_key)):
    app.state.qdrant.delete_collection(config.COLLECTION_NAME)
    app.state.qdrant.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE)
    )
    return {"status": "wiped and recreated"}
