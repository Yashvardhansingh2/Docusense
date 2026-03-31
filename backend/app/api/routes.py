import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.cache import get_cached, set_cached
from app.services.ingestion import ingest_document
from app.services.llm import generate_answer
from app.services.retriever import retrieve_chunks

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    doc_id: str


class ChunkTestRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /documents/upload")
    question: str = Field(..., description="Question used to retrieve the most relevant chunks")
    top_k: int = Field(5, ge=1, le=10, description="Number of chunks to return")


class ChunkItem(BaseModel):
    chunk_index: int
    score: float
    content: str


class ChunkTestResponse(BaseModel):
    doc_id: str
    question: str
    top_k: int
    returned_chunks: int
    chunks: list[ChunkItem]


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "docusense-api", "version": "1.0.0"}


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post(
    "/documents/upload",
    summary="Upload a PDF and index it into chunks",
    description="Uploads a PDF, extracts text, creates embeddings, and stores searchable chunks.",
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20 MB.")

    doc_id = str(uuid.uuid4())

    # Create document record
    await db.execute(
        text("""
            INSERT INTO documents (id, filename, storage_path, file_size_bytes)
            VALUES (:id, :name, '', :size)
        """),
        {"id": doc_id, "name": file.filename, "size": len(file_bytes)},
    )

    try:
        result = await ingest_document(file_bytes, file.filename, doc_id, db)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": result["chunks"],
        "characters_extracted": result["chars"],
        "status": "ready",
    }


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="Ask a question about one uploaded document",
    description="Runs retrieval + LLM answer generation for a document.",
)
async def query_document(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check Redis cache first
    cached = get_cached(body.doc_id, body.question)
    if cached:
        cached["from_cache"] = True
        return cached

    # Retrieve top-k similar chunks from pgvector
    chunks = await retrieve_chunks(body.question, body.doc_id, db)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found or contains no indexed content.",
        )

    # Generate answer via Gemini
    result = await generate_answer(body.question, chunks)
    result["doc_id"] = body.doc_id
    result["question"] = body.question
    result["from_cache"] = False

    # Cache the result
    set_cached(body.doc_id, body.question, result)

    # Persist query log
    await db.execute(
        text("""
            INSERT INTO query_logs
                (document_id, question, answer, sources_used, response_time_ms)
            VALUES (:doc_id, :q, :a, :s, :t)
        """),
        {
            "doc_id": body.doc_id,
            "q": body.question,
            "a": result["answer"],
            "s": len(result["sources"]),
            "t": result["response_time_ms"],
        },
    )

    return result


@router.post(
    "/chunks/test",
    response_model=ChunkTestResponse,
    summary="Test retrieved chunks (no LLM)",
    description="Returns top-k retrieved chunks for a question so you can validate retrieval quality in Swagger.",
)
async def test_chunks(
    body: ChunkTestRequest,
    db: AsyncSession = Depends(get_db),
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = await retrieve_chunks(body.question, body.doc_id, db, k=body.top_k)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found or contains no indexed content.",
        )

    return {
        "doc_id": body.doc_id,
        "question": body.question,
        "top_k": body.top_k,
        "returned_chunks": len(chunks),
        "chunks": chunks,
    }


# ── List documents ────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        text("""
            SELECT id, filename, chunk_count, status, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT 20
        """)
    )
    return {"documents": [dict(r._mapping) for r in rows.fetchall()]}


# ── Delete document ───────────────────────────────────────────────────────────

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(
        text("DELETE FROM documents WHERE id = :doc_id"),
        {"doc_id": doc_id},
    )
    return {"deleted": doc_id}
