import io
import logging

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Embedding model (free Gemini tier) ---
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.gemini_api_key,
)

# --- Text splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def _upload_to_supabase(file_bytes: bytes, filename: str, doc_id: str) -> str:
    """Upload file to Supabase storage and return storage path, or empty string on failure."""
    storage_path = f"documents/{doc_id}/{filename}"
    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        supabase.storage.from_(settings.supabase_bucket).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/pdf"},
        )
        return storage_path
    except Exception as exc:
        logger.warning("Supabase upload skipped: %s", exc)
        return ""


def extract_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
    db: AsyncSession,
) -> dict:
    """
    Full ingestion pipeline:
    1. Upload PDF to Supabase Storage
    2. Extract text
    3. Chunk text
    4. Embed chunks via Gemini
    5. Store embeddings in pgvector
    """
    # 1. Upload raw PDF to Supabase Storage (best-effort)
    storage_path = _upload_to_supabase(file_bytes, filename, doc_id)

    # 2. Extract text
    raw_text = extract_text(file_bytes)
    if not raw_text.strip():
        raise ValueError("Could not extract any text from this PDF.")

    # 3. Chunk
    chunks = splitter.split_text(raw_text)

    # 4. Embed all chunks in one batch call (saves quota)
    chunk_embeddings = await embeddings_model.aembed_documents(chunks)

    # 5. Bulk insert into pgvector
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        embedding_768 = embedding[:768]
        await db.execute(
            text("""
                INSERT INTO chunks (document_id, content, embedding, chunk_index)
                VALUES (:doc_id, :content, CAST(:embedding AS vector), :idx)
            """),
            {
                "doc_id": doc_id,
                "content": chunk_text,
                "embedding": str(embedding_768),
                "idx": idx,
            },
        )

    # 6. Mark document as ready
    await db.execute(
        text("""
            UPDATE documents
            SET status = 'ready',
                chunk_count = :count,
                storage_path = :path
            WHERE id = :doc_id
        """),
        {"count": len(chunks), "path": storage_path, "doc_id": doc_id},
    )

    return {"chunks": len(chunks), "chars": len(raw_text)}
