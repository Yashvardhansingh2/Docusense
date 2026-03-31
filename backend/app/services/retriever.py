from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.gemini_api_key,
    output_dimensionality=768,
)


async def retrieve_chunks(
    question: str,
    doc_id: str,
    db: AsyncSession,
    k: int = 5,
) -> list[dict]:
    """
    Embed the question, then find the k most similar chunks
    from this document using pgvector cosine similarity.
    """
    query_embedding = await embeddings_model.aembed_query(question)
    query_embedding_768 = query_embedding[:768]

    rows = await db.execute(
        text("""
            SELECT
                content,
                chunk_index,
                1 - (embedding <=> CAST(:emb AS vector)) AS score
            FROM chunks
            WHERE document_id = :doc_id
            ORDER BY embedding <=> CAST(:emb AS vector)
            LIMIT :k
        """),
        {"emb": str(query_embedding_768), "doc_id": doc_id, "k": k},
    )

    return [
        {
            "content": r.content,
            "chunk_index": r.chunk_index,
            "score": round(float(r.score), 3),
        }
        for r in rows.fetchall()
    ]
