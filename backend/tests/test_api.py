import pytest
from httpx import ASGITransport, AsyncClient

import app.services.ingestion as ingestion_module
import app.services.llm as llm_module
import app.services.retriever as retriever_module
from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "docusense-api"


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "DocuSense" in r.json()["message"]


@pytest.mark.asyncio
async def test_query_empty_question():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/api/v1/query",
            json={"question": "", "doc_id": "fake-id"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_non_pdf():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
    assert r.status_code == 400


def test_google_clients_not_initialized_at_import():
    """Regression test: importing the app must NOT trigger Google credential
    checks.  All Google API clients must remain None until the first real
    request that actually needs them (lazy initialisation pattern)."""
    assert retriever_module._embeddings_model is None, (
        "retriever._embeddings_model was initialised at import time; "
        "this causes DefaultCredentialsError in CI environments without ADC."
    )
    assert ingestion_module._embeddings_model is None, (
        "ingestion._embeddings_model was initialised at import time."
    )
    assert llm_module._llm is None, (
        "llm._llm was initialised at import time."
    )


@pytest.mark.asyncio
async def test_upload_file_too_large():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 21 MB payload
        large_content = b"0" * (21 * 1024 * 1024)
        r = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("large.pdf", large_content, "application/pdf")},
        )
    assert r.status_code == 400
    assert "File too large" in r.json()["detail"]


@pytest.mark.asyncio
async def test_chunks_test_validation():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Invalid top_k > 10
        r1 = await client.post(
            "/api/v1/chunks/test",
            json={"doc_id": "test-doc", "question": "test question", "top_k": 20},
        )
        assert r1.status_code == 422

        # Empty question
        r2 = await client.post(
            "/api/v1/chunks/test",
            json={"doc_id": "test-doc", "question": "  ", "top_k": 5},
        )
        assert r2.status_code == 400


def test_extract_text_and_splitter():
    import pathlib
    from app.services.ingestion import extract_text, splitter

    pdf_path = pathlib.Path(__file__).parent.parent / "test_sample.pdf"
    assert pdf_path.exists()
    content = pdf_path.read_bytes()
    text = extract_text(content)
    assert len(text) > 0
    assert "FastAPI" in text

    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    assert "FastAPI" in chunks[0]


def test_extract_text_empty_and_invalid():
    from app.services.ingestion import extract_text

    with pytest.raises(ValueError, match="empty"):
        extract_text(b"")

    with pytest.raises(ValueError, match="Failed to parse PDF"):
        extract_text(b"not a valid pdf content")


def test_cache_service():
    from app.services import cache

    key = cache._make_key("doc-123", "  Sample Question? ")
    assert key.startswith("docusense:query:")
    # Idempotent MD5
    assert key == cache._make_key("doc-123", "sample question?")

    # Safe fault tolerance when redis is unreachable or fake
    val = cache.get_cached("doc-123", "Sample Question?")
    assert val is None

    # Should not raise exception
    cache.set_cached("doc-123", "Sample Question?", {"answer": "test"})


@pytest.mark.asyncio
async def test_list_and_delete_documents_with_mock_db():
    from unittest.mock import AsyncMock, MagicMock
    from app.core.database import get_db

    mock_row = MagicMock()
    mock_row._mapping = {
        "id": "11111111-1111-1111-1111-111111111111",
        "filename": "test.pdf",
        "chunk_count": 5,
        "status": "ready",
        "created_at": "2026-04-01T00:00:00Z",
    }
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Test GET /documents
            r_list = await client.get("/api/v1/documents")
            assert r_list.status_code == 200
            data = r_list.json()
            assert "documents" in data
            assert len(data["documents"]) == 1
            assert data["documents"][0]["filename"] == "test.pdf"

            # Test DELETE /documents/{doc_id}
            r_del = await client.delete("/api/v1/documents/11111111-1111-1111-1111-111111111111")
            assert r_del.status_code == 200
            assert r_del.json() == {"deleted": "11111111-1111-1111-1111-111111111111"}
    finally:
        app.dependency_overrides.clear()


