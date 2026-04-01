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
