"""
tests/test_api.py
Updated for modular refactor.
"""

import io
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Stub heavy dependencies ──────────────────────────────────────────────────
def stub_module(name, **kwargs):
    stub = types.ModuleType(name)
    for k, v in kwargs.items():
        setattr(stub, k, v)
    sys.modules[name] = stub
    return stub

stub_module("llama_cpp", Llama=MagicMock())
stub_module("sentence_transformers", CrossEncoder=MagicMock())
stub_module("langchain_huggingface", HuggingFaceEmbeddings=MagicMock())
stub_module("pinecone", Pinecone=MagicMock(), ServerlessSpec=MagicMock())
stub_module("langchain_pinecone", PineconeVectorStore=MagicMock())
stub_module("langsmith", traceable=lambda *a, **kw: (lambda f: f), RunTree=MagicMock())

os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    with patch("api.core.database.create_client") as mock_supa, \
         patch("api.core.database.boto3.client") as mock_boto, \
         patch("api.core.database.TOSAssistant") as mock_assistant_cls:
        
        mock_supa.return_value = MagicMock()
        mock_boto.return_value = MagicMock()
        mock_assistant = MagicMock()
        mock_assistant_cls.return_value = mock_assistant
        
        # We MUST import these after patching
        import api.core.database as db
        db.supa_admin = mock_supa.return_value
        db.s3_client = mock_boto.return_value
        db.shared_assistant = mock_assistant
        
        from api.main import app as fastapi_app
        yield fastapi_app

@pytest.fixture()
def client(app):
    return TestClient(app)

@pytest.fixture(autouse=True)
def patch_security(app):
    # Bypass real JWT verification using FastAPI dependency overrides
    from api.core.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "user-abc", "token": "fake"}
    yield
    app.dependency_overrides = {}

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_auth_signup_rate_limit(client, monkeypatch):
    # Test rate limiting on signup
    responses = []
    for _ in range(7):
        r = client.post("/auth/signup", json={"email": "test@example.com", "password": "password123"})
        responses.append(r.status_code)
    assert 429 in responses

def test_ingest_upload_validation(client):
    # Too large file
    big_content = b"%PDF-1.4" + b"x" * (settings_max_bytes := 51 * 1024 * 1024)
    resp = client.post(
        "/ingest/upload",
        files={"file": ("test.pdf", io.BytesIO(big_content), "application/pdf")},
        data={"service_name": "Test"},
        headers={"Authorization": "Bearer fake"}
    )
    assert resp.status_code == 413

def test_chat_summary_requires_auth(client, app):
    # Clear overrides to test real auth requirement
    app.dependency_overrides = {}
    resp = client.get("/chat/summary/some-id")
    assert resp.status_code == 401

def test_history_list(client):
    from api.core import database
    # supa_admin is already a mock from the fixture, let's just configure it
    database.supa_admin.table().select().eq().order().execute.return_value = MagicMock(data=[{"id": "doc1"}])
    resp = client.get("/history/documents", headers={"Authorization": "Bearer fake"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

# Helper for settings in tests
from api.core.config import settings