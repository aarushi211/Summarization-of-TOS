"""
tests/test_api.py
"""

import io
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

# ── Stub heavy dependencies ───────────────────────────────────────────────────
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
stub_module("langsmith", 
    traceable=lambda *a, **kw: (lambda f: f), 
    RunTree=MagicMock(),
    Client=MagicMock(),
    get_tracing_context=MagicMock(),
)

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
    from api.core.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "user-abc", "token": "fake"}
    yield
    app.dependency_overrides = {}


# ── Existing smoke tests (unchanged) ─────────────────────────────────────────
def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_auth_signup_rate_limit(client, monkeypatch):
    responses = []
    for _ in range(7):
        r = client.post("/auth/signup", json={"email": "test@example.com", "password": "password123"})
        responses.append(r.status_code)
    assert 429 in responses


def test_ingest_upload_validation(client):
    big_content = b"%PDF-1.4" + b"x" * (51 * 1024 * 1024)
    resp = client.post(
        "/ingest/upload",
        files={"file": ("test.pdf", io.BytesIO(big_content), "application/pdf")},
        data={"service_name": "Test"},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 413


def test_chat_summary_requires_auth(client, app):
    app.dependency_overrides = {}
    resp = client.get("/chat/summary/some-id")
    assert resp.status_code == 401


def test_history_list(client):
    from api.core import database
    database.supa_admin.table().select().eq().order().execute.return_value = MagicMock(data=[{"id": "doc1"}])
    resp = client.get("/history/documents", headers={"Authorization": "Bearer fake"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


# ── RAG pipeline tests ────────────────────────────────────────────────────────
#
# WHY these tests matter:
#
# The existing tests only verify that HTTP endpoints return 200. They say
# nothing about whether the RAG pipeline actually retrieves relevant documents,
# whether the reranker is being called, or whether the streaming output has the
# right shape. These tests exercise the core value of the system — the part
# most likely to break silently when dependencies change.
#
# We test the TOSAssistant class directly (not via HTTP) so failures point
# at the ML logic, not the API layer.


@pytest.fixture()
def mock_assistant_instance():
    """
    A TOSAssistant with real method logic but mocked external calls.
    Pinecone, the LLM, and the cross-encoder are all stubbed so the test
    runs without any real models or network calls.
    """
    from src.RAG.engine import TOSAssistant
    from langchain_core.documents import Document

    # Fake retrieved documents that look like real Pinecone results
    fake_docs = [
        Document(
            page_content="We may share your data with advertising partners without notice.",
            metadata={"chunk_id": 0, "citation": "p.3 › Data Sharing", "section": "Data Sharing", "page": 2},
        ),
        Document(
            page_content="You can request deletion of your account within 30 days.",
            metadata={"chunk_id": 1, "citation": "p.7 › User Rights", "section": "User Rights", "page": 6},
        ),
        Document(
            page_content="Arbitration is required for all disputes. Class actions are waived.",
            metadata={"chunk_id": 2, "citation": "p.12 › Disputes", "section": "Disputes", "page": 11},
        ),
    ]

    with patch("src.RAG.engine.PineconeClient"), \
         patch("src.RAG.engine.HuggingFaceEmbeddings"), \
         patch("src.RAG.engine.Llama"), \
         patch("src.RAG.engine.CrossEncoder") as mock_ce_cls, \
         patch.object(TOSAssistant, "_ensure_pinecone_index"), \
         patch.object(TOSAssistant, "_search_vectorstore", return_value=fake_docs):

        # Cross-encoder scores: doc[0] is most relevant, doc[2] least
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.92, 0.65, 0.31]
        mock_ce_cls.return_value = mock_ce

        assistant = TOSAssistant.__new__(TOSAssistant)
        assistant._metrics = []
        assistant._pc = MagicMock()
        assistant._index_name = "test-index"
        assistant._dimension = 768
        assistant._pc_index = MagicMock()
        assistant.embed_model = MagicMock()
        assistant.cross_encoder = mock_ce
        assistant.llm = MagicMock()

        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        assistant.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "doc_title"), ("##", "section")],
            strip_headers=False,
        )
        assistant.sub_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=700)

        assistant._search_vectorstore = MagicMock(return_value=fake_docs)

        yield assistant, fake_docs, mock_ce


def test_rag_retrieval_returns_grounded_results(mock_assistant_instance):
    """
    The retrieval pipeline (Pinecone → BM25 RRF → cross-encoder reranking)
    should return documents that have citations and section metadata.

    This is the core contract of the RAG system: every returned chunk must
    be attributable to a specific location in the source document.
    Without this, the LLM has no grounding for its answers.
    """
    from src.RAG.schemas import SessionState

    assistant, fake_docs, mock_ce = mock_assistant_instance
    state = SessionState(
        pinecone_namespace="user_abc_doc_123",
        document_id="doc-123",
        cached_chunks=fake_docs,
    )

    docs, metrics = assistant._get_relevant_chunks(
        query="what data do you share with third parties?",
        state=state,
        top_k=5,
    )

    # Cross-encoder must be called — skipping it means we return random chunks
    assert mock_ce.predict.called, "Cross-encoder reranker was not called"

    # Every returned doc must have a citation (the grounding guarantee)
    for doc in docs:
        assert "citation" in doc.metadata, f"Doc missing citation: {doc.metadata}"
        assert doc.metadata["citation"], "Citation must not be empty"
        assert "section" in doc.metadata, f"Doc missing section: {doc.metadata}"

    # Results must be ordered by cross-encoder score (highest first)
    # Our mock returns scores [0.92, 0.65, 0.31] — chunk_id 0 should be first
    assert docs[0].metadata["chunk_id"] == 0, (
        "Reranker did not sort by score — highest-scored doc should be first"
    )

    # Retrieval timing must be recorded (needed for latency monitoring)
    assert "retrieval_s" in metrics, "Retrieval latency not recorded in metrics"
    assert metrics["retrieval_s"] >= 0


def test_rag_pipeline_handles_empty_index(mock_assistant_instance):
    """
    When Pinecone returns no results (new user, no document uploaded yet),
    the pipeline should return an empty list cleanly — not raise an exception
    or return malformed data that crashes the streaming generator.

    This edge case caused silent 500s in the original code because the
    streaming generator had no guard for an empty docs list.
    """
    from src.RAG.schemas import SessionState

    assistant, _, _ = mock_assistant_instance

    # Override to simulate an empty index
    assistant._search_vectorstore = MagicMock(return_value=[])

    state = SessionState(pinecone_namespace="user_new_doc_000", document_id="doc-000")

    docs, metrics = assistant._get_relevant_chunks(
        query="what is the refund policy?",
        state=state,
        top_k=7,
    )

    assert docs == [], "Empty index should return empty list, not raise"
    assert isinstance(metrics, dict), "Metrics dict must be returned even on empty results"


def test_rag_all_sources_returns_complete_attribution(mock_assistant_instance):
    """
    _all_sources() builds the source attribution list that is sent to the
    frontend alongside every answer. Every field the frontend depends on
    (tag, citation, section, page, excerpt) must be present and non-empty.

    If any field is missing, the frontend citation UI silently breaks.
    """
    assistant, fake_docs, _ = mock_assistant_instance

    sources = assistant._all_sources(fake_docs)

    assert len(sources) == len(fake_docs), "Must return one source entry per doc"

    required_fields = {"tag", "citation", "section", "page", "excerpt"}
    for i, source in enumerate(sources):
        missing = required_fields - set(source.keys())
        assert not missing, f"Source {i} missing fields: {missing}"
        assert source["tag"] == f"[SOURCE {i+1}]", f"Source tag wrong for index {i}"
        assert source["excerpt"], "Excerpt must not be empty"
        # Page numbers shown to users must be 1-indexed (not 0-indexed internal)
        assert source["page"] >= 1, f"Page number must be 1-indexed, got {source['page']}"


def test_streaming_output_schema_is_valid(mock_assistant_instance):
    """
    The streaming generator yields dicts that are serialised directly to SSE.
    If a dict is missing 'type' or has an unexpected type value, the frontend
    SSE parser silently breaks mid-stream with no error shown to the user.

    This test validates that every yielded message has a 'type' field and
    that the final message is always {"type": "done"}.
    """
    from src.RAG.schemas import SessionState

    assistant, fake_docs, _ = mock_assistant_instance

    # Stub the LLM to return a known token stream
    assistant.llm = MagicMock(return_value=[
        {"choices": [{"text": "We "}]},
        {"choices": [{"text": "share "}]},
        {"choices": [{"text": "your data."}]},
    ])

    state = SessionState(pinecone_namespace="user_abc_doc_123", document_id="doc-123")
    state.full_text = "some document text"

    assistant._get_relevant_chunks = MagicMock(return_value=(fake_docs, {"retrieval_s": 0.1}))
    messages = list(assistant.answer_question_stream("what data do you share?", state))

    valid_types = {"token", "sources", "done", "error"}
    for msg in messages:
        assert "type" in msg, f"Message missing 'type' field: {msg}"
        assert msg["type"] in valid_types, f"Unexpected message type: {msg['type']}"

    # Stream must always end with done
    assert messages[-1]["type"] == "done", (
        "Stream did not end with 'done' — frontend will wait forever for stream end"
    )