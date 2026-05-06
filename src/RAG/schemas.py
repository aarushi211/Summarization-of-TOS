import time
import dataclasses

# ── Typed errors ─────────────────────────────────────────────────────────────
class RAGError(Exception):
    code: str = "RAG_ERROR"

class DocumentNotLoadedError(RAGError):
    code = "DOCUMENT_NOT_LOADED"

class RetrievalError(RAGError):
    code = "RETRIEVAL_ERROR"

class InferenceError(RAGError):
    code = "INFERENCE_ERROR"

class IngestionError(RAGError):
    code = "INGESTION_ERROR"


# ── Per-session state ─────────────────────────────────────────────────────────
@dataclasses.dataclass
class SessionState:
    """
    All per-session mutable state. Never stored on TOSAssistant — passed to
    every method so concurrent requests cannot interfere with each other.
    """
    pinecone_namespace: str   = ""
    document_id:        str   = ""
    service_name:       str   = "Unknown Service"
    doc_type:           str   = "Unknown Document"
    full_text:          str   = ""
    last_accessed:      float = dataclasses.field(default_factory=time.time)
    cached_chunks:      list  = dataclasses.field(default_factory=list)

    @property
    def has_document(self) -> bool:
        return bool(self.pinecone_namespace)

    def reset(self):
        self.pinecone_namespace = ""
        self.document_id        = ""
        self.service_name       = "Unknown Service"
        self.doc_type           = "Unknown Document"
        self.full_text          = ""
        self.cached_chunks      = []

SUMMARY_TOPICS = [
    ("Data Collection",          "what personal data and information is collected from users"),
    ("Data Sharing / Selling",   "sharing selling transferring user data to third parties partners"),
    ("Data Retention & Deletion","data retention period deletion of user account and personal data"),
    ("User Rights",              "user rights account termination suspension appeals"),
    ("Refund & Cancellation",    "refund cancellation subscription termination policy"),
    ("Arbitration & Disputes",   "arbitration dispute resolution class action waiver"),
    ("Liability Limitations",    "limitation of liability damages indemnification"),
    ("IP & Content Ownership",   "intellectual property content ownership license user generated"),
    ("Policy Changes",           "changes to terms notice modification policy updates"),
    ("Governing Law",            "governing law jurisdiction venue applicable law"),
    ("Children & COPPA",         "children under 13 COPPA minors privacy"),
    ("Cookies & Tracking",       "cookies tracking pixels analytics advertising identifiers"),
]
