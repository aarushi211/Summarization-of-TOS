import logging
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.RAG.schemas import IngestionError

logger = logging.getLogger(__name__)

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_RE_UNICODE_REPLACEMENT = re.compile(r"\uFFFD")


def _sanitize_pdf_text(text: str) -> str:
    """Remove null bytes and control chars that break chunking and embeddings."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = _RE_UNICODE_REPLACEMENT.sub(" ", text)
    text = _RE_CONTROL.sub(" ", text)
    return text


def _load_pdf_pymupdf(file_path: str) -> list[Document]:
    import fitz  # pymupdf

    docs: list[Document] = []
    with fitz.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf):
            text = _sanitize_pdf_text(page.get_text("text"))
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_idx},
                )
            )
    return docs


def _load_pdf_pypdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = _sanitize_pdf_text(doc.page_content)
    return docs


def load_pdf(file_path: str) -> list[Document]:
    try:
        docs = _load_pdf_pymupdf(file_path)
        loader_name = "pymupdf"
    except ImportError:
        logger.warning("pymupdf not installed; falling back to PyPDFLoader for %s", file_path)
        docs = _load_pdf_pypdf(file_path)
        loader_name = "pypdf"
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s (%s); falling back to PyPDFLoader", file_path, exc)
        docs = _load_pdf_pypdf(file_path)
        loader_name = "pypdf"

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise IngestionError("PDF produced no pages.")
    logger.debug("Loaded %s with %s (%d pages)", file_path, loader_name, len(docs))
    return docs


def load_text(file_path: str) -> list[Document]:
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        if not docs or not docs[0].page_content.strip():
            raise IngestionError("Text file is empty.")
        return docs
    except Exception as e:
        logger.error("Failed to load text %s: %s", file_path, e)
        raise IngestionError(f"Failed to load text: {e}")
