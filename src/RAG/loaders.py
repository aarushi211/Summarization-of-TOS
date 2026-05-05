import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.RAG.schemas import IngestionError

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> list[Document]:
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise IngestionError("PDF produced no pages.")
        return docs
    except Exception as e:
        logger.error("Failed to load PDF %s: %s", file_path, e)
        raise IngestionError(f"Failed to load PDF: {e}")

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
