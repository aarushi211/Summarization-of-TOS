from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
import re
from langchain_community.document_loaders import TextLoader

class TOSAssistant:
    def __init__(self, model_path, index_dir="faiss_index"):
        self.index_dir = Path(index_dir)
        
        print(f"Loading RAG Model: {Path(model_path).name}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )

        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        local_embed_path = PROJECT_ROOT / "models" / "embeddings"
        local_cross_path = PROJECT_ROOT / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

        print("Loading Embeddings & Cross-Encoder...")
        self.embed_model = HuggingFaceEmbeddings(
            model_name=str(local_embed_path),
            model_kwargs={"trust_remote_code": True}
        )
        self.cross_encoder = CrossEncoder(str(local_cross_path))

        self.vector_store = None
        self.full_text = ""
        self.doc_type = "Unkown Document"
        self.service_name = "Unknown Service"

    def clean_text(self, text):
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl", 
            "": "",}
        
        for search, replace in replacements.items():
            text = text.replace(search, replace)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def ingest_document(self, pdf_path):
        print(f'Ingesting {pdf_path}...')
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        cleaned_pages = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            cleaned_pages.append(cleaned_content)

        self.full_text = '\n'.join(cleaned_pages)

        for doc, clean_text in zip(documents, cleaned_pages):
            doc.page_content = clean_text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        print("Ingestion complete. Vector Store Ready.")

    def ingest_text_file(self, txt_path):
        print(f'Ingesting Text File {txt_path}...')
        loader = TextLoader(txt_path, encoding='utf-8')
        documents = loader.load()
        
        # Reuse your existing splitter logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)
        
        self.full_text = documents[0].page_content
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        print("Text Ingestion complete.")

    def _get_relevant_chunks(self, query, top_k=5):
        candidates = self.vector_store.max_marginal_relevance_search(
            query, k=50, fetch_k=100, lambda_mult=0.5
        )

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]

    def format_qwen_prompt(self, system_msg, user_msg):
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_global_summary(self):
        if not self.full_text:
            return "No document loaded."

        truncated_text = self.full_text[:20000]

        system_prompt = "You are a legal expert. Summarize the following Terms of Service. Focus on user rights and data privacy."
        
        user_prompt = f"Service: {self.service_name}\nDoc Type: {self.doc_type}\n\nText:\n{truncated_text}"

        formatted_prompt = self.format_qwen_prompt(system_prompt, user_prompt)

        output = self.llm(
            formatted_prompt,
            max_tokens=500,
            temperature=0.1,
            repeat_penalty=1.2,
            stop=["<|im_end|>", "<|eot_id|>"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()

    def answer_question(self, query):
        if not self.vector_store:
            return "Please ingest a document first."

        relevant_docs = self._get_relevant_chunks(query, top_k=7)
        
        context_str = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])

        system_msg = (
            "You are a strict legal assistant. Answer the user's question using ONLY the provided Context sources. "
            "Do not infer 'selling' of data unless the text explicitly states 'we sell data'. "
            "Distinguish between 'sharing' (for functionality) and 'selling' (for profit). "
            "If the answer is not present in the sources, say 'NOT_IN_DOCUMENT'."
        )
        
        user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=400,
            repeat_penalty=1.1,
            stop=["<|im_end|>"]
        )

        return {
            "answer": output['choices'][0]['message']['content'],
            "sources": [d.page_content[:200] + "..." for d in relevant_docs]
        }