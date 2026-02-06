import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

class TOSAssistant:
    def __init__(self, rag_model_path, summary_model_path, index_dir="faiss_index"):
        self.index_dir = Path(index_dir)
        
        # 1. Initialize Models (Raw Llama-cpp)
        # We load them directly to avoid LangChain chain deprecations
        print(f"Loading RAG Model: {Path(rag_model_path).name}")
        self.rag_llm = Llama(
            model_path=rag_model_path,
            n_ctx=8192,
            n_gpu_layers=-1, # Offload all to GPU
            verbose=False
        )

        print(f"Loading Summary Model: {Path(summary_model_path).name}")
        self.summary_llm = Llama(
            model_path=summary_model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )

        # 2. Load Embeddings & Reranker
        print("Loading Embeddings & Cross-Encoder...")
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.vector_store = None

    def ingest_document(self, pdf_path):
        print(f'Ingesting {pdf_path}...')
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split text (Large chunks for context)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Build Index
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        print("Ingestion complete. Vector Store Ready.")

    def _get_relevant_chunks(self, query, top_k=5):
        """
        Retrieval Pipeline: 
        1. FAISS MMR (Gets diverse candidates) -> 
        2. Cross-Encoder (Reranks for high precision)
        """
        # Step 1: Semantic Search with MMR (Built-in to FAISS)
        # Fetch 20 diverse candidates to give the reranker good options
        candidates = self.vector_store.max_marginal_relevance_search(
            query, k=20, fetch_k=50, lambda_mult=0.5
        )

        # Step 2: Cross-Encoder Reranking
        # Create pairs: [[query, text1], [query, text2], ...]
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Sort by score (descending)
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Return top_k high-quality chunks
        return [doc for doc, score in scored_docs[:top_k]]

    def generate_global_summary(self):
        """
        Uses the Fine-Tuned Model (self.summary_llm)
        """
        if not self.vector_store:
            return "No document loaded."

        # Grab context
        docs = self.vector_store.similarity_search("", k=10)
        context = "\n".join([d.page_content for d in docs])
        
        # --- FIX: Match the Training Prompt Exactly ---
        
        # 1. Match the System Prompt from your training script
        system_msg = (
            "You are a legal expert. Summarize the following Terms of Service. "
            "Focus on user rights and data privacy."
        )

        # 2. Match the User Prompt Format (Service + Doc Type)
        # We use the class attributes defaults if real ones aren't available
        service_name = getattr(self, 'service_name', "Unknown Service")
        doc_type = getattr(self, 'doc_type', "Terms of Service")
        
        # This exact f-string structure matches your format_synthetic_dataset function
        user_msg = (
            f"Service: {service_name}\n"
            f"Doc Type: {doc_type}\n\n"
            f"Text:\n{context[:12000]}" # Truncated text
        )

        # Direct Chat Completion call
        response = self.summary_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    
    def answer_question(self, query):
        """
        Uses the Base Qwen Model (self.rag_llm) with RAG
        """
        if not self.vector_store:
            return "Please ingest a document first."

        # 1. Retrieve
        relevant_docs = self._get_relevant_chunks(query)
        
        # 2. Format Context
        context_str = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])

        # 3. Construct Prompt (Qwen Format)
        system_msg = (
            "You are a helpful legal assistant. Answer the user's question using ONLY the provided Context.\n"
            "If the answer is not present, say 'NOT_IN_DOCUMENT'."
        )
        user_msg = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}"
        )

        # 4. Generate
        output = self.rag_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=512
        )

        return {
            "answer": output['choices'][0]['message']['content'],
            "sources": [d.page_content[:200] + "..." for d in relevant_docs]
        }