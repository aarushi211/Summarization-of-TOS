import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

class TOSAssistant:
    def __init__(self, model_path, index_dir="faiss_index"):
        self.llm = Llama(
            model_path, 
            n_ctx = 8192,
            n_gpu_layers=-1,
            verbose=False
        )

        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        local_embed_path = PROJECT_ROOT / "models" / "embeddings"

        if local_embed_path.exists():
            print(f"Loading Embeddings from local cache at {local_embed_path}...")
            self.embed_model = HuggingFaceEmbeddings(
                model_name=str(local_embed_path),
                model_kwargs={"trust_remote_code": True}
            )
        else:
            print("Loading Embeddings from HuggingFace Hub...")
            self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("Loading Cross-Encoder for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.vector_store = None
        self.ensemble_retriever = None 
        self.full_text = ""
        self.doc_type = "Terms of Service"
        self.service_name = "Unknown Service"
        self.index_dir = Path(index_dir)
        self.chunk_metadata = {}

        # retrievers placeholders
        self.bm25_retriever = None
        self.faiss_retriever = None

    def ingest_document(self, pdf_path, persist = True):
        print(f'Ingesting {pdf_path}')
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        self.full_text = '\n'.join([doc.page_content for doc in documents])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400 
        )
        chunks = text_splitter.split_documents(documents)
        
        for i, doc in enumerate(chunks):
            chunk_id = f"{Path(pdf_path).stem}::chunk_{i}"
            doc.metadata = doc.metadata or {}
            doc.metadata.update({
                "chunk_id": chunk_id,
                "source": str(pdf_path),
            })
            self.chunk_metadata[chunk_id] = {
                "source": str(pdf_path),
                "page": doc.metadata.get("page", None)
            }

        # Persist/load FAISS index directory presence check:
        if persist and self.index_dir.exists():
            # load existing, then add (if add_documents is supported)
            self.vector_store = FAISS.load_local(str(self.index_dir), self.embed_model, allow_dangerous_deserialization=True)
            try:
                self.vector_store.add_documents(chunks)
            except Exception as e:
                # fallback: rebuild
                print("Could not add documents to existing index, rebuilding:", e)
                self.vector_store = FAISS.from_documents(chunks, self.embed_model)
                if persist:
                    self.vector_store.save_local(str(self.index_dir))
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embed_model)
            if persist:
                os.makedirs(self.index_dir, exist_ok=True)
                self.vector_store.save_local(str(self.index_dir))

        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 50})
        print("Ingest done; FAISS persisted:", persist)

        # create ensemble retriever (use self.*)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever], 
            weights=[0.5, 0.5]
        )
        print("Document ingested. Hybrid Retriever (BM25 + FAISS) ready.")

    def _mmr_select(self, query_embedding, candidate_embeddings, candidate_ids, top_k=3, lambda_param=0.5):
        # simple greedy MMR: pick one by one
        if len(candidate_ids) == 0:
            return []
        selected = []
        selected_ids = []
        similarities = cosine_similarity(candidate_embeddings, query_embedding.reshape(1, -1)).flatten()
        idxs = list(range(len(candidate_ids)))
        first = int(np.argmax(similarities))
        selected.append(candidate_embeddings[first]); selected_ids.append(candidate_ids[first])
        idxs.remove(first)
        while len(selected_ids) < top_k and idxs:
            mmr_scores = []
            for j in idxs:
                sim_to_query = similarities[j]
                if len(selected) > 0:
                    sim_to_selected = max(cosine_similarity(candidate_embeddings[j].reshape(1,-1), 
                                                            np.vstack(selected)).flatten())
                else:
                    sim_to_selected = 0
                mmr_scores.append(lambda_param * sim_to_query - (1-lambda_param) * sim_to_selected)
            best_idx = int(np.argmax(mmr_scores))
            best = idxs[best_idx]
            selected.append(candidate_embeddings[best]); selected_ids.append(candidate_ids[best])
            idxs.remove(best)
        return selected_ids
    
    def format_qwen_prompt(self, system_msg, user_msg):
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def generate_global_summary(self):
        if not self.full_text:
            return "No document loaded."

        truncated_text = self.full_text[:25000] 

        system_prompt = "You are a legal expert. Summarize the following Terms of Service. Focus on user rights and data privacy."
        user_prompt = f"Summarize the following {self.doc_type} for the service \"{self.service_name}\":\n\nDocument Text:\n{truncated_text}"

        formatted_prompt = self.format_qwen_prompt(system_prompt, user_prompt)

        output = self.llm(
            formatted_prompt,
            max_tokens=600, 
            temperature=0.0,
            stop=["<|im_end|>"],
            echo=False
        )
        return output['choices'][0]['text']

    def _calculate_coverage(self, answer, context_chunks):
        answer_lower = answer.lower()
    
        answer_clean = re.sub(r'\[[^\]]+\]', '', answer_lower)
        
        total_chars = len(answer_clean)
        if total_chars == 0:
            return 0.0
        
        covered_chars = 0
        
        for chunk in context_chunks:
            chunk_lower = chunk.page_content.lower()
            
            # Find longest common substrings
            matcher = SequenceMatcher(None, answer_clean, chunk_lower)
            for match in matcher.get_matching_blocks():
                if match.size >= 20:  # At least 20 characters
                    covered_chars += match.size
        
        coverage = min(covered_chars / total_chars, 1.0)
        return coverage
    
    def _extract_quotes(self, answer):
        # Match text in quotes
        quote_patterns = [
            r'"([^"]{10,})"',  # Double quotes, min 10 chars
            r"'([^']{10,})'",  # Single quotes
        ]
        
        quotes = []
        for pattern in quote_patterns:
            quotes.extend(re.findall(pattern, answer))
        
        return quotes
    
    def _extract_citations(self, text):
        pattern = r'\[([^\]]+::[^\]]+)\]'
        matches = re.findall(pattern, text)
        return set(matches)


    def answer_question(self, query, top_k=3, rerank_pool=30, answer_threshold=0.2, require_quote=True, min_coverage=0.3):
        if not self.ensemble_retriever:
            return "Please upload a document first."

        # Use the ensemble retriever you already created
        candidates = self.ensemble_retriever.invoke(query)[:rerank_pool]
        
        if not candidates:
            return "No relevant document chunks found."

        # Ensure all candidates have chunk_id
        for i, doc in enumerate(candidates):
            if 'chunk_id' not in doc.metadata:
                doc.metadata['chunk_id'] = f"unknown::{i}"

        # Cross-encoder rerank
        pairs = [[query, c.page_content] for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Answerability threshold
        if scored[0][1] < answer_threshold:
            return "I cannot find a confident answer in the provided document."

        # Apply MMR for diversity
        top_candidates = [c for c, s in scored[:min(10, len(scored))]]  # Take top 10 for MMR
        cand_texts = [c.page_content for c in top_candidates]
        cand_embs = np.array(self.embed_model.embed_documents(cand_texts))
        query_emb = np.array(self.embed_model.embed_query(query))
        
        selected_ids = self._mmr_select(
            query_emb, cand_embs, 
            [c.metadata['chunk_id'] for c in top_candidates], 
            top_k=top_k
        )
        final_chunks = [c for c in top_candidates if c.metadata['chunk_id'] in selected_ids]

        # Build context with citations
        context = "\n\n".join([
            f"[{c.metadata['chunk_id']}|page:{c.metadata.get('page', 'n/a')}]\n{c.page_content[:2000]}" 
            for c in final_chunks
        ])

        system_prompt = "You are a legal expert. Answer strictly using only the context below. Quote clauses verbatim when referring to them and cite chunk ids."
        user_prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer (include explicit [chunk_id] citations):"
        formatted = self.format_qwen_prompt(system_prompt, user_prompt)

        output = self.llm(formatted, max_tokens=400, temperature=0.0, stop=["<|im_end|>"], echo=False)
        
        answer_text = output['choices'][0]['text']
        cited_chunk_ids = self._extract_citations(answer_text)
        coverage = self._calculate_coverage(answer_text, final_chunks)
        quotes = self._extract_quotes(answer_text)
        quality_flags = {
            "coverage": coverage,
            "has_quotes": len(quotes) > 0,
            "num_quotes": len(quotes),
            "passes_coverage_threshold": coverage >= min_coverage,
            "passes_quote_requirement": not require_quote or len(quotes) > 0
        }

        if require_quote and len(quotes) == 0:
            return {
                "answer": "Answer lacks direct quotes from source material. Please rephrase your question.",
                "sources": [],
                "quality": quality_flags,
                "warning": "No direct quotes found"
            }
    
        if coverage < min_coverage:
            quality_flags["warning"] = f"Low coverage ({coverage:.1%}), answer may contain hallucinations"
        
        cited_chunk_ids = self._extract_citations(answer_text)
        sources = []

        for chunk in final_chunks:
            chunk_id = chunk.metadata['chunk_id']
            if chunk_id in cited_chunk_ids or not cited_chunk_ids:
                sources.append({
                    "chunk_id": chunk_id,
                    "page": chunk.metadata.get('page', 'n/a'),
                    "snippet": chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content,
                    "full_text": chunk.page_content,
                    "source_file": chunk.metadata.get('source', 'unknown')
                })
        
        return {
        "answer": answer_text,
        "sources": sources,
        "cited_chunks": list(cited_chunk_ids),
        "quality": quality_flags,
        "quotes": quotes
        }