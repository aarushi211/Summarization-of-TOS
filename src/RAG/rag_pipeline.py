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
            self.vector_store = FAISS.load_local(str(self.index_dir), self.embed_model, allow_dangerous_deserialization=True)
            try:
                self.vector_store.add_documents(chunks)
            except Exception as e:
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

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever], 
            weights=[0.5, 0.5]
        )
        print("Document ingested. Hybrid Retriever (BM25 + FAISS) ready.")

    def _mmr_select(self, query_embedding, candidate_embeddings, candidate_chunks, 
                top_k=3, lambda_param=0.5, use_keyword_boost=True):
        
        if len(candidate_chunks) == 0:
            return []
        
        selected = []
        selected_ids = []
        
        similarities = cosine_similarity(candidate_embeddings, query_embedding.reshape(1, -1)).flatten()
        
        if use_keyword_boost:
            query_text = ""  
            
            keyword_scores = np.array([
                self._calculate_keyword_relevance_score(chunk.page_content, query_text)
                for chunk in candidate_chunks
            ])
            
            similarities = similarities + (keyword_scores * 0.3) 
        
        idxs = list(range(len(candidate_chunks)))
        
        first = int(np.argmax(similarities))
        selected.append(candidate_embeddings[first])
        selected_ids.append(candidate_chunks[first].metadata['chunk_id'])
        idxs.remove(first)
        
        while len(selected_ids) < top_k and idxs:
            mmr_scores = []
            
            for j in idxs:
                sim_to_query = similarities[j]
                
                if len(selected) > 0:
                    sim_to_selected = max(
                        cosine_similarity(
                            candidate_embeddings[j].reshape(1, -1), 
                            np.vstack(selected)
                        ).flatten()
                    )
                else:
                    sim_to_selected = 0
                
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr_score)
            
            best_idx = int(np.argmax(mmr_scores))
            best = idxs[best_idx]
            
            selected.append(candidate_embeddings[best])
            selected_ids.append(candidate_chunks[best].metadata['chunk_id'])
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

        # truncated_text = self.full_text[:25000] 
        if len(self.full_text) > 25000:
            truncated_text = self.full_text[:12500] + "\n...[SNIP]...\n" + self.full_text[-12500:]
        else:
            truncated_text = self.full_text

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

    def _calculate_coverage(self, answer, context_chunks, min_match_size=10):
        answer_lower = answer.lower()
        answer_clean = re.sub(r'\[[^\]]+\]', '', answer_lower).strip()
        
        answer_words = re.findall(r'\b\w+\b', answer_clean)
        if len(answer_words) == 0:
            return 0.0
        
        matched_words = set()
        combined_chunks = ' '.join([c.page_content.lower() for c in context_chunks])
        
        for i in range(len(answer_words) - 2):
            trigram = ' '.join(answer_words[i:i+3])
            if trigram in combined_chunks:
                matched_words.add(i)
                matched_words.add(i+1)
                matched_words.add(i+2)
        
        coverage = len(matched_words) / len(answer_words)
        return coverage
    
    def _extract_quotes(self, answer):
        quotes = re.findall(r'"([^"]+)"', answer) + re.findall(r"'([^']+)'", answer)
        extra = [m[0] for m in re.findall(r'"([^"]+)"\s*\[([^\]]+)\]', answer)]
        return list(dict.fromkeys(quotes + extra))
    
    def _extract_citations(self, text):
        pattern = r'\[([^\]]+::[^\]]+)\]'
        matches = re.findall(pattern, text)
        return set(matches)
    
    def _has_verbatim_content(self, answer, context_chunks, min_length=15):
        answer_lower = re.sub(r'\[[^\]]+\]', '', answer.lower()).strip()
        
        for chunk in context_chunks:
            chunk_lower = chunk.page_content.lower()
            
            for i in range(len(answer_lower) - min_length + 1):
                snippet = answer_lower[i:i+min_length]
                if snippet in chunk_lower:
                    return True
        
        return False
    
    def _extract_verbatim_quotes(self, answer_text, final_chunks, min_length=40):  
        quotes_with_sources = []
        
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < min_length:
                continue
            
            for chunk in final_chunks:
                chunk_text = chunk.page_content
                chunk_id = chunk.metadata['chunk_id']
                
                matcher = SequenceMatcher(None, sentence_clean.lower(), chunk_text.lower())
                match = matcher.find_longest_match(0, len(sentence_clean), 0, len(chunk_text))
                
                if match.size >= min_length:  
                    start_in_chunk = match.b
                    end_in_chunk = match.b + match.size
                    verbatim_text = chunk_text[start_in_chunk:end_in_chunk].strip()
                    
                    if len(verbatim_text) >= min_length:
                        quotes_with_sources.append({
                            "quote": verbatim_text,
                            "chunk_id": chunk_id,
                            "page": chunk.metadata.get('page', 'n/a')
                        })
                        break  
        
        return quotes_with_sources

    def _format_answer_with_quotes(self, answer_text, quotes_with_sources):
        if not quotes_with_sources:
            return answer_text
        
        enhanced = answer_text
        
        for item in quotes_with_sources:
            quote = item['quote']
            chunk_id = item['chunk_id']
            
            pattern = re.escape(quote)
            
            def add_citation(match):
                return f'"{match.group(0)}" [{chunk_id}]'
            
            enhanced = re.sub(pattern, add_citation, enhanced, count=1, flags=re.IGNORECASE)
        
        return enhanced
    
    def _calculate_keyword_relevance_score(self, text, query):
        """
        Updated to fix the 'cause of action' penalty bug.
        """
        text_lower = text.lower()
        query_lower = query.lower()
        
        BOOST_KEYWORDS = [
            'terminate', 'termination', 'suspend', 'suspension', 
            'breach', 'violate', 'violation', 'remove', 'ban', 
            'liability', 'liable', 'dispute', 'arbitration', 
            'refund', 'cancel', 'payment', 'fee', 'charge',
            'privacy', 'data', 'collect', 'share', 'third-party',
            'cause of action', 'jurisdiction', 'law'  # Moved 'cause of action' here
        ]
        
        # Only penalize truly generic styling/software words
        PENALIZE_KEYWORDS = [
            'downloadable', 'royalty-free', 'non-exclusive', 'sublicensable',
            'format', 'layout', 'design' 
        ]
        
        score = 0.0
        
        for keyword in BOOST_KEYWORDS:
            if keyword in text_lower:
                score += 0.1
                if keyword in query_lower:
                    score += 0.2 # Strong boost if keyword matches query
        
        for keyword in PENALIZE_KEYWORDS:
            if keyword in text_lower:
                score -= 0.1
        
        return score

    def answer_question(self, query, top_k=3, rerank_pool=30, answer_threshold=0.2, 
                   min_coverage=0.3, auto_quote=True, min_quote_length=40):
        if not self.ensemble_retriever:
            return {"answer": "Please upload a document first.", "sources": []}

        candidates = self.ensemble_retriever.invoke(query)[:rerank_pool]
        if not candidates:
            return {"answer": "No relevant document chunks found.", "sources": []}

        for i, doc in enumerate(candidates):
            if 'chunk_id' not in doc.metadata:
                doc.metadata['chunk_id'] = f"unknown::{i}"

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        pairs = [[query, c.page_content] for c in candidates]
        raw_scores = self.cross_encoder.predict(pairs)
        prob_scores = sigmoid(raw_scores)
        
        adjusted_scores = []
        for i, score in enumerate(prob_scores):
            keyword_score = self._calculate_keyword_relevance_score(candidates[i].page_content, query)
            # Now a 0.2 boost is actually significant relative to a 0.7 probability
            adjusted_scores.append(score + keyword_score)
        
        scored = sorted(
            zip(candidates, adjusted_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if scored[0][1] < answer_threshold:
            return {"answer": "I cannot find a confident answer in the provided document.", "sources": []}

        top_candidates = [c for c, s in scored[:min(10, len(scored))]]
        cand_texts = [c.page_content for c in top_candidates]
        cand_embs = np.array(self.embed_model.embed_documents(cand_texts))
        query_emb = np.array(self.embed_model.embed_query(query))
        
        selected_ids = self._mmr_select_with_keywords(
            query, 
            query_emb, 
            cand_embs, 
            top_candidates,  
            top_k=top_k
        )
        
        final_chunks = [c for c in top_candidates if c.metadata['chunk_id'] in selected_ids]
        
        print("---- FINAL CHUNKS WITH KEYWORD SCORES ----")
        for c in final_chunks:
            keyword_score = self._calculate_keyword_relevance_score(c.page_content, query)
            print(f"CHUNK_ID: {c.metadata.get('chunk_id')}")
            print(f"PAGE: {c.metadata.get('page','n/a')}")
            print(f"KEYWORD_SCORE: {keyword_score:.3f}")
            print(c.page_content[:500])
            print("-----")
        print("---- END CHUNKS ----")

        context = "\n\n".join([
            f"[{c.metadata['chunk_id']}]\n{c.page_content}"
            for c in final_chunks
        ])

        system_prompt = (
            "You are a helpful legal assistant. Answer the user's question using ONLY the provided Context chunks.\n"
            "Format your answer as follows:\n"
            "1. ANSWER: A direct answer to the question.\n"
            "2. EVIDENCE: Exact quotes from the context supporting the answer, followed by the [chunk_id].\n"
            "If the answer is truly not present, reply: NOT_IN_DOCUMENT"
        )
        
        # We give it a fake example so it sees the pattern we want
        example_interaction = (
            "Context:\n"
            "[demo::chunk_1]\nUsers must be at least 18 years old to use the Service.\n"
            "[demo::chunk_2]\nWe collect your email address.\n\n"
            "Question: What is the age requirement?\n"
            "1. ANSWER: You must be 18 years or older.\n"
            "2. EVIDENCE: \"Users must be at least 18 years old\" [demo::chunk_1]"
        )

        user_prompt = (
            f"Example:\n{example_interaction}\n\n"
            f"Real Task:\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )

        formatted = self.format_qwen_prompt(system_prompt, user_prompt)
        output = self.llm(formatted, max_tokens=400, temperature=0.0, stop=["<|im_end|>"], echo=False)
        
        answer_text = output['choices'][0]['text'].strip()
        
        if auto_quote:
            verbatim_quotes = self._extract_verbatim_quotes(
                answer_text, 
                final_chunks, 
                min_length=min_quote_length
            )
            answer_text = self._format_answer_with_quotes(answer_text, verbatim_quotes)
        
        cited_chunk_ids = self._extract_citations(answer_text)
        quotes = self._extract_quotes(answer_text)
        coverage = self._calculate_coverage(answer_text, final_chunks, min_match_size=10)
        
        quality_flags = {
            "coverage": coverage,
            "has_quotes": len(quotes) > 0,
            "num_quotes": len(quotes),
            "has_citations": len(cited_chunk_ids) > 0,
            "verbatim_snippets_found": len(verbatim_quotes) if auto_quote else 0,
            "min_quote_length": min_quote_length
        }
        
        if coverage < min_coverage:
            quality_flags["warning"] = f"Coverage {coverage:.1%} - verify answer carefully"
        
        sources = []
        for chunk in final_chunks:
            chunk_id = chunk.metadata['chunk_id']
            keyword_score = self._calculate_keyword_relevance_score(chunk.page_content, query)
            
            sources.append({
                "chunk_id": chunk_id,
                "page": chunk.metadata.get('page', 'n/a'),
                "snippet": chunk.page_content[:400] + "..." if len(chunk.page_content) > 400 else chunk.page_content,
                "full_text": chunk.page_content,
                "source_file": chunk.metadata.get('source', 'unknown'),
                "was_cited": chunk_id in cited_chunk_ids,
                "keyword_relevance": keyword_score  # Add this for UI to show
            })
        
        return {
            "answer": answer_text,
            "sources": sources,
            "cited_chunks": list(cited_chunk_ids),
            "quality": quality_flags,
            "quotes": quotes if auto_quote else self._extract_quotes(answer_text)
        }

    def _mmr_select_with_keywords(self, query_text, query_embedding, candidate_embeddings, 
                                candidate_chunks, top_k=3, lambda_param=0.5):
        if len(candidate_chunks) == 0:
            return []
        
        selected = []
        selected_ids = []
        
        similarities = cosine_similarity(candidate_embeddings, query_embedding.reshape(1, -1)).flatten()
        
        keyword_scores = np.array([
            self._calculate_keyword_relevance_score(chunk.page_content, query_text)
            for chunk in candidate_chunks
        ])
        
        combined_similarities = similarities + (keyword_scores * 0.3)
        
        idxs = list(range(len(candidate_chunks)))
        
        first = int(np.argmax(combined_similarities))
        selected.append(candidate_embeddings[first])
        selected_ids.append(candidate_chunks[first].metadata['chunk_id'])
        idxs.remove(first)
        
        while len(selected_ids) < top_k and idxs:
            mmr_scores = []
            
            for j in idxs:
                sim_to_query = combined_similarities[j]  # Use keyword-enhanced similarity
                
                if len(selected) > 0:
                    sim_to_selected = max(
                        cosine_similarity(
                            candidate_embeddings[j].reshape(1, -1), 
                            np.vstack(selected)
                        ).flatten()
                    )
                else:
                    sim_to_selected = 0
                
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr_score)
            
            best_idx = int(np.argmax(mmr_scores))
            best = idxs[best_idx]
            
            selected.append(candidate_embeddings[best])
            selected_ids.append(candidate_chunks[best].metadata['chunk_id'])
            idxs.remove(best)
        
        return selected_ids