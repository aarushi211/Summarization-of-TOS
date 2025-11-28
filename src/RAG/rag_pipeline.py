import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama

class TOSAssistant:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path, 
            x_ctx = 8192,
            n_gpu_layers=-1,
            verbose=False
        )
        self.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_store = None
        self.full_text = ""
        self.doc_type = "Terms of Service" # Default
        self.service_name = "Unknown Service" #Default

    def ingest_doc(self, pdf_path):
        print(f'Ingesting {pdf_path}')
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        self.full_text = '\n'.join([doc.page_content for doc in documents])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        print("Document ingested and Vector Store created.")

    def format_llama3_prompt(self, system_msg, user_msg):
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    def generate_global_summary(self):
        if not self.full_text:
            return "No document loaded."

        truncated_text = self.full_text[:30000] 

        # 1. The Fine-Tuning Prompt (Recreating the magic)
        system_prompt = "You are a legal expert. Summarize the following Terms of Service. Focus on user rights and data privacy."
        user_prompt = f"Summarize the following {self.doc_type} for the service \"{self.service_name}\":\n\nDocument Text:\n{truncated_text}"

        formatted_prompt = self.format_llama3_prompt(system_prompt, user_prompt)

        # 2. Generate
        output = self.llm(
            formatted_prompt,
            max_tokens=500, # The abstractive summary length
            temperature=0.1,
            stop=["<|eot_id|>"],
            echo=False
        )
        return output['choices'][0]['text']

    def answer_question(self, query):
        if not self.vector_store:
            return "Please upload a document first."

        # 1. Retrieve Top 3 Relevant Chunks
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # 2. The RAG Prompt
        system_prompt = "You are a legal expert assistant. Answer the user's question based strictly on the provided context clauses."
        user_prompt = f"Question: {query}\n\nContext Clauses:\n{context}\n\nAnswer:"

        formatted_prompt = self.format_llama3_prompt(system_prompt, user_prompt)

        # 3. Generate
        output = self.llm(
            formatted_prompt,
            max_tokens=300,
            temperature=0.2,
            stop=["<|eot_id|>"],
            echo=False
        )
        return output['choices'][0]['text']