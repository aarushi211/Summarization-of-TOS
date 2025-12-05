import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from contextlib import asynccontextmanager
from pathlib import Path

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / 'model_artifact' / 'qwen-merged-q4_k_m.gguf' 

# Global variable
llm = None

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 512
    min_length: int = 50
    temperature: float = 0.2

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the GGUF model efficiently on startup.
    """
    global llm
    print(f"🔄 Loading GGUF Model from {MODEL_PATH}...")
    
    if not MODEL_PATH.exists():
        print(f"❌ Critical Error: Model file not found at {MODEL_PATH}")
        # We don't raise here to allow the server to start, but endpoints will fail 
    else:
        try:
            # Initialize Llama.cpp model
            # n_ctx=8192: Matches Qwen's context capabilities
            # n_gpu_layers=0: Force CPU mode for Cloud Run (set to -1 if you have a GPU)
            # verbose=False: Keeps logs clean
            llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=8192,
                n_gpu_layers=0, 
                verbose=False
            )
            print("✅ Qwen GGUF Model loaded successfully!")
        except Exception as e:
            print(f"❌ Critical Error initializing Llama: {e}")

    yield
    
    # Cleanup
    print("Cleaning up resources...")
    if llm:
        del llm

app = FastAPI(title="TOS Summarizer API (Qwen GGUF)", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {
        "status": "ready" if llm is not None else "error",
        "model_type": "Qwen-1.5B-GGUF"
    }

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not ready or failed to load.")

    try:
        # 1. Construct Prompt (Chat Template)
        # Qwen instruct models expect ChatML format. Llama.cpp handles this via
        # high-level chat completion if the model file contains the template metadata.
        messages = [
            {"role": "system", "content": "You are a legal expert. Summarize the following Terms of Service document concisely."},
            {"role": "user", "content": f"Document Text:\n{request.text[:25000]}"} # Truncate input for safety
        ]

        # 2. Generate
        # create_chat_completion handles tokenization and generation internally
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_length,
            temperature=request.temperature,
            top_p=0.95,
            stream=False
        )

        # 3. Extract Response
        summary = output['choices'][0]['message']['content']
        
        return {"summary": summary.strip()}

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))