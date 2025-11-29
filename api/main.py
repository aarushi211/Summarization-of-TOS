import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
from pathlib import Path

app = FastAPI(title='TOS Summarizer API')

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / 'data' / 'model_artifact' / 'led_onnx_quantized'

# Global variables
model = None
tokenizer = None

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 300
    min_length: int = 50

@app.lifespan('startup')
async def load_model():
    global model, tokenizer
    print(f"Loading ONNX Model from {MODEL_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, file_name = 'model_quantized.onnx')
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get('/health')
def health_check():
    return {"status": "active", "model": "Longformer-LED-ONNX"}

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            max_length=4096, 
            truncation=True, 
            padding=True
        )
        
        summary_ids = model.generate(
            inputs.input_ids, 
            max_length=request.max_length, 
            min_length=request.min_length,
            num_beams=2, 
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))