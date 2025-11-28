import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

BASE_MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = SCRIPT_DIR.parent.parent / 'data' / 'model_artifact' / 'final_model'
MERGED_PATH = SCRIPT_DIR.parent.parent / 'data' / 'model_artifact' / 'Mistral_merged'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Loading base model in fp16
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("2. Loading Adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="auto")

print("3. Merging...")
model = model.merge_and_unload() # This combines weights physically

print("4. Saving Merged Model...")
model.save_pretrained(MERGED_PATH, safe_serialization=True)

print("5. Saving Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(MERGED_PATH)

print("Merge Complete!")