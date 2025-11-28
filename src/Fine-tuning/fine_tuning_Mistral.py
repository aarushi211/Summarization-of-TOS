import torch
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from pathlib import Path

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config = bnb_config,
    device_map = 'auto',
    use_cache = False
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = 'right'

# LoRA Config
lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 32,
    lora_dropout=0.05,
    bias = 'none',
    task_type = 'CAUSAL_LM',
    target_modules = ['q_proj', 'v_proj']
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def format_synthetic_dataset(data):
    system_prompt = 'You are a legal expert who writes clear, simplified, abstractive summaries of policy documents.Your goal is to help users understand the key rules without legal jargon.'
    instruction = f"Summarize the following {data['doc_type']} for the service \"{data['service_name']}\":"
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n\nDocument Text:\n{data['original_text']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{data['summary']}<|eot_id|>"
    )
    return {"text": text}

SCRIPT_DIR = Path(__file__).resolve().parent
df_path = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'synthetic_dataset.csv'
OUTPUT_PATH = SCRIPT_DIR.parent.parent / 'data' / 'model_artifact'

df = pd.read_csv(df_path)
subset_df = df.sample(n=2000, random_state=42)
train_dataset = Dataset.from_pandas(subset_df)
remaining_df = df.drop(subset_df.index)
val_dataset = Dataset.from_pandas(remaining_df.sample(n=200, random_state=42))

dataset = DatasetDict({
    'train': train_dataset,
    'test': val_dataset
})

print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(val_dataset)} samples")

synthetic_dataset = dataset['train'].map(format_synthetic_dataset)
synthetic_eval_dataset = dataset['test'].map(format_synthetic_dataset)

sft_config = SFTConfig(
    output_dir = f'{OUTPUT_PATH}/checkpoints',
    max_length = 1024,
    dataset_text_field = 'text',
    packing = False,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=50, 
    eval_strategy='steps',
    eval_steps=100, 
    learning_rate=2e-4,
    fp16=True,
    group_by_length=True,
    report_to='none'
)

trainer = SFTTrainer(
    model=model,
    train_dataset=synthetic_dataset,
    eval_dataset=synthetic_eval_dataset,     
    peft_config=lora_config,
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()
trainer.save_model(f'{OUTPUT_PATH}/final_model')