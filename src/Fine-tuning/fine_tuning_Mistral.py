import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
DATA_PATH = '/project2/neiswang_1520/gamelen/TOS/synthetic_dataset.csv'
OUTPUT_DIR = '/project2/neiswang_1520/gamelen/TOS/model_artifact_v2' 

# 1. QUANTIZATION CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 2. LOAD MODEL & TOKENIZER
print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' 

# 3. LORA CONFIG
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 4. DATA PREPARATION (Maintaining your 8500/500 Split)
df = pd.read_csv(DATA_PATH)
subset_df = df.sample(n=8500, random_state=42) 
remaining_df = df.drop(subset_df.index)
test_df = remaining_df.sample(n=500, random_state=42)

# Save the test set now so you have it for evaluation later!
test_df.to_csv("final_unseen_test_data.csv", index=False)

train_dataset = Dataset.from_pandas(subset_df)
val_dataset = Dataset.from_pandas(test_df)

def format_synthetic_dataset(data):
    system_message = (
        "You are a legal expert who writes clear, simplified, abstractive summaries of policy documents. "
        "Focus on user rights and data privacy."
    )
    
    text = (
        f"<s>[INST] {system_message}\n\n"
        f"Summarize the following {data['doc_type']} for the service \"{data['service_name']}\":\n\n"
        f"{data['original_text']} [/INST] "
        f"{data['summary']} </s>"
    )
    return {"text": text}

print("Formatting dataset...")
synthetic_dataset = train_dataset.map(format_synthetic_dataset)
synthetic_eval_dataset = val_dataset.map(format_synthetic_dataset)

# 5. TRAINING CONFIG
sft_config = SFTConfig(
    output_dir=f'{OUTPUT_DIR}/checkpoints',
    max_seq_length=4096,
    dataset_text_field='text',
    packing=False,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100, 
    eval_strategy='steps',
    eval_steps=100, 
    learning_rate=2e-4,
    bf16=True,
    report_to='none'
)

# 6. TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=synthetic_dataset,
    eval_dataset=synthetic_eval_dataset,
    peft_config=lora_config,
    args=sft_config,
)

print("Starting Training...")
trainer.train()

trainer.save_model(f'{OUTPUT_DIR}/legal_mistral_adapter')
print("Done! Model saved to legal_mistral_adapter")