import torch
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
DATA_PATH = "/project2/neiswang_1520/gamelen/TOS/synthetic_dataset.csv"
OUTPUT_DIR = "/project2/neiswang_1520/gamelen/TOS/model_artifact_qwen"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False 
)

# 3. LoRA (Standard)
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 4. Formatting Function (Qwen Chat Template)
def format_synthetic_dataset(example):
    messages = [
        {"role": "system", "content": "You are a legal expert. Summarize the following Terms of Service. Focus on user rights and data privacy."},
        {"role": "user", "content": f"Service: {example['service_name']}\nDoc Type: {example['doc_type']}\n\nText:\n{example['original_text'][:20000]}"},
        {"role": "assistant", "content": example['summary']}
    ]
    return {"text": tokenizer.format_synthetic_dataset(messages, tokenize=False)}

print("Loading and splitting dataset...")
df = pd.read_csv(DATA_PATH)
train_df = df.sample(n=8500, random_state=42)
remaining_df = df.drop(train_df.index)
val_df = remaining_df.sample(n=500, random_state=42)
test_df = remaining_df.drop(val_df.index)
test_df.to_csv("test_dataset.csv", index=False)

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(val_df)
})

print(f"Training on {len(train_df)} samples")
print(f"Validating on {len(val_df)} samples")
print(f"Test set (saved to CSV) has {len(test_df)} samples")

print("Formatting dataset...")
synthetic_dataset = dataset['train'].map(format_synthetic_dataset)
synthetic_eval_dataset = dataset['test'].map(format_synthetic_dataset)

# 5. Trainer
sft_config = SFTConfig(
    output_dir=f'{OUTPUT_DIR}/checkpoints',
    max_length=4096,
    dataset_text_field='text', 
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100, 
    eval_strategy='steps',
    eval_steps=100, 
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
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

print("Starting Training...")
trainer.train()

print("Saving Final Model...")
trainer.save_model(f'{OUTPUT_DIR}/qwen_model')
print("Done!")