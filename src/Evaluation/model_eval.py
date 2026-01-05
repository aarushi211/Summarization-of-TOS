import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import os

# --- CONFIGURATION ---
MODEL_PATH = '/project2/neiswang_1520/gamelen/TOS/model_artifact_qwen/qwen_merged'
TEST_DATA_PATH = '/project2/neiswang_1520/gamelen/TOS/test_dataset.csv'
OUTPUT_CSV = "/project2/neiswang_1520/gamelen/TOS/Qwen_evaluation.csv"

# --- LOAD METRICS ---
print("Loading metrics...")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# 1. Load Model & Tokenizer
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Prepare Data
print(f"Loading test data from {TEST_DATA_PATH}...")
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

df = pd.read_csv(TEST_DATA_PATH).dropna(subset=['original_text', 'summary'])

print(f"Evaluating on {len(df)} samples...")

# 3. Generation Loop
generated_summaries = []
reference_summaries = df['summary'].tolist()
original_texts = df['original_text'].tolist()
filenames = df['filename'].tolist()

def format_prompt(row):
    system_message = "You are a legal expert who writes clear, simplified, abstractive summaries of policy documents. Focus on user rights and data privacy."
    return (
        f"<|im_start|>{system_message}\n\n"
        f"Summarize the following {row['doc_type']} for the service \"{row['service_name']}\":"
        f"\n\n{row['original_text']}<|im_end|>"
    )

print("Generating summaries...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    prompt = format_prompt(row)
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=4096 
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Calculate input length to strip the prompt from output
    input_len = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    generated_summaries.append(generated_text)
    
print("Generation complete. Saving intermediate outputs...")
intermediate_df = pd.DataFrame({
    "filename": filenames,
    "original_text": original_texts,
    "reference_summary": reference_summaries,
    "generated_summary": generated_summaries
})
intermediate_df.to_csv("intermediate_generations.csv", index=False)
print("Intermediate results saved to 'intermediate_generations.csv'")

# 4. Compute Metrics (Aggregated & Per Sample)
print("Computing metrics...")

# df = pd.read_csv('intermediate_generations.csv')
initial_count = len(intermediate_df)
intermediate_df = intermediate_df.dropna(subset=['generated_summary', 'reference_summary'])
intermediate_df = intermediate_df[intermediate_df['generated_summary'].str.strip().astype(bool)]
filtered_count = len(intermediate_df)

if filtered_count < initial_count:
    print(f"Dropped {initial_count - filtered_count} rows with empty generated summaries.")

generated_summaries = intermediate_df['generated_summary'].tolist()
reference_summaries = intermediate_df['reference_summary'].tolist()
filenames = intermediate_df['filename'].tolist()
original_texts = intermediate_df['original_text'].tolist()

# A. ROUGE (use_aggregator=False returns list of scores per sample)
rouge_scores = rouge.compute(predictions=generated_summaries, references=reference_summaries, use_aggregator=False)

# B. BERTScore (computes for all, returns lists)
# Note: batch_size helps with speed on GPU
bert_scores = bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en", batch_size=10, device="cuda")

# C. METEOR & BLEU (Iterative for per-sample scores)
# Standard evaluate.load() computes corpus-level stats by default, so we loop for row-level granularity
meteor_scores_list = []
bleu_scores_list = []

print("Computing row-level METEOR and BLEU...")
for pred, ref in zip(generated_summaries, reference_summaries):
    # METEOR
    m_score = meteor.compute(predictions=[pred], references=[ref])['meteor']
    meteor_scores_list.append(m_score)
    
    # BLEU
    # Note: BLEU expects references to be a list of possible references (hence the list wrapping)
    b_score = bleu.compute(predictions=[pred], references=[[ref]])['bleu']
    bleu_scores_list.append(b_score)

# 5. Save Results
results_df = pd.DataFrame({
    "filename": filenames,
    "original_text": original_texts,
    "reference_summary": reference_summaries,
    "generated_summary": generated_summaries,
    # ROUGE (Values are usually floats 0-1)
    "rouge1": rouge_scores['rouge1'],
    "rouge2": rouge_scores['rouge2'],
    "rougeL": rouge_scores['rougeL'],
    # BERTScore
    "bert_precision": bert_scores['precision'],
    "bert_recall": bert_scores['recall'],
    "bert_f1": bert_scores['f1'],
    # METEOR & BLEU
    "meteor": meteor_scores_list,
    "bleu": bleu_scores_list
})

# Calculate Aggregate Averages for display
print("\n--- AGGREGATE RESULTS ---")
print(f"Mean ROUGE-1: {results_df['rouge1'].mean():.4f}")
print(f"Mean ROUGE-2: {results_df['rouge2'].mean():.4f}")
print(f"Mean ROUGE-L: {results_df['rougeL'].mean():.4f}")
print(f"Mean BERT-F1: {results_df['bert_f1'].mean():.4f}")
print(f"Mean METEOR:  {results_df['meteor'].mean():.4f}")
print(f"Mean BLEU:    {results_df['bleu'].mean():.4f}")

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Detailed results saved to {OUTPUT_CSV}")