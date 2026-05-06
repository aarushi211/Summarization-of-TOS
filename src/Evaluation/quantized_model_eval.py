import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import evaluate
import os

MODEL_TYPE = "QWEN" 

if MODEL_TYPE == "MISTRAL":
    MODEL_PATH = '/path/to/legal_mistral.Q4_K_M.gguf'
    OUTPUT_CSV = "Mistral_GGUF_evaluation.csv"
    MAX_CTX = 4096
    STOP_TOKENS = ["</s>"]
elif MODEL_TYPE == "QWEN":
    MODEL_PATH = '/path/to/legal_qwen.Q4_K_M.gguf'
    OUTPUT_CSV = "Qwen_GGUF_evaluation.csv"
    MAX_CTX = 8192 
    STOP_TOKENS = ["<|im_end|>"]

TEST_DATA_PATH = '/project2/neiswang_1520/gamelen/TOS/test_dataset.csv'

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# 1. Load GGUF Model
print(f"Loading GGUF model: {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CTX,      
    n_gpu_layers=-1,    
    verbose=False       
)

# 2. Prepare Data
print(f"Loading test data from {TEST_DATA_PATH}...")
df = pd.read_csv(TEST_DATA_PATH).dropna(subset=['original_text', 'summary'])
print(f"Evaluating on {len(df)} samples...")

# 3. Prompt Formatting Functions
def format_prompt_mistral(row):
    system_msg = "You are a legal expert who writes clear, simplified, abstractive summaries of policy documents. Focus on user rights and data privacy."
    instruction = f"Summarize the following {row['doc_type']} for the service \"{row['service_name']}\":\n\n{row['original_text']}"
    # Mistral v3 Template
    return f"<s>[INST] {system_msg}\n\n{instruction} [/INST]"

def format_prompt_qwen(row):
    system_msg = "You are a legal expert who writes clear, simplified, abstractive summaries of policy documents. Focus on user rights and data privacy."
    instruction = f"Summarize the following {row['doc_type']} for the service \"{row['service_name']}\":\n\n{row['original_text']}"
    # Qwen ChatML Template
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# Select formatter based on config
format_fn = format_prompt_mistral if MODEL_TYPE == "MISTRAL" else format_prompt_qwen

# 4. Generation Loop
generated_summaries = []
reference_summaries = df['summary'].tolist()
original_texts = df['original_text'].tolist()
filenames = df['filename'].tolist()

print("Generating summaries...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    prompt = format_fn(row)
    
    output = llm.create_completion(
        prompt,
        max_tokens=512,
        temperature=0.1,   
        top_p=0.95,
        stop=STOP_TOKENS,  
        echo=False         
    )
    
    # Extract text
    generated_text = output['choices'][0]['text'].strip()
    generated_summaries.append(generated_text)

# 5. Compute Metrics (Same as before)
print("Computing metrics...")

rouge_scores = rouge.compute(predictions=generated_summaries, references=reference_summaries, use_aggregator=False)
bert_scores = bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en", batch_size=10, device="cuda")

meteor_scores_list = []
bleu_scores_list = []

print("Computing row-level METEOR and BLEU...")
for pred, ref in zip(generated_summaries, reference_summaries):
    m_score = meteor.compute(predictions=[pred], references=[ref])['meteor']
    meteor_scores_list.append(m_score)
    b_score = bleu.compute(predictions=[pred], references=[[ref]])['bleu']
    bleu_scores_list.append(b_score)

# 6. Save Results
results_df = pd.DataFrame({
    "filename": filenames,
    "original_text": original_texts,
    "reference_summary": reference_summaries,
    "generated_summary": generated_summaries,
    "rouge1": rouge_scores['rouge1'],
    "rouge2": rouge_scores['rouge2'],
    "rougeL": rouge_scores['rougeL'],
    "bert_f1": bert_scores['f1'],
    "meteor": meteor_scores_list,
    "bleu": bleu_scores_list
})

print("\n--- AGGREGATE RESULTS ---")
print(f"Mean ROUGE-1: {results_df['rouge1'].mean():.4f}")
print(f"Mean BERT-F1: {results_df['bert_f1'].mean():.4f}")

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Detailed results saved to {OUTPUT_CSV}")