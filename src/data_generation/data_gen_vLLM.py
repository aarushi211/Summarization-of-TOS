import os
import pandas as pd
from vllm import LLM, SamplingParams
import glob
from tqdm import tqdm
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'Original'
OUTPUT_PATH = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'synthetic_dataset.csv'
MODEL_ID = 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'
BATCH_SIZE = 500


# Loading processed files to avoid processing them again
processed_files = set()
file_exists = os.path.exists(OUTPUT_PATH)
if file_exists:
    try:
        existing_df = pd.read_csv(OUTPUT_PATH)
        processed_files = set(existing_df['filename'].unique())
        print(f'Loaded {len(existing_df)} existing summaries from checkpoint')
    except pd.errors.EmptyDataError:
        print('Checkpoint file exists but is empty')
else:
    print('No checkpoint file found. Starting fresh')


# Load files to process
all_files = glob.glob(f'{INPUT_DIR}/*.txt')
files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]
print(f'Total files found: {len(all_files)}')
print(f'Files remaining to process: {len(files_to_process)}')

if not files_to_process:
    print('All files already processed. Exiting script.')
    exit()


# Read all data into memory once
all_input_data = []
for f in tqdm(files_to_process, desc='Preparing Inputs'):
    filename_full = os.path.basename(f)
    parts = filename_full.split('_')
    service_name = parts[0]
    doc_type = '_'.join(parts[1:]).replace('.txt', '').replace('.TXT', '')

    with open(f, 'r', encoding='utf-8') as file:
        text_content = file.read()

    truncated_text = text_content[:8000]

    prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a legal expert who writes clear, simplified, abstractive summaries of policy documents.
    Your goal is to help users understand the key rules without legal jargon.

    Instructions:
    1. Summarize the document in 200–350 words.
    2. Use a single, fluent paragraph (no headings or bullets).
    3. Focus exclusively on the following areas: user responsibilities and liabilities, privacy rights, data usage/sharing/storage, payment/account rules, and termination clauses.
    4. Keep the tone simplified, factual, and neutral.
    5. **CRITICAL: Only include facts explicitly supported by the provided text.**
    6. **WARNING: If a required topic (e.g., refunds) is not covered in the text, you must omit it entirely and do not speculate or invent content.**
    7. Do not copy sentences; rewrite ideas in your own words.
    8. Do not add any conversational phrases, introductory sentences, or concluding remarks not part of the summary.
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Summarize the following {doc_type} for the service "{service_name}":

    Document Text:
    {truncated_text}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    [SUMMARY START]
    '''

    all_input_data.append({
        'prompt': prompt,
        'filename': filename_full,
        'service_name': service_name,
        'doc_type': doc_type,
        'original_text': text_content
    })


# Inference and batch saving 
llm = LLM(model=MODEL_ID, quantization='awq', dtype='half', gpu_memory_utilization=0.85, max_model_len=8192)
sampling_params = SamplingParams(temperature=0.1, max_tokens=500)

total_remaining = len(all_input_data)
for i in tqdm(range(0, total_remaining, BATCH_SIZE), desc='Processing Batches'):
    # slicing data for current batch
    batch_data = all_input_data[i:i + BATCH_SIZE]
    batch_prompts = [item['prompt'] for item in batch_data]
    print(f'\n--- Batch {i // BATCH_SIZE + 1}: Generating {len(batch_prompts)} documents ---')

    # Generation
    outputs = llm.generate(batch_prompts, sampling_params)
    results = [output.outputs[0].text for output in outputs]

    # Temporary df for this batch
    batch_df = pd.DataFrame({
        'filename': [item['filename'] for item in batch_data],
        'service_name': [item['service_name'] for item in batch_data],
        'doc_type': [item['doc_type'] for item in batch_data],
        'original_text': [item['original_text'] for item in batch_data],
        'summary': results
    })

    # Checkpoint saving
    batch_df.to_csv(
        OUTPUT_PATH,
        mode='a',
        header=(not file_exists),
        index=False
    )
    file_exists = True

print(f'\n✅ All batches processed. Final data saved to: {OUTPUT_PATH}')
