import os
import time
import pandas as pd
import google.generativeai as genai
from google.colab import drive
from tqdm.notebook import tqdm
from dotenv import load_dotenv
from pathlib import Path
import glob

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = 'gemini-2.5-flash'

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'Original'
OUTPUT_PATH = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'synthetic_dataset_gemini.csv'
BATCH_SIZE = 10

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

all_files = glob.glob(f'{INPUT_DIR}/*.txt')
files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]
print(f'Total files found: {len(all_files)}')
print(f'Files remaining to process: {len(files_to_process)}')

if not files_to_process:
    print('All files already processed. Exiting script.')
    exit()

all_input_data = []
for f in tqdm(files_to_process, desc='Preparing Inputs'):
    try:
        filename_full = os.path.basename(f)
        parts = filename_full.split('_')
        service_name = parts[0]
        doc_type = '_'.join(parts[1:]).replace('.txt', '').replace('.TXT', '')

        with open(f, 'r', encoding='utf-8') as file:
            text_content = file.read()

            prompt = f'''
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

            Summarize the following {doc_type} for the service "{service_name}":

            Document Text:
            {text_content[:30000]}
            '''

            # Generate
            response = model.generate_content(prompt)

            if response.text:
                summary = response.text

                batch_data.append({
                    'filename': filename_full,
                    'service_name': service_name,
                    'doc_type': doc_type,
                    'original_text': text_content,
                    'reference_summary': summary
                })

            # Save Batch
            if len(batch_data) >= BATCH_SIZE:
                df = pd.DataFrame(batch_data)
                use_header = not os.path.exists(OUTPUT_PATH)
                df.to_csv(OUTPUT_PATH, mode='a', header=use_header, index=False)
                batch_data = [] # Reset

            # Rate Limit Sleep (Crucial for Free Tier)
            time.sleep(5)

    except Exception as e:
        if "429" in str(e):
            print(f"Daily Limit Reached on {filename_full}. Stop and resume tomorrow.")
            break
        else:
            print(f"Error on {filename_full}: {e}")
            time.sleep(2)

if batch_data:
    df = pd.DataFrame(batch_data)
    use_header = not os.path.exists(OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, mode='a', header=use_header, index=False)
    print("Final batch saved.")