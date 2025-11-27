import os
import subprocess
import time
import pandas as pd
from tqdm.notebook import tqdm
import time
from pathlib import Path
import glob

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'Original'
OUTPUT_PATH = SCRIPT_DIR.parent.parent / 'data' / 'Dataset' / 'synthetic_dataset_Ollama.csv'
MODEL_PATH = SCRIPT_DIR.parent.parent / 'data' / 'Ollama_Model'

# Check if model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)
# Set environment variables for the model
os.environ["OLLAMA_MODELS"] = MODEL_PATH

OLLAMA_MODEL = "llama3.1:8b"
CTX_SIZE = 20000

# Check if Ollama Server is running
def is_ollama_server_running():
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ollama.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:  # Linux/macOS
            result = subprocess.run(['pgrep', '-f', 'ollama'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return result.returncode == 0  # If there's a running process, pgrep returns 0
    except Exception as e:
        print(f"Error checking Ollama server status: {e}")
        return False
    
# Function to check if the model has been downloaded
def is_model_downloaded(model_name):
    model_path = os.path.join(MODEL_PATH, model_name)
    return os.path.exists(model_path)

# Check if the Ollama server is running
if not is_ollama_server_running():
    print("Ollama server not running. Starting Ollama server...")
    # Start Ollama server in the background
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)  # Give time for the server to start (can be adjusted as needed)
else:
    print("Ollama server is already running.")

# Check if the model is already downloaded
if not is_model_downloaded(OLLAMA_MODEL):
    print(f"Model {OLLAMA_MODEL} not found. Pulling the model...")
    subprocess.run(["ollama", "pull", OLLAMA_MODEL])  # Pull the model if not present
else:
    print(f"Model {OLLAMA_MODEL} is already downloaded.")


import ollama

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

batch_data = []
BATCH_SIZE = 10

for f in tqdm(files_to_process, desc="Generating"):
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
            {text_content[:60000]}
            '''
        
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt},
        ], options={'num_ctx': CTX_SIZE}) 

        summary = response['message']['content']

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
            batch_data = []

    except Exception as e:
        print(f"⚠️ Error on {filename_full}: {e}")
        # If Ollama crashes (OOM), we might need to restart loop manually
        time.sleep(1)

# Final Save
if batch_data:
    df = pd.DataFrame(batch_data)
    use_header = not os.path.exists(OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, mode='a', header=use_header, index=False)

print("✅ Batch complete.")