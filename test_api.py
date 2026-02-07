import requests
import json
from langchain_community.document_loaders import PyPDFLoader

# URL of your running Docker container
url = "http://localhost:8000/summarize"

# Path to a sample TOS PDF on your laptop
PDF_PATH = "Terms of Service Youtube.pdf"  

def extract_text_from_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # Combine all pages into one long string
    full_text = "\n".join([doc.page_content for doc in documents])
    print(f"Extracted {len(full_text)} characters.")
    return full_text

# 1. Extract Text
try:
    tos_text = extract_text_from_pdf(PDF_PATH)
except Exception as e:
    print(f"Error reading PDF: {e}")
    exit()

# 2. Send to API
payload = {
    "text": tos_text,
    "max_length": 300,  # Adjust summary length
    "min_length": 50
}

headers = {"Content-Type": "application/json"}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("\nSUCCESS! Summary:")
        print("-" * 40)
        print(response.json()["summary"])
        print("-" * 40)
    else:
        print(f"\nError {response.status_code}:")
        print(response.text)
        
except Exception as e:
    print(f"\nConnection failed: {e}")
    print("Is your Docker container running?")