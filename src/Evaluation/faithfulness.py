import pandas as pd
import google.generativeai as genai
import json
from tqdm import tqdm
import time

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-3-flash-preview")

# Load CSV
df = pd.read_csv("/content/detailed_sample_metrics.csv")

df = df.sample(
    n=min(20, len(df)),
    random_state=42
)

results = []

PROMPT_TEMPLATE = """
You are a legal compliance reviewer evaluating a SUMMARY of a Terms of Service.

Important rules:
- A summary is NOT required to include all clauses from the original.
- Missing clauses are ONLY an error if the summary claims to be exhaustive.
- Do NOT penalize high-level abstraction or omission of details.
- Penalize ONLY:
  1. Statements that contradict the original
  2. Claims that are not supported by the original
  3. Meaning changes (e.g., may → must, added obligations, removed rights)

Original Terms of Service:
<<<ORIGINAL>>>

Generated Summary:
<<<SUMMARY>>>

Your task:
Evaluate whether EACH statement in the summary is supported by the original.

Respond ONLY in valid JSON:
{
  "hallucinations": true/false,
  "unsupported_claims": [brief list],
  "meaning_changes": [brief list],
  "faithfulness_score": number between 0 and 1
}
"""

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = PROMPT_TEMPLATE.replace("<<<ORIGINAL>>>", row["original_text"]) \
                            .replace("<<<SUMMARY>>>", row["generated_summary"])

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0}
        )

        text = response.text.strip()

        judge_output = json.loads(text)

        judge_output["index"] = idx
        results.append(judge_output)

        time.sleep(10)  

    except Exception as e:
        print(f"Error at index {idx}: {e}")