"""
TOS Summarizer — Answer Relevance Evaluation
=============================================
Measures whether the model's generated answer directly addresses the user's
question — regardless of whether it is factually grounded.

Metric
------
Answer Relevance = average score of how directly each answer addresses its question.
Score scale: 0.0 (completely irrelevant) → 1.0 (perfectly targeted)

For each question:
  1. Generate the full answer using the RAG engine.
  2. Ask the LLM judge: "On a scale 0–1, how well does this answer address
     the question? Was it specific, concise, and on-topic?"
  3. The judge returns a structured score + reasoning.

What this catches
-----------------
- Rambling answers that mention correct but unrelated facts
- Overly verbose summaries (especially common in 7B models)
- Valid abstentions ("I don't know") scored correctly as 0.0 since they don't
  answer the question — this cross-references against faithfulness to show
  safe abstentions are a feature, not a defect

Usage
-----
    python src/Evaluation/answer_relevance.py

Outputs
-------
    answer_relevance_results.csv  — per-question scores + judge reasoning
    Console summary table
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.RAG.schemas import SessionState


# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge
# ─────────────────────────────────────────────────────────────────────────────

class AnswerRelevanceJudge:
    """
    LLM judge for Answer Relevance.

    Evaluates the question ↔ answer relationship ONLY.
    Context is intentionally excluded from the prompt so the judge scores
    topical directness rather than factual groundedness (that's Faithfulness).

    Special case: If the answer is a safe abstention ("I do not have enough
    information…"), the score is 0.0 for relevance but this should be compared
    against Context Recall. If Context Recall is also 0 for that question,
    the abstention was correct behaviour.
    """

    SYSTEM_PROMPT = """\
You are evaluating the answer relevance of a legal document QA system.
Given a user question and an AI-generated answer, score how directly and
concisely the answer addresses the question.

Scoring rubric:
  1.0  - Directly answers the question, concise, no unnecessary tangents.
  0.75 - Answers the question but includes some unnecessary context.
  0.5  - Partially answers; misses a key aspect of the question.
  0.25 - Tangentially related but does not answer the question.
  0.0  - Completely off-topic, or is an abstention ("I don't know").

Note: An abstention answer ("I do not have enough information…") scores 0.0
for relevance regardless of whether it was the correct decision to abstain.
"""

    USER_TEMPLATE = """\
USER QUESTION:
{question}

AI-GENERATED ANSWER:
{answer}

How directly and concisely does this answer address the question?

Respond ONLY with valid JSON:
{{
  "score": <float 0.0–1.0>,
  "is_abstention": <true|false>,
  "reasoning": "one sentence explanation"
}}
"""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model  = model
        print(f"  Judge: Groq / {model}")

    def evaluate(self, question: str, answer: str) -> dict:
        prompt = self.USER_TEMPLATE.format(
            question=question[:500],
            answer=answer[:2000],
        )
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                return {
                    "score":         float(data.get("score", 0.0)),
                    "is_abstention": bool(data.get("is_abstention", False)),
                    "reasoning":     data.get("reasoning", ""),
                }
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Rate limit" in err_msg:
                    wait_time = 15 * (attempt + 1)
                    print(f"\n      [!] Rate limit. Waiting {wait_time}s…")
                    time.sleep(wait_time)
                elif attempt >= 2:
                    return {"score": 0.0, "is_abstention": False,
                            "reasoning": f"API error: {e}"}
                else:
                    time.sleep(3)

        return {"score": 0.0, "is_abstention": False,
                "reasoning": "Failed after 3 attempts."}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner():
    w = 62
    print("╔" + "═" * w + "╗")
    print("║" + "TOS Summarizer — Answer Relevance Evaluation".center(w) + "║")
    print("╠" + "═" * w + "╣")
    print("║" + f"  Date: {datetime.now():%Y-%m-%d %H:%M:%S}".ljust(w) + "║")
    print("╚" + "═" * w + "╝\n")

def _section(title: str):
    print("\n" + "─" * 62)
    print(f"  {title}")
    print("─" * 62)

def _safe_ns(filename: str, prefix: str = "ar") -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", filename)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return f"{prefix}-{safe[:55]}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(model: str):
    _banner()

    _section("Step 1: Loading Judge")
    if not os.getenv("GROQ_API_KEY"):
        print("  ✗ GROQ_API_KEY not set. Aborting.")
        return
    judge = AnswerRelevanceJudge(model=model)

    _section("Step 2: Loading RAG Engine")
    models_dir = PROJECT_ROOT / "models"
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print("  ✗ No GGUF model found in models/. Aborting.")
        return
    model_path   = str(gguf_files[0])
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    chroma_dir   = str(PROJECT_ROOT / "data" / "ar_chroma")

    from src.RAG.engine import TOSAssistant
    rag = TOSAssistant(
        model_path=model_path,
        pinecone_api_key=pinecone_key,
        use_local_vectorstore=not pinecone_key,
        data_dir=chroma_dir,
    )
    print("  RAG engine ready.")

    _section("Step 3: Loading Test Data")
    test_csv  = PROJECT_ROOT / "data" / "Test_data.csv"
    tos_dir   = PROJECT_ROOT / "data" / "TOS files"
    if not test_csv.exists():
        print(f"  ✗ {test_csv} not found"); return
    df        = pd.read_csv(test_csv)
    pdf_files = sorted(tos_dir.glob("*.pdf"))
    print(f"  {len(pdf_files)} docs | {len(df)} questions")

    _section("Step 4: Generating + Evaluating Answers")
    rows = []

    for pdf_path in pdf_files:
        doc_name = pdf_path.name
        doc_qs   = df[df["filename"] == doc_name]
        if doc_qs.empty:
            continue

        print(f"\n  📄 {doc_name} ({len(doc_qs)} questions)")

        state = SessionState(pinecone_namespace=_safe_ns(doc_name))
        row0  = doc_qs.iloc[0]
        state.service_name = str(row0.get("service_name", "Unknown"))
        state.doc_type     = str(row0.get("doc_type", "Document"))

        try:
            rag.ingest_document(str(pdf_path), state)
        except Exception as e:
            print(f"    ✗ Ingestion failed: {e}")
            continue

        for _, qrow in tqdm(doc_qs.iterrows(), total=len(doc_qs),
                            desc="    Relevance", ncols=70):
            question = str(qrow["question"])
            try:
                result = rag.answer_question(question, state)
            except Exception as e:
                print(f"      ✗ QA error: {e}")
                continue

            answer = result.get("answer", "")
            if not answer:
                continue

            eval_result = judge.evaluate(question, answer)
            rows.append({
                "document":     doc_name,
                "question":     question,
                "answer":       answer[:150].replace("\n", " "),
                "score":        eval_result["score"],
                "is_abstention": eval_result["is_abstention"],
                "reasoning":    eval_result["reasoning"],
            })

            time.sleep(4.0)

    _section("Step 5: Results")

    if not rows:
        print("  ✗ No results generated."); return

    results_df = pd.DataFrame(rows)
    out_path   = PROJECT_ROOT / "answer_relevance_results.csv"
    results_df.to_csv(out_path, index=False)

    col_w = 44
    abstentions = results_df["is_abstention"].sum()
    total       = len(results_df)

    print(f"\n  {'Document':<{col_w}} {'Relevance':>9}")
    print("  " + "─" * (col_w + 12))

    doc_scores = []
    for doc, grp in results_df.groupby("document"):
        avg = grp["score"].mean()
        doc_scores.append(avg)
        print(f"  {doc[:col_w]:<{col_w}} {avg:>9.3f}")

    overall = sum(doc_scores) / len(doc_scores)
    print("  " + "─" * (col_w + 12))
    print(f"  {'Overall':.<{col_w}} {overall:>9.3f}")
    print(f"\n  Abstentions: {abstentions}/{total} "
          f"({abstentions/total*100:.0f}%) — cross-reference with Context Recall")

    low = results_df[(results_df["score"] < 0.5) & (~results_df["is_abstention"])]
    if not low.empty:
        print(f"\n  ⚠  {len(low)} low-relevance answer(s) (not abstentions):")
        for _, r in low.iterrows():
            print(f"    • Q: {r['question'][:55]}")
            print(f"      A: {r['answer'][:60]}")
            print(f"      Reason: {r['reasoning'][:80]}")

    print(f"\n  Full results → {out_path.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOS Answer Relevance Evaluator")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    args = parser.parse_args()
    run(model=args.model)
