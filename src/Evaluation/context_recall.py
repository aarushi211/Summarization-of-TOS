"""
TOS Summarizer — Context Recall Evaluation
===========================================
Measures whether the retrieval stage fetches chunks that actually contain the
information needed to answer each question.

Metric
------
Context Recall = questions where retrieved context is sufficient / total questions

For each question:
  1. Retrieve the top-k chunks using the RAG engine (same as production).
  2. Ask the LLM judge: "Does ANY of these chunks contain enough information
     to answer this question? Yes / Partial / No"
  3. Score:
       Yes     → 1.0  (retrieval succeeded)
       Partial → 0.5  (retrieval got part of the answer)
       No      → 0.0  (retrieval failed — this is WHY the model abstains)

Why this matters
----------------
When faithfulness is low due to abstentions (model saying "I don't know"),
Context Recall diagnoses whether the problem is in:
  - The retrieval layer  → Context Recall is low (fix: better chunking/embeddings)
  - The LLM layer       → Context Recall is high but faithfulness is still low
                          (fix: better prompting or larger model)

Usage
-----
    python src/Evaluation/context_recall.py

Outputs
-------
    context_recall_results.csv   — per-question recall scores + judge reasoning
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

class ContextRecallJudge:
    """
    LLM judge for Context Recall.

    Unlike Faithfulness (answer vs context), Context Recall evaluates
    (question vs context) — entirely ignoring the generated answer.
    This cleanly isolates the retrieval stage from the generation stage.
    """

    SYSTEM_PROMPT = """\
You are evaluating a retrieval system for a legal document QA application.
Given a user's question and the chunks retrieved from the legal document,
you must assess whether the retrieved chunks contain enough information
to answer the question.

Verdict options:
- "yes"     : The chunks contain clear, sufficient information to answer the question.
- "partial" : The chunks contain some relevant info but are incomplete or ambiguous.
- "no"      : The chunks do not contain the information needed to answer the question.

Be pragmatic. If a careful reader could extract the answer from the chunks, say "yes".
"""

    USER_TEMPLATE = """\
USER QUESTION:
{question}

RETRIEVED CONTEXT CHUNKS (from the legal document):
{context}

Can a reader extract the answer to this question from the retrieved chunks?

Respond ONLY with valid JSON:
{{
  "verdict": "yes|partial|no",
  "reason": "one sentence explanation",
  "key_chunk": "quote the most relevant sentence from the context, or null if none found"
}}
"""

    SCORE_MAP = {"yes": 1.0, "partial": 0.5, "no": 0.0}

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model  = model
        print(f"  Judge: Groq / {model}")

    def evaluate(self, question: str, context: str) -> dict:
        prompt = self.USER_TEMPLATE.format(
            question=question[:500],
            context=context[:5000],
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
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                verdict = data.get("verdict", "no").lower()
                return {
                    "verdict":   verdict,
                    "score":     self.SCORE_MAP.get(verdict, 0.0),
                    "reason":    data.get("reason", ""),
                    "key_chunk": data.get("key_chunk"),
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
                    return {"verdict": "no", "score": 0.0,
                            "reason": f"API error: {e}", "key_chunk": None}
                else:
                    time.sleep(3)

        return {"verdict": "no", "score": 0.0,
                "reason": "Failed after 3 attempts.", "key_chunk": None}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner():
    w = 62
    print("╔" + "═" * w + "╗")
    print("║" + "TOS Summarizer — Context Recall Evaluation".center(w) + "║")
    print("╠" + "═" * w + "╣")
    print("║" + f"  Date: {datetime.now():%Y-%m-%d %H:%M:%S}".ljust(w) + "║")
    print("╚" + "═" * w + "╝\n")

def _section(title: str):
    print("\n" + "─" * 62)
    print(f"  {title}")
    print("─" * 62)

def _safe_ns(filename: str, prefix: str = "cr") -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", filename)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return f"{prefix}-{safe[:55]}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(model: str):
    _banner()

    # ── Load judge ────────────────────────────────────────────────────────────
    _section("Step 1: Loading Judge")
    if not os.getenv("GROQ_API_KEY"):
        print("  ✗ GROQ_API_KEY not set in .env. Context Recall requires an LLM judge.")
        return
    judge = ContextRecallJudge(model=model)

    # ── Load RAG engine ───────────────────────────────────────────────────────
    _section("Step 2: Loading RAG Engine")
    models_dir = PROJECT_ROOT / "models"
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print("  ✗ No GGUF model found in models/. Aborting.")
        return
    model_path   = str(gguf_files[0])
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    chroma_dir   = str(PROJECT_ROOT / "data" / "cr_chroma")

    from src.RAG.engine import TOSAssistant, _QA_TOP_K
    rag = TOSAssistant(
        model_path=model_path,
        pinecone_api_key=pinecone_key,
        use_local_vectorstore=not pinecone_key,
        data_dir=chroma_dir,
    )
    print("  RAG engine ready.")

    # ── Load test data ────────────────────────────────────────────────────────
    _section("Step 3: Loading Test Data")
    test_csv  = PROJECT_ROOT / "data" / "Test_data.csv"
    tos_dir   = PROJECT_ROOT / "data" / "TOS files"

    if not test_csv.exists():
        print(f"  ✗ {test_csv} not found"); return
    df        = pd.read_csv(test_csv)
    pdf_files = sorted(tos_dir.glob("*.pdf"))
    print(f"  {len(pdf_files)} docs | {len(df)} questions")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    _section("Step 4: Running Context Recall Evaluation")
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
                            desc="    Recall", ncols=70):
            question = str(qrow["question"])

            # Retrieve chunks — same path as production, but stop before LLM
            try:
                docs, _ = rag._get_relevant_chunks(question, state, top_k=5)
            except Exception as e:
                print(f"      ✗ Retrieval error: {e}")
                continue

            context = "\n\n".join(d.page_content for d in docs)
            result  = judge.evaluate(question, context)

            rows.append({
                "document":  doc_name,
                "question":  question,
                "verdict":   result["verdict"],
                "score":     result["score"],
                "reason":    result["reason"],
                "key_chunk": result.get("key_chunk", ""),
                "n_chunks_retrieved": len(docs),
            })

            time.sleep(4.0)  # Rate limit guard

    # ── Save & display ────────────────────────────────────────────────────────
    _section("Step 5: Results")

    if not rows:
        print("  ✗ No results generated."); return

    results_df = pd.DataFrame(rows)
    out_path   = PROJECT_ROOT / "context_recall_results.csv"
    results_df.to_csv(out_path, index=False)

    col_w = 44
    print(f"\n  {'Document':<{col_w}} {'Recall':>7}  {'Yes':>4}  {'Partial':>7}  {'No':>4}")
    print("  " + "─" * (col_w + 28))

    doc_scores = []
    for doc, grp in results_df.groupby("document"):
        avg = grp["score"].mean()
        doc_scores.append(avg)
        yes     = (grp["verdict"] == "yes").sum()
        partial = (grp["verdict"] == "partial").sum()
        no      = (grp["verdict"] == "no").sum()
        print(f"  {doc[:col_w]:<{col_w}} {avg:>7.3f}  {yes:>4}  {partial:>7}  {no:>4}")

    overall = sum(doc_scores) / len(doc_scores)
    print("  " + "─" * (col_w + 28))
    print(f"  {'Overall':.<{col_w}} {overall:>7.3f}")

    # Low recall questions
    low = results_df[results_df["score"] == 0.0]
    if not low.empty:
        print(f"\n  ⚠  {len(low)} question(s) with ZERO context recall (retrieval failed):")
        for _, r in low.iterrows():
            print(f"    • {r['question'][:60]}")
            print(f"      Reason: {r['reason'][:80]}")

    print(f"\n  Full results → {out_path.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOS Context Recall Evaluator")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    args = parser.parse_args()
    run(model=args.model)
