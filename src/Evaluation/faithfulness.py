"""
TOS Summarizer — Faithfulness Evaluation (LLM-as-Judge)
========================================================
Measures whether the RAG pipeline's outputs are grounded in retrieved context.

Metric
------
Faithfulness = supported_claims / total_claims

For each (retrieved context, generated answer) pair:
  1. Send context + answer to an LLM judge in a single structured prompt.
  2. The judge identifies individual claims and verdicts each:
       supported    → claim is directly backed by the context
       neutral      → claim is absent from context but not contradicted
       contradicted → claim contradicts the context (hallucination signal)
  3. Faithfulness = supported / (supported + neutral + contradicted)

Judge Model
-----------
Primary:  Groq API  (set GROQ_API_KEY in .env)
          Configurable via --model flag.
          Default: llama-3.3-70b-versatile

Fallback: cross-encoder/nli-deberta-v3-small  (local, no API key needed)
          Used automatically when GROQ_API_KEY is absent.
          Note: NLI is trained on formal text, not legal paraphrase.
          Scores will be conservative (~0.1–0.4 range).

Scope
-----
--mode qa       : evaluate Q&A answers only  (default, ~15 API calls)
--mode summary  : evaluate topic summaries   (~60 API calls, 12 topics × 5 docs)
--mode both     : evaluate everything        (~75 API calls)

Usage
-----
    # Set GROQ_API_KEY in .env first, then:
    python src/Evaluation/faithfulness.py
    python src/Evaluation/faithfulness.py --mode both
    python src/Evaluation/faithfulness.py --mode summary --model llama3-70b-8192

Outputs
-------
    faithfulness_results.csv   — per-claim detail with LLM reasoning
    faithfulness_summary.csv   — per-question/topic aggregate scores
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

# ── NLI fallback labels ───────────────────────────────────────────────────────
_NLI_LABELS = {0: "contradicted", 1: "supported", 2: "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# Judge backends
# ─────────────────────────────────────────────────────────────────────────────

class GroqJudge:
    """
    LLM-as-judge using the Groq API.

    Why LLM over NLI:
    - Understands legal paraphrase and abstraction (NLI cannot)
    - Returns human-readable reasoning per claim (auditable)
    - A single API call evaluates an entire answer (vs one NLI call per claim)
    - Calibrated scores — distinguishes "not mentioned" from "hallucinated"
    """

    SYSTEM_PROMPT = """\
You are a strict legal document faithfulness evaluator.
Your job is to assess whether each factual claim in an AI-generated legal answer
is supported by the provided source context.

Rules:
- "supported"    : The claim is directly stated or clearly implied by the context.
                   Paraphrase of context counts as supported.
- "neutral"      : The claim is not addressed in the context. It may be correct
                   but we cannot verify it from the given sources.
- "contradicted" : The claim contradicts or materially misrepresents the context.
                   This is the hallucination signal.

Be lenient with abstraction — legal summaries rephrase rather than quote verbatim.
Only mark "contradicted" if the meaning actually changes.
"""

    USER_TEMPLATE = """\
SOURCE CONTEXT (retrieved chunks from the legal document):
{context}

AI-GENERATED ANSWER:
{answer}

Task: Identify each factual claim in the answer and evaluate it.

Respond ONLY with valid JSON — no prose, no markdown fences:
{{
  "claims": [
    {{"claim": "...", "verdict": "supported|neutral|contradicted", "reason": "..."}}
  ],
  "faithfulness_score": <float 0.0–1.0>,
  "overall_assessment": "one sentence summary"
}}
"""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model  = model
        print(f"  Judge: Groq / {model}")

    def evaluate(self, context: str, answer: str) -> dict:
        """Return structured faithfulness evaluation from Groq."""
        prompt = self.USER_TEMPLATE.format(
            context=context[:6000],   # stay within context window
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
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content.strip()

                # Strip markdown fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                return json.loads(raw)

            except json.JSONDecodeError as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    # Return minimal structure on parse failure
                    return {
                        "claims": [{"claim": answer[:80], "verdict": "neutral",
                                    "reason": f"Parse error: {e}"}],
                        "faithfulness_score": 0.0,
                        "overall_assessment": "Evaluation failed (JSON parse error)",
                    }
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Rate limit" in err_msg:
                    wait_time = 15 * (attempt + 1)
                    print(f"\n      [!] Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt < 2:
                        time.sleep(3)
                    else:
                        return {
                            "claims": [],
                            "faithfulness_score": 0.0,
                            "overall_assessment": f"API error: {e}",
                        }
        
        # If we exhausted all attempts
        return {
            "claims": [],
            "faithfulness_score": 0.0,
            "overall_assessment": "Failed after 3 attempts due to rate limiting.",
        }


class NLIJudge:
    """
    Local NLI-based judge using cross-encoder/nli-deberta-v3-small.

    Tradeoffs vs LLM judge:
    + No API key needed, fully offline, no rate limits
    - Trained on MNLI/SNLI, not legal text → conservative scores (~0.1–0.4)
    - Penalises paraphrase as NEUTRAL (artificially low faithfulness)
    - No human-readable reasoning
    - One model call per sentence instead of one call per answer
    """

    def __init__(self):
        from sentence_transformers import CrossEncoder
        nli_model = "cross-encoder/nli-deberta-v3-small"
        local_path = PROJECT_ROOT / "models" / "nli-deberta-v3-small"
        if local_path.exists():
            self.model = CrossEncoder(str(local_path))
        else:
            print(f"  Downloading {nli_model} (~86 MB, one-time) …")
            local_path.mkdir(parents=True, exist_ok=True)
            self.model = CrossEncoder(nli_model)
            self.model.save(str(local_path))
        print(f"  Judge: Local NLI  (cross-encoder/nli-deberta-v3-small)")
        print("  ⚠  NLI scores are conservative for legal paraphrase.")
        print("     Set GROQ_API_KEY in .env for LLM-as-judge.\n")

    def _split(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 15]

    def evaluate(self, context: str, answer: str) -> dict:
        claims = self._split(answer)
        if not claims:
            return {"claims": [], "faithfulness_score": 0.0,
                    "overall_assessment": "No claims extracted"}

        pairs  = [[context, c] for c in claims]
        scores = self.model.predict(pairs)

        result_claims = []
        supported = 0
        for claim, sv in zip(claims, scores):
            idx     = int(sv.argmax())
            verdict = _NLI_LABELS[idx]
            if verdict == "supported":
                supported += 1
            result_claims.append({
                "claim":   claim,
                "verdict": verdict,
                "reason":  f"NLI confidence: {float(sv[idx]):.3f}",
            })

        score = supported / len(claims)
        return {
            "claims": result_claims,
            "faithfulness_score": round(score, 4),
            "overall_assessment": f"{supported}/{len(claims)} claims supported",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner():
    width = 62
    print("╔" + "═" * width + "╗")
    print("║" + "TOS Summarizer — Faithfulness Evaluation".center(width) + "║")
    print("╠" + "═" * width + "╣")
    print("║" + f"  Date: {datetime.now():%Y-%m-%d %H:%M:%S}".ljust(width) + "║")
    print("╚" + "═" * width + "╝\n")


def _section(title: str):
    print("\n" + "─" * 62)
    print(f"  {title}")
    print("─" * 62)


def _safe_ns(filename: str, prefix: str = "faith") -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", filename)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return f"{prefix}-{safe[:55]}"


def _sources_to_context(sources: list[dict]) -> str:
    return " ".join(s.get("excerpt", "") for s in sources)


def _load_judge(model: str) -> "GroqJudge | NLIJudge":
    if os.getenv("GROQ_API_KEY"):
        try:
            return GroqJudge(model=model)
        except ImportError:
            print("  ⚠  groq package not installed. Falling back to NLI judge.")
            print("     Run: pip install groq")
    else:
        print("  ℹ  GROQ_API_KEY not set → using local NLI fallback.")
    return NLIJudge()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def _eval_qa(rag, judge, df: pd.DataFrame, pdf_files: list, claim_rows: list,
             summary_rows: list):
    """Evaluate faithfulness of Q&A answers."""
    _section("Q&A Faithfulness Evaluation")

    for pdf_path in pdf_files:
        doc_name = pdf_path.name
        doc_qs   = df[df["filename"] == doc_name]
        if doc_qs.empty:
            continue

        print(f"\n  📄 {doc_name} ({len(doc_qs)} questions)")

        state = SessionState(pinecone_namespace=_safe_ns(doc_name, "fqa"))
        row0  = doc_qs.iloc[0]
        state.service_name = str(row0.get("service_name", "Unknown"))
        state.doc_type     = str(row0.get("doc_type", "Document"))

        try:
            rag.ingest_document(str(pdf_path), state)
        except Exception as e:
            print(f"    ✗ Ingestion failed: {e}")
            continue

        for _, qrow in tqdm(doc_qs.iterrows(), total=len(doc_qs),
                            desc="    Q&A", ncols=70):
            question = str(qrow["question"])
            try:
                result  = rag.answer_question(question, state)
            except Exception as e:
                print(f"      ✗ QA error: {e}")
                continue

            answer  = result.get("answer", "")
            sources = result.get("sources", [])
            context = _sources_to_context(sources)

            if not answer or not context:
                continue

            eval_result = judge.evaluate(context, answer)

            for c in eval_result.get("claims", []):
                claim_rows.append({
                    "mode":       "qa",
                    "document":   doc_name,
                    "input":      question,
                    "claim":      c.get("claim", ""),
                    "verdict":    c.get("verdict", ""),
                    "reason":     c.get("reason", ""),
                })

            summary_rows.append({
                "mode":              "qa",
                "document":          doc_name,
                "input":             question,
                "faithfulness":      eval_result.get("faithfulness_score", 0.0),
                "overall_assessment": eval_result.get("overall_assessment", ""),
                "n_claims":          len(eval_result.get("claims", [])),
                "supported":  sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "supported"),
                "neutral":    sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "neutral"),
                "contradicted": sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "contradicted"),
                "answer_preview": answer[:100].replace("\n", " "),
            })

            # Rate limit guard for Groq
            if isinstance(judge, GroqJudge):
                time.sleep(4.0) # Conservative sleep to avoid TPM limits


def _eval_summary(rag, judge, pdf_files: list, claim_rows: list,
                  summary_rows: list):
    """Evaluate faithfulness of topic-level global summaries."""
    _section("Summary Faithfulness Evaluation")

    for pdf_path in pdf_files:
        doc_name = pdf_path.name
        print(f"\n  📄 {doc_name}")

        state = SessionState(pinecone_namespace=_safe_ns(doc_name, "fsum"))

        try:
            rag.ingest_document(str(pdf_path), state)
        except Exception as e:
            print(f"    ✗ Ingestion failed: {e}")
            continue

        try:
            summary_result = rag.generate_global_summary(state)
        except Exception as e:
            print(f"    ✗ Summary generation failed: {e}")
            continue

        topics = summary_result.get("topics", [])
        for topic in tqdm(topics, desc="    Topics", ncols=70):
            label   = topic.get("label", "Unknown")
            text    = topic.get("summary", "")
            sources = topic.get("sources", [])
            context = _sources_to_context(sources)

            if not text or not context:
                continue

            eval_result = judge.evaluate(context, text)

            for c in eval_result.get("claims", []):
                claim_rows.append({
                    "mode":     "summary",
                    "document": doc_name,
                    "input":    label,
                    "claim":    c.get("claim", ""),
                    "verdict":  c.get("verdict", ""),
                    "reason":   c.get("reason", ""),
                })

            summary_rows.append({
                "mode":               "summary",
                "document":           doc_name,
                "input":              label,
                "faithfulness":       eval_result.get("faithfulness_score", 0.0),
                "overall_assessment": eval_result.get("overall_assessment", ""),
                "n_claims":           len(eval_result.get("claims", [])),
                "supported":  sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "supported"),
                "neutral":    sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "neutral"),
                "contradicted": sum(1 for c in eval_result.get("claims", []) if c.get("verdict") == "contradicted"),
                "answer_preview": text[:100].replace("\n", " "),
            })

            if isinstance(judge, GroqJudge):
                time.sleep(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str, model: str):
    _banner()

    # ── Load judge ────────────────────────────────────────────────────────────
    _section("Step 1: Loading Judge")
    judge = _load_judge(model)

    # ── Load RAG engine ───────────────────────────────────────────────────────
    _section("Step 2: Loading RAG Engine")
    models_dir = PROJECT_ROOT / "models"
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print("  ✗ No GGUF model found in models/. Aborting.")
        return
    model_path  = str(gguf_files[0])
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    chroma_dir   = str(PROJECT_ROOT / "data" / "faithfulness_chroma")

    from src.RAG.engine import TOSAssistant
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
    if not tos_dir.exists():
        print(f"  ✗ {tos_dir} not found"); return

    df        = pd.read_csv(test_csv)
    pdf_files = sorted(tos_dir.glob("*.pdf"))
    print(f"  Mode: {mode.upper()} | {len(pdf_files)} docs | {len(df)} Q&A questions")

    # ── Run evaluation ────────────────────────────────────────────────────────
    claim_rows:   list[dict] = []
    summary_rows: list[dict] = []

    if mode in ("qa", "both"):
        _eval_qa(rag, judge, df, pdf_files, claim_rows, summary_rows)

    if mode in ("summary", "both"):
        _eval_summary(rag, judge, pdf_files, claim_rows, summary_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    _section("Step 4: Saving Results")

    if not summary_rows:
        print("  ✗ No results generated."); return

    claims_df  = pd.DataFrame(claim_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_claims  = PROJECT_ROOT / "faithfulness_results.csv"
    out_summary = PROJECT_ROOT / "faithfulness_summary.csv"
    claims_df.to_csv(out_claims,  index=False)
    summary_df.to_csv(out_summary, index=False)
    print(f"  Claim detail → {out_claims.name}")
    print(f"  Summary      → {out_summary.name}")

    # ── Console table ─────────────────────────────────────────────────────────
    _section("Faithfulness Results")

    col_w = 44
    modes = summary_df["mode"].unique()
    for m in modes:
        mdf = summary_df[summary_df["mode"] == m]
        print(f"\n  [{m.upper()}]")
        print(f"  {'Document':<{col_w}} {'Faith':>7}  {'Sup':>5}  {'Neu':>5}  {'Con':>5}")
        print("  " + "─" * (col_w + 28))
        doc_scores = []
        for doc, grp in mdf.groupby("document"):
            avg = grp["faithfulness"].mean()
            doc_scores.append(avg)
            sup = grp["supported"].sum()
            neu = grp["neutral"].sum()
            con = grp["contradicted"].sum()
            print(f"  {doc[:col_w]:<{col_w}} {avg:>7.3f}  {sup:>5}  {neu:>5}  {con:>5}")
        overall = sum(doc_scores) / len(doc_scores) if doc_scores else 0
        print("  " + "─" * (col_w + 28))
        print(f"  {'Overall':.<{col_w}} {overall:>7.3f}")

    # Worst answers
    worst = summary_df.nsmallest(3, "faithfulness")[
        ["mode", "document", "input", "faithfulness", "overall_assessment"]
    ]
    if not worst.empty:
        print("\n  Lowest-faithfulness outputs:")
        for _, r in worst.iterrows():
            print(f"    [{r['faithfulness']:.2f}] [{r['mode'].upper()}] {r['input'][:55]}")
            print(f"           {r['overall_assessment'][:80]}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOS Faithfulness Evaluator")
    parser.add_argument(
        "--mode", choices=["qa", "summary", "both"], default="qa",
        help="What to evaluate (default: qa)"
    )
    parser.add_argument(
        "--model", default="llama-3.3-70b-versatile",
        help="Groq model name (default: llama-3.3-70b-versatile)"
    )
    args = parser.parse_args()
    run(mode=args.mode, model=args.model)