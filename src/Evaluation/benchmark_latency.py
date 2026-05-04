"""
TOS Summarizer — Latency & Performance Benchmark
=================================================
Runs the full RAG pipeline against all test documents and questions,
collecting per-stage timing metrics and token throughput data.

Usage:
    python src/Evaluation/benchmark_latency.py

Outputs:
    - latency_benchmark_report.csv  (per-operation detail)
    - latency_summary.csv           (aggregate statistics)
    - Console summary table
"""

import os
import sys
import time
import platform
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Fix Windows Console Encoding ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# --- Project Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))


def get_memory_rss_mb():
    """Get current process RSS memory in MB (cross-platform)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: no psutil available
        return None


def format_time(seconds):
    """Human-readable time formatting."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def safe_stat(series, stat_fn, default=0):
    """Safely compute a statistic on a series, returning default if empty."""
    clean = series.dropna()
    if len(clean) == 0:
        return default
    return stat_fn(clean)


def print_banner():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " TOS Summarizer — Latency & Performance Benchmark ".center(62) + "║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54}║")
    print(f"║  Platform: {platform.platform()[:50]:<51}║")
    print(f"║  Python: {platform.python_version():<53}║")
    print("╚" + "═" * 62 + "╝")
    print()


def print_section(title):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def print_metric(label, value, indent=0):
    prefix = "│  " * indent + ("├─ " if indent > 0 else "")
    print(f"  {prefix}{label:<36} {value:>18}")


def print_metric_last(label, value, indent=0):
    prefix = "│  " * (indent - 1) + ("└─ " if indent > 0 else "")
    print(f"  {prefix}{label:<36} {value:>18}")


def run_benchmark():
    """Main benchmark execution."""
    print_banner()

    # ──────────────────────────────────────────────
    # 1. COLD-START: Model Loading
    # ──────────────────────────────────────────────
    print_section("Phase 1: Cold-Start Model Loading")

    mem_before = get_memory_rss_mb()

    model_path = str(PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf")
    if not Path(model_path).exists():
        print(f"  ✗ Model not found at {model_path}")
        print("  Aborting benchmark.")
        return

    t_total_start = time.perf_counter()

    from src.RAG.rag_pipeline import TOSAssistant
    rag = TOSAssistant(model_path)

    t_init_done = time.perf_counter()
    total_init_time = t_init_done - t_total_start

    mem_after = get_memory_rss_mb()

    # Extract init sub-timings from the metrics
    init_metrics = [m for m in rag.get_metrics() if m.get("stage") == "init"]
    init_breakdown = {m["sub_stage"]: m["latency_s"] for m in init_metrics}

    print_metric("Model Load (Total)", format_time(total_init_time))
    print_metric("GGUF Model", format_time(init_breakdown.get("gguf_model_load", 0)), indent=1)
    print_metric("Embedding Model", format_time(init_breakdown.get("embedding_load", 0)), indent=1)
    print_metric_last("Cross-Encoder", format_time(init_breakdown.get("cross_encoder_load", 0)), indent=1)

    if mem_before and mem_after:
        print_metric("Memory (RSS after load)", f"{mem_after:.0f} MB")
        print_metric("Memory (delta)", f"+{mem_after - mem_before:.0f} MB")

    # ──────────────────────────────────────────────
    # 2. DOCUMENT INGESTION + GLOBAL SUMMARIES
    # ──────────────────────────────────────────────
    print_section("Phase 2: Document Ingestion & Global Summarization")

    test_csv = PROJECT_ROOT / "data" / "Test_data.csv"
    tos_dir = PROJECT_ROOT / "data" / "TOS_files"

    if not test_csv.exists():
        print(f"  ✗ Test data not found at {test_csv}")
        return

    df = pd.read_csv(test_csv)
    pdf_files = sorted(tos_dir.glob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDF documents, {len(df)} test questions\n")

    detailed_rows = []

    # Init rows (for the CSV)
    for m in init_metrics:
        detailed_rows.append({
            "operation": "init",
            "document": "—",
            "sub_stage": m["sub_stage"],
            "latency_s": m["latency_s"],
        })

    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        print(f"  📄 {pdf_name}")

        # Reset metrics for clean per-doc tracking
        rag.reset_metrics()

        # --- Ingestion ---
        rag.ingest_document(str(pdf_path))

        # Find service name/doc_type from test data
        doc_rows = df[df["filename"] == pdf_name]
        if len(doc_rows) > 0:
            rag.service_name = doc_rows.iloc[0].get("service_name", "Unknown")
            rag.doc_type = doc_rows.iloc[0].get("doc_type", "Document")

        ingest_metrics = [m for m in rag.get_metrics() if m.get("stage") == "ingest_document"]
        if ingest_metrics:
            im = ingest_metrics[-1]
            print_metric("Ingestion Total", format_time(im.get("total_ingest_s", 0)), indent=1)
            print_metric("PDF Parse", format_time(im.get("pdf_parse_s", 0)), indent=2)
            print_metric("Text Cleaning", format_time(im.get("text_clean_s", 0)), indent=2)
            print_metric("Chunking", format_time(im.get("chunking_s", 0)), indent=2)
            print_metric_last("FAISS Indexing", format_time(im.get("faiss_index_s", 0)), indent=2)
            print_metric("Pages / Chunks / Chars", 
                         f"{im.get('page_count', '?')} / {im.get('chunk_count', '?')} / {im.get('total_chars', '?')}", indent=1)
            
            detailed_rows.append({
                "operation": "ingest_document",
                "document": pdf_name,
                "sub_stage": "total",
                "latency_s": im.get("total_ingest_s", 0),
                "pdf_parse_s": im.get("pdf_parse_s", 0),
                "text_clean_s": im.get("text_clean_s", 0),
                "chunking_s": im.get("chunking_s", 0),
                "faiss_index_s": im.get("faiss_index_s", 0),
                "page_count": im.get("page_count", 0),
                "chunk_count": im.get("chunk_count", 0),
                "total_chars": im.get("total_chars", 0),
            })

        # --- Global Summary ---
        summary = rag.generate_global_summary()

        summary_metrics = [m for m in rag.get_metrics() if m.get("stage") == "global_summary"]
        if summary_metrics:
            sm = summary_metrics[-1]

            if "topics" in sm and sm["topics"]:
                total_llm_s = sum(t.get("llm_inference_s", 0) for t in sm["topics"])
                total_prompt = sum(t.get("prompt_tokens", 0) for t in sm["topics"])
                total_comp = sum(t.get("completion_tokens", 0) for t in sm["topics"])
                sm["llm_inference_s"] = total_llm_s
                sm["prompt_tokens"] = total_prompt
                sm["completion_tokens"] = total_comp
                if total_llm_s > 0:
                    sm["tokens_per_sec"] = total_comp / total_llm_s
                else:
                    sm["tokens_per_sec"] = 0

            print_metric("Global Summary Total", format_time(sm.get("total_summary_s", 0)), indent=1)
            print_metric("LLM Inference", format_time(sm.get("llm_inference_s", 0)), indent=2)
            tok_sec = sm.get("tokens_per_sec", 0)
            print_metric_last("Tokens/sec", f"{tok_sec:.1f} tok/s" if tok_sec else "N/A", indent=2)
            print_metric("Tokens (prompt/completion)", 
                         f"{sm.get('prompt_tokens', '?')} / {sm.get('completion_tokens', '?')}", indent=1)

            detailed_rows.append({
                "operation": "global_summary",
                "document": pdf_name,
                "sub_stage": "total",
                "latency_s": sm.get("total_summary_s", 0),
                "llm_inference_s": sm.get("llm_inference_s", 0),
                "prompt_tokens": sm.get("prompt_tokens", 0),
                "completion_tokens": sm.get("completion_tokens", 0),
                "tokens_per_sec": sm.get("tokens_per_sec", 0),
                "input_chars": sm.get("input_chars", 0),
            })

        # --- Q&A for this document ---
        questions = doc_rows["question"].tolist() if len(doc_rows) > 0 else []
        if questions:
            print(f"     ╔ Running {len(questions)} Q&A queries...")

        for i, question in enumerate(questions):
            rag.reset_metrics()  # Keep clean per-question
            # Re-add init metrics since reset preserves them
            
            result = rag.answer_question(question)

            qa_metrics = [m for m in rag.get_metrics() if m.get("stage") == "qa_inference"]
            if qa_metrics:
                qm = qa_metrics[-1]
                tok_sec = qm.get("tokens_per_sec", 0)
                print(f"     ║ Q{i+1}: {format_time(qm.get('total_qa_s', 0)):>8}  "
                      f"(retrieval: {format_time(qm.get('total_retrieval_s', 0))}, "
                      f"LLM: {format_time(qm.get('llm_inference_s', 0))}, "
                      f"{tok_sec:.1f} tok/s)")

                detailed_rows.append({
                    "operation": "qa_inference",
                    "document": pdf_name,
                    "query": question[:80],
                    "sub_stage": "total",
                    "latency_s": qm.get("total_qa_s", 0),
                    "mmr_search_s": qm.get("mmr_search_s", 0),
                    "rerank_s": qm.get("rerank_s", 0),
                    "total_retrieval_s": qm.get("total_retrieval_s", 0),
                    "llm_inference_s": qm.get("llm_inference_s", 0),
                    "prompt_tokens": qm.get("prompt_tokens", 0),
                    "completion_tokens": qm.get("completion_tokens", 0),
                    "tokens_per_sec": qm.get("tokens_per_sec", 0),
                    "context_chars": qm.get("context_chars", 0),
                    "mmr_candidates": qm.get("mmr_candidates", 0),
                })

        if questions:
            print(f"     ╚ Done ({len(questions)} questions)")
        print()

    # ──────────────────────────────────────────────
    # 3. AGGREGATE RESULTS
    # ──────────────────────────────────────────────
    t_total_end = time.perf_counter()
    total_benchmark_time = t_total_end - t_total_start

    detailed_df = pd.DataFrame(detailed_rows)

    # --- Compute aggregates ---
    print_section("Aggregate Results")

    # Ingestion stats
    ingest_df = detailed_df[detailed_df["operation"] == "ingest_document"]
    if len(ingest_df) > 0:
        print_metric("Document Ingestion (avg)", format_time(ingest_df["latency_s"].mean()))
        print_metric("  PDF Parse (avg)", format_time(safe_stat(ingest_df["pdf_parse_s"], lambda s: s.mean())))
        print_metric("  FAISS Index (avg)", format_time(safe_stat(ingest_df["faiss_index_s"], lambda s: s.mean())))

    # Summary stats
    summary_df = detailed_df[detailed_df["operation"] == "global_summary"]
    if len(summary_df) > 0:
        print_metric("Global Summary (avg)", format_time(summary_df["latency_s"].mean()))
        print_metric("  LLM Inference (avg)", format_time(safe_stat(summary_df["llm_inference_s"], lambda s: s.mean())))
        print_metric("  Tokens/sec (avg)", f"{safe_stat(summary_df['tokens_per_sec'], lambda s: s.mean()):.1f} tok/s")

    # Q&A stats
    qa_df = detailed_df[detailed_df["operation"] == "qa_inference"]
    if len(qa_df) > 0:
        print()
        print_metric("Q&A Total (avg)", format_time(qa_df["latency_s"].mean()))
        print_metric("Q&A Total (median)", format_time(qa_df["latency_s"].median()))
        print_metric("Q&A Total (P95)", format_time(qa_df["latency_s"].quantile(0.95)))
        print_metric("Q&A Total (max)", format_time(qa_df["latency_s"].max()))
        print()
        print_metric("  Retrieval (avg)", format_time(safe_stat(qa_df["total_retrieval_s"], lambda s: s.mean())))
        print_metric("    MMR Search (avg)", format_time(safe_stat(qa_df["mmr_search_s"], lambda s: s.mean())))
        print_metric("    Reranking (avg)", format_time(safe_stat(qa_df["rerank_s"], lambda s: s.mean())))
        print_metric("  LLM Inference (avg)", format_time(safe_stat(qa_df["llm_inference_s"], lambda s: s.mean())))
        print_metric("  Tokens/sec (avg)", f"{safe_stat(qa_df['tokens_per_sec'], lambda s: s.mean()):.1f} tok/s")

    # Overall
    print()
    print_metric("Total Questions Benchmarked", str(len(qa_df)))
    print_metric("Total Documents Benchmarked", str(len(ingest_df)))
    print_metric("Total Benchmark Time", format_time(total_benchmark_time))

    mem_final = get_memory_rss_mb()
    if mem_final:
        print_metric("Final Memory (RSS)", f"{mem_final:.0f} MB")

    # ──────────────────────────────────────────────
    # 4. SAVE REPORTS
    # ──────────────────────────────────────────────
    print_section("Saving Reports")

    # Detailed report
    detail_path = PROJECT_ROOT / "latency_benchmark_report.csv"
    detailed_df.to_csv(detail_path, index=False)
    print(f"  ✓ Detailed report: {detail_path}")

    # Summary report
    summary_rows = []
    summary_rows.append({"metric": "model_load_total_s", "value": total_init_time})
    for sub, val in init_breakdown.items():
        summary_rows.append({"metric": f"model_load_{sub}_s", "value": val})

    if len(ingest_df) > 0:
        summary_rows.append({"metric": "ingest_avg_s", "value": ingest_df["latency_s"].mean()})
        summary_rows.append({"metric": "ingest_faiss_avg_s", "value": safe_stat(ingest_df["faiss_index_s"], lambda s: s.mean())})

    if len(summary_df) > 0:
        summary_rows.append({"metric": "summary_avg_s", "value": summary_df["latency_s"].mean()})
        summary_rows.append({"metric": "summary_tokens_per_sec_avg", "value": safe_stat(summary_df["tokens_per_sec"], lambda s: s.mean())})

    if len(qa_df) > 0:
        summary_rows.append({"metric": "qa_avg_s", "value": qa_df["latency_s"].mean()})
        summary_rows.append({"metric": "qa_median_s", "value": qa_df["latency_s"].median()})
        summary_rows.append({"metric": "qa_p95_s", "value": qa_df["latency_s"].quantile(0.95)})
        summary_rows.append({"metric": "qa_max_s", "value": qa_df["latency_s"].max()})
        summary_rows.append({"metric": "qa_retrieval_avg_s", "value": safe_stat(qa_df["total_retrieval_s"], lambda s: s.mean())})
        summary_rows.append({"metric": "qa_llm_inference_avg_s", "value": safe_stat(qa_df["llm_inference_s"], lambda s: s.mean())})
        summary_rows.append({"metric": "qa_tokens_per_sec_avg", "value": safe_stat(qa_df["tokens_per_sec"], lambda s: s.mean())})

    summary_rows.append({"metric": "total_benchmark_time_s", "value": total_benchmark_time})
    summary_rows.append({"metric": "total_questions", "value": len(qa_df)})
    summary_rows.append({"metric": "total_documents", "value": len(ingest_df)})
    if mem_final:
        summary_rows.append({"metric": "memory_rss_mb", "value": mem_final})

    summary_rows.append({"metric": "benchmark_date", "value": datetime.now().isoformat()})
    summary_rows.append({"metric": "platform", "value": platform.platform()})

    summary_path = PROJECT_ROOT / "latency_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  ✓ Summary report:  {summary_path}")

    # Final banner
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " ✓ Benchmark Complete ".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()


if __name__ == "__main__":
    run_benchmark()
