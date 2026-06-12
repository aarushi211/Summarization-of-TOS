import os
import sys
from pathlib import Path

# Ensure the root directory is in your Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import your actual pipeline functions
from src.RAG.loaders import load_pdf
from src.RAG.processors import pages_to_source_text, convert_to_markdown
from langchain_text_splitters import MarkdownHeaderTextSplitter

def inspect_my_pipeline(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find PDF at {pdf_path}")
        return

    print("\n" + "="*60)
    print(f"🎬 STARTING PIPELINE INSPECTION FOR: {Path(pdf_path).name}")
    print("="*60)

    # ─────────────────────────────────────────────────────────
    # STEP 1: Raw PDF Text Extraction
    # ─────────────────────────────────────────────────────────
    print("\n--- [STAGE 1] RAW PDF EXTRACTION ---")
    raw_docs = load_pdf(pdf_path)
    print(f"✅ Successfully extracted {len(raw_docs)} pages.")
    
    # Print the first page's raw content as a sample
    print(f"\n📄 [Sample] Raw Content of Page 1 (First 400 chars):")
    print("-" * 40)
    print(raw_docs[0].page_content[:400] + "...")
    print("-" * 40)

    # ─────────────────────────────────────────────────────────
    # STEP 2: Heuristic Text Cleaning & Page Token Stitching
    # ─────────────────────────────────────────────────────────
    print("\n--- [STAGE 2] STITCHED RAW SOURCE TEXT ---")
    stitched_text = pages_to_source_text(raw_docs)
    print(f"✅ Stitched text length: {len(stitched_text)} characters.")
    
    print(f"\n📄 [Sample] Stitched Text showing injected Page Comments:")
    print("-" * 40)
    # Print the first 300 characters to see the <!-- page:1 --> tag
    print(stitched_text[:300] + "\n...")
    print("-" * 40)

    # ─────────────────────────────────────────────────────────
    # STEP 3: Heuristic Regex Markdown Conversion
    # ─────────────────────────────────────────────────────────
    print("\n--- [STAGE 3] HEURISTIC REGEX MARKDOWN STRING ---")
    markdown_text = convert_to_markdown(stitched_text)
    
    print(f"\n📄 [Sample] Lines matching our structural regex injections:")
    print("-" * 40)
    # Find and print lines that successfully got # or ## injected
    md_lines = markdown_text.split("\n")
    headers_found = [line for line in md_lines if line.startswith("#")]
    for h in headers_found[:8]:  # Show up to the first 8 detected headers
        print(f"Found Markdown Tag: {h}")
    if not headers_found:
        print("⚠️ No structural headers matched your regex layout rules.")
    print("-" * 40)

    # ─────────────────────────────────────────────────────────
    # STEP 4: In-Memory Markdown Header Splitting
    # ─────────────────────────────────────────────────────────
    print("\n--- [STAGE 4] MARKDOWN HEADER SPLITTER OUTPUT ---")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "doc_title"), 
            ("##", "section"),
            ("###", "subsection"), 
            ("####", "clause"),
        ],
        strip_headers=False,
    )
    
    structural_sections = header_splitter.split_text(markdown_text)
    print(f"✅ Isolated into {len(structural_sections)} distinct structural sections.")
    
    # Let's inspect the first 2 sections to look at their inherited metadata
    print(f"\n📄 Detailed Structural Breakdown (First 2 sections):")
    for i, section in enumerate(structural_sections[:2]):
        print("-" * 40)
        print(f"📍 Section Document {i+1}:")
        print(f"   ↳ Inherited Metadata: {section.metadata}")
        print(f"   ↳ Text Content (First 200 chars): \n   {section.page_content[:200].strip()}...")
    print("-" * 40)

if __name__ == "__main__":
    # 💡 REPLACE THIS path with a path to a PDF on your laptop
    TARGET_PDF = r"C:\Users\aarus\Desktop\College\Projects\TOS-Summarization\data\TOS files\Netflix _ Terms & Conditions.pdf" 
    inspect_my_pipeline(TARGET_PDF)