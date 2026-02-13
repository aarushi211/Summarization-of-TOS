import os
import sys
import time
import warnings
import pandas as pd
from pathlib import Path
from datasets import Dataset 
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.append(str(project_root))

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from src.RAG.rag_pipeline import TOSAssistant

LOCAL_JUDGE_MODEL = "qwen2.5:7b"

local_llm = ChatOllama(model=LOCAL_JUDGE_MODEL, temperature=0, num_ctx=8192, timeout=300)
judge_llm = LangchainLLMWrapper(local_llm)

local_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
judge_embeddings = LangchainEmbeddingsWrapper(local_embed_model)

metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
for m in metrics:
    m.llm = judge_llm
    if hasattr(m, 'embeddings'):
        m.embeddings = judge_embeddings

def generate_test_data_with_refusal_tracking(csv_input):
    csv_path = project_root / "data" / csv_input
    if not csv_path.exists():
        csv_path = project_root / "data" / "Test_data.csv"
    
    df = pd.read_csv(csv_path)
    base_model = str(project_root / "models" / "legal_qwen.Q4_K_M.gguf")
    rag = TOSAssistant(base_model)
    
    answered_results = {
        "question": [], "answer": [], "contexts": [], "ground_truth": []
    }
    
    refusal_tracking = []

    for filename, group in df.groupby("filename"):
        pdf_path = project_root / "data" / "TOS_files" / filename
        
        if pdf_path.exists():
            print(f"Processing {filename}...")
            rag.ingest_document(str(pdf_path))
            rag.service_name = group.iloc[0].get('service_name', 'Unknown')
            rag.doc_type = group.iloc[0].get('doc_type', 'Document')
            
            for _, row in group.iterrows():
                print(f"{row['question'][:40]}...")
                output = rag.answer_question(row['question'])
                
                refused = "NOT_IN_DOCUMENT" in output["answer"].upper()
                
                ground_truth_str = str(row['ground_truth']).strip().lower()
                ground_truth_empty = (
                    pd.isna(row['ground_truth']) or 
                    ground_truth_str == "" or
                    "not_in_document" in ground_truth_str
                )
                
                appropriate_refusal = refused and ground_truth_empty
                inappropriate_refusal = refused and not ground_truth_empty
                
                if refused:
                    print(f"Refused")
                    refusal_tracking.append({
                        'question': row['question'],
                        'refused': True,
                        'appropriate': appropriate_refusal,
                        'inappropriate': inappropriate_refusal,
                        'ground_truth_present': not ground_truth_empty
                    })
                else:
                    print(f"Answered")
                    answered_results["question"].append(row['question'])
                    answered_results["answer"].append(output["answer"])
                    answered_results["contexts"].append(output["sources"]) 
                    answered_results["ground_truth"].append(row['ground_truth'])
                    
                    refusal_tracking.append({
                        'question': row['question'],
                        'refused': False,
                        'appropriate': True,  
                        'inappropriate': False,
                        'ground_truth_present': not ground_truth_empty
                    })

    return Dataset.from_dict(answered_results), pd.DataFrame(refusal_tracking)

if __name__ == "__main__":
    try:
        csv_filename = "Test_data.csv"
        
        print("\nLoading and processing data...")
        answered_dataset, refusal_df = generate_test_data_with_refusal_tracking(csv_filename)
        
        print("\n" + "="*70)
        print("REFUSAL ANALYSIS")
        print("="*70)
        
        total_questions = len(refusal_df)
        total_refused = refusal_df['refused'].sum()
        total_answered = total_questions - total_refused
        appropriate_refusals = refusal_df['appropriate'].sum()
        inappropriate_refusals = refusal_df['inappropriate'].sum()
        
        print(f"\nTotal questions: {total_questions}")
        print(f"Answered: {total_answered}")
        print(f"Refused: {total_refused}")
        print(f"✓ Appropriate refusals: {appropriate_refusals}")
        print(f"✗ Inappropriate refusals: {inappropriate_refusals}")
        
        if total_refused > 0:
            refusal_accuracy = (appropriate_refusals / total_refused) * 100
            print(f"\nRefusal accuracy: {refusal_accuracy:.1f}%")
        
        # Save refusal analysis
        refusal_file = project_root / "refusal_analysis.csv"
        refusal_df.to_csv(refusal_file, index=False)
        print(f"\nRefusal analysis saved: {refusal_file}")
    
       
        accumulated_results = []
        
        for i in range(len(answered_dataset)):
            single_item = answered_dataset.select([i])
            
            print(f"\n[{i+1}/{len(answered_dataset)}] {single_item['question'][0][:60]}...")
            
            try:
                start = time.time()
                
                result = evaluate(
                    dataset=single_item,
                    metrics=metrics,
                    run_config=RunConfig(timeout=600, max_retries=3, max_workers=1, max_wait=30)
                )
                
                print(f"Done in {time.time() - start:.1f}s")
                
                result_df = result.to_pandas()
                accumulated_results.append(result_df)
                
                for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    if col in result_df.columns:
                        score = result_df[col].iloc[0]
                        if pd.notna(score):
                            print(f"      {col}: {score:.3f}")
                
            except Exception as e:
                print(f"Error: {str(e)[:100]}")
                error_row = pd.DataFrame({
                    'question': [single_item['question'][0]],
                    'answer': [single_item['answer'][0]],
                    'contexts': [str(single_item['contexts'][0])],
                    'ground_truth': [single_item['ground_truth'][0]],
                    'faithfulness': [None], 'answer_relevancy': [None],
                    'context_precision': [None], 'context_recall': [None]
                })
                accumulated_results.append(error_row)
            
            time.sleep(2)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        
        final_df = pd.concat(accumulated_results, ignore_index=True)
        output_file = project_root / "ragas_answered_questions_report.csv"
        final_df.to_csv(output_file, index=False)
        
        print("\nRAGAS Statistics (Answered Questions Only):")
        metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        print(final_df[metric_cols].describe().round(3))
        
        successful = final_df[metric_cols].notna().all(axis=1).sum()
        print(f"\nSuccessful: {successful}/{len(answered_dataset)}")
        
        print(f"\nReports saved:")
        print(f"- RAGAS scores: {output_file}")
        print(f"- Refusal analysis: {refusal_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()