import os
import json
import logging
from dotenv import load_dotenv

# Add project root to sys.path if running independently
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import TelecomRAGPipeline
from main import expert_client, call_expert

logger = logging.getLogger(__name__)

def generate_evaluation_dataset(prompts_json_path: str, output_dataset_path: str, sample_size: int = 10):
    """
    Generates a dataset for evaluating Faithfulness and Accuracy.
    
    How it works:
    1. Reads the generated anomaly queries from `anomaly_prompts.json` (created by Path A).
    2. Runs each query through the RAG Pipeline to get `retrieved_context`.
    3. Sends the query + context to the Generator LLM (expert_client) to get `generated_answer`.
    4. Saves the triplet (question, context, answer) to be consumed by the evaluators.
    """
    load_dotenv()
    
    if not os.path.exists(prompts_json_path):
        raise FileNotFoundError(f"Anomaly prompts not found at {prompts_json_path}. Run anomaly/run_pipeline.py first.")
        
    with open(prompts_json_path, 'r') as f:
        anomaly_windows = json.load(f)
        
    pipeline = TelecomRAGPipeline()
    dataset = []
    
    # Take a sample of high-severity windows
    windows_to_process = anomaly_windows[:sample_size]
    
    print(f"Generating evaluation dataset for {len(windows_to_process)} samples...")
    
    for idx, window in enumerate(windows_to_process):
        query = window.get('rag_diagnostic_query')
        if not query:
            continue
            
        print(f"[{idx+1}/{len(windows_to_process)}] Processing Window {window.get('window_id')}...")
        
        # 1. Retrieve Context
        try:
            retrieved_chunks = pipeline.retrieve_context(query=query, top_k=3)
            context_text = "\n\n".join([c.get("content", "") for c in retrieved_chunks])
        except Exception as e:
            print(f"  -> Skipping due to retrieval error: {e}")
            continue
            
        # 2. Generate Answer
        full_prompt = f"=== 3GPP SPECS ===\n{context_text}\n=== TASK ===\n{query}"
        try:
            generated_answer = call_expert(full_prompt)
        except Exception as e:
            print(f"  -> Skipping due to generation error: {e}")
            continue
            
        # 3. Store Dataset Record
        record = {
            "id": window.get("window_id"),
            "domain": window.get("domain"),
            "question": query,
            "retrieved_context": context_text,
            "generated_answer": generated_answer,
            # For anomaly reports, we don't have a single "ground truth" reference answer in a dataset,
            # but we can optionally add a "reference_answer" later via a human expert or powerful Oracle model.
            "reference_answer": None 
        }
        dataset.append(record)
        
    with open(output_dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n[SUCCESS] Saved {len(dataset)} evaluation records to {output_dataset_path}")
    print("You can now run `evals/faithfulness.py` and `evals/accuracy.py` iterating over this JSON file.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    prompts_path = os.path.join(base_dir, 'Data', 'anomaly_prompts.json')
    out_path = os.path.join(base_dir, 'evals', 'eval_dataset.json')
    
    # Generate a small dataset
    generate_evaluation_dataset(prompts_path, out_path, sample_size=5)
