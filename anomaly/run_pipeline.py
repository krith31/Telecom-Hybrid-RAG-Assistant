import os
import sys
import time

# Import sub-modules
from merge_kpis import merge_spotlight_kpis
from clean_dataset import clean_spotlight_kpi
from detectors import EnsembleDetector
from window_detector import detect_anomaly_windows
from anomaly_to_prompt import convert_all_windows_to_prompts
from extract_dataset import extract_anomaly_dataset
import pandas as pd
import json

def run_full_anomaly_pipeline():
    start_t = time.time()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    extracted_data_dir = os.path.join(data_dir, 'dataset_extracted')
    
    merged_csv = os.path.join(data_dir, 'merged_kpi.csv')
    cleaned_csv = os.path.join(data_dir, 'cleaned_kpi.csv')
    detected_csv = os.path.join(data_dir, 'detected_kpi.csv')
    windows_json = os.path.join(data_dir, 'anomaly_windows.json')
    windows_summary_csv = os.path.join(data_dir, 'anomaly_windows.csv')
    prompts_json = os.path.join(data_dir, 'anomaly_prompts.json')
    prompts_txt = os.path.join(data_dir, 'anomaly_prompts.txt')

    print("\n==========================================================================")
    print("  SPOTLIGHT ANOMALY DETECTION & 3GPP RAG PROMPT PIPELINE (PATH A)")
    print("==========================================================================\n")
    
    # Step 0: Extract zipped dataset
    print(">>> STAGE 0: Extracting Raw Anomaly Dataset Zips...")
    if not os.path.exists(extracted_data_dir):
        extract_anomaly_dataset()
    else:
        print(f"    [*] Found existing extracted dataset at {extracted_data_dir}")

    # Step 1: Merge KPIs (cached — source data doesn't change)
    print("\n>>> STAGE 1: Merging SpotLight Radio & Platform KPIs...")
    if not os.path.exists(merged_csv):
        merge_spotlight_kpis(extracted_data_dir, merged_csv)
    else:
        print(f"    [*] Found existing merged dataset at {merged_csv}")

    # Step 2: Clean Dataset (cached — source data doesn't change)
    print("\n>>> STAGE 2: Cleaning & Normalizing Data...")
    if not os.path.exists(cleaned_csv):
        clean_spotlight_kpi(merged_csv, cleaned_csv)
    else:
        print(f"    [*] Found existing cleaned dataset at {cleaned_csv}")

    # Step 3: Ensemble Anomaly Detection — always re-run with updated detectors
    print("\n>>> STAGE 3: Running Per-Domain Isolation Forest & Rolling Z-Score Detection...")
    df_clean = pd.read_csv(cleaned_csv)
    detector = EnsembleDetector(contamination=0.18, z_window=30, z_threshold=3.0)
    df_detected, _ = detector.run_detection(df_clean)
    df_detected.to_csv(detected_csv, index=False)
    print(f"    [SUCCESS] Saved detection results to {detected_csv}")

    # Step 4: Group Contiguous Anomaly Windows — always re-run
    print("\n>>> STAGE 4: Grouping Contiguous Anomaly Windows (per-domain baselines)...")
    windows, summary_df = detect_anomaly_windows(detected_csv)
    with open(windows_json, 'w') as f:
        json.dump(windows, f, indent=2)
    summary_df.to_csv(windows_summary_csv, index=False)

    # Step 5: Convert Anomaly Windows to RAG Prompts — always re-run
    print("\n>>> STAGE 5: Converting Anomaly Windows -> Natural Language RAG Prompts...")
    prompts = convert_all_windows_to_prompts(windows_json, prompts_json, prompts_txt, max_samples=None)

    # Step 6: Cleanup Intermediate Files
    print("\n>>> STAGE 6: Cleaning up intermediate dataset files...")
    import shutil
    files_to_remove = [merged_csv, cleaned_csv, detected_csv, windows_json, windows_summary_csv, prompts_txt]
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(extracted_data_dir):
        shutil.rmtree(extracted_data_dir)
    print("    [SUCCESS] Intermediate files removed. Kept only anomaly_prompts.json")

    total_t = time.time() - start_t
    print("\n==========================================================================")
    print(" [SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Total Execution Time : {total_t:.2f} seconds")
    print(f" Total Samples        : {len(df_detected):,}")
    print(f" Detected Windows     : {len(windows):,}")
    print(f" Generated RAG Prompts: {len(prompts):,} (Saved to anomaly_prompts.txt)")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_full_anomaly_pipeline()
