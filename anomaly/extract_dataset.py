import os
import zipfile
import shutil

def extract_anomaly_dataset():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    source_dir = os.path.join(base_dir, 'data', 'Anomaly Dataset')
    target_dir = os.path.join(base_dir, 'data', 'dataset_extracted')
    
    if not os.path.exists(source_dir):
        print(f"[!] Source dataset not found at {source_dir}. Ensure 'Anomaly Dataset' is uploaded to the data folder.")
        return
        
    print(f"[*] Extracting zipped datasets from {source_dir} to {target_dir}...")
    
    # Clean previous extractions
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    extracted_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                # Compute relative path from source_dir to preserve structure
                rel_path = os.path.relpath(root, source_dir)
                
                # Use the zip filename (without .zip) as the leaf directory so radio.csv and platform.csv are inside
                leaf_dir = file.replace('.zip', '')
                extract_path = os.path.join(target_dir, rel_path, leaf_dir)
                
                os.makedirs(extract_path, exist_ok=True)
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    extracted_count += 1
                except zipfile.BadZipFile:
                    print(f"    [!] Failed to extract {zip_path}: Bad Zip File")
                    
    print(f"[SUCCESS] Extracted {extracted_count} datasets to {target_dir}")

if __name__ == "__main__":
    extract_anomaly_dataset()
