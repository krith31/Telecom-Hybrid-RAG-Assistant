import os
import pandas as pd
import numpy as np

def clean_spotlight_kpi(input_file, output_file):
    print(f"[*] Loading merged KPI dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"    -> Raw shape: {df.shape}")
    
    # 1. Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Separate metadata columns vs feature columns
    meta_cols = ['timestamp', 'domain', 'anomaly']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # 3. Replace infinite values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # 4. Handle missing values per column
    for col in feature_cols:
        # Interpolate linear over time
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        # Fallback to median if whole column was NaN
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            
    # 5. Remove invariant constant columns
    invariant_cols = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            invariant_cols.append(col)
            
    if invariant_cols:
        print(f"[*] Dropping {len(invariant_cols)} invariant (constant) columns: {invariant_cols}")
        feature_cols = [c for c in feature_cols if c not in invariant_cols]
        df = df[meta_cols + feature_cols]
        
    # 6. Ensure numeric types for feature columns
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    print(f"[SUCCESS] Cleaned dataset shape: {df.shape}")
    print(f"          Anomalous samples count: {df['anomaly'].sum()} / {len(df)} ({df['anomaly'].mean()*100:.2f}%)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved clean dataset to {output_file}")
    return df

if __name__ == "__main__":
    in_csv = os.path.join(os.path.dirname(__file__), 'Data', 'merged_kpi.csv')
    out_csv = os.path.join(os.path.dirname(__file__), 'Data', 'cleaned_kpi.csv')
    clean_spotlight_kpi(in_csv, out_csv)
