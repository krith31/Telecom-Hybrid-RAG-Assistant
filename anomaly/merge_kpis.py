import os
import glob
import pandas as pd
import numpy as np

# Core 3GPP & O-RAN KPI Mapping Specification
SELECTED_RADIO_KPIS = [
    'timestamp',
    'mac_sinr_update_mean',
    'mac_sinr_update_min',
    'mac_dl_harq_nack_rate',
    'mac_dl_harq_NACK',
    'mac_dl_harq_total',
    'mac_ul_CRC_Loss_Rate',
    'mac_ul_CRC_Loss',
    'l1_avg_mcs',
    'l1_avg_UL_mcs',
    'l1_avg_prbSize',
    'l1_avg_UL_prbSize',
    'l1_avg_tbs',
    'l1_avg_UL_tbs',
    'rlc_mac_size_mean',
    'f1u_rlc_size_mean',
    'mac_dl_bo_mean',
    'mac_bsr_update_mean',
    'anomaly'
]

SELECTED_PLATFORM_KPIS = [
    'timestamp',
    'gnb_du_layer2_f1_du_worker_0_total_runtime',
    'gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime',
    'gnb_cu_pdcp_0_0_pdcp_master_0_total_runtime',
    'l1app_main_ebbupool_act_0_total_runtime',
    'l1app_main_fh_main_poll-22_total_runtime',
    'in_octet_total_Ethernet15/2',
    'out_octet_total_Ethernet31/1',
    'anomaly'
]

def find_dataset_folders(base_path):
    """Recursively find all directories containing radio.csv and platform.csv."""
    folder_pairs = []
    for root, dirs, files in os.walk(base_path):
        if 'radio.csv' in files and 'platform.csv' in files:
            # Determine domain from path
            domain = "UNKNOWN"
            upper_root = root.upper().replace('\\', '/')
            if 'RADIO' in upper_root:
                domain = 'RADIO'
            elif 'MAC' in upper_root:
                domain = 'MAC'
            elif 'NETWORK' in upper_root:
                domain = 'NETWORK'
            elif 'PDCP' in upper_root:
                domain = 'PDCP'
            elif 'MIXED' in upper_root:
                domain = 'MIXED'
            
            folder_pairs.append({
                'domain': domain,
                'radio_path': os.path.join(root, 'radio.csv'),
                'platform_path': os.path.join(root, 'platform.csv')
            })
    return folder_pairs

def load_and_merge_pair(item):
    domain = item['domain']
    radio_path = item['radio_path']
    platform_path = item['platform_path']
    
    print(f"[*] Processing domain [{domain}] from {os.path.dirname(radio_path)}...")
    
    # Load radio data
    df_radio = pd.read_csv(radio_path)
    avail_radio_cols = [c for c in SELECTED_RADIO_KPIS if c in df_radio.columns]
    df_radio = df_radio[avail_radio_cols].copy()
    
    # Load platform data
    df_plat = pd.read_csv(platform_path)
    avail_plat_cols = [c for c in SELECTED_PLATFORM_KPIS if c in df_plat.columns]
    df_plat = df_plat[avail_plat_cols].copy()
    
    # Convert timestamps
    df_radio['timestamp'] = pd.to_datetime(df_radio['timestamp'])
    df_plat['timestamp'] = pd.to_datetime(df_plat['timestamp'])
    
    df_radio = df_radio.sort_values('timestamp')
    df_plat = df_plat.sort_values('timestamp')
    
    # Drop duplicate timestamps if any
    df_radio = df_radio.drop_duplicates(subset=['timestamp'])
    df_plat = df_plat.drop_duplicates(subset=['timestamp'])
    
    # Merge asof with 500ms tolerance
    merged = pd.merge_asof(
        df_radio, 
        df_plat, 
        on='timestamp', 
        suffixes=('_radio', '_plat'),
        direction='nearest',
        tolerance=pd.Timedelta('500ms')
    )
    
    # Combine anomaly labels if both exist
    if 'anomaly_radio' in merged.columns and 'anomaly_plat' in merged.columns:
        a_radio = merged['anomaly_radio'].fillna(0).astype(float)
        a_plat = merged['anomaly_plat'].fillna(0).astype(float)
        merged['anomaly'] = np.where((a_radio > 0) | (a_plat > 0), 1, 0)
        merged.drop(columns=['anomaly_radio', 'anomaly_plat'], inplace=True)
    elif 'anomaly_radio' in merged.columns:
        merged['anomaly'] = merged['anomaly_radio'].fillna(0).astype(int)
        merged.drop(columns=['anomaly_radio'], inplace=True)
    elif 'anomaly_plat' in merged.columns:
        merged['anomaly'] = merged['anomaly_plat'].fillna(0).astype(int)
        merged.drop(columns=['anomaly_plat'], inplace=True)
        
    merged['domain'] = domain
    return merged

def merge_spotlight_kpis(data_dir, output_file):
    folder_pairs = find_dataset_folders(data_dir)
    if not folder_pairs:
        raise FileNotFoundError(f"No radio.csv and platform.csv pairs found in {data_dir}")
        
    print(f"[+] Found {len(folder_pairs)} SpotLight anomaly data pairs.")
    
    all_dfs = []
    for pair in folder_pairs:
        try:
            df = load_and_merge_pair(pair)
            all_dfs.append(df)
            print(f"    -> Loaded {len(df)} rows for {pair['domain']}")
        except Exception as e:
            print(f"    [!] Error merging {pair}: {e}")
            
    if not all_dfs:
        raise ValueError("Failed to process any SpotLight dataset pairs.")
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    full_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved merged dataset with {len(full_df)} total rows to {output_file}")
    return full_df

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), 'Data')
    out_csv = os.path.join(os.path.dirname(__file__), 'Data', 'merged_kpi.csv')
    merge_spotlight_kpis(base_dir, out_csv)
