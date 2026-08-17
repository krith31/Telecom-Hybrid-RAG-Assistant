import os
import json
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# KPI human-readable names for clean prompt language
# ─────────────────────────────────────────────────────────────────────────────
KPI_LABELS = {
    'mac_sinr_update_mean':                         'Average DL SINR',
    'mac_sinr_update_min':                          'Minimum DL SINR',
    'mac_dl_harq_nack_rate':                        'DL HARQ NACK Rate',
    'mac_dl_harq_NACK':                             'DL HARQ NACK Count',
    'mac_dl_harq_total':                            'DL HARQ Total Transmission Count',
    'mac_ul_CRC_Loss_Rate':                         'UL CRC Frame Error Rate',
    'mac_ul_CRC_Loss':                              'UL CRC Error Count',
    'l1_avg_mcs':                                   'DL Modulation & Coding Scheme (MCS) Index',
    'l1_avg_UL_mcs':                                'UL Modulation & Coding Scheme (MCS) Index',
    'l1_avg_prbSize':                               'DL Physical Resource Blocks (PRB) Allocated',
    'l1_avg_UL_prbSize':                            'UL Physical Resource Blocks (PRB) Allocated',
    'l1_avg_tbs':                                   'DL Transport Block Size (TBS)',
    'l1_avg_UL_tbs':                                'UL Transport Block Size (TBS)',
    'rlc_mac_size_mean':                            'RLC-to-MAC PDU Payload Size',
    'f1u_rlc_size_mean':                            'F1-U Interface RLC Payload Size',
    'mac_dl_bo_mean':                               'DL Buffer Occupancy (MAC BO)',
    'mac_bsr_update_mean':                          'UL Buffer Status Report (BSR) Update',
    'gnb_du_layer2_f1_du_worker_0_total_runtime':   'gNB DU Layer-2 F1 Worker CPU Runtime',
    'gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime':  'PDCP Worker Thread CPU Runtime',
    'gnb_cu_pdcp_0_0_pdcp_master_0_total_runtime':  'PDCP Master Thread CPU Runtime',
    'l1app_main_ebbupool_act_0_total_runtime':       'L1 eBBU Pool Thread Runtime',
    'l1app_main_fh_main_poll-22_total_runtime':     'L1 Front-Haul Poll Thread Runtime',
    'in_octet_total_Ethernet15/2':                  'Ethernet Port RX Bytes (Front-Haul In)',
    'out_octet_total_Ethernet31/1':                 'Ethernet Port TX Bytes (Front-Haul Out)',
}

# KPIs where going to 0 means IDLE/DISCONNECTED, NOT a performance anomaly
IDLE_INDICATOR_KPIS = {
    'mac_dl_harq_total',
    'mac_dl_harq_NACK',
    'mac_ul_CRC_Loss',
}


def _classify_deviation(kpi, delta, pct_change, anomaly_mean):
    """
    Returns (is_significant, direction_str, interpretation_note).
    Handles the idle-vs-fault ambiguity for HARQ/CRC KPIs.
    """
    if abs(pct_change) <= 20.0 or abs(delta) <= 0.01:
        return False, None, None

    direction = "increased" if delta > 0 else "decreased"

    # ── Special case: HARQ/CRC count going to zero means link idle, not fault ──
    if kpi in IDLE_INDICATOR_KPIS and anomaly_mean < 0.5 and delta < 0:
        return True, "dropped to zero", "link became idle or UE disconnected during this window"

    return True, direction, None


def detect_anomaly_windows(detected_csv, max_gap_seconds=2.0):
    print(f"[*] Extracting anomaly windows from {detected_csv}...")
    df = pd.read_csv(detected_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    meta_cols = [
        'timestamp', 'domain', 'anomaly', 'if_score', 'if_flag',
        'z_score_max', 'z_flag', 'z_top_kpi', 'ensemble_score',
        'ensemble_flag',
    ]
    kpi_cols = [c for c in df.columns if c not in meta_cols]

    # ── FIX: per-domain baseline so RADIO normals don't pollute PDCP baseline ──
    domain_baselines = {}
    for domain, ddf in df.groupby('domain'):
        normal_slice = ddf[ddf['ensemble_flag'] == 0]
        if normal_slice.empty:
            normal_slice = ddf
        domain_baselines[domain] = normal_slice[kpi_cols].mean()

    # ── Sort within domain, then concatenate ──
    parts = []
    for domain, ddf in df.groupby('domain'):
        ddf = ddf.sort_values('timestamp').reset_index(drop=True)
        parts.append(ddf)
    df = pd.concat(parts, ignore_index=True)

    # Only anomalous rows
    anom_df = df[df['ensemble_flag'] == 1].copy()
    if anom_df.empty:
        print("[!] No anomalies flagged by detector.")
        return [], pd.DataFrame()

    # Group contiguous rows: new window when time gap > threshold OR domain changes
    anom_df = anom_df.sort_values(['domain', 'timestamp']).reset_index(drop=True)
    anom_df['time_diff']      = anom_df['timestamp'].diff().dt.total_seconds()
    anom_df['domain_changed'] = anom_df['domain'] != anom_df['domain'].shift(1)
    anom_df['new_window']     = (anom_df['time_diff'] > max_gap_seconds) | anom_df['domain_changed']
    anom_df['window_group']   = anom_df['new_window'].cumsum()

    windows      = []
    summary_rows = []

    for _, group in anom_df.groupby('window_group'):
        if len(group) < 3:   # discard noise-level micro-windows
            continue

        domain   = group['domain'].iloc[0]
        start_t  = group['timestamp'].min()
        end_t    = group['timestamp'].max()
        duration = (end_t - start_t).total_seconds()

        peak_z   = float(group['z_score_max'].max())
        peak_ens = float(group['ensemble_score'].max())

        # Primary KPI = most frequently the highest-Z feature in this window
        top_kpi_counts = group['z_top_kpi'].value_counts()
        primary_kpi    = top_kpi_counts.index[0] if not top_kpi_counts.empty else "unknown"

        # ── Per-domain baseline for delta computation ──
        baseline_means = domain_baselines.get(domain, domain_baselines[list(domain_baselines.keys())[0]])

        kpi_deltas       = {}
        affected_details = []       # human-readable deviation strings
        fault_kpis       = []       # KPIs that are truly degraded (not just idle)
        idle_detected    = False

        for kpi in kpi_cols:
            base_val   = float(baseline_means.get(kpi, 0.0))
            anom_val   = float(group[kpi].mean())
            delta      = anom_val - base_val
            pct_change = (delta / (abs(base_val) + 1e-6)) * 100.0

            kpi_deltas[kpi] = {
                'baseline_mean': round(base_val, 4),
                'anomaly_mean':  round(anom_val, 4),
                'delta':         round(delta, 4),
                'pct_change':    round(pct_change, 2),
            }

            significant, direction, note = _classify_deviation(kpi, delta, pct_change, anom_val)
            if not significant:
                continue

            label = KPI_LABELS.get(kpi, kpi)

            if note:            # idle indicator
                idle_detected = True
                affected_details.append(
                    f"{label} dropped to zero (link idle / UE detached during window)"
                )
            else:
                fault_kpis.append(kpi)
                affected_details.append(
                    f"{label} {direction} by {abs(pct_change):.1f}% "
                    f"(from {base_val:.2f} to {anom_val:.2f})"
                )

        win_dict = {
            'window_id':           f"WIN_{len(windows)+1:03d}",
            'domain':              domain,
            'start_time':          start_t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'end_time':            end_t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'duration_sec':        round(duration, 2),
            'sample_count':        len(group),
            'peak_ensemble_score': round(peak_ens, 4),
            'peak_z_score':        round(peak_z, 2),
            'primary_affected_kpi': primary_kpi,
            'idle_detected':       idle_detected,
            'fault_kpis':          fault_kpis,
            'kpi_summary_points':  affected_details[:6],
            'kpi_deltas':          kpi_deltas,
        }

        windows.append(win_dict)
        summary_rows.append({
            'window_id':    win_dict['window_id'],
            'domain':       domain,
            'start_time':   win_dict['start_time'],
            'end_time':     win_dict['end_time'],
            'duration_sec': win_dict['duration_sec'],
            'sample_count': win_dict['sample_count'],
            'peak_z_score': win_dict['peak_z_score'],
            'primary_kpi':  primary_kpi,
            'idle_detected': idle_detected,
            'top_deviation': affected_details[0] if affected_details else "",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(f"[SUCCESS] Extracted {len(windows)} distinct anomaly windows.")
    return windows, summary_df


if __name__ == "__main__":
    detected_csv      = os.path.join(os.path.dirname(__file__), 'Data', 'detected_kpi.csv')
    out_json          = os.path.join(os.path.dirname(__file__), 'Data', 'anomaly_windows.json')
    out_summary_csv   = os.path.join(os.path.dirname(__file__), 'Data', 'anomaly_windows.csv')

    if os.path.exists(detected_csv):
        windows, summary_df = detect_anomaly_windows(detected_csv)
        with open(out_json, 'w') as f:
            json.dump(windows, f, indent=2)
        summary_df.to_csv(out_summary_csv, index=False)
        print(f"[SUCCESS] Saved window details to {out_json} and {out_summary_csv}")
