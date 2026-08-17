import os
import json


# ─────────────────────────────────────────────────────────────────────────────
# Domain → Telecom Layer name (for human context only, NOT for LLM query)
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_LAYER = {
    'RADIO':   'L1 Physical Layer & RF Channel (5G NR Radio)',
    'MAC':     'Layer 2 MAC Sublayer (MAC Scheduler)',
    'NETWORK': 'Transport Layer (F1-U / Front-Haul Interface)',
    'PDCP':    'Layer 2 PDCP & RLC Sublayer',
    'MIXED':   'Cross-Layer (PHY + MAC + Transport)',
}

# ─────────────────────────────────────────────────────────────────────────────
# Fault pattern classifier — reads which KPIs moved and infers a fault label
# This is what drives the diagnostic question, not the domain name.
# ─────────────────────────────────────────────────────────────────────────────
def _classify_fault_pattern(kpi_deltas, idle_detected, fault_kpis, domain):
    """
    Returns (pattern_label, severity_description) based on which KPIs deviated.
    """
    # Shorthand
    mcs_drop    = kpi_deltas.get('l1_avg_mcs', {}).get('pct_change', 0)
    prb_drop    = kpi_deltas.get('l1_avg_prbSize', {}).get('pct_change', 0)
    ul_prb_drop = kpi_deltas.get('l1_avg_UL_prbSize', {}).get('pct_change', 0)
    nack_rate   = kpi_deltas.get('mac_dl_harq_nack_rate', {}).get('pct_change', 0)
    crc_rate    = kpi_deltas.get('mac_ul_CRC_Loss_Rate', {}).get('pct_change', 0)
    sinr_drop   = kpi_deltas.get('mac_sinr_update_mean', {}).get('pct_change', 0)
    bsr_change  = kpi_deltas.get('mac_bsr_update_mean', {}).get('pct_change', 0)
    pdcp_runtime= kpi_deltas.get('gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime', {}).get('pct_change', 0)
    du_runtime  = kpi_deltas.get('gnb_du_layer2_f1_du_worker_0_total_runtime', {}).get('pct_change', 0)
    eth_rx      = kpi_deltas.get('in_octet_total_Ethernet15/2', {}).get('pct_change', 0)

    # ── Pattern detection rules ──

    # 1. Radio interference / channel degradation
    if mcs_drop < -50 and (crc_rate > 200 or sinr_drop < -20):
        return (
            "Radio Channel Degradation / Interference",
            f"Downlink MCS index collapsed by {abs(mcs_drop):.0f}%, forcing low-order modulation. "
            f"Uplink CRC error rate spiked by {crc_rate:.0f}%, indicating severe interference or "
            f"path loss increase causing link quality collapse."
        )

    # 2. PRB starvation / scheduler breakdown
    if prb_drop < -40 and ul_prb_drop < -40 and mcs_drop < -20:
        return (
            "PHY/MAC Scheduler Resource Starvation",
            f"Both downlink PRB allocation ({abs(prb_drop):.0f}% drop) and uplink PRB allocation "
            f"({abs(ul_prb_drop):.0f}% drop) collapsed simultaneously alongside MCS degradation "
            f"({abs(mcs_drop):.0f}% drop), indicating the MAC scheduler stopped granting resources."
        )

    # 3. HARQ-driven throughput collapse
    if nack_rate > 100 and mcs_drop < -30:
        return (
            "HARQ Retransmission Overload",
            f"Downlink HARQ NACK rate spiked by {nack_rate:.0f}%, overwhelming the retransmission "
            f"buffer. Concurrent MCS drop ({abs(mcs_drop):.0f}%) indicates link adaptation could not "
            f"keep pace with channel deterioration, causing throughput collapse."
        )

    # 4. Uplink CRC-only fault (UL path issue)
    if crc_rate > 500 and mcs_drop > -20:
        return (
            "Uplink Frame Integrity Failure (CRC Errors)",
            f"Uplink CRC error rate surged by {crc_rate:.0f}% while downlink metrics remained "
            f"relatively stable, indicating an uplink-specific issue such as UE transmit power "
            f"limitation, timing misalignment, or interference only on the UL band."
        )

    # 5. PDCP / RLC processing overload
    if pdcp_runtime > 50 or du_runtime > 50:
        which = []
        if pdcp_runtime > 50:
            which.append(f"PDCP worker CPU runtime increased by {pdcp_runtime:.0f}%")
        if du_runtime > 50:
            which.append(f"DU L2 worker CPU runtime increased by {du_runtime:.0f}%")
        return (
            "PDCP/RLC Software Processing Overload",
            ". ".join(which) + ". This indicates software thread contention or buffer "
            "congestion in the PDCP/RLC stack, causing packet re-ordering delays or drops "
            "without any RF channel degradation."
        )

    # 6. BSR-driven UL congestion
    if bsr_change > 100 and ul_prb_drop < -30:
        return (
            "Uplink MAC Buffer Status Report (BSR) Congestion",
            f"UL Buffer Status Reports surged by {bsr_change:.0f}% while UL PRB allocation "
            f"dropped by {abs(ul_prb_drop):.0f}%, indicating UE UL data backlog grew but the "
            f"gNB scheduler could not grant sufficient uplink resources to clear it."
        )

    # 7. Front-Haul / Transport layer
    if eth_rx < -40 or (domain == 'NETWORK' and len(fault_kpis) > 0):
        return (
            "Front-Haul Transport Layer Congestion or Link Failure",
            f"Ethernet port RX byte throughput dropped by {abs(eth_rx):.0f}%, indicating "
            f"front-haul bandwidth saturation, cable/switch fault, or QoS flow throttling "
            f"between the RU and DU over the F1-U interface."
        )

    # 8. Link idle / UE detached (not a true fault)
    if idle_detected and len(fault_kpis) == 0:
        return (
            "Link Idle / UE Detachment",
            "No active UE transmissions were observed during this window. "
            "HARQ and CRC counts dropped to zero because the link was idle, "
            "not because of a performance fault."
        )

    # 9. Generic multi-KPI degradation
    if len(fault_kpis) >= 3:
        return (
            "Correlated Multi-KPI Degradation",
            f"Multiple KPIs across {domain} degraded simultaneously — "
            f"{', '.join(fault_kpis[:3])} among others — suggesting a single upstream root cause "
            f"rather than independent metric failures."
        )

    return ("Anomalous Network Behavior", "One or more KPIs deviated significantly from their domain baseline.")


# ─────────────────────────────────────────────────────────────────────────────
# Build the actual question that will be fed to the LLM via RAG
# No document names. No "according to TS 38.xxx". Just clear telecom language.
# ─────────────────────────────────────────────────────────────────────────────
def _build_rag_query(pattern_label, kpi_deltas, fault_kpis, domain, duration_sec, idle_detected):
    """
    Generates a targeted diagnostic question derived entirely from the
    observed data. The RAG retriever decides which documents are relevant.
    """
    # Pure idle window with no active fault KPIs
    if idle_detected and not fault_kpis:
        return (
            "Why would a 5G NR cell show zero HARQ transmissions and zero CRC errors "
            "during a monitoring window, and what does this indicate about UE connectivity state?"
        )

    # Collect active fault KPI observations (NOT the idle-consequence ones)
    notable = []
    mcs     = kpi_deltas.get('l1_avg_mcs', {})
    ul_mcs  = kpi_deltas.get('l1_avg_UL_mcs', {})
    crc     = kpi_deltas.get('mac_ul_CRC_Loss_Rate', {})
    nack    = kpi_deltas.get('mac_dl_harq_nack_rate', {})
    prb     = kpi_deltas.get('l1_avg_prbSize', {})
    ul_prb  = kpi_deltas.get('l1_avg_UL_prbSize', {})
    sinr    = kpi_deltas.get('mac_sinr_update_mean', {})
    pdcp    = kpi_deltas.get('gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime', {})
    bsr     = kpi_deltas.get('mac_bsr_update_mean', {})
    f1u     = kpi_deltas.get('f1u_rlc_size_mean', {})

    if mcs.get('pct_change', 0) < -20:
        notable.append(
            f"downlink MCS index fell from {mcs['baseline_mean']:.1f} to "
            f"{mcs['anomaly_mean']:.1f} ({abs(mcs['pct_change']):.0f}% reduction — "
            f"gNB link adaptation forced the scheduler to use lower-order modulation)"
        )
    if ul_mcs.get('pct_change', 0) < -20:
        notable.append(
            f"uplink MCS index fell from {ul_mcs['baseline_mean']:.1f} to "
            f"{ul_mcs['anomaly_mean']:.1f} ({abs(ul_mcs['pct_change']):.0f}% reduction — "
            f"UE transmit power or UL channel quality degraded)"
        )
    if crc.get('pct_change', 0) > 100:
        notable.append(
            f"uplink CRC frame error rate spiked by {crc['pct_change']:.0f}% "
            f"(from {crc['baseline_mean']:.3f} to {crc['anomaly_mean']:.3f})"
        )
    if nack.get('pct_change', 0) > 100:
        notable.append(
            f"downlink HARQ NACK rate increased by {nack['pct_change']:.0f}% "
            f"(from {nack['baseline_mean']:.3f} to {nack['anomaly_mean']:.3f}), "
            f"driving excessive retransmissions and buffer backlog"
        )
    if prb.get('pct_change', 0) < -30:
        notable.append(
            f"downlink PRB allocation fell by {abs(prb['pct_change']):.0f}% "
            f"(from {prb['baseline_mean']:.1f} to {prb['anomaly_mean']:.1f} resource blocks — "
            f"scheduler resource grants stopped or were heavily reduced)"
        )
    if ul_prb.get('pct_change', 0) < -30:
        notable.append(
            f"uplink PRB allocation fell by {abs(ul_prb['pct_change']):.0f}% "
            f"(from {ul_prb['baseline_mean']:.1f} to {ul_prb['anomaly_mean']:.1f} resource blocks)"
        )
    if sinr.get('pct_change', 0) < -15:
        notable.append(
            f"average downlink SINR dropped by {abs(sinr['pct_change']):.0f}% "
            f"(from {sinr['baseline_mean']:.1f} to {sinr['anomaly_mean']:.1f} dB)"
        )
    if pdcp.get('pct_change', 0) > 50:
        notable.append(
            f"PDCP worker thread CPU runtime increased by {pdcp['pct_change']:.0f}%, "
            f"indicating software stack congestion or buffer saturation"
        )
    if bsr.get('pct_change', 0) > 100:
        notable.append(
            f"UL Buffer Status Report values surged by {bsr['pct_change']:.0f}%, "
            f"indicating a growing UE uplink data backlog"
        )
    if f1u.get('pct_change', 0) > 50:
        notable.append(
            f"F1-U interface RLC payload size increased by {f1u['pct_change']:.0f}%, "
            f"indicating backhaul buffering or front-haul congestion"
        )

    if not notable:
        notable.append(f"multiple {domain}-layer KPIs deviated significantly from their domain-specific baseline")

    observed_str = "; ".join(notable)

    # When idle is detected alongside real fault KPIs, add the RLF causal sequence
    rlf_context = ""
    if idle_detected and fault_kpis:
        rlf_context = (
            f" The cell also stopped all HARQ transmissions by the end of the window, "
            f"suggesting the degradation progressed to a Radio Link Failure (RLF) or UE detachment event."
        )

    query = (
        f"A 5G NR gNB cell experienced a '{pattern_label}' lasting {duration_sec:.1f} seconds. "
        f"The following active fault indicators were observed: {observed_str}.{rlf_context} "
        f"What is the most likely sequence of events that produced this pattern? "
        f"What are the specific root causes for the MCS fallback and PRB starvation, "
        f"and what recovery procedures should the operator execute to restore normal cell performance?"
    )
    return query


# ─────────────────────────────────────────────────────────────────────────────
# Main function — build full structured prompt for one anomaly window
# ─────────────────────────────────────────────────────────────────────────────
def generate_rag_prompt_for_window(window):
    w_id          = window['window_id']
    domain        = window['domain']
    start_t       = window['start_time']
    end_t         = window['end_time']
    dur           = window['duration_sec']
    peak_z        = window['peak_z_score']
    summary_pts   = window.get('kpi_summary_points', [])
    kpi_deltas    = window.get('kpi_deltas', {})
    fault_kpis    = window.get('fault_kpis', [])
    idle_detected = window.get('idle_detected', False)

    layer_name = DOMAIN_LAYER.get(domain, domain)

    # Classify fault pattern from data
    pattern_label, pattern_description = _classify_fault_pattern(
        kpi_deltas, idle_detected, fault_kpis, domain
    )

    # Build the plain-English diagnostic query for the LLM
    rag_query = _build_rag_query(pattern_label, kpi_deltas, fault_kpis, domain, dur, idle_detected)

    # ── Separate fault KPI observations from idle-consequence KPIs ──
    # Idle indicators (HARQ/CRC=0) are consequences of the fault, not the fault itself.
    # Presenting them mixed together confuses the LLM into hedging.
    idle_pts  = [pt for pt in summary_pts if 'idle' in pt or 'dropped to zero' in pt]
    active_pts = [pt for pt in summary_pts if pt not in idle_pts]

    if active_pts:
        active_block = "\n".join(f"  - {pt}" for pt in active_pts)
    else:
        active_block = "  - No dominant active fault KPI identified; review raw time-series for transient glitches."

    if idle_pts:
        idle_block = "\nCONSEQUENCE (link became idle / UE detached by end of window):\n" + \
                     "\n".join(f"  - {pt}" for pt in idle_pts)
    else:
        idle_block = ""

    # Causal sequence note: if we have BOTH active faults AND idle outcome,
    # tell the LLM this represents a progression (degradation → RLF → detach)
    if idle_pts and active_pts:
        causal_note = (
            f"\nCAUSAL SEQUENCE:\n"
            f"  The active fault KPIs (MCS collapse, PRB starvation) were observed DURING "
            f"the window. The HARQ/CRC activity then dropped to zero by the window's end, "
            f"indicating the channel degradation progressed to a Radio Link Failure (RLF) "
            f"or UE detachment. The MCS and PRB degradation are the CAUSE; the HARQ idle state is the EFFECT."
        )
    else:
        causal_note = ""

    # ── Prompt structure: what goes INTO the LLM ──
    # No document names. The RAG retriever handles document selection.
    prompt_text = f"""=== 5G O-RAN ANOMALY REPORT [{w_id}] ===

Time Window     : {start_t}  to  {end_t}  (Duration: {dur}s)
Fault Pattern   : {pattern_label}
Peak Z-Score    : {peak_z:.2f}

ACTIVE FAULT INDICATORS:
{active_block}{idle_block}{causal_note}

WHAT IT MEANS:
{pattern_description}

DIAGNOSTIC QUESTION:
{rag_query}"""

    return {
        'window_id':          w_id,
        'full_prompt_text':   prompt_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch conversion utility
# ─────────────────────────────────────────────────────────────────────────────
def convert_all_windows_to_prompts(windows_json_path, out_json_path, out_txt_path, max_samples=50):
    print(f"[*] Reading anomaly windows from {windows_json_path}...")
    with open(windows_json_path, 'r') as f:
        windows = json.load(f)

    # Sort by peak Z-score descending (most severe first)
    windows_sorted = sorted(windows, key=lambda x: x['peak_z_score'], reverse=True)

    # Exclude pure idle windows from top-N export (they're not useful RAG queries)
    non_idle = [w for w in windows_sorted if not w.get('idle_detected', False) or w.get('fault_kpis')]
    selected = non_idle[:max_samples]

    print(f"[*] Generating prompts for {len(selected)} high-severity windows (idle-only excluded)...")

    prompts     = []
    text_blocks = []

    for win in selected:
        prompt_obj = generate_rag_prompt_for_window(win)
        prompts.append(prompt_obj)
        text_blocks.append(prompt_obj['full_prompt_text'])

    with open(out_json_path, 'w') as f:
        json.dump(prompts, f, indent=2)

    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write(("\n\n" + ("=" * 80) + "\n\n").join(text_blocks))

    print(f"[SUCCESS] Saved {len(prompts)} structured RAG prompts to:")
    print(f"          {out_json_path}")
    print(f"          {out_txt_path}")
    return prompts


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    win_json = os.path.join(data_dir, 'anomaly_windows.json')
    out_json = os.path.join(data_dir, 'anomaly_prompts.json')
    out_txt  = os.path.join(data_dir, 'anomaly_prompts.txt')

    if os.path.exists(win_json):
        convert_all_windows_to_prompts(win_json, out_json, out_txt)
