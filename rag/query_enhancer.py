import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

TELECOM_GLOSSARY: Dict[str, str] = {
    "SSB": "SSB (Synchronization Signal Block)",
    "gNB": "gNB (Next Generation Node B / 5G Base Station)",
    "RSRP": "RSRP (Reference Signal Received Power)",
    "RSRQ": "RSRQ (Reference Signal Received Quality)",
    "SINR": "SINR (Signal-to-Interference-plus-Noise Ratio)",
    "PUSCH": "PUSCH (Physical Uplink Shared Channel)",
    "PDCCH": "PDCCH (Physical Downlink Control Channel)",
    "PDSCH": "PDSCH (Physical Downlink Shared Channel)",
    "AMF": "AMF (Access and Mobility Management Function)",
    "SMF": "SMF (Session Management Function)",
    "UPF": "UPF (User Plane Function)",
    "DU": "DU (Distributed Unit)",
    "CU": "CU (Central Unit)",
    "RU": "RU (Radio Unit)",
    "MIMO": "MIMO (Multiple-Input Multiple-Output)",
    "URLLC": "URLLC (Ultra-Reliable Low-Latency Communication)",
    "eMBB": "eMBB (Enhanced Mobile Broadband)",
    "mMTC": "mMTC (Massive Machine-Type Communications)",
    "S-NSSAI": "S-NSSAI (Single Network Slice Selection Assistance Information)",
    "NSSAI": "NSSAI (Network Slice Selection Assistance Information)",
    "NGAP": "NGAP (Next Generation Application Protocol)",
    "XnAP": "XnAP (Xn Application Protocol)",
    "PDU": "PDU (Protocol Data Unit)",
    "RAN": "RAN (Radio Access Network)",
    "O-RAN": "O-RAN (Open Radio Access Network)",
    "RIC": "RIC (RAN Intelligent Controller)",
    "HO": "HO (Handover)",
    "RLF": "RLF (Radio Link Failure)",
    "QoS": "QoS (Quality of Service)",
    "5QI": "5QI (5G QoS Identifier)",
}

class QueryEnhancer:
    """
    Telecom query processing engine for acronym expansion, sub-query decomposition,
    and Path A signal anomaly text conversion.
    """
    def __init__(self, glossary: Dict[str, str] = TELECOM_GLOSSARY):
        self.glossary = glossary

    def expand_acronyms(self, query: str) -> str:
        """
        Identify telecom acronyms in the query and append their expanded definitions.
        """
        enhanced_terms = []
        words = re.findall(r'\b[A-Za-z0-9\-]+\b', query)

        for word in words:
            upper_word = word.upper()
            if upper_word in self.glossary:
                enhanced_terms.append(self.glossary[upper_word])

        if not enhanced_terms:
            return query

        expanded_str = ", ".join(list(set(enhanced_terms)))
        return f"{query} [Context Expansions: {expanded_str}]"

    def decompose_query(self, query: str) -> List[str]:
        """
        Generate expanded sub-queries for multi-faceted telecom questions.
        """
        sub_queries = [query]
        # Add expanded version
        expanded = self.expand_acronyms(query)
        if expanded != query:
            sub_queries.append(expanded)
        return sub_queries

    def format_anomaly_report(self, anomaly_data: Dict[str, Any]) -> str:
        """
        Path A Signal-to-Language Bridge: Converts mathematical KPI anomaly data into
        structured technical text suitable for RAG root cause analysis.
        """
        cell_id = anomaly_data.get("cell_id", "Unknown Cell")
        timestamp = anomaly_data.get("timestamp", "N/A")
        anomalies = anomaly_data.get("anomalies", [])
        confidence = anomaly_data.get("ensemble_confidence", "High")

        report_lines = [
            f"ANOMALY REPORT — Cell ID: {cell_id} [{timestamp}] (Confidence: {confidence})",
            "Observed Degradations:"
        ]

        for item in anomalies:
            metric = item.get("metric", "KPI")
            val = item.get("value", 0)
            baseline = item.get("baseline", 0)
            z_score = item.get("z_score", 0)
            trend = item.get("trend", "drop")
            report_lines.append(
                f" • {metric}: current value {val} (baseline: {baseline}, Z-score: {z_score}, trend: {trend})"
            )

        report_lines.append(
            f"RAG Query: What are the root causes and 3GPP troubleshooting procedures when {cell_id} experiences "
            f"correlated degradation in {', '.join([a.get('metric', '') for a in anomalies])}?"
        )

        return "\n".join(report_lines)

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Full query enhancement pipeline.
        """
        enhanced_text = self.expand_acronyms(query)
        sub_queries = self.decompose_query(query)

        return {
            "raw_query": query,
            "enhanced_query": enhanced_text,
            "sub_queries": sub_queries
        }
