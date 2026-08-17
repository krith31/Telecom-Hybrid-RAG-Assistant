import json
import os
import logging
from typing import Optional
from agno.tools import tool

logger = logging.getLogger(__name__)

# ─── Load anomaly database at module import (O(1) lookups at runtime) ──────────
_ANOMALY_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "anomaly_prompts.json"
)

def _load_anomaly_db() -> dict:
    """Load anomaly_prompts.json and build a window_id -> record dict."""
    if not os.path.exists(_ANOMALY_DB_PATH):
        logger.error(f"Anomaly database not found at: {_ANOMALY_DB_PATH}")
        return {}
    with open(_ANOMALY_DB_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    # Build index — support both WIN_33 and WIN_033 style lookups (case-insensitive)
    index = {}
    for record in records:
        wid = record["window_id"].upper()
        index[wid] = record
    logger.info(f"Anomaly database loaded: {len(index)} records indexed.")
    return index

_ANOMALY_INDEX: dict = _load_anomaly_db()


# ─── Tool ─────────────────────────────────────────────────────────────────────

@tool
def fetch_anomaly_window(window_id: str) -> str:
    """
    Fetches the full anomaly report for a specific network anomaly window.

    Use this tool when the user asks about a specific anomaly window (e.g. 'WIN_033',
    'window 42', 'analyse WIN_1741') or wants to understand what happened during a
    specific detected anomaly event. Do NOT call this for general telecom questions.

    Args:
        window_id: The anomaly window identifier. Accepts formats like 'WIN_033',
                   'WIN_33', 'win033', '033', or just '33'.

    Returns:
        The full structured anomaly report text for the specified window.
    """
    # Normalize: strip spaces, uppercase, ensure WIN_ prefix
    normalized = window_id.strip().upper()
    if not normalized.startswith("WIN_"):
        normalized = "WIN_" + normalized.lstrip("WIN_").lstrip("_")

    # Direct lookup
    record = _ANOMALY_INDEX.get(normalized)

    # Fallback: try stripping leading zeros from the numeric part (WIN_033 -> WIN_33)
    if record is None:
        parts = normalized.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stripped_id = f"WIN_{int(parts[1])}"
            record = _ANOMALY_INDEX.get(stripped_id)

    # Fallback: try with zero-padding up to 4 digits (WIN_33 -> WIN_033 etc.)
    if record is None:
        parts = normalized.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            for pad in range(2, 5):
                padded_id = f"WIN_{parts[1].zfill(pad)}"
                record = _ANOMALY_INDEX.get(padded_id)
                if record:
                    break

    if record is None:
        available = sorted(_ANOMALY_INDEX.keys())[:10]
        return (
            f"Anomaly window '{window_id}' was not found in the database.\n"
            f"Available examples: {', '.join(available)} (and {len(_ANOMALY_INDEX) - 10} more)."
        )

    return record["full_prompt_text"]
