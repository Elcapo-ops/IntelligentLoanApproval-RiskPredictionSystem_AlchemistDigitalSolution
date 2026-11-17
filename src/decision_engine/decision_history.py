"""Decision history and audit trail persistence.

Uses file-based JSON storage following the project's existing artifact patterns.
No new database is introduced. Records are stored under
artifacts/decision_engine/ and can be viewed, searched, and filtered.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decision_engine.schemas import DecisionReport
from utils.exceptions import DataValidationError

HISTORY_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "decision_engine"
HISTORY_FILE = HISTORY_DIR / "decisions.jsonl"


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def record_decision(report: DecisionReport) -> None:
    """Append a decision report to the audit trail.

    Each decision is stored as a single JSON line. Sensitive personal
    information is limited to what is already in the decision report.
    """
    _ensure_dir()
    record = report_to_history_dict(report)
    with HISTORY_FILE.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, allow_nan=False) + "\n")


def report_to_history_dict(report: DecisionReport) -> dict[str, Any]:
    """Convert a DecisionReport to a history-storage dictionary."""
    from decision_engine.explanations import report_to_dict

    base = report_to_dict(report)
    base["recorded_at"] = datetime.now(timezone.utc).isoformat()
    return base


def get_all_decisions() -> list[dict[str, Any]]:
    """Return all recorded decisions, newest first."""
    _ensure_dir()
    if not HISTORY_FILE.is_file():
        return []
    records: list[dict[str, Any]] = []
    with HISTORY_FILE.open("r", encoding="utf-8") as history_file:
        for line in history_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r.get("decision_timestamp", ""), reverse=True)
    return records


def find_decision(application_id: str) -> dict[str, Any] | None:
    """Find a decision by application reference."""
    for record in get_all_decisions():
        if record.get("application_id") == application_id:
            return record
    return None


def filter_decisions(
    *,
    decision: str | None = None,
    application_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter decisions by decision outcome and/or application ID."""
    records = get_all_decisions()
    if application_id:
        records = [r for r in records if r.get("application_id") == application_id]
    if decision:
        records = [r for r in records if r.get("decision") == decision]
    if limit is not None:
        records = records[:limit]
    return records


def get_decision_counts() -> dict[str, int]:
    """Return counts of decisions by outcome."""
    counts: dict[str, int] = {"APPROVE": 0, "REJECT": 0, "REFER FOR MANUAL REVIEW": 0}
    for record in get_all_decisions():
        outcome = record.get("decision", "UNKNOWN")
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


__all__ = [
    "record_decision",
    "get_all_decisions",
    "find_decision",
    "filter_decisions",
    "get_decision_counts",
    "HISTORY_FILE",
]
