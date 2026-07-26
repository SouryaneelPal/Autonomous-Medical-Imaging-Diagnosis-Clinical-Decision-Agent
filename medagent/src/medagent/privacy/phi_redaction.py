"""
PHI-adjacent audit logging and redaction for the clinical decision
pipeline (PRD 2.3 "Data Privacy").

Called from orchestrator.py's finalize_report_node and archive_case_node
to record every human decision (approve/revise/reject) to an
append-only local audit log, with any free-text reviewer feedback
scrubbed of common PHI patterns before it's written to disk.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("medagent.privacy")

AUDIT_LOG_PATH = Path("data/audit_log.jsonl")

# (label, compiled pattern). More specific patterns are listed first;
# none of these currently overlap in a way that ordering would change
# the result, but keep specific-before-general if you add more.
_PHI_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    # SSN: 123-45-6789 or 123.45.6789 (not bare 9-digit runs -- too many
    # false positives against MRNs, phone numbers, and other IDs).
    ("SSN", re.compile(r"\b\d{3}[-.]\d{2}[-.]\d{4}\b")),
    # Mock MRN: "MRN" followed by 6-10 digits, e.g. "MRN-1029384" or "MRN 102938471".
    # Real MRN formats vary by institution; adjust this pattern to match yours.
    ("MRN", re.compile(r"\bMRN[-:\s]*\d{6,10}\b", re.IGNORECASE)),
    # US-style phone numbers: (123) 456-7890, 123-456-7890, 123.456.7890,
    # +1 123 456 7890.
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    # Dates: MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD. Catches DOB-shaped values
    # along with any other date in the same format -- regex alone can't
    # tell "date of birth" from "date of imaging" by context, so this is
    # deliberately broad rather than falsely precise.
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
]


def redact_phi(text: str) -> str:
    """
    Strips common PHI-shaped substrings (SSNs, phone numbers,
    date-of-birth-shaped dates, mock MRNs) from free text, replacing
    each match with a labeled placeholder like "[REDACTED-SSN]".

    This is a lightweight regex pass, not a clinical-grade
    de-identification tool: it will miss PHI that doesn't match these
    shapes (names, addresses, unusual local ID formats) and can
    occasionally over-redact look-alike numbers. Treat it as a floor,
    not a guarantee -- swap in a proper NLP-based de-identifier (e.g.
    Presidio) before this handles real patient data at scale.
    """
    if not text:
        return text

    redacted = text
    for label, pattern in _PHI_PATTERNS:
        redacted = pattern.sub(f"[REDACTED-{label}]", redacted)
    return redacted


def write_audit_entry(
    case_id: str,
    event: str,
    decision: str | None = None,
    reviewer_feedback: str | None = None,
) -> None:
    """
    Appends one JSON line to the local append-only audit log recording a
    clinical decision event.

    NOTE ON THIS SIGNATURE: orchestrator.py's finalize_report_node and
    archive_case_node call this exact shape --
        write_audit_entry(case_id=..., event=..., decision=..., reviewer_feedback=...)
    -- today. `reviewer_feedback` is the one free-text field a clinician
    could type PHI into, so it's the only field passed through
    redact_phi() before being written.

    Never raises: an audit-logging failure (disk full, permission
    error) must not be allowed to crash the graph after a clinician has
    already made a real decision. Failures are logged loudly instead of
    propagating.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "event": event,
        "decision": decision,
        "reviewer_feedback": redact_phi(reviewer_feedback) if reviewer_feedback else reviewer_feedback,
    }

    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Audit entry written for case=%s event=%s", case_id, event)
    except OSError:
        logger.exception(
            "Failed to write audit entry for case=%s event=%s -- continuing without "
            "blocking the graph, but this entry is LOST and needs manual follow-up.",
            case_id, event,
        )