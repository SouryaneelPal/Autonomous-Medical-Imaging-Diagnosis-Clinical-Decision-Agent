"""
Verifier Agent -- Verification Layer (TRD 3.1).

Cross-checks consistency across the diagnosis, retrieved evidence, and
drafted report text *before* a human clinician ever sees it. This is a
quality gate, not a diagnostic authority: it can only send the draft
back to report_agent (bounded by MAX_REGENERATIONS in orchestrator.py)
or wave it through to human_review. It never approves or finalizes a
case itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

from medagent.agents.state import AgentState
from medagent.llm.client import get_llm_client

logger = logging.getLogger("medagent.agents.verifier")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verifier_prompt.txt"


def verifier_agent_node(state: AgentState) -> dict:
    llm = get_llm_client()

    findings = state.get("diagnosis_findings") or {}

    prompt = _PROMPT_PATH.read_text().format(
        finding_label=findings.get("finding_label", "Undetermined"),
        severity=findings.get("severity", "high"),
        clinical_reasoning=findings.get("clinical_reasoning", ""),
        retrieved_evidence=state.get("retrieved_evidence") or "No evidence retrieved.",
        draft_report=state["draft_report"],
    )

    verdict = llm.invoke(prompt)

    is_consistent = verdict.get("is_consistent", False)
    notes = verdict.get("notes", "")

    status = "consistent" if is_consistent else "flagged"

    if status == "flagged":
        logger.info("Case %s flagged by verifier: %s", state["case_id"], notes)

    return {
        "verification_status": status,
        "verification_notes": notes,
        "regeneration_count": state.get("regeneration_count", 0) + (0 if is_consistent else 1),
    }
