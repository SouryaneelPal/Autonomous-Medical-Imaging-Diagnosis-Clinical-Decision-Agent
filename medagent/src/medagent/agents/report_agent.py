"""
Report Agent -- drafts the structured clinical report that the
clinician ultimately sees in the Human Oversight Dashboard (PRD 2.2).

Runs both on the "happy path" and whenever the Verifier Agent loops
execution back here after flagging an inconsistency -- in the latter
case `state["verification_notes"]` is populated and should be used to
correct the regenerated draft.
"""
from __future__ import annotations

import logging
from pathlib import Path

from medagent.agents.state import AgentState
from medagent.llm.client import get_llm_client

logger = logging.getLogger("medagent.agents.report")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "report_prompt.txt"


def report_agent_node(state: AgentState) -> dict:
    llm = get_llm_client()

    correction_notes = (
        state.get("verification_notes")
        if state.get("verification_status") == "flagged"
        else "None -- first draft."
    )

    findings = state.get("diagnosis_findings") or {}

    prompt = _PROMPT_PATH.read_text().format(
        finding_label=findings.get("finding_label", "Undetermined"),
        anatomical_region=findings.get("anatomical_region", "unknown"),
        severity=findings.get("severity", "high"),
        clinical_reasoning=findings.get("clinical_reasoning", ""),
        detections=state["detections"],
        retrieved_evidence=state.get("retrieved_evidence") or "No evidence retrieved.",
        patient_metadata=state["patient_metadata"],
        correction_notes=correction_notes,
    )

    draft = llm.invoke(prompt)

    return {"draft_report": draft.get("report_text", draft) if isinstance(draft, dict) else draft}
