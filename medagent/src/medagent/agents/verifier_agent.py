"""
Verifier Agent -- Verification Layer (TRD 3.1).

Runs after the Report Agent. Checks the draft report for contradictions
or unsupported claims against the Diagnosis Agent's finding and the
Evidence Agent's retrieved guideline text.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from medagent.agents.state import AgentState
from medagent.utils.settings import get_settings

logger = logging.getLogger("medagent.agents.verifier")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verifier_prompt.txt"

_FALLBACK_PROMPT = """\
You are an expert AI clinical quality assurance verifier. 
Your job is to check if the drafted clinical report contradicts the diagnosis findings or the retrieved medical evidence.
If the draft report adds fabricated information (e.g., wrong anatomy, incorrect severity, or hallucinated evidence), you MUST flag it.
If the report is entirely consistent with the facts provided, you must pass it.

FINDING LABEL: {finding_label}
SEVERITY: {severity}
REASONING: {clinical_reasoning}
EVIDENCE: {retrieved_evidence}

DRAFT REPORT TO VERIFY:
{draft_report}

RESPOND ONLY WITH VALID JSON. The JSON must contain exactly two keys:
"status": either "passed" or "flagged"
"notes": a string explaining the specific issues if flagged, or an empty string if passed.
"""

class VerificationResult(BaseModel):
    """Structured verdict the LLM is constrained to return."""
    status: Literal["passed", "flagged"] = Field(
        ...,
        description="'passed' if the draft report is fully consistent... 'flagged' if it contains any contradiction."
    )
    notes: str = Field(
        ...,
        description="Specific issue(s) found if flagged; empty string if passed.",
    )


def verifier_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Verifier Agent.
    """
    case_id = state.get("case_id", "unknown")
    current_count = state.get("regeneration_count") or 0

    try:
        settings = get_settings()
        findings = state.get("diagnosis_findings") or {}

        if _PROMPT_PATH.exists():
            prompt_template = _PROMPT_PATH.read_text()
            if "JSON" not in prompt_template:
                prompt_template += "\n\nRESPOND ONLY WITH VALID JSON containing exactly two keys: 'status' and 'notes'."
        else:
            logger.warning("verifier_prompt.txt not found. Using fallback prompt.")
            prompt_template = _FALLBACK_PROMPT

        format_args = {
            "finding_label": findings.get("finding_label", "Undetermined"),
            "severity": findings.get("severity", "high"),
            "clinical_reasoning": findings.get("clinical_reasoning", ""),
            "retrieved_evidence": state.get("retrieved_evidence") or "No evidence retrieved.",
            "draft_report": state.get("draft_report") or "",
            "diagnosis_findings": json.dumps(findings, indent=2),
            "case_id": case_id
        }

        # Bypass .format() completely
        prompt = prompt_template
        for key, value in format_args.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        llm = ChatOllama(
            model="llama3.1",
            temperature=0.0,
            format="json",  
            num_predict=settings.llm_max_new_tokens,
        )

        response = llm.invoke(prompt)
        
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        parsed_data = json.loads(content.strip())
        
        # --- THE FIX: CLEANUP AI OUTPUT VARIATIONS ---
        # 1. Rename verification_status to status
        if "verification_status" in parsed_data:
            parsed_data["status"] = parsed_data.pop("verification_status")
            
        # 2. Force status to lowercase
        if "status" in parsed_data and isinstance(parsed_data["status"], str):
            parsed_data["status"] = parsed_data["status"].lower()
        
        # 3. Rename reason to notes
        if "reason" in parsed_data and "notes" not in parsed_data:
            parsed_data["notes"] = parsed_data.pop("reason")
            
        result = VerificationResult(**parsed_data)

        new_count = current_count + 1 if result.status == "flagged" else current_count

        if result.status == "flagged":
            logger.warning(
                "Case %s flagged by verifier (attempt %d): %s", case_id, new_count, result.notes
            )
        else:
            logger.info("Case %s: verifier passed", case_id)

        return {
            "verification_status": result.status,
            "verification_notes": result.notes,
            "regeneration_count": new_count,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("verifier_agent_node failed for case=%s", case_id)
        return {"errors": [f"verifier_agent_node: {type(exc).__name__}: {exc}"]}


# ─────────────────────────────────────────────────────────────────────
# Standalone Execution / Testing Block
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("\n[INIT] Starting Verifier Agent Test Run...")
    
    mock_state: AgentState = {
        "case_id": "TEST-VERIFIER-001",
        "diagnosis_findings": {
            "finding_label": "Pneumonia",
            "anatomical_region": "Right lower lobe",  
            "severity": "high",
            "clinical_reasoning": "Consolidation detected."
        },
        "retrieved_evidence": "[1] ATS/IDSA Guidelines: Empiric antibiotic therapy...",
        # HALLUCINATION INJECTED BELOW (Says Left Upper Lobe instead of Right Lower):
        "draft_report": "FINDINGS: Pneumonia with consolidation detected in the left upper lobe. IMPRESSION: High-severity pneumonia.",
        "regeneration_count": 0,
        "errors": []
    }
    
    print("\n[INPUT] Feeding mock data to Verifier Agent (Intentional error included)...")
    print("[PROCESSING] Asking Ollama to verify the report (this may take a few seconds)...")
    
    result_state = verifier_agent_node(mock_state)
    
    print("\n[OUTPUT] Final Verdict from Verifier Agent:")
    print(json.dumps(result_state, indent=2))