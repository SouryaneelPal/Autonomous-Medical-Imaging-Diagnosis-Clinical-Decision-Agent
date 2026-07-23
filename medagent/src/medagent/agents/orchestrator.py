"""
Main agentic state machine for the Autonomous Medical Imaging Diagnosis &
Clinical Decision Agent.

Pipeline (TRD 3.1):

    START
      -> perceive_image        (AI Perception Layer: classify + localize + Grad-CAM)
      -> diagnosis_agent        (Agentic Intelligence Layer -> diagnosis_findings)
      -> evidence_agent          (Knowledge Layer / RAG over ATS-IDSA guidelines
                                   -> a single formatted retrieved_evidence string)
      -> report_agent             (drafts structured clinical report)
      -> verifier_agent            (Verification Layer; loops back to report_agent
                                    on inconsistency, up to MAX_REGENERATIONS times)
      -> human_review                (hard interrupt: NOTHING is finalized without
                                       a clinician decision -- see PRD "1. Complete
                                       Problem Statement", Human-in-the-Loop paradigm)
      -> finalize_report | archive_case
      -> END

Changelog vs. the previous version of this file: diagnosis_agent now
writes a single `diagnosis_findings` dict (finding_label,
anatomical_region, severity, clinical_reasoning) instead of separate
`preliminary_diagnosis` / `confidence_score` fields, and evidence_agent
now writes `retrieved_evidence` as one pre-formatted citable string
instead of a list of evidence dicts. The human_review interrupt payload
below has been updated to match -- it no longer references
`confidence_score`, which no node has written since that change and
which was silently reaching the UI as None.

Run this module directly for a smoke test:  `python -m medagent.agents.orchestrator`
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from medagent.agents.diagnosis_agent import diagnosis_agent_node
from medagent.agents.evidence_agent import evidence_agent_node
from medagent.agents.report_agent import report_agent_node
from medagent.agents.state import AgentState, PatientMetadata, new_case_state
from medagent.agents.verifier_agent import verifier_agent_node
from medagent.utils.settings import get_settings
from medagent.vision.inference import run_perception

logger = logging.getLogger("medagent.orchestrator")

MAX_REGENERATIONS = 2


# ─────────────────────────────────────────────────────────────────────
# Nodes that don't warrant their own agent module
# ─────────────────────────────────────────────────────────────────────

def perceive_image_node(state: AgentState) -> dict:
    """
    AI Perception Layer. Not an LLM agent -- a deterministic CNN/ViT
    inference call (classification + bounding boxes + Grad-CAM), wrapped
    as a graph node so its outputs flow into the same typed state as
    everything downstream. See vision/inference.py for the actual model
    calls; kept out-of-line here to keep this file orchestration-only.
    """
    try:
        result = run_perception(
            image_path=state["image_path"],
            metadata=state["patient_metadata"],
        )
        return {
            "classification": result["classification"],
            "detections": result["detections"],
            "gradcam_heatmap_path": result["gradcam_heatmap_path"],
            "heatmap_bbox_alignment_score": result["heatmap_bbox_alignment_score"],
        }
    except Exception as exc:  # noqa: BLE001 - surface into graph state, don't crash the run
        logger.exception("perceive_image_node failed for case=%s", state["case_id"])
        return {"errors": [f"perceive_image_node: {exc}"]}


def human_review_node(state: AgentState) -> Command[Literal["finalize_report", "archive_case"]]:
    """
    Hard Human-in-the-Loop checkpoint (PRD 1, PRD 2.2 "Human Oversight
    Dashboard"). Execution pauses here -- durably, via the checkpointer --
    until a clinician acts in the Streamlit dashboard and the thread is
    resumed with `graph.invoke(Command(resume={...}), config)`.

    Expected resume payload shape:
        {"action": "approve"}
        {"action": "revise", "edited_text": "<clinician-edited report>"}
        {"action": "reject", "reason": "<free text>"}
    """
    decision = interrupt(
        {
            "type": "clinical_review_required",
            "case_id": state["case_id"],
            "patient_metadata": state["patient_metadata"],
            "classification": state["classification"],
            "detections": state["detections"],
            "gradcam_heatmap_path": state["gradcam_heatmap_path"],
            "heatmap_bbox_alignment_score": state["heatmap_bbox_alignment_score"],
            "diagnosis_findings": state["diagnosis_findings"],
            "retrieved_evidence": state["retrieved_evidence"],
            "draft_report": state["draft_report"],
        }
    )

    action = decision.get("action")

    if action == "approve":
        return Command(
            update={
                "human_decision": "approved",
                "final_report": state["draft_report"],
            },
            goto="finalize_report",
        )

    if action == "revise":
        edited = decision.get("edited_text", state["draft_report"])
        return Command(
            update={
                "human_decision": "revised",
                "human_feedback": decision.get("edited_text", ""),
                "final_report": edited,
            },
            goto="finalize_report",
        )

    # action == "reject" (or anything unrecognized -> fail safe to archive, not autofinalize)
    return Command(
        update={
            "human_decision": "rejected",
            "human_feedback": decision.get("reason", ""),
        },
        goto="archive_case",
    )


def finalize_report_node(state: AgentState) -> dict:
    """Persist the clinician-approved/revised report and emit an audit trail entry."""
    from medagent.privacy.phi_redaction import write_audit_entry

    write_audit_entry(
        case_id=state["case_id"],
        event="report_finalized",
        decision=state["human_decision"],
        reviewer_feedback=state.get("human_feedback"),
    )
    logger.info("Case %s finalized with decision=%s", state["case_id"], state["human_decision"])
    return {}


def archive_case_node(state: AgentState) -> dict:
    """Clinician rejected the draft outright -- archive for manual radiologist workup."""
    from medagent.privacy.phi_redaction import write_audit_entry

    write_audit_entry(
        case_id=state["case_id"],
        event="case_archived",
        decision="rejected",
        reviewer_feedback=state.get("human_feedback"),
    )
    logger.warning("Case %s archived (clinician rejected draft)", state["case_id"])
    return {}


# ─────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────

def route_after_verification(state: AgentState) -> Literal["report_agent", "human_review"]:
    """
    Loop the draft back through report_agent when the Verifier flags an
    inconsistency between diagnosis / evidence / report text (TRD 3.1,
    Verification Layer), bounded by MAX_REGENERATIONS so a stubborn case
    always reaches a human rather than looping forever.
    """
    if (
        state.get("verification_status") == "flagged"
        and state.get("regeneration_count", 0) < MAX_REGENERATIONS
    ):
        return "report_agent"
    return "human_review"


# ─────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> CompiledStateGraph:
    settings = get_settings()
    builder = StateGraph(AgentState)

    builder.add_node("perceive_image", perceive_image_node)
    builder.add_node("diagnosis_agent", diagnosis_agent_node)
    builder.add_node("evidence_agent", evidence_agent_node)
    builder.add_node("report_agent", report_agent_node)
    builder.add_node("verifier_agent", verifier_agent_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize_report", finalize_report_node)
    builder.add_node("archive_case", archive_case_node)

    builder.add_edge(START, "perceive_image")
    builder.add_edge("perceive_image", "diagnosis_agent")
    builder.add_edge("diagnosis_agent", "evidence_agent")
    builder.add_edge("evidence_agent", "report_agent")
    builder.add_edge("report_agent", "verifier_agent")
    builder.add_conditional_edges(
        "verifier_agent",
        route_after_verification,
        {"report_agent": "report_agent", "human_review": "human_review"},
    )
    # human_review routes dynamically via Command(goto=...); no static edge needed.
    builder.add_edge("finalize_report", END)
    builder.add_edge("archive_case", END)

    if checkpointer is None:
        # A checkpointer is *mandatory* here, not optional -- interrupt()
        # cannot suspend/resume a case without persisted state. SQLite is
        # the offline-friendly default; swap for Postgres in multi-replica
        # deployments (see docs/architecture.md).
        checkpointer = SqliteSaver.from_conn_string(settings.checkpoint_db_path)

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────
# Public entry points used by the Streamlit dashboard / API layer
# ─────────────────────────────────────────────────────────────────────

def run_case(
    graph: CompiledStateGraph,
    case_id: str,
    image_path: str,
    patient_metadata: PatientMetadata,
) -> dict:
    """
    Starts a new case. Runs until the graph either completes or hits the
    human_review interrupt, whichever comes first. Returns the current
    state snapshot; check `__interrupt__` in the returned dict to know
    whether clinician input is now required.
    """
    config = {"configurable": {"thread_id": case_id}}
    initial_state = new_case_state(
        case_id=case_id,
        thread_id=case_id,
        image_path=image_path,
        patient_metadata=patient_metadata,
    )
    return graph.invoke(initial_state, config)


def resume_case(graph: CompiledStateGraph, case_id: str, clinician_decision: dict) -> dict:
    """
    Resumes a case paused at human_review. `clinician_decision` must match
    the payload shape documented on human_review_node, e.g.:
        {"action": "approve"}
    """
    config = {"configurable": {"thread_id": case_id}}
    return graph.invoke(Command(resume=clinician_decision), config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_graph = build_graph()

    demo_metadata: PatientMetadata = {
        "age": 47,
        "sex": "F",
        "view_position": "PA",
        "patient_id": "DEMO-0001",
    }

    result = run_case(
        demo_graph,
        case_id="demo-case-001",
        image_path="data/raw/sample_cxr.png",
        patient_metadata=demo_metadata,
    )
    print("Paused for clinician review:" if "__interrupt__" in result else "Completed:", result)