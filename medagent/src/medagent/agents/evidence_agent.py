"""
Evidence Agent -- RAG node in the clinical decision LangGraph pipeline.

Follows the Diagnosis Agent. Takes the diagnosis label from
state["diagnosis_findings"]["finding_label"], retrieves the top 3 most
relevant ATS/IDSA guideline chunks from the local FAISS vector store,
and returns them as a single formatted string for the Report Agent to
cite from.
"""
from __future__ import annotations

import logging

from medagent.agents.state import AgentState
from medagent.rag.retriever import get_retriever

logger = logging.getLogger("medagent.agents.evidence")

TOP_K = 3


def evidence_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Evidence Agent.

    Reads `diagnosis_findings.finding_label` off the shared graph state,
    queries the FAISS-backed ATS/IDSA guideline retriever, and formats
    the top 3 chunks into a single citable string.

    Args:
        state: Current `AgentState`. Only `diagnosis_findings` is read.

    Returns:
        `{"retrieved_evidence": <formatted evidence string>}` on success.
        On any failure (missing label, retriever error, empty index),
        `retrieved_evidence` still holds a plain-text explanation rather
        than raising -- this node must never crash the graph.
    """
    case_id = state.get("case_id", "unknown")

    try:
        diagnosis_findings = state.get("diagnosis_findings") or {}
        finding_label = (diagnosis_findings.get("finding_label") or "").strip()

        if not finding_label:
            raise ValueError("diagnosis_findings.finding_label is missing or empty")

        retriever = get_retriever()
        hits = retriever.retrieve(finding_label)[:TOP_K]

        if not hits:
            logger.warning("Case %s: no guideline evidence found for %r", case_id, finding_label)
            return {
                "retrieved_evidence": (
                    f"No matching ATS/IDSA guideline evidence found for '{finding_label}'."
                )
            }

        formatted = "\n\n".join(
            f"[{i}] Source: {hit.source} (relevance {hit.score:.2f})\n{hit.text.strip()}"
            for i, hit in enumerate(hits, start=1)
        )
        return {"retrieved_evidence": formatted}

    except Exception as exc:  # noqa: BLE001 - never let retrieval failure crash the graph
        logger.exception("evidence_agent_node failed for case=%s", case_id)
        return {"retrieved_evidence": f"Evidence retrieval failed: {type(exc).__name__}: {exc}"}