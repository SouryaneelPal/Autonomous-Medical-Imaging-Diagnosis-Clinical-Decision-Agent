"""
Diagnosis Agent -- Agentic Intelligence Layer (TRD 3.1).

LangGraph node that synthesizes the AI Perception Layer's outputs
(classification, bounding-box detections) and patient demographics into
a single structured clinical finding, using a local, offline LLM
constrained to a fixed JSON schema via LangChain's
`.with_structured_output()`.

This agent deliberately does NOT cite guidelines or draft report prose
-- that's evidence_agent.py and report_agent.py downstream. Keeping the
concerns separate lets verifier_agent.py check that the finding, the
evidence, and the report text all agree with each other.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from medagent.agents.state import AgentState
from medagent.utils.settings import get_settings

logger = logging.getLogger("medagent.agents.diagnosis")


# ─────────────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────────────

class DiagnosisOutput(BaseModel):
    """A single structured clinical finding produced by the Diagnosis Agent.

    This is the exact JSON shape the local LLM is constrained to return
    via `.with_structured_output()` -- it is also what downstream agents
    (evidence_agent, report_agent, verifier_agent) key off of, so
    changing field names here is a breaking change across the graph.
    """

    finding_label: str = Field(
        ...,
        description=(
            "Short clinical label for the primary finding, e.g. "
            "'Right lower lobe consolidation' or 'No acute abnormality'."
        ),
    )
    anatomical_region: str = Field(
        ...,
        description=(
            "Anatomical location the finding pertains to, e.g. "
            "'right lower lobe', 'bilateral lower zones', 'not applicable'."
        ),
    )
    severity: Literal["high", "moderate", "low"] = Field(
        ...,
        description=(
            "Clinical severity/urgency of the finding. This system "
            "prioritizes recall over precision (PRD 2.3) -- when the "
            "underlying evidence is ambiguous, do not default to 'low' "
            "defensively; reserve 'low' for genuinely unremarkable cases."
        ),
    )
    clinical_reasoning: str = Field(
        ...,
        description=(
            "One to three sentences explaining how the classification, "
            "detections, and patient demographics support this finding. "
            "Must not introduce facts not present in the supplied data."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Clinical Diagnostician AI supporting a radiologist's review \
of a frontal chest X-ray. You do not have access to the raw image -- \
only the structured outputs of an upstream computer-vision pipeline \
and the patient's demographic metadata, both provided below.

Rules:
- Base your finding strictly on the supplied classification, \
detections, and metadata. Do not infer findings the data does not \
support, and do not speculate about history, labs, or symptoms you \
were not given.
- If the classification is "Normal" and there are no detections, \
report a low-severity, no-acute-finding result rather than inventing \
an abnormality.
- If the data is ambiguous or the detector's confidence is low, say so \
explicitly in `clinical_reasoning` rather than resolving the ambiguity \
yourself -- a human clinician makes the final call, not you.
- Respond only in the requested structured schema."""

_HUMAN_TEMPLATE = """\
CLASSIFICATION (from vision model):
{classification}

DETECTIONS (bounding boxes from vision model, normalized coordinates):
{detections}

PATIENT METADATA:
{patient_metadata}

Produce the single most clinically relevant finding for this case."""

_DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_TEMPLATE),
    ]
)


# ─────────────────────────────────────────────────────────────────────
# LLM initialization
# ─────────────────────────────────────────────────────────────────────

def _build_chat_model() -> BaseChatModel:
    """
    Local, offline chat-model factory (TRD 3.2: Llama 3.1 8B or Qwen2.5
    7B/14B, 4-bit quantized). Backend is selected via `LLM_BACKEND` in
    `.env` so this is a config change, not a code change:

      - "ollama" (default): a local Ollama server serving an
        already-quantized GGUF build, e.g. `ollama pull llama3.1:8b` or
        `ollama pull qwen2.5:14b`. Ollama's native tool-calling support
        is what makes `.with_structured_output()` reliable without a
        hosted API.
      - "hf_endpoint": a self-hosted HuggingFace Text Generation
        Inference (TGI) endpoint serving a bitsandbytes 4-bit model,
        wrapped as a LangChain chat model. Prefer this if the LLM needs
        to share a process/GPU allocation with the vision models rather
        than run as a separate Ollama daemon.
    """
    settings = get_settings()
    backend = getattr(settings, "llm_backend", "ollama")

    if backend == "hf_endpoint":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            endpoint_url=getattr(settings, "llm_endpoint_url", "http://localhost:8080"),
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=0.1,  # low temperature: clinical determinism over creativity
        )
        return ChatHuggingFace(llm=endpoint)

    # Default: local Ollama server. Fully offline once the model is pulled
    # (PRD 2.3 Data Privacy -- no PHI-adjacent data leaves the process).
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=getattr(settings, "ollama_model", "llama3.1:8b"),
        temperature=0.1,
        num_predict=settings.llm_max_new_tokens,
    )


@lru_cache(maxsize=1)
def _get_diagnosis_chain() -> Runnable:
    """
    Builds and caches the prompt | structured-LLM chain once per process.
    `method="json_schema"` constrains decoding to the schema directly
    (more reliable locally than function-calling for smaller quantized
    models); swap to `method="function_calling"` if your served model
    doesn't yet support the json_schema response format.
    """
    llm = _build_chat_model()
    structured_llm = llm.with_structured_output(DiagnosisOutput, method="json_schema")
    return _DIAGNOSIS_PROMPT | structured_llm


# ─────────────────────────────────────────────────────────────────────
# Graph node
# ─────────────────────────────────────────────────────────────────────

def diagnosis_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Diagnosis Agent.

    Reads `classification`, `detections`, and `patient_metadata` off the
    shared graph state, invokes the local LLM constrained to
    `DiagnosisOutput`, and returns a partial state update.

    Args:
        state: Current `AgentState`. Only the three fields above are read.

    Returns:
        On success: `{"diagnosis_findings": <DiagnosisOutput.model_dump()>}`.
        On failure: a fail-safe `DiagnosisOutput` with `severity="high"`
        (an undetermined case must never be silently deprioritized, per
        this project's recall-over-precision priority) plus an
        `errors` entry describing what went wrong.
    """
    case_id = state.get("case_id", "unknown")

    try:
        chain = _get_diagnosis_chain()
        result: DiagnosisOutput = chain.invoke(
            {
                "classification": str(state.get("classification")),
                "detections": str(state.get("detections")),
                "patient_metadata": str(state.get("patient_metadata")),
            }
        )

        logger.info(
            "Case %s: diagnosis_agent finding=%r region=%r severity=%s",
            case_id,
            result.finding_label,
            result.anatomical_region,
            result.severity,
        )
        return {"diagnosis_findings": result.model_dump()}

    except Exception as exc:  # noqa: BLE001 - never let a bad LLM call crash the graph
        logger.exception("diagnosis_agent_node failed for case=%s", case_id)

        fallback = DiagnosisOutput(
            finding_label="UNDETERMINED",
            anatomical_region="unknown",
            severity="high",
            clinical_reasoning=(
                f"Automated diagnosis generation failed ({type(exc).__name__}: {exc}). "
                "This case requires direct clinician review of the source image and "
                "raw vision-model output before any finding is reported."
            ),
        )
        return {
            "diagnosis_findings": fallback.model_dump(),
            "errors": [f"diagnosis_agent_node: {exc}"],
        }