"""
Centralized settings for the clinical decision pipeline, loaded from
.env via pydantic-settings. Every module reads config through
get_settings() rather than os.environ directly, so there is exactly
one place that defines defaults before a case ever touches PHI.
"""
from __future__ import annotations

import functools
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: str = "ollama"
    ollama_model: str = "llama3.1:8b"     # generic fallback; per-agent fields below take precedence -- see llm/loader.py
    llm_max_new_tokens: int = 512

    # Adjusted to a plain file path to avoid sqlite3 connection string errors
    checkpoint_db_path: str = "checkpoints.db"

    # --- Per-agent LLM routing (llm/loader.py::get_llm()) ---
    # Model *names* here are Ollama tags (whatever `ollama pull <tag>` has
    # actually been run) -- keep them in sync with what's pulled locally.
    # configs/model_config.yaml's llms: section describes a richer intended
    # routing (including non-Ollama providers, e.g. verifier_llm's
    # HuggingFace bitsandbytes profile, and cloud fallbacks) that
    # llm/loader.py doesn't implement yet -- see its module docstring.
    diagnosis_llm_model: str = "qwen2.5:14b-instruct"
    diagnosis_llm_temperature: float = 0.2
    evidence_llm_model: str = "qwen2.5:7b-instruct"
    evidence_llm_temperature: float = 0.0
    report_llm_model: str = "llama3.1:8b-instruct"
    report_llm_temperature: float = 0.3
    # Deliberately NOT report_llm_model -- a verifier grading its own
    # generator's output with the same model risks self-grading bias
    # (agent_config.yaml's own guardrail for this agent).
    verifier_llm_model: str = "qwen2.5:7b-instruct"
    verifier_llm_temperature: float = 0.0

    # --- AI Perception Layer (vision/) ---
    # "auto" resolves to cuda > mps > cpu at runtime; force a specific backend
    # (e.g. for a CUDA-only deployment target) by setting DEVICE explicitly.
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    # torchxrayvision weight identifier for the DenseNet-121 backbone -- see
    # https://github.com/mlmed/torchxrayvision for the full catalogue.
    # "-all" is trained across NIH/PadChest/CheXpert/MIMIC/Kaggle-RSNA and
    # covers 18 pathologies (not just pneumonia), giving a meaningful signal
    # for state.py's "Other Abnormality" bucket as well as "Lung Opacity".
    vision_weights_id: str = "densenet121-res224-all"
    vision_input_size: int = 224
    # None = torchxrayvision's own default cache dir (~/.torchxrayvision).
    vision_cache_dir: str | None = None

    # Fine-tuned checkpoint paths -- see vision/models/*.py. Absent by
    # default (Phase 1 training hasn't produced these yet); each loader logs
    # a loud warning and degrades to pretrained/random weights when missing.
    classifier_checkpoint_path: str = "./models/checkpoints/pneumonet_cxr_v2_3.pt"
    detector_checkpoint_path: str = "./models/checkpoints/detector_frcnn_r50.pt"

    detector_score_threshold: float = 0.5
    detector_nms_iou_threshold: float = 0.4
    detector_max_detections: int = 10

    gradcam_output_dir: str = "./data/processed/gradcam"

    # --- RAG / Vector Store (FAISS) -- Phase 2 item 2 ---
    faiss_index_dir: str = "./vectorstore/faiss_index"
    faiss_index_name: str = "medagent_guidelines"
    # Shared HMAC-SHA256 secret between rag/ingest.py (signs the index
    # after building it) and rag/vectorstore.py (verifies before
    # loading) -- see security/artifact_signing.py. Blank by default so
    # a missing key fails LOUD (SecurityError) rather than silently
    # signing/verifying with an empty, guessable key.
    faiss_signing_key: str = ""

    # --- Audit trail (security/audit_logger.py) -- Phase 2 item 3 ---
    audit_log_path: str = "data/audit_log.jsonl"

@functools.lru_cache
def get_settings() -> Settings:
    return Settings()