"""
FAISS vector store initialization for the ATS/IDSA guideline RAG index.
Loads a persisted index from disk in production; if none exists yet
(fresh dev environment, CI, before make build-index has been run
against the real guideline PDF), falls back to a small in-memory index
built from sample guideline text so the pipeline still runs end-to-end
rather than crashing.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from medagent.utils.settings import get_settings

logger = logging.getLogger("medagent.rag.vectorstore")

_SAMPLE_GUIDELINE_CHUNKS = [
    "Pneumonia guidelines: Empiric antibiotic therapy for community-acquired "
    "pneumonia should be guided by severity assessment (e.g. CURB-65) and "
    "local resistance patterns. Amoxicillin is first-line for low-severity, "
    "outpatient-managed cases.",
    "Pneumonia guidelines: Patients with high-severity community-acquired "
    "pneumonia and signs of sepsis should be assessed for ICU admission and "
    "started on combination antibiotic therapy covering atypical organisms.",
    "Fracture guidelines: Suspected long-bone fractures should be immobilized "
    "and radiographically confirmed before weight-bearing is permitted. "
    "Displaced fractures typically require orthopedic referral for reduction.",
    "Fracture guidelines: Stress fractures may not be visible on initial "
    "plain-film radiographs; if clinical suspicion remains high despite a "
    "negative X-ray, MRI or repeat imaging in 1-2 weeks is recommended.",
]

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        model_name = getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings

def _build_sample_vectorstore() -> FAISS:
    """
    Fail-safe fallback: builds a small in-memory FAISS index from a
    handful of sample guideline chunks. This is NOT a substitute for the
    real ATS/IDSA guideline index -- it exists purely so the pipeline
    runs end-to-end in local dev/testing before make build-index has
    been pointed at real guideline documents.
    """
    logger.warning(
        "No persisted FAISS index found on disk -- building a small "
        "in-memory sample index for local testing. Do NOT use this in "
        "production; run make build-index against the real ATS/IDSA "
        "guideline PDF first."
    )
    return FAISS.from_texts(_SAMPLE_GUIDELINE_CHUNKS, _get_embeddings())

def get_vectorstore() -> FAISS:
    """
    Loads the persisted ATS/IDSA guideline FAISS index from disk. Falls
    back to an in-memory sample index (see _build_sample_vectorstore)
    if no index has been built yet, rather than raising and taking down
    the whole graph on a missing index file.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()
    index_dir = Path(getattr(settings, "faiss_index_dir", "./vectorstore/faiss_index"))

    if (index_dir / "index.faiss").exists():
        try:
            _vectorstore = FAISS.load_local(
                str(index_dir), _get_embeddings(), allow_dangerous_deserialization=True
            )
            return _vectorstore
        except Exception:
            logger.exception(
                "Failed to load FAISS index from %s -- falling back to sample index", index_dir
            )

    _vectorstore = _build_sample_vectorstore()
    return _vectorstore