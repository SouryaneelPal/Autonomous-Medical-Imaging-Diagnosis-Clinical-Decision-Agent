"""
Centralized settings for the clinical decision pipeline, loaded from
.env via pydantic-settings. Every module reads config through
get_settings() rather than os.environ directly, so there is exactly
one place that defines defaults before a case ever touches PHI.
"""
from __future__ import annotations

import functools
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    llm_max_new_tokens: int = 512
    
    # Adjusted to a plain file path to avoid sqlite3 connection string errors
    checkpoint_db_path: str = "checkpoints.db"

@functools.lru_cache
def get_settings() -> Settings:
    return Settings()