from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Without a key the app still works: BM25 retrieval + extractive answers.
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.8-flash"
    # Set to an empty string to disable dense retrieval and use BM25 only.
    embedding_model: str = "gemini-embedding-001"

    storage_dir: str = "storage"
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=800, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    min_similarity: float = 0.3
    # Truncate words to N characters for BM25 (simple stemming for Uzbek/Turkic suffixes). 0 = off.
    bm25_prefix_len: int = Field(default=5, ge=0)
    max_upload_mb: int = 20
    seed_samples: bool = True  # index the PDFs in samples_dir on first start so the demo is not empty
    samples_dir: str = "samples"  # relative to the working directory (/app in Docker)
