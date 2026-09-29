"""Application settings, loaded from the environment or a local `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SearchProvider = Literal["tavily", "none"]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden with an environment variable of the same name
    (case-insensitive), or through a `.env` file next to the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    groq_api_key: str = Field(default="", description="API key for Groq.")
    model: str = Field(default="llama-3.3-70b-versatile", description="Groq model id.")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Search ------------------------------------------------------------
    tavily_api_key: str = Field(default="", description="API key for Tavily search.")
    search_provider: SearchProvider = Field(
        default="tavily",
        description="'tavily' for live web search, 'none' to disable the tool.",
    )
    search_max_results: int = Field(default=4, ge=1, le=20)

    # --- Orchestration -----------------------------------------------------
    max_questions: int = Field(
        default=5, ge=1, le=20, description="Questions the planner may generate per round."
    )
    max_review_cycles: int = Field(
        default=2, ge=0, le=10, description="How many times the critic may force a revision."
    )

    # --- Storage -----------------------------------------------------------
    runs_dir: Path = Field(default=Path("runs"), description="Where finished reports are written.")
    cache_file: Path = Field(default=Path(".cache/search.json"), description="Search cache path.")
    cache_ttl_hours: int = Field(default=24, ge=0)

    @field_validator("runs_dir", "cache_file", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    def require_llm_key(self) -> None:
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GROQ_API_KEY before running."
            )

    def require_search_key(self) -> None:
        if self.search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set but SEARCH_PROVIDER=tavily. "
                "Add a key, or set SEARCH_PROVIDER=none to run without web search."
            )
