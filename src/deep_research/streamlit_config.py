"""Settings plumbing for the Streamlit deployment.

Streamlit Community Cloud does not read this repository's local `.env` file.
Operators enter secrets in the Streamlit dashboard instead, where they appear to
the app as `st.secrets`. This module translates that secrets mapping into the
same `Settings` object used by the CLI and FastAPI server, without ever logging
or displaying secret values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import SEARCH_KEY_ENV, Settings


def _flattened_secrets(secrets: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one level of TOML nesting, preserving explicit values.

    Both of these spellings work:

        GROQ_API_KEY = "..."
        TAVILY_API_KEY = "..."

        [deep_research]
        groq_api_key = "..."
        tavily_api_key = "..."
    """
    flattened: dict[str, Any] = {}
    for key, value in secrets.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                flattened[f"{key}.{nested_key}"] = nested_value
        else:
            flattened[key] = value
    return flattened


def _candidate_names(key: object) -> list[str]:
    """Normalise a secrets key the way environment-variable lookup does."""
    name = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    if "." in name:
        return [name, name.rsplit(".", 1)[-1]]
    return [name]


def settings_from_secrets(secrets: Mapping[str, Any], base: Settings | None = None) -> Settings:
    """Build application settings from a Streamlit secrets mapping.

    Unknown keys are ignored so dashboard-only entries do not break validation.
    Values already present in the process environment or local `.env` remain in
    force unless the same setting is explicitly present in `secrets`.
    """
    base_settings = base or Settings()
    known = {name.lower(): name for name in Settings.model_fields}
    overrides: dict[str, Any] = {}
    for key, value in _flattened_secrets(secrets).items():
        # Dashboard pastes often carry a trailing newline or space. A key with
        # invisible whitespace is present but invalid, and Groq answers 403.
        if isinstance(value, str):
            value = value.strip()
        for candidate in _candidate_names(key):
            if candidate in known:
                overrides[known[candidate]] = value
                break
    if not overrides:
        return base_settings
    return Settings(**{**base_settings.model_dump(), **overrides})


def settings_with_overrides(base: Settings, **overrides: Any) -> Settings:
    """Apply validated Streamlit widget values to existing settings."""
    return Settings(**{**base.model_dump(), **overrides})


def missing_configuration(settings: Settings) -> list[str]:
    """Environment-variable names the operator still needs to provide."""
    missing: list[str] = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if settings.search_provider != "none" and not settings.api_key_for(settings.search_provider):
        missing.append(SEARCH_KEY_ENV[settings.search_provider])
    return missing


def public_configuration(settings: Settings) -> dict[str, Any]:
    """Settings values that are safe to show in the Streamlit sidebar."""
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "search_provider": settings.search_provider,
        "search_max_results": settings.search_max_results,
        "max_questions": settings.max_questions,
        "max_review_cycles": settings.max_review_cycles,
        "max_output_tokens": settings.max_output_tokens,
        "max_output_ceiling": settings.max_output_ceiling,
        "runs_dir": str(settings.runs_dir),
        "cache_file": str(settings.cache_file),
    }
