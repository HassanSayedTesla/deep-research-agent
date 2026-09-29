"""LLM construction and the structured-output helper.

The course notebooks talked to OpenAI (through a DataLab proxy). This project
switches to Groq, which serves Llama 3.3 70B with native tool calling and is
fast enough to fan out a research run without the run dragging on for minutes.
Swapping providers is a one-line change: return any `llama_index.core.llms.LLM`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from llama_index.core.llms import LLM
from pydantic import BaseModel, ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"

T = TypeVar("T", bound=BaseModel)


def build_llm(settings: Settings) -> LLM:
    """Create the chat LLM used by every agent in the pipeline."""
    settings.require_llm_key()

    from llama_index.llms.groq import Groq

    return Groq(
        model=settings.model or DEFAULT_MODEL,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
    )


async def ask_structured(llm: LLM, output_cls: type[T], prompt: str) -> T | None:
    """Ask the model for an instance of `output_cls`, tolerating a bad reply.

    Tries native structured output first. If the provider cannot do it, or the
    model returns something that will not validate, falls back to a plain
    completion parsed leniently. Returns `None` only if both routes fail, which
    the caller treats as "degrade gracefully", never as an exception.
    """
    try:
        return await llm.astructured_predict(output_cls, prompt)
    except Exception as exc:
        logger.warning("structured predict failed (%s); falling back to plain completion", exc)

    try:
        response = await llm.acomplete(prompt)
    except Exception as exc:
        logger.error("plain completion failed too: %s", exc)
        return None

    return _salvage(response.text, output_cls)


def _salvage(text: str, output_cls: type[T]) -> T | None:
    """Best-effort parse of `text` into `output_cls`.

    Handles a fenced code block, a bare object, and an object with trailing
    commentary after it — all things models do when asked for JSON.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        _, _, candidate = candidate.partition("\n")
        candidate = candidate.rsplit("```", 1)[0]

    try:
        return output_cls.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return output_cls.model_validate(json.loads(candidate[start : end + 1]))
            except (json.JSONDecodeError, ValidationError):
                continue

    logger.warning("could not parse a %s out of the model reply", output_cls.__name__)
    return None


async def complete_text(llm: LLM, prompt: str) -> str:
    """Plain, non-streamed completion. Used for short, cheap calls."""
    response: Any = await llm.acomplete(prompt)
    return str(response.text)
