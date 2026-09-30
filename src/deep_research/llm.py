"""LLM construction and the structured-output helper.

The course notebooks talked to OpenAI (through a DataLab proxy). This project
switches to Groq, which is fast enough to fan out a research run without the
run dragging on for minutes. Swapping providers is a one-line change: return any
`llama_index.core.llms.LLM`.

The model default is a moving target. Groq retired `llama-3.3-70b-versatile`
mid-project, which produced a bare 404 on every call and a report that could
never be written. `check_model_available` exists so that failure mode arrives as
one clear line naming the models your key can actually reach.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import httpx
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from pydantic import BaseModel, ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

# Chosen against the live API, not from memory: it is the only candidate tested
# that both parses native structured output *and* will call the tool when Groq
# demands it. See README "Choosing a model".
DEFAULT_MODEL = "qwen/qwen3.8-27b"

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

T = TypeVar("T", bound=BaseModel)


class ModelUnavailableError(RuntimeError):
    """The configured model is not reachable with this API key."""


def build_llm(settings: Settings) -> LLM:
    """Create the chat LLM used by every agent in the pipeline."""
    settings.require_llm_key()

    from llama_index.llms.groq import Groq

    return Groq(
        model=settings.model or DEFAULT_MODEL,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
    )


async def available_models(api_key: str) -> list[str]:
    """Model ids this key can reach, sorted. Empty if the listing is refused."""
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(GROQ_MODELS_URL, headers=headers)
    if response.status_code != 200:
        return []
    return sorted(entry["id"] for entry in response.json().get("data", []))


async def check_model_available(settings: Settings) -> str:
    """Fail early and usefully if `settings.model` has been retired.

    A 404 part-way through a run wastes the whole run and reads like a quota
    problem. Returns the model name when it is fine, and raises
    `ModelUnavailableError` listing the alternatives when it is not. A listing
    failure is not fatal, so an unreachable catalogue does not block a valid run.
    """
    model = settings.model or DEFAULT_MODEL
    try:
        models = await available_models(settings.groq_api_key)
    except Exception as exc:  # noqa: BLE001 - a preflight must never block a good run
        logger.warning("could not list Groq models (%s); skipping preflight", exc)
        return model

    if not models or model in models:
        return model

    raise ModelUnavailableError(
        f"MODEL={model} is not available to your Groq key. "
        f"Available: {', '.join(models)}. Set MODEL in .env to one of those."
    )


async def ask_structured(llm: LLM, output_cls: type[T], prompt: str) -> T | None:
    """Ask the model for an instance of `output_cls`, tolerating a bad reply.

    Tries native structured output first. If the provider cannot do it, or the
    model returns something that will not validate, falls back to a plain
    completion parsed leniently. Returns `None` only if both routes fail, which
    the caller treats as "degrade gracefully", never as an exception.

    The prompt is wrapped in a `RichPromptTemplate` on purpose.
    `astructured_predict` declares `prompt: PromptTemplate`, and passing a bare
    `str` fails validation before the request is ever sent. That failure is
    silent from the caller's side: it lands in the except branch below, and the
    lenient salvage then has to carry every plan and verdict on its own.
    """
    template = RichPromptTemplate(template_str=prompt)
    try:
        return await llm.astructured_predict(output_cls, template)
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
    commentary after it - all things models do when asked for JSON.
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
