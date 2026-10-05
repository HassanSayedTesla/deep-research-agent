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

import asyncio
import json
import logging
import os
import re
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
        # A hard output cap, and the single most important number in this
        # module. Groq's free tier allows 1000 output tokens/minute. Left to
        # itself the adapter requests far more than that per call, so every
        # request is rejected with
        #
        #   429 Request too large ... OTPM: Limit 1000, Requested 1108
        #
        # which no amount of waiting or throttling can fix, because the *next*
        # request will ask for the same amount. 512 leaves headroom for the
        # structured-output calls, which also pay for schema tokens.
        max_tokens=settings.max_output_tokens,
        # Turn OFF the SDK's own retry loop. LlamaIndex wraps every chat call in
        # `llm_retry_decorator`, which retries *any* 429 up to `max_retries`
        # times with up to 20s of jittered backoff. It cannot tell a transient
        # per-minute limit from a dead daily budget, so on a day the quota is
        # gone it sleeps through the entire retry budget and only then hands the
        # error up. Observed in a live run:
        #
        #   Retrying llama_index.llms.openai.base.OpenAI._achat in 120 seconds ...
        #     on tokens per day (TPD): Limit 200000, Used 199601
        #
        # That defeats `with_rate_limit_retry` entirely, which exists precisely
        # to fail fast on a long window and let the caller degrade to something
        # useful. The transport-level `max_retries` is set to match, because the
        # openai client retries 429s on its own too. With both disabled,
        # `with_rate_limit_retry` is the single place that decides whether to
        # wait, and it can act on the distinction the SDK throws away.
        max_retries=0,
    )


class LLMGate:
    """An async context manager that paces concurrent calls to one LLM.

    Fan-out is bounded by `num_workers`, which is a statement about the
    workflow, not about the provider. Every researcher re-sends the system
    prompt, the tool schema and its own memory, so four at once can exceed a
    per-minute *input token* limit and earn a 429 - even with request headroom
    to spare. Groq's free tier allows 7000 input tokens/minute, and a research
    turn costs roughly 2.8k, so only two fit.

    The gate keeps a little parallelism (searching is not free) while staying
    under the budget. It is a soft control: a caller can still blow through it,
    which is why `retry_on_rate_limit` exists as the backstop.
    """

    def __init__(self, max_concurrency: int = 2) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self._semaphore: asyncio.Semaphore | None = None
        self.peak_concurrency = 0
        self._active = 0

    def _get(self) -> asyncio.Semaphore:
        # Built lazily so the gate is safe to construct outside a loop, and so
        # each test or run gets a semaphore bound to its own loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def __aenter__(self) -> LLMGate:
        await self._get().acquire()
        self._active += 1
        self.peak_concurrency = max(self.peak_concurrency, self._active)
        return self

    async def __aexit__(self, *_: object) -> None:
        self._active -= 1
        self._get().release()


def llm_concurrency_limit() -> int:
    """How many LLM turns may be in flight at once, from the environment."""
    try:
        return max(1, min(8, int(os.environ.get("LLM_CONCURRENCY", "2"))))
    except ValueError:
        return 2


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
    except Exception as exc:
        logger.warning("could not list Groq models (%s); skipping preflight", exc)
        return model

    if not models or model in models:
        return model

    raise ModelUnavailableError(
        f"MODEL={model} is not available to your Groq key. "
        f"Available: {', '.join(models)}. Set MODEL in .env to one of those."
    )


# Groq says how long to wait, in seconds, inside a 429. Honour it when present
# rather than guessing, so a retry lands just after the window reopens.
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)

# The budget windows long enough that retrying cannot pay off: the day's, the
# week's or the month's allowance is gone, and it does not come back inside a
# research run. Groq still offers a "try again in 6m49s" in these messages,
# which is misleading enough to be worth matching on explicitly.
_LONG_WINDOW = re.compile(
    r"tokens per (day|week|month)|\b(TPD|RPD|MPD)\b|requests per (day|week|month)",
    re.IGNORECASE,
)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying `exc`."""
    match = _RETRY_AFTER.search(str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return min(2.0**attempt, 30.0)


def is_quota_error(exc: Exception) -> bool:
    """Did the provider refuse this call because of a quota?

    True for every rate-limit refusal, transient or not. Callers use this to
    decide whether the run can still produce something useful by other means -
    a dead writer can fall back to the research notes, for instance.
    """
    status = getattr(exc, "status_code", None)
    if status == 413 or "too large" in str(exc).lower():
        return True
    return status == 429 or "RateLimit" in type(exc).__name__ or "rate_limit" in str(exc).lower()


def is_auth_error(exc: Exception) -> bool:
    """Did the provider reject the call because of credentials or access?

    A key can be present but still invalid, revoked, scoped to another
    organization, or blocked by network policy. Groq reports this as 401/403,
    while the SDK may surface it as `PermissionDeniedError` or
    `AuthenticationError`.
    """
    status = getattr(exc, "status_code", None)
    if status == 401 or status == 403:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "permissiondenied",
        "authentication",
        "unauthorized",
        "forbidden",
        "access denied",
        "invalid api key",
        "incorrect api key",
    )
    return any(marker in text for marker in markers)


def provider_failure_note(exc: Exception) -> str:
    """Reader-safe category for a provider failure, without provider details."""
    status = getattr(exc, "status_code", None)
    suffix = f" (HTTP {status})" if isinstance(status, int) else ""
    if is_auth_error(exc):
        return f"the provider rejected the configured API key{suffix}"
    if is_quota_error(exc):
        return "the provider's quota was exhausted"
    return f"the provider returned {type(exc).__name__}"


def is_retryable(exc: Exception) -> bool:
    """Would waiting plausibly let this call succeed?

    Groq overloads one 429 code for three quite different situations, and only
    one of them is worth retrying:

      429 "... on tokens per minute (TPM): Limit 7000, Requested 8308.
           Please try again in 6.95s"
          Transient. The window is seconds away, so waiting works.

      429 "... on tokens per day (TPD): Limit 200000, Used 199383"
          The *day's* budget is gone. The seconds-scale wait in the message is a
          courtesy, not a real hint: the next request is just as large. Retrying
          three times in four seconds cannot succeed, it only spends the wall
          clock and then fails anyway.

      413 "Request too large ... on tokens per request (TPR): Limit 8192"
          One request exceeds the ceiling. No wait helps, ever, because the next
          request will be just as large.

    So a quota error is only retried when the window is short enough that the
    pipeline will still be running when it reopens.
    """
    if not is_quota_error(exc):
        return False
    if _LONG_WINDOW.search(str(exc)):
        return False
    status = getattr(exc, "status_code", None)
    if status == 413 or "too large" in str(exc).lower():
        return False
    return True


async def with_rate_limit_retry(
    operation: Any,
    attempts: int = 3,
    base_delay: float = 2.0,
    should_retry: Any | None = None,
) -> Any:
    """Await `operation()`, retrying a 429 with backoff.

    A short-window rate limit is a transient condition, not a reason to lose a
    run that has already paid for its planner call and its research fan-out.
    Without this, one unlucky minute kills the whole pipeline and the report is
    never written. A long-window limit is not retried at all - see
    `is_retryable` - and fails fast so the caller can degrade instead.

    `should_retry(exc)` can veto a retry that would otherwise be attempted. It
    exists for the writer, which streams tokens straight to the browser: once
    some have been sent, a retry appends a second draft to the first, so a
    later refusal has to be handed back as a partial draft rather than retried.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as exc:
            if should_retry is not None and not should_retry(exc):
                raise
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            last = exc
            delay = _retry_delay(exc, attempt)
            logger.warning(
                "rate limited (attempt %d/%d, waiting %.1fs): %s",
                attempt + 1,
                attempts,
                delay,
                str(exc)[:120],
            )
            await asyncio.sleep(delay if delay > base_delay else base_delay * (attempt + 1))
    if last is not None:  # pragma: no cover - the loop always returns or raises
        raise last
    return None


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
        return await with_rate_limit_retry(lambda: llm.astructured_predict(output_cls, template))
    except Exception as exc:
        logger.warning("structured predict failed (%s); falling back to plain completion", exc)

    try:
        response = await with_rate_limit_retry(lambda: llm.acomplete(prompt))
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


def settings_override(settings: Settings, **changes: Any) -> Settings:
    """A copy of `settings` with `changes` applied.

    Used where one call needs a different budget from the rest of the run, e.g.
    the writer retrying at a larger `max_tokens`. Copying keeps the change local:
    a shared object would silently raise the cap for every later call too.
    """
    return settings.model_copy(update=changes)


async def complete_text(llm: LLM, prompt: str) -> str:
    """Plain, non-streamed completion. Used for short, cheap calls."""
    response: Any = await llm.acomplete(prompt)
    return str(response.text)
