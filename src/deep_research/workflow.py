"""The pipeline, as a LlamaIndex `Workflow`.

Shape of the graph, and what each edge costs in model calls:

    planner ──questions──> research#N ──findings──> writer ──draft──> critic
       ^                                                       |
       └────────────── feedback (loop back) ───────────────────┤
                                                                       |
                            writer ◄───── accepted (terminal) ─────────┘

Every arrow is a `send_event`, every stage is a `@step`, and the loop is the
reflection cycle from exercise 11. Three things differ deliberately from the
notebook version:

1. Dependencies (LLM, agent, cache, settings) live in `ctx.store`, not on
   `self`. The course assigned them to instance attributes in the first step,
   which makes the workflow object single-use; keeping them in the run's context
   lets one workflow instance serve many runs.
2. Progress is published on an `EventBridge` the caller supplies, so the
   workflow does not care whether it is watched by a browser, a terminal, or a
   test.
3. A step declares the event type it *produces* even when it only calls
   `ctx.send_event` and returns nothing. The 2.x runtime validates the graph
   statically from annotations, and will not start a workflow whose events it
   cannot account for.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from llama_index.core.llms import LLM
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from .agents import build_research_agent
from .config import Settings
from .events import (
    EDGE_FLOW,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    EventBridge,
)
from .graph import NODE_BY_ID, parse_node_id
from .llm import (
    ask_structured,
    build_llm,
    is_quota_error,
    settings_override,
    with_rate_limit_retry,
)
from .prompts import (
    CRITIC_PROMPT,
    PLANNER_PROMPT,
    PLANNER_REVISION_PROMPT,
    WRITER_PROMPT,
)
from .schemas import ResearchPlan, ReviewVerdict
from .tools.web_search import WebSearcher

logger = logging.getLogger(__name__)

DEPS_KEY = "deep_research.deps"


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, default))))
    except ValueError:
        return default


# How many researchers run at the same time. `num_workers` is frozen into the
# step at class-creation time, so this is read from the environment at import
# rather than from Settings. Raising it costs API calls, not money to read.
RESEARCH_WORKERS = _env_int("CONCURRENCY", 4, 1, 32)

# A 20-agent fan-out would otherwise build a prompt the model truncates or
# refuses. The tail of a long note is detail, not signal.
MAX_NOTE_CHARS = 2500


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
class PlanRequest(Event):
    """Ready to turn the topic into questions."""


class RevisionRequest(Event):
    """The critic rejected the draft; re-plan with its feedback."""

    topic: str
    feedback: str


class QuestionEvent(Event):
    """One question, handed to one researcher."""

    index: int
    question: str


class FindingEvent(Event):
    """A researcher's answer, waiting to be collected."""

    index: int
    question: str
    answer: str


class DraftReady(Event):
    """A finished draft, on its way to the critic."""

    draft: str


class ProgressEvent(Event):
    """Human-readable progress, for the CLI."""

    msg: str


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------
@dataclass(slots=True)
class ResearchDeps:
    """Everything a run needs, passed in rather than built inside the workflow."""

    llm: LLM
    searcher: WebSearcher
    settings: Settings
    bridge: EventBridge
    # A prebuilt agent, shared across questions. Only tests inject one; the
    # factory below is how each researcher gets its own search allowance.
    research_agent: Any = None
    llm_gate: Any = None
    # Builds the agent for question `index`. When set, it wins over
    # `research_agent`, which is what lets `runner` give each researcher its own
    # search allowance.
    research_agent_factory: Callable[[int], Any] | None = None

    def llm_slot(self) -> Any:
        """An async context manager that paces calls to the LLM.

        `num_workers` bounds how many researchers exist, not how many requests
        the provider will accept at once. A researcher's turn re-sends the
        system prompt, the tool schema and any memory, so a handful in flight
        can exceed a per-minute input-token limit and earn a 429. The gate
        keeps some parallelism while staying under the budget, and `nullcontext`
        when unset keeps the workflow usable with an injected fake.
        """
        return self.llm_gate if self.llm_gate is not None else nullcontext()

    def research_agent_for(self, index: int) -> Any:
        """The agent that answers question `index`, with its own search allowance.

        Preference order:

        1. an injected factory, which is how `runner` gets one agent per question;
        2. a prebuilt `research_agent`, which is how tests inject a fake - it is
           shared, so it also shares one search allowance;
        3. otherwise a real agent built per question over the shared LLM and a
           per-researcher view of the searcher.
        """
        if self.research_agent_factory is not None:
            return self.research_agent_factory(index)
        if self.research_agent is not None:
            return self.research_agent
        return build_research_agent(self.llm, self.searcher.for_agent())

    async def node_started(self, node: str, detail: str = "") -> None:
        base, index = parse_node_id(node)
        spec = NODE_BY_ID.get(base)
        await self.bridge.emit(
            NODE_ACTIVATED,
            node=node,
            base=base,
            index=index,
            label=spec.label if spec else base,
            agent=spec.agent if spec else base,
            detail=detail,
        )

    async def node_finished(self, node: str, summary: str = "") -> None:
        await self.bridge.emit(NODE_FINISHED, node=node, summary=summary)

    async def flow(self, source: str, target: str, label: str, preview: str = "") -> None:
        await self.bridge.emit(
            EDGE_FLOW, source=source, target=target, label=label, preview=preview
        )

    async def log(self, message: str) -> None:
        await self.bridge.emit("log", message=message)


def _assemble_report(topic: str, findings: list[FindingEvent]) -> str:
    """A report built from the findings themselves, with no model call.

    Used when the writer cannot run at all, so the research that already
    happened is still readable. It is deliberately plainer than the written
    briefing, and it says so at the top rather than passing itself off as a
    finished piece. Links are carried through untouched.
    """
    lines = [
        f"# {topic}",
        "",
        "> **Unprocessed research notes.** The writing stage could not run, so this "
        "is the research output assembled directly: accurate and cited, but not "
        "edited into a briefing.",
        "",
    ]
    for finding in findings:
        lines.append(f"## {finding.question}")
        lines.append("")
        lines.append(finding.answer.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _short(exc: BaseException, limit: int = 160) -> str:
    """A one-line, credential-free description of an exception.

    Provider errors carry request ids and org ids, which are not secrets but are
    not useful to a reader either, so keep the summary tight.
    """
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return _truncate(text, limit)


def _is_shutdown(exc: BaseException) -> bool:
    """Is this a cancellation rather than a genuine research failure?

    Importing `WorkflowCancelledByUser` is not an option: `workflows` is an
    optional dependency of this package, and this module must import without it.
    So the check is by class name. It has to name the *real* class - the
    original list contained a `WorkflowCancelledError` that does not exist, so
    a user pressing "stop" was treated as a failed researcher: the run carried
    on, logged "continuing without it", and stood a fabricated "could not be
    researched" finding in place of the cancelled question.
    """
    if isinstance(exc, asyncio.CancelledError):
        return True
    return type(exc).__name__ in {
        "WorkflowCancelledByUser",
        "WorkflowCancelledError",
        "CancelledError",
        "GeneratorExit",
    }


def _larger_draft_budget(settings: Settings) -> int:
    """A bigger output allowance to retry a truncated draft with.

    Tripling looked generous and was in fact impossible on a free tier: a live run
    set `MAX_OUTPUT_TOKENS=512`, the retry asked for 1536, and Groq refused with
    `OTPM: Limit 1000, Requested 130` - the request itself is over the ceiling, so
    waiting can never help. Asking for more than the provider will ever grant
    turns a recoverable truncation into an unconditional fallback to raw notes.

    So the retry is clamped to what the provider will actually grant, so that a
    recoverable truncation does not turn into an unconditional fallback.
    """
    ceiling = settings.max_output_ceiling
    wanted = settings.max_output_tokens * 3
    return min(wanted, 2048, ceiling)


def _hit_length_cap(chunk: Any) -> bool:
    """Did the provider stop generating because it ran out of output tokens?

    This is the authoritative truncation signal. The provider sets
    `finish_reason="length"` when it stops at `max_tokens`, so the caller never
    has to infer it from the shape of the text. Inferring it is what made a
    report ending in a bullet or a closing `**` look cut off, and a wrong
    guess costs a second provider call on a quota that may already be spent.
    """
    raw = getattr(chunk, "raw", None)
    choices = getattr(raw, "choices", None) or ()
    return any(getattr(choice, "finish_reason", None) == "length" for choice in choices)


def _looks_truncated(text: str) -> bool:
    """Does this draft stop because it ran out of tokens rather than finished?

    A model cut off by a `max_tokens` cap leaves a hard signal: the text ends
    without closing the markdown construct it was in. A finished report ends
    with punctuation. Guessing from length alone would misfire, so this only
    reports a cut when the evidence is there, and the caller retries once.

    The default is "finished". Ending on a bare word, a bullet, a closing
    `**`, a URL or a full-width character is all normal for a report, and
    treating any of those as a cut costs a wasted retry on essentially every
    real document - and, because the retry is another provider call, it costs a
    wasted call on a quota that may already be gone.
    """
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped[-1] in ".!?:;`)]}\"'":
        return False
    # Unclosed bold, a dangling code fence, or a cut mid-heading.
    if stripped.count("**") % 2 == 1:
        return True
    if stripped.count("```") % 2 == 1:
        return True
    if stripped.rsplit("\n", 1)[-1].startswith("#"):
        return True
    return False


def _truncate(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _truncate_notes(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    """Truncate a research note without dropping the citations.

    A hard cut at `MAX_NOTE_CHARS` was landing in the middle of a source list, so
    the writer received notes with no links in them and produced a report with
    zero citations - a research report nobody can check. When a note is cut, the
    links it contained are appended so the attribution survives the trim.
    """
    body = " ".join(str(text).split())
    if len(body) <= limit:
        return body

    links = list(dict.fromkeys(re.findall(r"https?://[^\s\)\]]+", body)))
    trimmed = body[:limit].rstrip()
    if not links:
        return trimmed + " [...]"

    kept = [link for link in links if link in trimmed]
    dropped = [link for link in links if link not in trimmed]
    tail = ", ".join(dropped) if dropped else ""
    suffix = f"\n\n[Sources for this note: {', '.join(kept + ([tail] if tail else []))}]"
    return f"{trimmed} [...]{suffix}"


def _format_notes(findings: list[FindingEvent]) -> str:
    blocks = [
        f"## Question {finding.index + 1}: {finding.question}\n{_truncate_notes(finding.answer)}"
        for finding in findings
    ]
    return "\n\n".join(blocks)


def _clean_questions(raw: list[str], topic: str, limit: int) -> list[str]:
    """Strip numbering and preamble, dedupe, cap the count, never return empty."""
    seen: set[str] = set()
    questions: list[str] = []
    for item in raw:
        text = " ".join(str(item).split()).lstrip("-*0123456789. ").strip()
        if len(text) < 8:
            continue
        key = text.lower().rstrip("?")
        if key in seen:
            continue
        seen.add(key)
        questions.append(text)
        if len(questions) == limit:
            break
    # A planner that returns nothing must not deadlock the collector below.
    return questions or [f"What are the most important things to know about {topic}?"]


# --------------------------------------------------------------------------
# workflow
# --------------------------------------------------------------------------
class DeepResearchWorkflow(Workflow):
    """Plan, research in parallel, write, then review and revise until settled."""

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> PlanRequest:
        await ctx.store.set(DEPS_KEY, ev.deps)
        await ctx.store.set("topic", ev.topic)
        await ctx.store.set("review_cycles", 0)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Researching: {ev.topic}"))
        return PlanRequest()

    @step
    async def plan(self, ctx: Context, ev: PlanRequest | RevisionRequest) -> QuestionEvent:
        """Emit one QuestionEvent per question. Returns nothing itself."""
        deps: ResearchDeps = await ctx.store.get(DEPS_KEY)
        settings = deps.settings
        topic: str = await ctx.store.get("topic", "")

        is_revision = isinstance(ev, RevisionRequest)
        await deps.node_started(
            "planner", detail="revising the plan" if is_revision else "new plan"
        )
        if is_revision:
            await deps.flow("critic", "planner", "feedback", preview=_truncate(ev.feedback, 200))
            previous: list[str] = await ctx.store.get("questions", [])
            prompt = PLANNER_REVISION_PROMPT.format(
                topic=topic,
                feedback=ev.feedback,
                previous_questions="\n".join(f"- {q}" for q in previous),
                min_questions=2,
                max_questions=settings.max_questions,
            )
        else:
            prompt = PLANNER_PROMPT.format(
                topic=topic, min_questions=2, max_questions=settings.max_questions
            )

        plan = await ask_structured(deps.llm, ResearchPlan, prompt)
        # `ask_structured` returns None on a quota refusal as well as on bad
        # JSON, and those two are nothing alike to the user. Falling through to
        # the single generic question in `_clean_questions` used to make an
        # exhausted daily budget look like a successful, merely-thinner report.
        # The critic path already reports an unusable verdict this way; the
        # planner must not be the one place that degrades silently.
        if plan is None:
            logger.warning("the planner returned nothing; using one generic question")
            await deps.bridge.emit(
                "log",
                msg=(
                    "The planner could not be reached, so this run is using a single "
                    "generic question instead of a researched breakdown of the topic."
                ),
            )
        questions = _clean_questions(plan.questions if plan else [], topic, settings.max_questions)

        # The writer blocks until it has collected this many findings, so the
        # count has to be set before the first question goes out.
        await ctx.store.set("total_questions", len(questions))
        await ctx.store.set("questions", questions)

        await deps.node_finished("planner", summary=f"{len(questions)} questions")
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Planned {len(questions)} questions on: {topic}")
        )

        for index, question in enumerate(questions):
            ctx.send_event(QuestionEvent(index=index, question=question))

    @step(num_workers=RESEARCH_WORKERS)
    async def research(self, ctx: Context, ev: QuestionEvent) -> FindingEvent:
        """One researcher per question. This is the fan-out.

        Concurrency is capped by an LLM-wide rate limiter rather than by
        `num_workers`. Each of these calls re-sends the system prompt, the tool
        schema and the agent's memory, so four at once can exceed a per-minute
        *input token* budget even when there is plenty of request headroom. The
        limiter queues them instead, so a 429 does not end the run.
        """
        deps: ResearchDeps = await ctx.store.get(DEPS_KEY)
        node = f"research#{ev.index}"

        await deps.node_started(node, detail=ev.question)
        await deps.flow("planner", node, "question", preview=_truncate(ev.question, 120))

        try:
            # A fresh agent per question, so each researcher gets its own search
            # allowance (`searcher.for_agent`). Sharing one agent meant one
            # shared search budget, and in a live run a researcher that looped to
            # its iteration cap spent the whole run's allowance: its sibling was
            # told the budget was gone having retrieved nothing, wrote a finding
            # from memory, and the writer cited it as research. The provider
            # client, cache and run-wide ceiling stay shared either way.
            agent = deps.research_agent_for(ev.index)
            async with deps.llm_slot():
                result = await with_rate_limit_retry(
                    lambda: agent.run(
                        user_msg=(
                            f"Research the answer to this question:\n"
                            f"<question>{ev.question}</question>\n\n"
                            f"Search the web as often as you need. "
                            "Return only the answer itself, with no preamble "
                            "and no markdown headings."
                        ),
                        # Bound the tool-calling loop. Left unbounded, a researcher
                        # keeps searching: once the shared search budget is spent the
                        # tool only replies "answer now", the model asks again, and
                        # the pair spins until LlamaIndex aborts the run with
                        # "Max iterations of 20 reached" - which took down a live run
                        # that had already produced one good answer. Asking for a
                        # generated early stop makes the agent write up what it found
                        # instead of raising.
                        max_iterations=deps.settings.researcher_max_iterations,
                        early_stopping_method="generate",
                    )
                )
            answer = str(result)
        except Exception as exc:
            # One researcher failing should cost one finding, not the run. The
            # writer still gets the others, and the report says what is missing
            # rather than disappearing behind an exception. Errors that look like
            # cancellation (the client hanging up, the run shutting down) are
            # re-raised so shutdown stays prompt.
            if _is_shutdown(exc):
                raise
            logger.warning("researcher %s failed, continuing without it: %s", ev.index, exc)
            # The finding goes to the writer and ends up in the report, so it must
            # not carry the provider's raw error body. A live run put the whole
            # thing in the finished document:
            #
            #   This question could not be researched: Error code: 429 - {'error':
            #   {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` in
            #   organization `org_01ksmg...` service tier `on_demand` on input
            #   tokens per minute (ITPM): Limit 7000, Used 6227, ...
            #
            # Truncating it was not enough: the first 90 characters still carried
            # the account id. So the note says what kind of failure it was and
            # nothing else - the reader needs to know the question went unanswered
            # and that nothing in it is verified. The detail belongs in the log,
            # where `logger.warning` above already recorded all of it.
            reason = (
                "the provider's quota was exhausted"
                if is_quota_error(exc)
                else f"the provider returned {type(exc).__name__}"
            )
            answer = (
                f"This question could not be researched: {reason}. Treat the whole "
                "point as unverified, answer from your own knowledge if you can, and "
                "do not present it as sourced."
            )
            await deps.bridge.emit(
                "log",
                msg=f"Researcher {ev.index + 1} failed ({_short(exc)}); continuing without it.",
            )
            ctx.write_event_to_stream(
                ProgressEvent(msg=f"Researcher {ev.index + 1} failed; continuing without it.")
            )

        await deps.node_finished(node, summary=_truncate(answer, 140))
        await deps.flow(node, "writer", "finding", preview=_truncate(answer, 140))
        ctx.write_event_to_stream(ProgressEvent(msg=f"Answered: {ev.question}"))

        return FindingEvent(index=ev.index, question=ev.question, answer=answer)

    @step
    async def write(self, ctx: Context, ev: FindingEvent) -> DraftReady:
        """Fires on every finding, but only writes once the last one lands."""
        deps: ResearchDeps = await ctx.store.get(DEPS_KEY)

        total: int = await ctx.store.get("total_questions", 1)
        collected = ctx.collect_events(ev, [FindingEvent] * total)
        if collected is None:
            ctx.write_event_to_stream(ProgressEvent(msg="Collecting findings..."))
            return None

        findings = sorted(
            (FindingEvent(**f.model_dump()) for f in collected), key=lambda f: f.index
        )
        await ctx.store.set("findings", [f.model_dump() for f in findings])

        topic: str = await ctx.store.get("topic", "")
        await deps.node_started("writer", detail=f"drafting from {len(findings)} findings")
        ctx.write_event_to_stream(ProgressEvent(msg="Writing the report..."))

        prompt = WRITER_PROMPT.format(topic=topic, notes=_format_notes(findings))
        try:
            draft = await self._stream_draft(ctx, deps, prompt)
        except Exception as exc:
            if _is_shutdown(exc) or not is_quota_error(exc):
                raise
            # The writer is the last step, and a provider that has run out of
            # quota will not answer the critic either. Returning nothing throws
            # away a minute of finished research, so assemble the report from
            # the notes instead: no LLM call is needed, which is the whole point
            # when the quota is what failed. A live run lost 64s of research to
            # "TPD: Limit 200000, Used 199383" at exactly this point.
            logger.warning("writer could not stream, assembling the report from notes: %s", exc)
            draft = _assemble_report(topic, findings)
            await ctx.store.set("writer_degraded", True)
            await deps.bridge.emit(
                "log",
                msg=(
                    "The writer ran out of provider quota, so this report is the raw "
                    "research notes assembled directly rather than a written briefing."
                ),
            )
            ctx.write_event_to_stream(
                ProgressEvent(msg="Writer unavailable; assembling the report from research notes.")
            )

        if not draft:  # pragma: no cover - provider-specific failure
            raise RuntimeError("The writer returned an empty report.")

        # The provider's per-minute output budget forces a token cap, and a
        # report longer than that cap comes back mid-word. A live run produced
        # "...18CrNiMo" as the last characters of a "finished" report. Rather
        # than ship a truncated document, detect it and retry once at a larger
        # allowance, which is affordable because this is the last big call.
        #
        # Not when the draft is already the assembled-notes fallback, though:
        # that text ends wherever the last finding ended, so it trips the
        # truncation check every time, and retrying means another call to a
        # provider that has already said no. A live run fell back to the notes
        # and then died retrying, throwing away the report it had just built.
        degraded: bool = await ctx.store.get("writer_degraded", False)
        # The provider's own `finish_reason` wins when it is available; the
        # text shape is only a fallback for providers that do not report it.
        capped: bool = await ctx.store.get("writer_hit_cap", False) or _looks_truncated(draft)
        if not degraded and capped and not await ctx.store.get("writer_retried", False):
            await ctx.store.set("writer_retried", True)
            bigger = _larger_draft_budget(deps.settings)
            logger.info(
                "draft hit the %d-token cap; retrying the writer at %d",
                deps.settings.max_output_tokens,
                bigger,
            )
            ctx.write_event_to_stream(
                ProgressEvent(msg="Draft hit the output cap; retrying with more room...")
            )
            await deps.bridge.emit(
                "log", msg=f"Draft was cut off at the output cap; retrying at {bigger} tokens."
            )
            retry = build_llm(settings_override(deps.settings, max_output_tokens=bigger))
            retry_parts: list[str] = []

            async def retry_attempt() -> str:
                # Same discipline as the first draft: this stream is on screen
                # too, so a refusal part-way through keeps what arrived instead
                # of retrying and appending a second copy of the report.
                stream = await retry.astream_complete(prompt)
                local: list[str] = []
                try:
                    async for chunk in stream:
                        delta = chunk.delta or ""
                        if delta:
                            local.append(delta)
                            await deps.bridge.emit(REPORT_DELTA, text=delta)
                        if _hit_length_cap(chunk):
                            await ctx.store.set("writer_hit_cap", True)
                            break
                finally:
                    retry_parts[:] = local
                return "".join(local).strip()

            try:
                longer = await with_rate_limit_retry(
                    retry_attempt,
                    should_retry=lambda _exc: not retry_parts,
                )
                # Always prefer the retry. The first draft is *known* to have
                # been cut off at the cap, so it is the worse artifact even when
                # it happens to contain more characters. Keeping it only if the
                # retry was longer shipped a report ending mid-word, and then
                # blamed the token cap for it.
                if longer:
                    draft = longer
            except Exception as exc:
                if _is_shutdown(exc) or not is_quota_error(exc):
                    raise
                # The retry itself failed because of quota, so fall back to notes.
                logger.warning("writer retry failed, assembling the report from notes: %s", exc)
                draft = _assemble_report(topic, findings)
                await ctx.store.set("writer_degraded", True)
                await deps.bridge.emit(
                    "log",
                    msg=(
                        "The writer ran out of provider quota while retrying; this "
                        "report is the raw research notes assembled directly."
                    ),
                )

        # L1: only blame the token cap when the cap is what actually stopped the
        # writer. Reporting a healthy report as truncated - and advising the
        # operator to raise a limit that was never the problem - is worse than
        # saying nothing, and a degraded draft is a different failure entirely.
        if await ctx.store.get("writer_hit_cap", False):
            await deps.bridge.emit(
                "log",
                msg=(
                    "The report is still truncated at the provider's output cap. "
                    "Raise MAX_OUTPUT_TOKENS on a paid tier, or lower MAX_QUESTIONS."
                ),
            )
        elif degraded:
            await deps.bridge.emit(
                "log",
                msg=(
                    "This report is the raw research notes: the writer had no quota "
                    "left to write them up, so it is unedited and unverified."
                ),
            )

        await deps.node_finished("writer", summary=f"{len(draft)} characters")
        await deps.flow("writer", "critic", "draft", preview=_truncate(draft, 140))
        return DraftReady(draft=draft)

    async def _stream_draft(self, ctx: Context, deps: ResearchDeps, prompt: str) -> str:
        """Stream the draft to the browser token by token, and return it whole.

        The only provider call in the pipeline that was not routed through
        `with_rate_limit_retry`, and a live run showed why that mattered: the
        writer was refused with `OTPM: Limit 1000 ... Please try again in 5.94s`
        - a six-second wait - and the caller treated it as terminal, so the run
        degraded to raw research notes. Every other call would have waited.

        Retrying is only safe while nothing has been streamed: once tokens have
        reached the browser, a retry would append a second draft to the first
        one. A mid-stream refusal therefore keeps what was written and stops,
        rather than duplicating the report.
        """
        parts: list[str] = []

        async def attempt() -> str:
            stream = await deps.llm.astream_complete(prompt)
            local: list[str] = []
            try:
                async for chunk in stream:
                    delta = chunk.delta or ""
                    if delta:
                        local.append(delta)
                        await deps.bridge.emit(REPORT_DELTA, text=delta)
                    if _hit_length_cap(chunk):
                        # The provider told us it stopped at `max_tokens`, so no
                        # further chunk can carry content. Recorded for the retry
                        # decision, which asks for a larger allowance.
                        await ctx.store.set("writer_hit_cap", True)
                        break
            finally:
                # Whatever arrived is real output, even if the stream then failed.
                # Without this, a refusal part-way through raised with `parts`
                # still empty, and the text the browser had already been sent was
                # thrown away in favour of a hard failure.
                parts[:] = local
            return "".join(local).strip()

        try:
            # `should_retry` is what makes the docstring's promise true: once a
            # token has reached the browser, a retry would append a second draft
            # to the first, so the refusal is handed back as a partial instead.
            return await with_rate_limit_retry(attempt, should_retry=lambda _exc: not parts)
        except Exception as exc:
            if not parts or _is_shutdown(exc):
                raise
            # Tokens are already on the screen, so this is a partial draft and
            # not a failure to retry. Hand back what arrived.
            logger.warning(
                "writer stream failed after %d characters (%s); keeping the partial draft",
                len("".join(parts)),
                exc,
            )
            return "".join(parts).strip()

    @step
    async def review(self, ctx: Context, ev: DraftReady) -> StopEvent | RevisionRequest:
        """Accept the draft, or hand specific feedback back to the planner."""
        deps: ResearchDeps = await ctx.store.get(DEPS_KEY)
        settings = deps.settings
        topic: str = await ctx.store.get("topic", "")
        findings = [FindingEvent(**f) for f in await ctx.store.get("findings", [])]
        cycles: int = await ctx.store.get("review_cycles", 0)

        await deps.node_started("critic", detail=f"review {cycles + 1}")

        verdict = await ask_structured(
            deps.llm,
            ReviewVerdict,
            CRITIC_PROMPT.format(topic=topic, notes=_format_notes(findings), draft=ev.draft),
        )

        # A critic that could not be reached is not a critic that approved.
        #
        # Treating an unreadable verdict as "acceptable" is the worst option
        # available: the report ships, the metadata says "approved", and nothing
        # anywhere records that the review never happened. A live run hit this
        # when the verdict call died on an output-token limit, and the run
        # reported a clean approval of a draft that had never been reviewed.
        #
        # So an unreadable verdict costs one review cycle and says so out loud.
        # The loop is already bounded by max_review_cycles, so this cannot
        # deadlock, and a genuine provider outage is recorded as a failure
        # instead of being laundered into a green result.
        if verdict is None:
            await ctx.store.set("verdict", "critic_unavailable")
            await deps.node_finished("critic", summary="could not review")
            await deps.bridge.emit(
                "log",
                msg=(
                    "The critic could not be reached, so the report is unverified. "
                    "Treat it as a draft, not a reviewed result."
                ),
            )
            await deps.flow("critic", "writer", "unverified", preview="no verdict")
            ctx.write_event_to_stream(
                ProgressEvent(msg="Critic unavailable; report left unverified.")
            )
            return StopEvent(result=ev.draft)

        acceptable = verdict.acceptable
        feedback = verdict.feedback
        exhausted = cycles >= settings.max_review_cycles
        degraded: bool = await ctx.store.get("writer_degraded", False)

        if acceptable or exhausted:
            reason = "approved" if acceptable else "out of review cycles"
            await deps.node_finished("critic", summary=reason)
            await deps.flow("critic", "writer", "accepted", preview=topic)
            await ctx.store.set("verdict", "acceptable" if acceptable else "cycle_limit")
            ctx.write_event_to_stream(ProgressEvent(msg=f"Report {reason}."))
            return StopEvent(result=ev.draft)

        if degraded:
            # A revision cannot fix this. The draft fell back to the raw notes
            # because the writer hit a quota, so sending it back re-runs the
            # research and the writer and lands in exactly the same wall. A live
            # run did precisely that, spending the remaining 800 seconds of a
            # 900-second budget on a second round that had no way to succeed.
            # The critic's feedback is still worth keeping; the report is still
            # worth returning, clearly labelled as unreviewed-and-improvable.
            await deps.node_finished("critic", summary="changes wanted, but no budget to act")
            await ctx.store.set("verdict", "unverified")
            await ctx.store.set("reviewer_feedback", feedback)
            await deps.flow("critic", "writer", "unverified", preview=_truncate(feedback, 200))
            await deps.bridge.emit(
                "log",
                msg=(
                    "The critic asked for changes, but the writer had already fallen "
                    "back to the raw notes, so a revision would hit the same limit. "
                    "Returning the notes with the feedback attached."
                ),
            )
            ctx.write_event_to_stream(
                ProgressEvent(msg="Changes wanted, but out of provider budget; stopping here.")
            )
            return StopEvent(result=ev.draft)

        # Recorded only when the critic actually asked for changes, so
        # `reviewer_feedback` reads as "why this took more than one pass".
        await ctx.store.set("review_cycles", cycles + 1)
        await ctx.store.set("reviewer_feedback", feedback)
        await deps.node_finished("critic", summary="requested changes")
        await deps.flow("critic", "planner", "feedback", preview=_truncate(feedback, 200))
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Reviewer asked for changes (cycle {cycles + 1})")
        )
        return RevisionRequest(topic=topic, feedback=feedback)


def build_workflow(settings: Settings, timeout: int = 900) -> DeepResearchWorkflow:
    """Create the workflow. Research parallelism is set by `RESEARCH_WORKERS`."""
    return DeepResearchWorkflow(timeout=timeout, verbose=False)
