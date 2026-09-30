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

import logging
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from llama_index.core.llms import LLM
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from .config import Settings
from .events import (
    EDGE_FLOW,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    EventBridge,
)
from .graph import NODE_BY_ID, parse_node_id
from .llm import ask_structured, build_llm, settings_override, with_rate_limit_retry
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
    research_agent: Any
    searcher: WebSearcher
    settings: Settings
    bridge: EventBridge
    llm_gate: Any = None

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


def _looks_truncated(text: str) -> bool:
    """Does this draft stop because it ran out of tokens rather than finished?

    A model cut off by a `max_tokens` cap leaves a hard signal: the text ends
    without closing the markdown construct it was in. A finished report ends
    with punctuation. Guessing from length alone would misfire, so this only
    reports a cut when the evidence is there, and the caller retries once.
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
    return True


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

        async with deps.llm_slot():
            result = await with_rate_limit_retry(
                lambda: deps.research_agent.run(
                    user_msg=(
                        f"Research the answer to this question:\n"
                        f"<question>{ev.question}</question>\n\n"
                        f"Search the web as often as you need. "
                        f"Return only the answer itself, with no preamble and no markdown headings."
                    )
                )
            )
        answer = str(result)

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
        parts: list[str] = []
        stream = await deps.llm.astream_complete(prompt)
        async for chunk in stream:
            delta = chunk.delta or ""
            if delta:
                parts.append(delta)
                await deps.bridge.emit(REPORT_DELTA, text=delta)

        draft = "".join(parts).strip()
        if not draft:  # pragma: no cover - provider-specific failure
            raise RuntimeError("The writer returned an empty report.")

        # The provider's per-minute output budget forces a token cap, and a
        # report longer than that cap comes back mid-word. A live run produced
        # "...18CrNiMo" as the last characters of a "finished" report. Rather
        # than ship a truncated document, detect it and retry once at a larger
        # allowance, which is affordable because this is the last big call.
        if _looks_truncated(draft) and not await ctx.store.get("writer_retried", False):
            await ctx.store.set("writer_retried", True)
            bigger = min(deps.settings.max_output_tokens * 3, 2048)
            logger.info(
                "draft looks truncated at the %d-token cap; retrying the writer at %d",
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
            retry_stream = await retry.astream_complete(prompt)
            async for chunk in retry_stream:
                delta = chunk.delta or ""
                if delta:
                    retry_parts.append(delta)
                    await deps.bridge.emit(REPORT_DELTA, text=delta)
            longer = "".join(retry_parts).strip()
            if len(longer) > len(draft):
                draft = longer

        if _looks_truncated(draft):
            # Still short after the retry: say so rather than implying the
            # report is complete, and let the critic weigh it.
            await deps.bridge.emit(
                "log",
                msg=(
                    "The report is still truncated at the provider's output cap. "
                    "Raise MAX_OUTPUT_TOKENS on a paid tier, or lower MAX_QUESTIONS."
                ),
            )

        await deps.node_finished("writer", summary=f"{len(draft)} characters")
        await deps.flow("writer", "critic", "draft", preview=_truncate(draft, 140))
        return DraftReady(draft=draft)

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

        if acceptable or exhausted:
            reason = "approved" if acceptable else "out of review cycles"
            await deps.node_finished("critic", summary=reason)
            await deps.flow("critic", "writer", "accepted", preview=topic)
            await ctx.store.set("verdict", "acceptable" if acceptable else "cycle_limit")
            ctx.write_event_to_stream(ProgressEvent(msg=f"Report {reason}."))
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
