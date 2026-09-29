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
from .llm import ask_structured
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


def _truncate(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _format_notes(findings: list[FindingEvent]) -> str:
    blocks = [
        f"## Question {finding.index + 1}: {finding.question}\n{_truncate(finding.answer)}"
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
        """One researcher per question. This is the fan-out."""
        deps: ResearchDeps = await ctx.store.get(DEPS_KEY)
        node = f"research#{ev.index}"

        await deps.node_started(node, detail=ev.question)
        await deps.flow("planner", node, "question", preview=_truncate(ev.question, 120))

        result = await deps.research_agent.run(
            user_msg=(
                f"Research the answer to this question:\n"
                f"<question>{ev.question}</question>\n\n"
                f"Search the web as often as you need. "
                f"Return only the answer itself, with no preamble and no markdown headings."
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

        # A failed verdict call counts as "no objections raised", never as a crash.
        acceptable = True if verdict is None else verdict.acceptable
        feedback = "" if verdict is None else verdict.feedback
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
