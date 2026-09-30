"""Running a research topic end to end.

`ResearchRunner` owns the parts a run needs but the workflow should not know
about: configuration, the event bridge, the on-disk archive, and the wiring
between the LLM, the researcher agent and the workflow. The CLI and the HTTP
API both drive a run through the same object, so both see identical events.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from llama_index.core.workflow import Context

from .agents import agent_tool_names, build_research_agent
from .config import Settings
from .events import (
    DONE,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
    EventBridge,
    RunEvent,
)
from .graph import graph_payload
from .llm import LLMGate, build_llm, check_model_available, llm_concurrency_limit
from .storage import RunMeta, RunStore, new_run_id
from .tools.web_search import WebSearcher
from .workflow import RESEARCH_WORKERS, ResearchDeps, build_workflow

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchResult:
    """The finished artefact: the report, where it went, and how it went."""

    run_id: str
    topic: str
    report: str
    meta: RunMeta
    report_path: str = ""
    events: list[RunEvent] = field(default_factory=list)


class ResearchRunner:
    """Drives one research run and streams its events to a caller."""

    def __init__(
        self,
        topic: str,
        settings: Settings | None = None,
        store: RunStore | None = None,
        llm: Any | None = None,
        searcher: WebSearcher | None = None,
        research_agent: Any | None = None,
        bridge: EventBridge | None = None,
    ) -> None:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("A research topic is required.")

        self.settings = settings or Settings()
        self.store = store or RunStore(self.settings.runs_dir)
        self.run_id = new_run_id(self.topic)
        self.bridge = bridge or EventBridge(self.run_id)

        # Injection points: the tests pass fakes here, the CLI passes nothing.
        self._llm = llm
        self._searcher = searcher
        self._research_agent = research_agent
        self._context: Context | None = None
        self._result: ResearchResult | None = None

    @property
    def result(self) -> ResearchResult | None:
        return self._result

    # -- public api ---------------------------------------------------------
    async def run(self) -> AsyncIterator[RunEvent]:
        """Execute the run, yielding every event as it happens.

        The iterator always ends with a `done` event, including on failure,
        so consumers can rely on a clean end-of-stream.
        """
        task = asyncio.create_task(self._execute(), name=f"research:{self.run_id}")
        try:
            async for event in self.bridge.events():
                yield event
                if event.kind == DONE:
                    break
        finally:
            # Never leave the task orphaned if the consumer walks away early.
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        # Only await a task that actually finished. Awaiting a *cancelled* task
        # re-raises `CancelledError` - a `BaseException`, so it escapes `run()`
        # entirely. A consumer that stops reading once it sees `done` cancels the
        # cleanup tail of the task, and that must not turn a complete, persisted
        # result into a traceback.
        if not task.cancelled():
            await task

    # -- internals ----------------------------------------------------------
    async def _execute(self) -> None:
        started = datetime.now(UTC)
        searcher: WebSearcher | None = self._searcher
        # Only a searcher this runner built may be closed below; an injected one
        # may be shared with the caller, and closing it is their decision.
        owns_searcher = self._searcher is None
        try:
            self.settings.require_llm_key()
            self.settings.require_search_key()

            # Skipped when a fake LLM is injected, so the offline suite never
            # touches the network. A retired model otherwise surfaces as a 404
            # on the first call, after the run has already started.
            if self._llm is None:
                await check_model_available(self.settings)

            llm = self._llm or build_llm(self.settings)
            searcher = self._searcher or WebSearcher(self.settings)

            # One agent per question, each with its own search allowance over the
            # shared searcher. A single shared agent meant a single shared search
            # budget, and a researcher that looped to its cap could spend the whole
            # run's allowance, leaving a sibling to report having no sources.
            injected = self._research_agent
            research_agent_factory = None
            research_agent = injected
            if injected is None:

                def research_agent_factory(_index: int) -> Any:
                    return build_research_agent(llm, searcher.for_agent())

                # Only used for run metadata (`tools_used`); the workflow asks the
                # factory once per question.
                research_agent = research_agent_factory(0)
            workflow = build_workflow(self.settings)

            await self.bridge.emit(
                RUN_STARTED,
                run_id=self.run_id,
                topic=self.topic,
                graph=graph_payload(),
                config=self.config_payload(),
            )

            deps = ResearchDeps(
                llm=llm,
                research_agent=research_agent,
                research_agent_factory=research_agent_factory,
                searcher=searcher,
                settings=self.settings,
                bridge=self.bridge,
                # A soft cap on in-flight LLM turns. num_workers still bounds how
                # many researchers exist; this bounds how many hit the provider at
                # once, which is a different and much tighter limit.
                llm_gate=LLMGate(llm_concurrency_limit()),
            )

            # Hold the context so the finished run's state can be read back
            # without threading a mutable result object through every step.
            self._context = Context(workflow)
            report = str(await workflow.run(topic=self.topic, deps=deps, ctx=self._context))

            self._result = await self._persist(
                report, searcher, research_agent, started, self._context
            )
            await self.bridge.emit(DONE)
            # Re-archive: the first save happened before `report_done` and `done`
            # were emitted, so only this write has the whole stream.
            self.store.save_events(self.run_id, self.bridge.replay())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Research run %s failed", self.run_id)
            message = f"{type(exc).__name__}: {exc}"
            await self.bridge.emit(RUN_FAILED, error=message)
            await self.bridge.emit(DONE)
            # A failed run is still worth keeping: the partial event log is
            # usually the only clue about what went wrong.
            await self._archive_failure(message, started, self._context)
        finally:
            # Release the search provider's connection pool, but only if this
            # runner is the one that created it.
            if owns_searcher and searcher is not None:
                await searcher.aclose()
            await self.bridge.close()

    async def _archive_failure(
        self,
        message: str,
        started: datetime,
        context: Context | None,
    ) -> None:
        """Record a failed run so it shows up in `deep-research runs`."""
        questions: list[str] = []
        if context is not None:
            try:
                questions = [str(q) for q in await context.store.get("questions", [])]
            except Exception:
                questions = []
        try:
            self.store.save_meta(
                RunMeta(
                    run_id=self.run_id,
                    topic=self.topic,
                    model=self.settings.model,
                    started_at=started.isoformat(),
                    finished_at=datetime.now(UTC).isoformat(),
                    duration_seconds=self.bridge.duration_seconds,
                    status="failed",
                    error=message,
                    questions=questions,
                )
            )
            self.store.save_events(self.run_id, self.bridge.replay())
        except OSError:
            logger.exception("Could not archive the failed run %s", self.run_id)

    async def _persist(
        self,
        report: str,
        searcher: WebSearcher,
        research_agent: Any,
        started: datetime,
        context: Context,
    ) -> ResearchResult:
        ctx = context
        findings: list[dict[str, str]] = list(await ctx.store.get("findings", []))
        questions: list[str] = list(await ctx.store.get("questions", []))
        cycles = int(await ctx.store.get("review_cycles", 0))
        verdict = str(await ctx.store.get("verdict", "acceptable"))
        feedback = str(await ctx.store.get("reviewer_feedback", ""))
        cache_stats = searcher.cache.stats() if searcher.cache else {}

        meta = RunMeta(
            run_id=self.run_id,
            topic=self.topic,
            model=self.settings.model,
            started_at=started.isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            duration_seconds=self.bridge.duration_seconds,
            review_cycles=cycles,
            questions=questions,
            findings=findings,
            search_calls=searcher.calls,
            cache=cache_stats,
            verdict=verdict,
            reviewer_feedback=feedback,
            tools_used=agent_tool_names(research_agent),
        )

        report_path = self.store.save_report(self.run_id, report)
        meta_path = self.store.save_meta(meta)
        self.store.save_events(self.run_id, self.bridge.replay())

        await self.bridge.emit(
            REPORT_DONE,
            run_id=self.run_id,
            markdown=report,
            report_path=str(report_path),
            meta_path=str(meta_path),
            questions=questions,
            review_cycles=cycles,
            verdict=verdict,
            cache=cache_stats,
            search_calls=searcher.calls,
        )
        return ResearchResult(
            run_id=self.run_id,
            topic=self.topic,
            report=report,
            meta=meta,
            report_path=str(report_path),
            events=self.bridge.replay(),
        )

    def config_payload(self) -> dict[str, Any]:
        """The run configuration, echoed to the UI so it can show what was used."""
        return {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "search_provider": self.settings.search_provider,
            "max_questions": self.settings.max_questions,
            "max_review_cycles": self.settings.max_review_cycles,
            "concurrency": RESEARCH_WORKERS,
        }
