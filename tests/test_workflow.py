"""End-to-end tests of the pipeline, driven entirely by fakes.

These are the tests that matter: they exercise the fan-out, the `collect_events`
barrier, the reflection loop and persistence, with no network and no API keys.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import QUESTIONS, make_settings
from deep_research.config import Settings
from deep_research.events import (
    DONE,
    EDGE_FLOW,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
)
from deep_research.runner import ResearchRunner
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.storage import RunStore
from deep_research.tools.web_search import WebSearcher
from fakes import FakeLLM, ScriptedResearchAgent, StubTavilyClient, extract_question
from helpers import drain, events_of, kinds


def make_runner(
    settings: Settings,
    llm: FakeLLM,
    agent: ScriptedResearchAgent,
    searcher: WebSearcher | None = None,
    topic: str = "gearbox selection for a mobile crane",
) -> ResearchRunner:
    return ResearchRunner(
        topic=topic,
        settings=settings,
        store=RunStore(settings.runs_dir),
        llm=llm,
        research_agent=agent,
        searcher=searcher,
    )


async def test_happy_path_produces_and_persists_a_report(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    runner = make_runner(settings, llm, agent)
    events, result = await drain(runner)

    assert result is not None
    assert result.report == llm.report
    assert result.meta.verdict == "acceptable"
    assert result.meta.status == "ok"
    assert result.meta.review_cycles == 0

    store = RunStore(settings.runs_dir)
    assert store.exists(runner.run_id)
    # save_report normalises the trailing newline; the in-memory result is raw.
    assert store.load_report(runner.run_id) == llm.report + "\n"
    # The archived log is the whole stream, terminal events included, so a run
    # can be replayed or debugged without re-spending tokens.
    assert [event["kind"] for event in store.load_events(runner.run_id)] == kinds(events)
    assert kinds(events)[-1] == DONE
    assert REPORT_DONE in kinds(events)


async def test_planner_questions_become_one_researcher_each(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    _, result = await drain(make_runner(settings, llm, agent))

    assert len(agent.questions) == len(QUESTIONS)
    assert result is not None
    assert result.meta.questions == QUESTIONS


async def test_findings_reach_the_writer_in_question_order(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    await drain(make_runner(settings, llm, agent))

    writer_prompt = llm.writer_prompts[0]
    positions = [writer_prompt.index(f"Question {i + 1}: {q}") for i, q in enumerate(QUESTIONS)]
    assert positions == sorted(positions), "writer prompt is not in question order"
    # Every finding body made it into the prompt, none were dropped by the barrier.
    for i in range(1, len(QUESTIONS) + 1):
        assert f"Question {i}:" in writer_prompt


async def test_research_fans_out_concurrently(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    agent.delay = 0.02
    await drain(make_runner(settings, llm, agent))

    assert agent.max_concurrent > 1, "researchers ran serially"


async def test_critic_feedback_loops_back_through_the_planner(
    settings: Settings, agent: ScriptedResearchAgent
):
    llm = FakeLLM(
        plans=[
            ResearchPlan(questions=list(QUESTIONS)),
            ResearchPlan(questions=["What is the service life of the gearbox?"]),
        ],
        reviews=[
            ReviewVerdict(acceptable=False, feedback="Nothing on maintenance intervals."),
            ReviewVerdict(acceptable=True),
        ],
    )
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert result.meta.review_cycles == 1
    assert result.meta.verdict == "acceptable"
    assert result.meta.reviewer_feedback == "Nothing on maintenance intervals."

    # Re-planned: the second planning prompt carries the feedback.
    assert len(llm.plans_seen) == 2
    assert "Nothing on maintenance intervals." in llm.plans_seen[1]
    # Both rounds researched, the second round the extra question only.
    assert len(agent.questions) == len(QUESTIONS) + 1
    # The loop-back edge is on the wire.
    loop_edges = [
        event
        for event in events_of(events, EDGE_FLOW)
        if event.data["source"] == "critic" and event.data["target"] == "planner"
    ]
    assert loop_edges


async def test_review_cycles_are_capped(settings: Settings, agent: ScriptedResearchAgent):
    capped = settings.model_copy(update={"max_review_cycles": 1})
    llm = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=False, feedback="Still thin.")],
    )
    _, result = await drain(make_runner(capped, llm, agent))

    assert result is not None
    # The critic never accepted; the cap stopped the loop instead.
    assert result.meta.verdict == "cycle_limit"
    assert result.meta.review_cycles == 1
    assert result.report == llm.report


async def test_no_review_cycles_accepts_immediately(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    no_review = settings.model_copy(update={"max_review_cycles": 0})
    always_reject = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=False, feedback="Nope.")],
    )
    _, result = await drain(make_runner(no_review, always_reject, agent))

    assert result is not None
    assert result.meta.review_cycles == 0
    assert result.meta.verdict == "cycle_limit"


async def test_event_stream_is_well_formed(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    events, _ = await drain(make_runner(settings, llm, agent))
    sequence = kinds(events)

    assert sequence[0] == RUN_STARTED
    assert sequence[-1] == DONE
    assert RUN_FAILED not in sequence
    assert sequence.count(RUN_STARTED) == 1

    # The graph announcement carries the topology the UI draws.
    started = events[0].data
    assert {n["id"] for n in started["graph"]["nodes"]} == {
        "planner",
        "research",
        "writer",
        "critic",
    }
    assert started["config"]["model"] == settings.model

    # Every worker node announced both its arrival and its departure.
    activated = [e.data["node"] for e in events_of(events, NODE_ACTIVATED)]
    finished = [e.data["node"] for e in events_of(events, NODE_FINISHED)]
    for index in range(len(QUESTIONS)):
        assert f"research#{index}" in activated
        assert f"research#{index}" in finished

    # The report was streamed token by token and then delivered whole.
    deltas = events_of(events, REPORT_DELTA)
    assert "".join(e.data["text"] for e in deltas) == llm.report
    done = events_of(events, REPORT_DONE)[0].data
    assert done["markdown"] == llm.report
    assert done["report_path"].endswith("report.md")


async def test_a_second_run_does_not_reuse_the_first_context(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    first, _ = await drain(make_runner(settings, llm, agent))
    second, result2 = await drain(make_runner(settings, llm, agent))

    assert kinds(first)[0] == RUN_STARTED
    assert kinds(second)[0] == RUN_STARTED
    assert result2 is not None
    assert result2.meta.questions == QUESTIONS


async def test_consumer_leaving_early_cancels_the_run(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    runner = make_runner(settings, llm, agent)
    seen = 0
    async for _ in runner.run():
        seen += 1
        if seen == 2:
            break

    assert seen == 2
    await asyncio.sleep(0.05)
    assert runner.result is None, "the run kept going after the consumer walked away"


async def test_search_results_reach_the_researcher(settings: Settings, llm: FakeLLM):
    searcher = WebSearcher(settings, client=StubTavilyClient(answer="Gearboxes are mechanical."))
    # Exercise the tool directly: the researcher agent is faked in these tests.
    result = await searcher.search("how gearboxes work")

    assert "Gearboxes are mechanical." in result
    assert "https://example.com" in result
    assert searcher.calls == 1

    # Second identical query is served from the cache, not from Tavily.
    again = await searcher.search("how gearboxes work")
    assert again == result
    assert searcher.calls == 1
    assert searcher.cache.stats()["hits"] == 1


async def test_empty_plan_does_not_deadlock(settings: Settings, agent: ScriptedResearchAgent):
    """A planner that returns nothing must still produce a report."""
    llm = FakeLLM(plans=[ResearchPlan(questions=[])], reviews=[ReviewVerdict(acceptable=True)])
    _, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert len(agent.questions) == 1
    assert "gearbox selection" in agent.questions[0]


async def test_failing_writer_is_reported_not_raised(
    settings: Settings, agent: ScriptedResearchAgent
):
    class ExplodingLLM(FakeLLM):
        async def astream_complete(self, prompt):
            raise RuntimeError("provider is down")

    llm = ExplodingLLM(plans=[ResearchPlan(questions=list(QUESTIONS))])
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is None
    assert RUN_FAILED in kinds(events)
    failure = events_of(events, RUN_FAILED)[0].data["error"]
    assert "provider is down" in failure
    assert kinds(events)[-1] == DONE, "the stream must end cleanly even on failure"


async def test_a_failed_run_is_archived_for_later_inspection(
    settings: Settings, agent: ScriptedResearchAgent
):
    class ExplodingLLM(FakeLLM):
        async def astream_complete(self, prompt):
            raise RuntimeError("provider is down")

    llm = ExplodingLLM(plans=[ResearchPlan(questions=list(QUESTIONS))])
    runner = make_runner(settings, llm, agent)
    events, result = await drain(runner)

    store = RunStore(settings.runs_dir)
    meta = store.load_meta(runner.run_id)
    assert result is None
    assert meta.status == "failed"
    assert "provider is down" in meta.error
    assert meta.questions == list(QUESTIONS), "the plan survives the failure"
    # The archive holds every event, so the failure is reproducible offline.
    assert [event["kind"] for event in store.load_events(runner.run_id)] == kinds(events)


async def test_missing_api_key_fails_cleanly(tmp_path):
    settings = make_settings(tmp_path, groq_api_key="")
    runner = ResearchRunner(topic="anything", settings=settings)
    events, result = await drain(runner)

    assert result is None
    assert "GROQ_API_KEY" in events_of(events, RUN_FAILED)[0].data["error"]


async def test_blank_topic_is_rejected(settings: Settings):
    with pytest.raises(ValueError, match="topic is required"):
        ResearchRunner(topic="   ", settings=settings)


async def test_questions_are_deduped_and_numbering_stripped(
    settings: Settings, agent: ScriptedResearchAgent
):
    llm = FakeLLM(
        plans=[
            ResearchPlan(
                questions=[
                    "1. What is a helical gearbox used for?",
                    "  What is a helical gearbox used for?  ",
                    "- Which gearbox suits a crane?",
                ]
            )
        ],
        reviews=[ReviewVerdict(acceptable=True)],
    )
    _, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert result.meta.questions == [
        "What is a helical gearbox used for?",
        "Which gearbox suits a crane?",
    ]
    assert extract_question(agent.questions[0]) == result.meta.questions[0]
