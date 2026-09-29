"""HTTP surface tests.

The whole point of `runner_factory` is to be here: a full research run happens
over the real SSE endpoint with fake agents, so the framing, the graph payload
and the report path are all covered without a network call.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import QUESTIONS
from deep_research.config import Settings
from deep_research.runner import ResearchRunner
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.server import create_app
from fakes import FakeLLM, ScriptedResearchAgent


def frames(body: str) -> list[dict]:
    """Parse an SSE body back into the frames a browser would see."""
    out = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


def kinds(parsed: list[dict]) -> list[str]:
    return [frame["kind"] for frame in parsed]


@pytest.fixture
def client(settings: Settings) -> TestClient:
    def factory(topic: str, config: Settings, store):
        return ResearchRunner(
            topic=topic,
            settings=config,
            store=store,
            llm=FakeLLM(
                plans=[ResearchPlan(questions=list(QUESTIONS))],
                reviews=[ReviewVerdict(acceptable=True)],
            ),
            research_agent=ScriptedResearchAgent(),
        )

    return TestClient(create_app(settings, runner_factory=factory))


def test_health_reports_the_configuration(client: TestClient):
    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["search_provider"] == "none"
    assert payload["concurrency"] >= 1


def test_health_flags_missing_keys(tmp_path):
    from conftest import make_settings

    app = create_app(make_settings(tmp_path, groq_api_key="", tavily_api_key=""))
    payload = TestClient(app).get("/api/health").json()

    assert payload["status"] == "needs_config"


def test_graph_endpoint_returns_the_topology(client: TestClient):
    payload = client.get("/api/graph").json()

    assert [node["id"] for node in payload["nodes"]] == [
        "planner",
        "research",
        "writer",
        "critic",
    ]
    assert any(edge["kind"] == "loop" for edge in payload["edges"])


def test_research_streams_the_whole_run(client: TestClient):
    response = client.post("/api/research", json={"topic": "gearbox selection for a crane"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    parsed = frames(response.text)
    sequence = kinds(parsed)

    assert sequence[0] == "run_started"
    assert sequence[-1] == "done"
    assert "run_failed" not in sequence
    assert "report_done" in sequence
    assert "report_delta" in sequence

    deltas = "".join(f["text"] for f in parsed if f["kind"] == "report_delta")
    done = next(f for f in parsed if f["kind"] == "report_done")
    assert deltas == done["markdown"]
    assert done["questions"] == QUESTIONS
    assert done["review_cycles"] == 0


def test_the_first_frame_carries_what_the_ui_draws(client: TestClient):
    parsed = frames(client.post("/api/research", json={"topic": "gearboxes"}).text)
    started = parsed[0]

    assert started["topic"] == "gearboxes"
    assert {node["id"] for node in started["graph"]["nodes"]} == {
        "planner",
        "research",
        "writer",
        "critic",
    }
    assert started["config"]["model"] == "llama-3.3-70b-versatile"


def test_the_run_is_archived_and_listed(client: TestClient):
    parsed = frames(client.post("/api/research", json={"topic": "gearboxes"}).text)
    done = next(f for f in parsed if f["kind"] == "report_done")

    listed = client.get("/api/runs").json()["runs"]
    assert [row["run_id"] for row in listed] == [done["run_id"]]
    assert listed[0]["topic"] == "gearboxes"

    fetched = client.get(f"/api/runs/{done['run_id']}").json()
    # The archive normalises the trailing newline; the stream carries the raw text.
    assert fetched["report"] == done["markdown"] + "\n"
    assert fetched["meta"]["model"] == "llama-3.3-70b-versatile"


def test_unknown_run_is_a_404(client: TestClient):
    assert client.get("/api/runs/nope").status_code == 404


def test_a_too_short_topic_is_rejected(client: TestClient):
    assert client.post("/api/research", json={"topic": "x"}).status_code == 422


def test_a_missing_topic_is_rejected(client: TestClient):
    assert client.post("/api/research", json={}).status_code == 422


def test_the_ui_is_served(client: TestClient):
    index = client.get("/")

    assert index.status_code == 200
    assert "Deep Research Agent" in index.text
    assert "/static/app.js" in index.text

    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


def test_a_failing_run_is_reported_in_band(tmp_path):
    """A provider error must arrive as an event, not as a 500."""
    from conftest import make_settings

    class ExplodingLLM(FakeLLM):
        async def astream_complete(self, prompt):
            raise RuntimeError("provider is down")

    def factory(topic: str, config: Settings, store):
        return ResearchRunner(
            topic=topic,
            settings=config,
            store=store,
            llm=ExplodingLLM(plans=[ResearchPlan(questions=list(QUESTIONS))]),
            research_agent=ScriptedResearchAgent(),
        )

    client = TestClient(create_app(make_settings(tmp_path), runner_factory=factory))
    response = client.post("/api/research", json={"topic": "gearboxes"})

    assert response.status_code == 200
    parsed = frames(response.text)
    failure = next(f for f in parsed if f["kind"] == "run_failed")

    assert "provider is down" in failure["error"]
    assert kinds(parsed)[-1] == "done"
