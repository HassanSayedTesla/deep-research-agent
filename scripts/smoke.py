"""Boot the real ASGI server and drive it end to end over HTTP.

Not part of the test suite — this is the check that the HTTP surface works
outside `TestClient`: a real uvicorn socket, a real streamed response, and a
real client disconnect.

    python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import httpx
import uvicorn

from deep_research.config import Settings
from deep_research.runner import ResearchRunner
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.server import create_app
from fakes import FakeLLM, ScriptedResearchAgent

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"

QUESTIONS = [
    "What is a helical gearbox?",
    "How do planetary gearboxes handle load?",
    "Which gearbox suits a mobile crane?",
]


def factory(topic: str, config: Settings, store):
    """A run that takes the whole path: plan, research, reject, revise, accept."""
    llm = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[
            ReviewVerdict(acceptable=False, feedback="Nothing on maintenance intervals."),
            ReviewVerdict(acceptable=True),
        ],
        report="# Gearbox selection\n\nPlanetary units suit variable loads.\n",
    )
    return ResearchRunner(
        topic=topic,
        settings=config,
        store=store,
        llm=llm,
        research_agent=ScriptedResearchAgent(),
    )


async def wait_for_server(server: uvicorn.Server, budget_seconds: float = 15.0) -> None:
    for _ in range(int(budget_seconds / 0.1)):
        if server.started:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("uvicorn never reported itself started")


async def stream_whole_run(client: httpx.AsyncClient) -> list[dict]:
    """POST a run and read the response frame by frame, as a browser would."""
    kinds: list[str] = []
    cycles = 0
    async with client.stream(
        "POST",
        "/api/research",
        json={"topic": "Choosing a crane gearbox"},
    ) as r:
        assert r.status_code == 200, r.status_code
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            kinds.append(event["kind"])
            if event["kind"] == "node_activated":
                print(f"  node   {event['node']:<12} {event.get('detail', '')[:40]}")
            elif event["kind"] == "edge_flow":
                print(f"  edge   {event['source']} -> {event['target']} ({event['label']})")
            elif event["kind"] == "report_done":
                cycles = event["review_cycles"]
                print(f"  saved  {Path(event['report_path']).name}")
            elif event["kind"] in {"run_started", "run_failed", "done"}:
                print(f"  {event['kind']}")
    print(f"frames: {len(kinds)}, review cycles: {cycles}")
    return kinds


async def stream_then_disconnect(client: httpx.AsyncClient) -> int:
    """Hang up mid-run. The server must cancel the job, not wedge or crash."""
    seen = 0
    async with client.stream(
        "POST", "/api/research", json={"topic": "A run that gets abandoned"}
    ) as r:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                seen += 1
                if seen == 3:
                    break  # leaving the block closes the connection
    print(f"aborted after {seen} frames")
    return seen


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            groq_api_key="smoke",
            search_provider="none",
            runs_dir=Path(tmp) / "runs",
            cache_file=Path(tmp) / "cache.json",
            max_questions=3,
            max_review_cycles=2,
        )
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(settings, runner_factory=factory),
                host="127.0.0.1",
                port=PORT,
                log_level="warning",
            )
        )
        task = asyncio.create_task(server.serve())
        await wait_for_server(server)
        print(f"server up on {PORT}")

        async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
            health = (await client.get("/api/health")).json()
            print(f"health: {health['status']} model={health['model']}")

            index = await client.get("/")
            print(f"index: {index.status_code} ({len(index.text)} bytes)")
            for asset in ("/static/app.js", "/static/styles.css"):
                got = await client.get(asset)
                print(f"{asset}: {got.status_code} ({len(got.text)} bytes)")

            print("full run:")
            kinds = await stream_whole_run(client)
            assert kinds[0] == "run_started", kinds
            assert kinds[-1] == "done", kinds
            assert "run_failed" not in kinds, kinds
            # The critic rejected once, so the pipeline must have looped.
            assert kinds.count("node_activated") > len(QUESTIONS) + 3, kinds

            runs = (await client.get("/api/runs")).json()["runs"]
            print(f"archived runs: {[r['run_id'] for r in runs]}")
            assert runs and runs[0]["status"] == "ok", runs

            print("disconnect:")
            await stream_then_disconnect(client)
            after = (await client.get("/api/health")).json()
            assert after["status"] == "ok", after
            print("server still healthy after the disconnect")

        server.should_exit = True
        await task

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
