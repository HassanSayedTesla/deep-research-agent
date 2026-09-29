"""HTTP API: streams a research run to the browser as server-sent events.

The endpoint is deliberately one round trip. `POST /api/research` responds with
`text/event-stream` and every event of the run, so the page gets the graph
topology, each node activation, each hand-off and every report token over a
single connection. There is no polling, no run registry and no session state to
keep in sync: closing the browser tab cancels the run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .graph import graph_payload
from .runner import ResearchRunner
from .storage import RunStore
from .workflow import RESEARCH_WORKERS

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Proxies and browsers drop idle connections; a research run is bursty, and a
# gap between the last finding and the first token can exceed a minute.
SSE_KEEPALIVE_SECONDS = 15


class ResearchRequest(BaseModel):
    """Body of `POST /api/research`."""

    topic: str = Field(
        min_length=3,
        max_length=500,
        description="What to research.",
        examples=["How to choose a gearbox for a mobile crane"],
    )


def create_app(
    settings: Settings | None = None,
    runner_factory: Callable[[str, Settings, RunStore], ResearchRunner] | None = None,
) -> FastAPI:
    """Build the ASGI app.

    `settings` is resolved once, at startup. `runner_factory` exists so the
    tests can drive the HTTP surface with fake agents and a fake LLM.
    """
    app_settings = settings or Settings()
    store = RunStore(app_settings.runs_dir)
    factory = runner_factory or (
        lambda topic, config, run_store: ResearchRunner(
            topic=topic, settings=config, store=run_store
        )
    )

    app = FastAPI(
        title="Deep Research Agent",
        version="0.1.0",
        description="A multi-agent research pipeline with a live flow graph.",
    )
    app.state.settings = app_settings
    app.state.store = store

    @app.get("/api/health")
    async def health() -> JSONResponse:
        """Liveness plus whether the app is configured enough to run."""
        ready = bool(app_settings.groq_api_key) and (
            app_settings.search_provider == "none" or bool(app_settings.tavily_api_key)
        )
        return JSONResponse(
            {
                "status": "ok" if ready else "needs_config",
                "model": app_settings.model,
                "search_provider": app_settings.search_provider,
                "max_questions": app_settings.max_questions,
                "max_review_cycles": app_settings.max_review_cycles,
                "concurrency": RESEARCH_WORKERS,
            }
        )

    @app.get("/api/graph")
    async def graph() -> dict[str, Any]:
        """The agent graph on its own, for rendering the idle diagram."""
        return graph_payload()

    @app.get("/api/runs")
    async def list_runs(limit: int = 25) -> dict[str, Any]:
        return {"runs": store.list_runs(limit=limit)}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        try:
            meta = store.load_meta(run_id)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        try:
            report = store.load_report(run_id)
        except FileNotFoundError:
            report = ""
        return JSONResponse({"meta": meta.to_dict(), "report": report})

    @app.post("/api/research")
    async def research(request: ResearchRequest) -> StreamingResponse:
        """Run a research job, streaming every event as server-sent events."""

        async def publish() -> AsyncIterator[str]:
            runner = factory(request.topic, app_settings, store)
            try:
                async for event in runner.run():
                    yield _sse(event.kind, event.data)
            except Exception:
                logger.exception("Streaming failed for %r", request.topic)
                yield _sse("run_failed", {"error": "The server lost the run."})
                yield _sse("done", {})

        # When the client disconnects, Starlette closes this generator, which
        # exits the `async for` above and cancels the in-flight run.
        return StreamingResponse(
            publish(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable nginx response buffering
            },
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _sse(kind: str, data: dict[str, Any]) -> str:
    """Frame one event in the SSE wire format."""
    body = json.dumps({"kind": kind, **data}, ensure_ascii=False)
    # A single line keeps the frame trivial to parse on the client.
    return f"data: {body}\n\n"


app = create_app()
