"""The event bridge between the workflow and whoever is watching it.

The workflow is written once and consumed from three places: the web UI (over
server-sent events), the CLI (as rich console output), and the test suite. All
three subscribe to the same stream, so the workflow never needs to know how it
is being observed.

`EventBridge` is that stream: an async-iterator fan-out with a replayable
history, small enough that a lock-free single-consumer design is fine.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

# Event kinds, mirrored by the frontend's SSE handlers.
RUN_STARTED = "run_started"
NODE_ACTIVATED = "node_activated"
NODE_FINISHED = "node_finished"
EDGE_FLOW = "edge_flow"
REPORT_DELTA = "report_delta"
REPORT_DONE = "report_done"
LOG = "log"
RUN_FAILED = "run_failed"
DONE = "done"

TERMINAL_KINDS = frozenset({REPORT_DONE, RUN_FAILED, DONE})

# A counter, not a clock. `time.monotonic()` has ~15ms granularity on Windows,
# so a fast run would give every event the same timestamp and `seq` would be
# useless for ordering a replayed log.
_SEQUENCE = itertools.count(1)


@dataclass(slots=True)
class RunEvent:
    """One observation from a running pipeline."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.monotonic)
    seq: int = field(default_factory=lambda: next(_SEQUENCE))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data, "seq": self.seq}

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}: {self.data}"


class EventBridge:
    """Collects run events and fans them out to a live consumer."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self.history: list[RunEvent] = []
        self.started_at = time.time()
        self.finished_at: float | None = None

    async def emit(self, kind: str, **data: Any) -> RunEvent:
        event = RunEvent(kind=kind, data=data)
        self.history.append(event)
        self._queue.put_nowait(event)
        return event

    async def close(self) -> None:
        """Signal end-of-stream. Safe to call more than once."""
        if self._queue.empty():
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[RunEvent]:
        """Yield events as they are emitted, replaying nothing.

        Call before starting the workflow so no early event is missed.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    def replay(self) -> list[RunEvent]:
        """Everything emitted so far, for persisting or post-mortem inspection."""
        return list(self.history)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 2)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for event in self.history:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return {"run_id": self.run_id, "duration_seconds": self.duration_seconds, "events": counts}
