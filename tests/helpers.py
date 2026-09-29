"""Test-only helpers."""

from __future__ import annotations

from deep_research.events import RunEvent
from deep_research.runner import ResearchResult, ResearchRunner


async def drain(runner: ResearchRunner) -> tuple[list[RunEvent], ResearchResult | None]:
    """Run a research job to completion. Returns (events, result)."""
    events = [event async for event in runner.run()]
    return events, runner.result


def kinds(events: list[RunEvent]) -> list[str]:
    return [event.kind for event in events]


def events_of(events: list[RunEvent], kind: str) -> list[RunEvent]:
    return [event for event in events if event.kind == kind]
