"""Where finished runs live on disk.

Each run gets its own directory:

    runs/20260929-141233-gearbox-types/
        report.md      the human-facing result
        meta.json      topic, model, timings, counts, cache stats
        events.jsonl   the full event stream, for replay or debugging

`events.jsonl` is what makes the UI's history possible later: the run is
reproducible from its event log without re-spending tokens.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .events import RunEvent

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 40) -> str:
    """Lowercase ASCII slug, safe for use in a directory name."""
    normalised = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = _SLUG_RE.sub("-", normalised.lower()).strip("-")
    return slug[:max_length].strip("-") or "run"


def new_run_id(topic: str, now: datetime | None = None) -> str:
    """Timestamp-prefixed, topic-derived id: `20260929-141233-gearbox-types`.

    The microseconds are not decoration. The id is to the second, so two runs
    started in the same second - two browser tabs, or a double-click on
    "research" - resolved to the same id and wrote over each other's
    `report.md`, `meta.json` and `events.jsonl` in one directory. Only one of
    the two then appeared in the run list.
    """
    moment = (now or datetime.now(UTC)).astimezone()
    stamp = moment.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{moment.microsecond:06d}-{slugify(topic)}"


@dataclass(slots=True)
class RunMeta:
    """Everything worth knowing about a run after it finishes."""

    run_id: str
    topic: str
    model: str
    started_at: str
    status: str = "ok"
    finished_at: str = ""
    duration_seconds: float = 0.0
    review_cycles: int = 0
    questions: list[str] = field(default_factory=list)
    findings: list[dict[str, str]] = field(default_factory=list)
    search_calls: int = 0
    cache: dict[str, Any] = field(default_factory=dict)
    verdict: str = "acceptable"
    reviewer_feedback: str = ""
    error: str = ""
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The metadata schema as data, so a file written by another version can still be
# read back instead of hard-failing on a field this version does not know.
_META_FIELDS = frozenset(f.name for f in fields(RunMeta))


class RunStore:
    """Filesystem-backed archive of research runs."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / "meta.json").exists()

    def create(self, run_id: str) -> Path:
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_report(self, run_id: str, markdown: str) -> Path:
        directory = self.create(run_id)
        path = directory / "report.md"
        path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        return path

    def save_meta(self, meta: RunMeta) -> Path:
        directory = self.create(meta.run_id)
        path = directory / "meta.json"
        path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return path

    def save_events(self, run_id: str, events: list[RunEvent]) -> Path:
        directory = self.create(run_id)
        path = directory / "events.jsonl"
        lines = [json.dumps(event.to_dict(), ensure_ascii=False) for event in events]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def load_report(self, run_id: str) -> str:
        path = self.run_dir(run_id) / "report.md"
        if not path.exists():
            raise FileNotFoundError(f"No report for run {run_id!r}")
        return path.read_text(encoding="utf-8")

    def load_meta(self, run_id: str) -> RunMeta:
        path = self.run_dir(run_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"No metadata for run {run_id!r}")
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return RunMeta(**{k: v for k, v in stored.items() if k in _META_FIELDS})
        except (json.JSONDecodeError, TypeError) as exc:
            # `list_runs` reads the raw dict, so a half-written or
            # schema-mismatched `meta.json` still gets listed - and then the
            # follow-up detail request raised an uncaught `TypeError` and
            # returned a 500 for a run the list had just advertised. Raising the
            # one error callers already handle keeps that a clean 404.
            raise FileNotFoundError(f"Metadata for run {run_id!r} is unreadable: {exc}") from exc

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.run_dir(run_id) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent first. Unreadable directories are skipped, not fatal."""
        if not self.root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            meta_path = directory / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            meta["has_report"] = (directory / "report.md").is_file()
            rows.append(meta)
            if len(rows) >= limit:
                break
        return rows
