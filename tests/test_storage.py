"""Unit tests for the run archive."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from deep_research.events import RunEvent
from deep_research.storage import RunMeta, RunStore, new_run_id, slugify


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Gearbox Selection", "gearbox-selection"),
        ("  Helical vs Planetary  ", "helical-vs-planetary"),
        ("Café / Crème brûlée", "cafe-creme-brulee"),
        ("What's a 10:1 ratio?!", "what-s-a-10-1-ratio"),
        ("---", "run"),
        ("", "run"),
    ],
)
def test_slugify(raw: str, expected: str):
    assert slugify(raw) == expected


def test_slugify_is_length_bounded():
    assert len(slugify("a" * 200)) == 40


def test_run_id_is_timestamp_prefixed_and_slugged():
    run_id = new_run_id("Gearbox Selection", now=datetime(2026, 9, 30, 14, 12, 33))
    assert run_id == "20260930-141233-000000-gearbox-selection"


def test_two_runs_in_the_same_second_get_different_ids():
    """A double-click, or two browser tabs, must not share a directory.

    The id was to the second, so both runs resolved to the same directory and
    overwrote each other's `report.md`, `meta.json` and `events.jsonl` - with
    only one of the two then appearing in the run list.
    """
    moment = datetime(2026, 9, 30, 14, 12, 33)
    first = new_run_id("Gearboxes", now=moment)
    second = new_run_id("Gearboxes", now=moment.replace(microsecond=1))

    assert first != second
    # And the readable prefix is unchanged.
    assert first.startswith("20260930-141233-")
    assert first.endswith("-gearboxes")


def test_save_and_reload_a_run(tmp_path: Path):
    store = RunStore(tmp_path)
    meta = RunMeta(
        run_id="r1",
        topic="Gearboxes",
        model="qwen/qwen3.8-27b",
        started_at="2026-09-30T00:00:00+00:00",
        questions=["What is a gearbox?"],
    )

    store.save_report("r1", "# Report\n\nbody")
    store.save_meta(meta)
    store.save_events("r1", [RunEvent(kind="log", data={"message": "hi"})])

    assert store.exists("r1")
    assert store.load_report("r1") == "# Report\n\nbody\n"
    assert store.load_meta("r1") == meta
    assert store.load_events("r1")[0]["data"] == {"message": "hi"}


def test_missing_run_raises(tmp_path: Path):
    store = RunStore(tmp_path)

    assert not store.exists("nope")
    with pytest.raises(FileNotFoundError):
        store.load_report("nope")
    with pytest.raises(FileNotFoundError):
        store.load_meta("nope")


def test_load_events_of_a_run_with_none(tmp_path: Path):
    assert RunStore(tmp_path).load_events("never-ran") == []


def test_list_runs_is_newest_first(tmp_path: Path):
    store = RunStore(tmp_path)
    for run_id, topic in [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")]:
        store.save_report(run_id, f"# {topic}")
        store.save_meta(RunMeta(run_id=run_id, topic=topic, model="m", started_at="t"))

    rows = store.list_runs()
    assert [row["run_id"] for row in rows] == ["c", "b", "a"]
    assert all(row["has_report"] for row in rows)


def test_list_runs_respects_the_limit(tmp_path: Path):
    store = RunStore(tmp_path)
    for index in range(5):
        store.save_meta(RunMeta(run_id=f"r{index}", topic="t", model="m", started_at="t"))

    assert len(store.list_runs(limit=2)) == 2


def test_list_runs_on_a_missing_root(tmp_path: Path):
    assert RunStore(tmp_path / "absent").list_runs() == []


def test_list_runs_skips_unreadable_entries(tmp_path: Path):
    store = RunStore(tmp_path)
    store.save_meta(RunMeta(run_id="good", topic="t", model="m", started_at="t"))
    (store.run_dir("broken")).mkdir(parents=True)
    (store.run_dir("broken") / "meta.json").write_text("{oops", encoding="utf-8")
    store.run_dir("empty").mkdir(parents=True)

    assert [row["run_id"] for row in store.list_runs()] == ["good"]


def test_meta_is_written_as_readable_json(tmp_path: Path):
    store = RunStore(tmp_path)
    store.save_meta(RunMeta(run_id="r1", topic="Gearboxes", model="m", started_at="t"))

    raw = (store.run_dir("r1") / "meta.json").read_text(encoding="utf-8")
    assert json.loads(raw)["topic"] == "Gearboxes"
    assert "\n  " in raw, "meta.json should be indented for human reading"


def test_events_are_one_json_object_per_line(tmp_path: Path):
    store = RunStore(tmp_path)
    store.save_events(
        "r1",
        [RunEvent(kind="log", data={"n": 1}), RunEvent(kind="done", data={})],
    )

    lines = (store.run_dir("r1") / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "log"


def test_save_events_with_no_events(tmp_path: Path):
    store = RunStore(tmp_path)
    store.save_events("r1", [])

    assert (store.run_dir("r1") / "events.jsonl").read_text(encoding="utf-8") == ""


def test_a_failed_run_is_archived_with_its_error(tmp_path: Path):
    store = RunStore(tmp_path)
    store.save_meta(
        RunMeta(
            run_id="r1",
            topic="Gearboxes",
            model="m",
            started_at="t",
            status="failed",
            error="RuntimeError: provider is down",
        )
    )

    meta = store.load_meta("r1")
    assert meta.status == "failed"
    assert "provider is down" in meta.error
    assert store.exists("r1")
    # No report was ever written, and the UI has to cope with that.
    assert store.list_runs()[0]["has_report"] is False
    with pytest.raises(FileNotFoundError):
        store.load_report("r1")


def test_meta_from_a_newer_version_still_loads(tmp_path: Path):
    """A metadata file outlives the code that wrote it. Extra keys are ignored."""
    store = RunStore(tmp_path)
    (store.run_dir("r1")).mkdir(parents=True)
    (store.run_dir("r1") / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "topic": "Gearboxes",
                "model": "m",
                "started_at": "t",
                "token_usage": {"total": 4321},  # from a future release
            }
        ),
        encoding="utf-8",
    )

    assert store.load_meta("r1").topic == "Gearboxes"
