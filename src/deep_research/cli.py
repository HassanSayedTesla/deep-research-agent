"""Command line interface.

    deep-research research "how to choose a gearbox"   run in the terminal
    deep-research serve                                open the flow-graph UI
    deep-research runs                                 list archived runs
    deep-research show <run-id>                        print a saved report
    deep-research diagram                              write the workflow graph

The terminal renderer and the browser read the exact same event stream, so a
run looks the same either way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Settings
from .events import (
    DONE,
    EDGE_FLOW,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
    RunEvent,
)
from .graph import graph_payload
from .runner import ResearchRunner
from .storage import RunStore
from .workflow import RESEARCH_WORKERS, DeepResearchWorkflow

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A multi-agent deep research pipeline built on LlamaIndex Workflows.",
)


def _glyph(char: str, fallback: str) -> str:
    """Use `char` only if the stream being written to can carry it.

    Windows consoles still default to cp1252, where a stray '✓' is a
    UnicodeEncodeError rather than a slightly prettier line. The encoding that
    matters is the console's own, not `sys.stdout`: rich may be writing to a
    different stream, and the two do not always agree.
    """
    encoding = (
        getattr(console.file, "encoding", None) or getattr(sys.stdout, "encoding", None) or "utf-8"
    )
    try:
        char.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return fallback
    return char


console = Console()


# Resolved per call, not once at import: the console can be redirected, piped or
# re-created after this module is loaded, and a glyph chosen too early would be
# written to a stream that cannot encode it.
def ok_mark() -> str:
    return _glyph("✓", "OK")


def fail_mark() -> str:
    return _glyph("✗", "x")


def arrow() -> str:
    return _glyph("↳", "->")


def to() -> str:
    return _glyph("→", "->")


def dot() -> str:
    return _glyph("·", "|")


NODE_ICON = {
    "planner": "[bold cyan]PLAN[/]  ",
    "research": "[bold blue]RESEARCH[/]",
    "writer": "[bold magenta]WRITE[/] ",
    "critic": "[bold yellow]CRITIC[/] ",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


def _node_style(node_id: str) -> str:
    return NODE_ICON.get(node_id.split("#")[0], "       ")


def _render_event(event: RunEvent) -> None:
    """Print one event. The report text itself is printed at the end."""
    data = event.data
    if event.kind == RUN_STARTED:
        config = data["config"]
        console.print(
            f"[dim]run[/] [bold]{data['run_id']}[/]  "
            f"[dim]{config['model']} {dot()} {config['search_provider']} search {dot()} "
            f"{RESEARCH_WORKERS} workers[/]"
        )
    elif event.kind == NODE_ACTIVATED:
        style = _node_style(data["node"])
        detail = data.get("detail", "")
        console.print(f"{style} [bold]{data.get('label', data['node'])}[/] {detail}")
    elif event.kind == NODE_FINISHED:
        if data.get("summary"):
            console.print(f"        [dim]{data['summary']}[/]")
    elif event.kind == EDGE_FLOW:
        console.print(
            f"        [dim]{arrow()} {data['source']} {to()} {data['target']} ({data['label']})[/]"
        )
    elif event.kind == REPORT_DONE:
        console.print(f"\n[green]{ok_mark()} report saved[/] [dim]{data['report_path']}[/]")
    elif event.kind == RUN_FAILED:
        console.print(f"\n[red]{fail_mark()} {data['error']}[/]")


async def _run(topic: str, settings: Settings) -> int:
    console.print(Panel(f"[bold]{topic}[/]", title="deep research", expand=False))
    runner = ResearchRunner(topic=topic, settings=settings)

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("starting", total=None)

        async for event in runner.run():
            if event.kind == REPORT_DELTA:
                # Live feedback without a wall of text; the report prints at the end.
                progress.update(
                    task, description=f"writing ({len(event.data['text'])} chars this run)"
                )
            elif event.kind == DONE:
                progress.update(task, description="finishing")
            else:
                progress.update(task, description=event.kind)
            _render_event(event)

    if runner.result is None:
        return 1

    result = runner.result
    console.print(
        Panel(
            result.report,
            title="[bold]report[/]",
            border_style="green",
        )
    )
    _print_run_summary(result.meta, result.report_path)
    return 0


def _print_run_summary(meta: Any, report_path: str) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("run", meta.run_id)
    table.add_row("model", meta.model)
    table.add_row("questions", str(len(meta.questions)))
    table.add_row("review cycles", str(meta.review_cycles))
    table.add_row("verdict", meta.verdict)
    table.add_row("searches", f"{meta.search_calls} billed")
    table.add_row("duration", f"{meta.duration_seconds}s")
    table.add_row("saved", report_path)
    if meta.reviewer_feedback:
        table.add_row("reviewer asked", meta.reviewer_feedback)
    console.print(Panel(table, title="summary", border_style="dim"))


@app.command()
def research(
    topic: str = typer.Argument(..., help="What to research."),
    questions: int = typer.Option(None, "--questions", "-q", help="Max questions per round."),
    cycles: int = typer.Option(None, "--cycles", "-c", help="Max review cycles."),
    out: Path = typer.Option(None, "--out", "-o", help="Directory for saved runs."),
) -> None:
    """Run a research topic and print the report."""
    overrides: dict[str, Any] = {}
    if questions is not None:
        overrides["max_questions"] = questions
    if cycles is not None:
        overrides["max_review_cycles"] = cycles
    if out is not None:
        overrides["runs_dir"] = out

    try:
        settings = _settings(**overrides)
    except Exception as exc:
        console.print(f"[red]Invalid configuration:[/] {exc}")
        raise typer.Exit(code=2) from exc

    # `typer.Exit` subclasses `RuntimeError`, so raising it inside this `try`
    # lets a *successful* run be caught by its own handler and re-raised as
    # exit code 1 - every successful research reported failure to the shell,
    # breaking any `&&` chain or CI step that checked `$?`. Compute the code
    # inside the `try` and raise outside it.
    try:
        code = asyncio.run(_run(topic, settings))
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=code)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to bind."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Reload on source changes (development)."),
) -> None:
    """Serve the flow-graph UI."""
    import uvicorn

    console.print(
        f"[green]Deep Research Agent[/] on [bold]http://{host}:{port}[/]  [dim](ctrl-c to stop)[/]"
    )
    uvicorn.run("deep_research.server:app", host=host, port=port, reload=reload)


def _summarise(event: dict[str, Any]) -> str:
    """One readable line for a replayed event."""
    data = event.get("data", {})
    for key in ("node", "summary", "label", "message", "error", "text"):
        if data.get(key):
            return str(data[key])[:70]
    if data.get("source"):
        return f"{data['source']} -> {data.get('target', '')} ({data.get('label', '')})"
    return ""


@app.command()
def runs(out: Path = typer.Option(None, "--out", help="Directory to list.")) -> None:
    """List archived runs, newest first."""
    settings = _settings()
    store = RunStore(out or settings.runs_dir)
    rows = store.list_runs(limit=25)
    if not rows:
        console.print(f"[dim]No runs archived under {store.root}[/]")
        return

    # Rich falls back to 80 columns when stdout is not a terminal, and a 43-char
    # run id would then squeeze every other column to a single character. Size
    # the columns against the actual console width instead.
    available = max(60, console.width)
    id_width = min(43, 34)
    topic_width = max(14, available - id_width - 26)

    table = Table(header_style="bold", expand=False, pad_edge=False)
    # "ellipsis" would emit U+2026, which a cp1252 Windows console cannot print.
    # Crop keeps the timestamp, fold keeps the full topic, and both stay ASCII.
    table.add_column("run id", style="cyan", no_wrap=True, width=id_width, overflow="crop")
    table.add_column("topic", width=topic_width, overflow="fold")
    table.add_column("cycles", justify="right", width=6)
    table.add_column("status", no_wrap=True, width=9, overflow="crop")

    for row in rows:
        meta = store.load_meta(row["run_id"])
        status = row.get("verdict", "") if meta.status == "ok" else meta.status
        table.add_row(
            row["run_id"],
            row["topic"],
            str(row.get("review_cycles", 0)),
            status,
        )
    console.print(table)


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run id, e.g. 20260930-141233-gearboxes."),
    out: Path = typer.Option(None, "--out", help="Directory to read from."),
    meta: bool = typer.Option(False, "--meta", help="Also print the run metadata."),
    events: bool = typer.Option(False, "--events", help="Also replay the event stream."),
) -> None:
    """Print a saved report."""
    store = RunStore(out or _settings().runs_dir)
    try:
        console.print(Panel(store.load_report(run_id), title=run_id, border_style="green"))
        if meta:
            console.print_json(json.dumps(store.load_meta(run_id).to_dict()))
        if events:
            for event in store.load_events(run_id):
                head = f"[dim]{event['seq']:>8}[/] [cyan]{event['kind']:<14}[/]"
                console.print(f"{head} {_summarise(event)}")
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


@app.command()
def graph() -> None:
    """Print the agent graph as JSON (same payload the UI draws)."""
    console.print_json(json.dumps(graph_payload(), indent=2))


@app.command()
def diagram(
    output: Path = typer.Option(Path("docs/diagrams/workflow.html"), "--output", "-o"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result."),
) -> None:
    """Render the workflow's own graph to a standalone HTML file.

    This is LlamaIndex's static visualiser: it inspects the `@step` annotations
    and draws every path through the workflow, including the reflection loop.
    """
    try:
        from llama_index.utils.workflow import draw_all_possible_flows
    except ImportError:  # pragma: no cover
        console.print("[red]Missing the visualiser:[/] pip install llama-index-utils-workflow")
        raise typer.Exit(code=1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    draw_all_possible_flows(DeepResearchWorkflow, filename=str(output))
    console.print(f"[green]{ok_mark()} wrote[/] {output}")
    if open_browser:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())


@app.command()
def cache(
    prune: bool = typer.Option(False, "--prune", help="Drop expired entries."),
    clear: bool = typer.Option(False, "--clear", help="Delete the whole cache."),
) -> None:
    """Inspect or clear the search cache."""
    settings = _settings()
    from .cache import SearchCache

    store = SearchCache(settings.cache_file, ttl_seconds=settings.cache_ttl_hours * 3600)
    if clear:
        store.clear()
        console.print(f"[green]cleared[/] {settings.cache_file}")
        return
    if prune:
        removed = store.prune()
        console.print(f"[green]pruned[/] {removed} expired entries")
        return

    stats = store.stats()
    console.print(f"cache: {settings.cache_file}")
    console.print_json(json.dumps(stats))


def main() -> None:  # pragma: no cover - console-script entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
