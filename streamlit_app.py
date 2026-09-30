"""Deep Research Agent as a Streamlit app.

This is a deployment entry point, not a second implementation. It reuses the
same `ResearchRunner`, `Settings`, and `RunStore` as the CLI and FastAPI UI,
then renders the runner's event stream with Streamlit widgets.

Deploy on Streamlit Community Cloud with:

    repository root / streamlit_app.py
    repository root / requirements.txt
    repository root / .streamlit/config.toml

Do not commit API keys. On Community Cloud, add them under the app's Secrets
settings using the same variable names as `.env.example`; locally, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from deep_research.config import Settings
from deep_research.events import (
    DONE,
    EDGE_FLOW,
    LOG,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
)
from deep_research.runner import ResearchRunner
from deep_research.storage import RunStore
from deep_research.streamlit_config import (
    missing_configuration,
    public_configuration,
    settings_from_secrets,
    settings_with_overrides,
)

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_secrets() -> dict[str, Any]:
    """Return Streamlit secrets as plain mappings, without displaying values."""
    try:
        secrets = st.secrets
    except Exception:
        return {}
    if not isinstance(secrets, Mapping):
        return {}
    plain: dict[str, Any] = {}
    for key, value in secrets.items():
        if isinstance(value, Mapping):
            plain[key] = dict(value)
        else:
            plain[key] = value
    return plain


def _summarise_event(kind: str, data: Mapping[str, Any]) -> str:
    """One compact line for the live event log."""
    if kind == NODE_ACTIVATED:
        return f"{data.get('node', 'node')}: {str(data.get('detail', ''))[:120]}"
    if kind == NODE_FINISHED:
        return f"{data.get('node', 'node')} finished: {str(data.get('summary', ''))[:120]}"
    if kind == EDGE_FLOW:
        source = data.get("source", "?")
        target = data.get("target", "?")
        label = data.get("label", "")
        return f"{source} -> {target} ({label})"
    if kind == LOG:
        return str(data.get("msg", ""))[:200]
    if kind == RUN_STARTED:
        return f"Run started: {data.get('run_id', '')}"
    if kind == REPORT_DONE:
        return (
            f"Report finished: verdict={data.get('verdict', '')}, "
            f"review_cycles={data.get('review_cycles', 0)}, "
            f"searches={data.get('search_calls', 0)}"
        )
    if kind == RUN_FAILED:
        return f"Run failed: {str(data.get('error', ''))[:300]}"
    if kind == DONE:
        return "Run stream finished."
    return kind


async def _drive_run(
    runner: ResearchRunner,
    status: Any,
    report_view: Any,
    progress: Any,
) -> Any:
    """Consume one runner stream and update Streamlit placeholders progressively."""
    report_text = ""
    log_lines: list[str] = []
    events_seen = 0
    failed: str | None = None

    async for event in runner.run():
        events_seen += 1
        data = event.data if isinstance(event.data, Mapping) else {}
        if event.kind == REPORT_DELTA:
            report_text += str(data.get("text", ""))
            # Re-render only at readable intervals; token-by-token rerenders turn
            # a long draft into UI churn.
            if events_seen % 8 == 0 or len(report_text) < 2000:
                report_view.markdown(report_text)
        elif event.kind in {NODE_ACTIVATED, NODE_FINISHED, EDGE_FLOW, LOG}:
            line = _summarise_event(event.kind, data)
            if line:
                log_lines.append(line)
                status.update(label=log_lines[-1])
                progress.progress(min(0.99, 0.05 + 0.01 * events_seen))
        elif event.kind == RUN_STARTED:
            status.update(label=_summarise_event(event.kind, data))
        elif event.kind == REPORT_DONE:
            report_text = str(data.get("markdown", report_text))
        elif event.kind == RUN_FAILED:
            failed = str(data.get("error", "The run failed."))
        elif event.kind == DONE:
            break

    report_view.markdown(report_text)
    progress.progress(1.0)
    return {
        "report": report_text,
        "log": log_lines[-250:],
        "events_seen": events_seen,
        "error": failed,
        "result": runner.result,
    }


base_settings = settings_from_secrets(_read_secrets())

st.sidebar.header("Run configuration")
topic = st.sidebar.text_input(
    "Research topic",
    value="How do planetary gear reducers handle shock loading?",
    max_chars=500,
)
search_provider = st.sidebar.selectbox(
    "Search provider",
    options=["tavily", "serper", "none"],
    index=["tavily", "serper", "none"].index(base_settings.search_provider),
    help="Use `none` to answer only from the model's own knowledge.",
)
model = st.sidebar.text_input("Model", value=base_settings.model)
max_questions = st.sidebar.slider("Max questions per round", 1, 4, base_settings.max_questions)
max_review_cycles = st.sidebar.slider("Max review cycles", 0, 2, base_settings.max_review_cycles)
search_max_results = st.sidebar.slider("Results per search", 1, 8, base_settings.search_max_results)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, float(base_settings.temperature), 0.05)

settings: Settings = settings_with_overrides(
    base_settings,
    model=model.strip() or base_settings.model,
    temperature=temperature,
    search_provider=search_provider,
    search_max_results=search_max_results,
    max_questions=max_questions,
    max_review_cycles=max_review_cycles,
    runs_dir=Path("runs"),
    cache_file=Path(".cache/search.json"),
)
missing = missing_configuration(settings)
store = RunStore(settings.runs_dir)

st.title("Deep Research Agent")
st.caption(
    "Planner, parallel researchers, streaming writer, and critic. "
    "Reports on Streamlit Community Cloud are kept in the app's ephemeral "
    "filesystem; download anything worth keeping."
)

with st.expander("Secrets and configuration", expanded=bool(missing)):
    st.write(
        "Set `GROQ_API_KEY` and the key for the selected search provider. "
        "Values are never displayed by this app."
    )
    st.code(
        'GROQ_API_KEY = "gsk_..."\nTAVILY_API_KEY = "tvly_..."\n# SERPER_API_KEY = "..."',
        language="toml",
    )
    st.json(
        {
            "missing": missing,
            "configuration": public_configuration(settings),
        }
    )

run_tab, history_tab = st.tabs(["Run research", "Archived runs"])

with run_tab:
    can_run = len(topic.strip()) >= 3 and not missing
    if missing:
        st.error(f"Missing configuration: {', '.join(missing)}")
    if settings.search_provider == "none":
        st.warning("Web search is disabled; researchers will use model knowledge only.")

    if st.button("Run research", type="primary", disabled=not can_run):
        status = st.status("Starting research…", expanded=True)
        progress = st.progress(0.0)
        report_view = st.empty()
        try:
            outcome = asyncio.run(
                _drive_run(ResearchRunner(topic, settings=settings), status, report_view, progress)
            )
        except Exception as exc:
            status.update(label="Run failed before completion.", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            status.update(label="Run finished.", state="complete")
            st.session_state["streamlit_outcome"] = outcome

    outcome = st.session_state.get("streamlit_outcome")
    if outcome:
        if outcome.get("error"):
            st.error(outcome["error"])
        result = outcome.get("result")
        if result is not None:
            meta = result.meta
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Report")
                st.markdown(outcome.get("report", ""))
                st.download_button(
                    "Download report",
                    data=outcome.get("report", ""),
                    file_name=f"{meta.run_id}.md",
                    mime="text/markdown",
                )
            with right:
                st.subheader("Run summary")
                st.table(
                    {
                        "run": meta.run_id,
                        "model": meta.model,
                        "questions": len(meta.questions),
                        "review cycles": meta.review_cycles,
                        "verdict": meta.verdict,
                        "searches billed": meta.search_calls,
                        "duration": f"{meta.duration_seconds}s",
                    }
                )
                if meta.reviewer_feedback:
                    st.info(meta.reviewer_feedback)
        with st.expander(f"Event log ({outcome.get('events_seen', 0)} events)", expanded=False):
            st.code("\n".join(outcome.get("log", [])) or "No log lines captured.")

with history_tab:
    rows = store.list_runs(limit=25)
    if not rows:
        st.info("No archived runs are available in this Streamlit session yet.")
    else:
        labels = {
            row["run_id"]: f"{row['run_id']} — {row.get('topic', '')} "
            f"({row.get('status', '')}, {row.get('verdict', '')})"
            for row in rows
            if "run_id" in row
        }
        selected = st.selectbox("Archived run", options=list(labels), format_func=labels.get)
        if selected:
            try:
                meta = store.load_meta(selected)
                report = store.load_report(selected)
            except FileNotFoundError as exc:
                st.error(str(exc))
            else:
                st.markdown(report)
                st.download_button(
                    "Download archived report",
                    data=report,
                    file_name=f"{meta.run_id}.md",
                    mime="text/markdown",
                )
                with st.expander("Archived metadata", expanded=False):
                    st.json(meta.to_dict())
