# Deep Research Agent

A multi-agent research pipeline on [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/understanding/agent/), with a live flow-graph UI that shows which agent is working and what it just passed where.

Give it a topic. A planner breaks it into questions, several researchers answer them in parallel, a writer streams a report, and a critic either approves it or sends it back for another round. You watch the whole thing happen in the browser.

- **Provider-agnostic LLM.** Runs on [Groq](https://console.groq.com/) out of the box; the course notebooks used OpenAI.
- **Live visualisation.** Every node activation, every edge, and every draft token is streamed over SSE and animated in SVG. No framework, no build step.
- **Actually testable.** The entire suite runs offline: the LLM and the search backend are faked, so CI needs no keys and cannot be rate-limited.
- **Persisted runs.** Every run is archived with its report, metadata and complete event log.

> Want the LlamaIndex-rendered version of the graph, with every path through the
> workflow drawn out? Run `deep-research diagram --open`.

---

## Contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Course mapping](#course-mapping)
- [The web UI](#the-web-ui)
- [CLI](#cli)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Docker](#docker)
- [Extending it](#extending-it)

---

## Quickstart

Requires Python 3.11+ (3.12 recommended).

```bash
git clone https://github.com/HassanSayedTesla/deep-research-agent.git
cd deep-research-agent

python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env      # then put your Groq key in it
```

Get a key at [console.groq.com](https://console.groq.com/keys), add it to `.env`:

```dotenv
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...   # optional: search falls back to local knowledge
```

Then either start the web UI:

```bash
deep-research serve          # http://127.0.0.1:8000
```

or run it in the terminal:

```bash
deep-research research "How do planetary gear reducers handle shock loading?"
```

A run with no search key still works end to end — the researchers fall back to
what the model already knows, and the UI tells you the provider it used.

---

## How it works

```
              ┌──────────┐
   topic ────▶│ planner  │  breaks the topic into N questions
              └────┬─────┘
        questions  │  (one per line, broadcast)
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐      each researcher searches,
│research0│   │research1│   │research2│      reads, and writes findings
└────┬────┘   └────┬────┘   └────┬────┘      independently, in parallel
     └─────────────┼─────────────┘
             findings  │        collect_events is the barrier:
        ┌─────────────┴─────────────┐   nothing downstream starts until
        ▼                          ▼   every researcher has reported
   ┌─────────┐   draft    ┌──────────┐
   │ writer  │───────────▶│  critic  │
   └─────────┘            └────┬─────┘
        ▲                     │
        └──── acceptable ─────┤
                              │ feedback
                              ▼
                         back to planner
```

**The three things that make it work.**

1. **Dynamic fan-out.** The planner returns a list, and the workflow spawns one
   researcher per question inside a single `@step(num_workers=...)`. Adding a
   fourth question needs no code change.
2. **A barrier, not a sleep.** `collect_events` waits for exactly as many
   findings as there are questions. A workflow that just awaited each step in a
   loop would serialise the research and gain nothing from the concurrency.
3. **Reflection is a loop, not an error.** The critic returns a structured
   verdict. If it is not acceptable, the workflow routes back to the planner with
   the feedback and starts a new round, bounded by `MAX_REVIEW_CYCLES`.

Every observation is published to an `EventBridge`, which is a single async
stream with a replayable history. The web UI, the CLI and the tests all consume
that one stream, so the workflow never knows how it is being watched.

---

## Course mapping

This project is the course *"Building Agentic Workflows with LlamaIndex"*, consolidated into something you would actually ship. Each exercise maps to real code:

| Exercise | Topic | Where it lives |
| --- | --- | --- |
| 01 | Getting started with LlamaIndex | `llm.py` (provider setup), `config.py` |
| 02 | Creating an agent | `agents.py` → `build_research_agent()` |
| 03 | Maintaining state across runs | `tools/web_search.py` → `SearchCache`, backed by `cache.py` |
| 04 | Streaming output and events | `events.py` → `EventBridge`; `workflow.py` → the writer's `astream_complete` loop |
| 06 | Multi-agent system with `AgentWorkflow` | `agents.py` — the same role split, but hand-rolled for visibility |
| 07 | Building agentic workflows from scratch | `workflow.py` → `build_workflow()`, replacing the black-box orchestrator with explicit steps |
| 08 | Custom events for multi-step workflows | `workflow.py` → custom event types; `graph.py` → the node/edge contract |
| 09 | Concurrency and event collection | `workflow.py` → `collect_events` barrier and `RESEARCH_WORKERS` fan-out |
| 10 | Putting together a multi-agent system | `runner.py` → `ResearchRunner`, the single entry point the CLI and API share |
| 11 | Adding self-reflection | `workflow.py` → the critic/planner revision loop, bounded by `MAX_REVIEW_CYCLES` |

### What changed from the notebooks, and why

The notebooks target an earlier LlamaIndex API. Three things had to move:

| Notebook | This project | Reason |
| --- | --- | --- |
| `ctx.get("key")` / `ctx.set("key", v)` | `await ctx.store.get("key", default)` / `await ctx.store.set("key", v)` | `Context.get/set` were removed; state now lives in the typed store. |
| `Workflow(num_workers=4)` | `@step(num_workers=RESEARCH_WORKERS)` | Concurrency is declared per step, not per workflow. |
| a step that only calls `send_event` | the step declares the event it produces in its return annotation | The workflow validator now infers step output types from annotations. |
| OpenAI `gpt-4o` | Groq `llama-3.3-70b-versatile` | Faster, and free-tier friendly. The LLM is created in one place (`llm.py`), so switching back is a one-line change. |

---

## The web UI

`deep-research serve` serves a single page with no build step and no npm.

- **The graph** is hand-rendered SVG. Lanes for planner, research, writer and
  critic; a card per agent; curved edges that animate a packet along the path
  when content moves between agents.
- **Live state** — the active card glows, a worker badge shows the per-agent
  token count, and searching agents report their query and result count.
- **The report** streams token by token in the panel on the right, and the event
  log below it records every event as it arrives.
- **Honest failure** — if the provider errors, the page says so instead of
  hanging on a spinner.

Everything is driven by the same `graph_payload()` the CLI uses, so the diagram
cannot drift out of sync with the code that runs.

---

## CLI

```bash
deep-research research "topic"        # run it, watch it in the terminal
deep-research serve                   # web UI on 127.0.0.1:8000
deep-research serve --port 9000       # or somewhere else
deep-research runs                    # archived runs, newest first
deep-research show <run-id>           # the report
deep-research show <run-id> --meta    # the metadata
deep-research show <run-id> --events  # the full event stream
deep-research graph                   # print the topology
deep-research diagram --open          # render it to HTML and open it
deep-research cache                   # hit rate and entry count
```

```bash
python -m deep_research research "topic"   # equivalent, without the entry point
```

`deep-research diagram` uses LlamaIndex's own static visualiser: it inspects the
`@step` annotations and draws every path through the workflow, including the
reflection loop. If the graph in the notebooks and the graph in your head ever
disagree, this is the tiebreaker.

---

## Configuration

Everything is environment-driven, with sensible defaults. Copy `.env.example` and
edit, or set the variables in your shell.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` | — | **Required.** The app refuses to start a run without it. |
| `TAVILY_API_KEY` | — | Optional. Without it, search degrades to model knowledge. |
| `SEARCH_PROVIDER` | `tavily` | `tavily` or `none`. `none` forces the local-knowledge path. |
| `MODEL` | `llama-3.3-70b-versatile` | Any model Groq serves. |
| `TEMPERATURE` | `0.3` | Low on purpose: this pipeline wants precision, not prose. |
| `MAX_QUESTIONS` | `5` | Upper bound the planner is asked to respect. |
| `MAX_REVIEW_CYCLES` | `2` | How many times the critic can send the work back. |
| `CONCURRENCY` | `4` | Researchers running at once. Raise it for Groq's rate limit. |
| `RUNS_DIR` | `runs` | Where reports are archived. |
| `CACHE_FILE` | `.cache/search.json` | Search cache. Delete it to force a refresh. |

---

## HTTP API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/api/health` | Status, model, provider, whether keys are present. |
| `GET` | `/api/graph` | The node/edge topology on its own. |
| `GET` | `/api/runs?limit=25` | Archived runs, newest first. |
| `GET` | `/api/runs/{run_id}` | Metadata, report and event log. |
| `POST` | `/api/research` | A streamed run. |

`POST /api/research` takes `{"topic": "..."}` and responds with
`text/event-stream`. Each frame is a `RunEvent`:

```bash
curl -N -X POST http://127.0.0.1:8000/api/research \
  -H 'Content-Type: application/json' \
  -d '{"topic":"How do planetary gear reducers handle shock loading?"}'
```

```
data: {"kind":"run_started","data":{"run_id":"20260930-141233-crane-gearbox", ...}}
data: {"kind":"node_activated","data":{"node":"research#0","label":"Researcher", ...}}
data: {"kind":"edge_flow","data":{"source":"planner","target":"research#0", ...}}
data: {"kind":"report_delta","data":{"text":"## Summary\n"}}
data: {"kind":"report_done","data":{"markdown":"...","report_path":"...","review_cycles":1}}
data: {"kind":"done","data":{}}
```

Event kinds: `run_started`, `node_activated`, `node_finished`, `edge_flow`,
`report_delta`, `report_done`, `log`, `run_failed`, `done`. A failure arrives as
a `run_failed` frame followed by `done` — it never turns into a dropped
connection or a 500, and the partial run is still archived so you can read
`events.jsonl` afterwards.

---

## Project layout

```
src/deep_research/
  workflow.py       the pipeline: steps, fan-out, barrier, reflection loop
  runner.py         one run end to end: config, events, persistence, failure
  agents.py         the agent roles and their tools
  llm.py            provider setup and structured-output helpers
  prompts.py        every prompt, in one place
  schemas.py        structured outputs (plan, verdict)
  events.py         the event bridge: one stream, three consumers
  graph.py          the node/edge contract the UI draws
  cache.py          TTL search cache
  storage.py        the run archive
  server.py         FastAPI + SSE
  cli.py            Typer + Rich
  static/           index.html, styles.css, app.js
tests/              offline suite: fakes, conftest, workflow, server, cache, storage
scripts/smoke.py    real uvicorn, real stream, real disconnect
docs/diagrams/      generated by `deep-research diagram`
```

---

## Testing

```bash
pytest                    # the full suite, offline
pytest --cov=deep_research
ruff check . && ruff format --check .
python scripts/smoke.py   # real HTTP, real SSE, real client disconnect
```

`tests/fakes.py` implements the LLM and the search backend, so a full run —
including the reflection loop — completes in about four seconds with no network
and no keys. The same fakes are what make the CI matrix (3.11 / 3.12 / 3.13)
affordable.

The tests that matter are in `tests/test_workflow.py`: they drive the whole
pipeline and assert on the event stream, not on internal return values. If the
fan-out breaks, the reflection loop stops looping, or a provider error escapes
as an exception instead of a `run_failed` event, they fail.

---

## Docker

```bash
cp .env.example .env      # add your keys
docker compose up --build
```

Then open <http://127.0.0.1:8000>. Reports and the search cache live in a named
volume, so they survive a rebuild. The image runs as a non-root user and has a
healthcheck wired to `/api/health`.

---

## Extending it

Some things worth trying, roughly in order of effort:

- **More providers.** `llm.py` is the only file that knows about Groq. Any
  `llama-index` LLM integration works — return it from `build_llm()`.
- **Different roles.** Add a `SourceVerifier` between research and writing that
  checks each finding against its citations.
- **Deeper research.** Give the planner the previous round's findings so round
  two fills gaps instead of re-asking.
- **Cost control.** Record token usage per agent in the run metadata, and cap
  the number of search calls per question.
- **A real UI framework.** The event protocol is the contract; the current client
  is deliberately small. Replay `events.jsonl` to rebuild any run without
  re-spending tokens.

---

## Licence

MIT. See [LICENSE](LICENSE).
