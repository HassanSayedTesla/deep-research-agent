/**
 * Deep Research Agent — live agent graph.
 *
 * Plain ES modules, no build step. The graph topology arrives from the server
 * (`/api/graph`, and again on every `run_started`), so this file never hardcodes
 * which agents exist. It only knows how to lay out lanes, animate a hand-off,
 * and stream text into a panel.
 *
 * Event protocol (see deep_research/events.py):
 *   run_started  graph, config, run_id      node_activated  node, label, agent, detail
 *   edge_flow    source, target, label      node_finished  node, summary
 *   report_delta text                       report_done     markdown, report_path, ...
 *   log          message                    run_failed / done
 */

const NODE_W = 136;
const NODE_H = 54;
const SPINE_X = 360;
const PER_ROW = 3;
const SVG_NS = "http://www.w3.org/2000/svg";

const el = {
  form: document.getElementById("run-form"),
  topic: document.getElementById("topic"),
  run: document.getElementById("run"),
  stop: document.getElementById("stop"),
  banner: document.getElementById("banner"),
  svg: document.getElementById("graph"),
  lanes: document.getElementById("lanes"),
  edges: document.getElementById("edges"),
  packets: document.getElementById("packets"),
  nodes: document.getElementById("nodes"),
  stats: document.getElementById("stats"),
  report: document.getElementById("report"),
  reportStatus: document.getElementById("report-status"),
  log: document.getElementById("log"),
  clearLog: document.getElementById("clear-log"),
};

const state = {
  spec: null,
  nodes: new Map(), // id -> { id, base, index, status, detail, x, y }
  edges: new Map(), // "src->tgt" -> { key, kind, path, labelEl }
  cycle: 0,
  startedAt: null,
  reportText: "",
  streaming: false,
  renderTimer: null,
  controller: null,
  activeWorkers: 0,
  maxWorkers: 0,
  searches: 0,
};

const supportsOffsetPath =
  typeof CSS !== "undefined" && CSS.supports && CSS.supports("offset-path", 'path("M0 0")');

// ---------------------------------------------------------------- utilities
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const short = (text, max) => {
  const clean = String(text ?? "").replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
};
const svg = (tag, attrs = {}) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
};
const clock = () =>
  new Date().toLocaleTimeString("en-GB", { hour12: false });

// ----------------------------------------------------------------- markdown
/** Escape first, then format. Model output is untrusted text. */
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

/** A deliberately small markdown subset: headings, lists, quotes, code, rules. */
function renderMarkdown(source) {
  const lines = String(source ?? "").split("\n");
  const out = [];
  let list = null;
  let fence = null;

  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const line of lines) {
    const fenceMatch = line.match(/^```(\w*)\s*$/);
    if (fenceMatch) {
      if (fence === null) {
        closeList();
        fence = [];
        out.push(`<pre><code class="lang-${escapeHtml(fenceMatch[1])}">`);
      } else {
        out.push(`${escapeHtml(fence.join("\n"))}</code></pre>`);
        fence = null;
      }
      continue;
    }
    if (fence !== null) {
      fence.push(line);
      continue;
    }

    if (!line.trim()) {
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(6, heading[1].length);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^(\s*[-*_]\s*){3,}$/.test(line)) {
      closeList();
      out.push("<hr>");
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      closeList();
      out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const want = bullet ? "ul" : "ol";
      if (list !== want) {
        closeList();
        out.push(`<${want}>`);
        list = want;
      }
      out.push(`<li>${inline((bullet ?? numbered)[1])}</li>`);
      continue;
    }

    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }

  if (fence !== null) out.push(`${escapeHtml(fence.join("\n"))}</code></pre>`);
  closeList();
  return out.join("");
}

// --------------------------------------------------------------------- log
function logLine(kind, message, level = "") {
  const li = document.createElement("li");
  if (level) li.className = `lvl-${level}`;
  li.innerHTML =
    `<span class="t">${clock()}</span><span class="k">${escapeHtml(kind)}</span>` +
    `<span class="m">${escapeHtml(short(message, 200))}</span>`;
  el.log.append(li);
  while (el.log.children.length > 300) el.log.firstElementChild.remove();
  el.log.scrollTop = el.log.scrollHeight;
}

el.clearLog.addEventListener("click", () => {
  el.log.replaceChildren();
});

// -------------------------------------------------------------------- graph
/** Seed the diagram from the server's graph spec (idle, before any run). */
function loadSpec(spec) {
  state.spec = spec;
  state.nodes.clear();
  state.edges.clear();
  el.nodes.replaceChildren();
  el.edges.replaceChildren();
  el.lanes.replaceChildren();

  for (const node of spec.nodes) {
    // Only the first instance of a spawning lane exists before the run.
    state.nodes.set(node.id, {
      ...node,
      base: node.id,
      index: null,
      status: "idle",
      detail: "",
      x: SPINE_X,
      y: 0,
    });
  }
  layout();
}

function instancesOf(baseId) {
  return [...state.nodes.values()].filter((node) => node.base === baseId);
}

/** Position every node, then redraw the edges to match. */
function layout() {
  if (!state.spec) return;

  // Research workers are spawned one per question and wrap onto extra rows, so
  // the lane's height is not known until the planner has run.
  const workers = instancesOf("research");
  const rows = Math.max(1, Math.ceil(workers.length / PER_ROW));

  const yPlanner = 70;
  const yResearchTop = yPlanner + 108;
  const yWriter = yResearchTop + (rows - 1) * 82 + 106;
  const yCritic = yWriter + 96;
  const height = yCritic + 74;

  el.svg.setAttribute("viewBox", `0 0 720 ${height}`);

  const laneY = { 0: yPlanner, 1: yResearchTop, 2: yWriter, 3: yCritic };

  for (const node of state.nodes.values()) {
    if (node.base === "research" && node.index !== null) {
      const position = workers.findIndex((w) => w.id === node.id);
      const row = Math.floor(position / PER_ROW);
      const column = position % PER_ROW;
      const inRow = Math.min(PER_ROW, workers.length - row * PER_ROW);
      const spacing = Math.min(196, 620 / inRow);
      node.x = SPINE_X + (column - (inRow - 1) / 2) * spacing;
      node.y = yResearchTop + row * 82;
    } else {
      node.y = laneY[node.lane] ?? yPlanner;
    }
    drawNode(node);
  }

  drawLanes(laneY);
  drawEdges();
}

function drawLanes(laneY) {
  el.lanes.replaceChildren();
  for (const lane of state.spec.lanes ?? []) {
    const y = laneY[lane.index];
    if (y === undefined) continue;
    el.lanes.append(svg("text", { x: 14, y: y - 28, class: "lane-label" })).textContent =
      String(lane.label).toUpperCase();
    el.lanes.append(
      svg("line", { x1: 14, x2: 706, y1: y - 18, y2: y - 18, class: "lane-rule" })
    );
  }
}

function drawNode(node) {
  let group = el.nodes.querySelector(`[data-id="${cssEscape(node.id)}"]`);
  if (!group) {
    group = svg("g", { class: "node", "data-id": node.id });
    group.append(svg("rect", { class: "node-box", width: NODE_W, height: NODE_H, x: -NODE_W / 2, y: -NODE_H / 2 }));
    group.append(svg("rect", { class: "node-pulse", width: NODE_W + 16, height: NODE_H + 16, x: -NODE_W / 2 - 8, y: -NODE_H / 2 - 8 }));
    group.append(svg("text", { class: "node-title", y: -4 }));
    group.append(svg("text", { class: "node-agent", y: 11 }));
    group.append(svg("text", { class: "node-detail", y: 25 }));
    el.nodes.append(group);
  }
  group.setAttribute("class", `node is-${node.base} is-${node.status}`);
  group.setAttribute("transform", `translate(${node.x} ${node.y})`);

  const [title, agent, detail] = group.querySelectorAll("text");
  title.textContent = node.index === null ? node.label : `${node.label} ${node.index + 1}`;
  agent.textContent = node.agent;
  detail.textContent = short(node.detail, 22);
}

const cssEscape = (value) => (window.CSS?.escape ? CSS.escape(value) : String(value).replace(/"/g, '\\"'));

/** Path from a node's edge to another node's edge, chosen by relative position. */
function edgePath(source, target) {
  const key = `${source.base}->${target.base}`;
  const sameLane = Math.abs(source.x - target.x) < 1;

  if (key === "critic->writer" || key === "critic->planner") {
    // Leave from the left of the critic, arc up the margin, enter from the left.
    const x0 = source.x - NODE_W / 2;
    const x1 = target.x - NODE_W / 2 - 26;
    const midY = (source.y + target.y) / 2;
    return `M ${x0} ${source.y} C ${x0 - 70} ${midY}, ${x1 - 70} ${midY}, ${x1} ${target.y}`;
  }
  if (sameLane) {
    return `M ${source.x} ${source.y} L ${target.x} ${target.y}`;
  }
  if (source.base === "research" || target.base === "research") {
    // Fan-out / convergence: leave the top, arrive at the top.
    return `M ${source.x} ${source.y - NODE_H / 2} C ${source.x} ${source.y - 56}, ${target.x} ${target.y - 56}, ${target.x} ${target.y - NODE_H / 2}`;
  }
  return `M ${source.x} ${source.y + NODE_H / 2} C ${source.x} ${source.y + 52}, ${target.x} ${target.y - 52}, ${target.x} ${target.y - NODE_H / 2}`;
}

function drawEdges() {
  el.edges.replaceChildren();
  state.edges.clear();
  for (const spec of state.spec.edges) {
    const source = [...state.nodes.values()].find((n) => n.base === spec.source);
    const target = [...state.nodes.values()].find((n) => n.base === spec.target);
    if (!source || !target) continue;

    const d = edgePath(source, target);
    const cls = `edge edge-${spec.kind}`;
    const line = svg("path", { d, class: cls, "marker-end": "url(#arrow)" });
    el.edges.append(line);

    const label = svg("text", { class: "edge-label", x: (source.x + target.x) / 2, y: (source.y + target.y) / 2 });
    label.textContent = spec.label;
    el.edges.append(label);

    state.edges.set(spec.source + "->" + spec.target, { d, line, label, kind: spec.kind });
  }
}

function setNodeState(id, status, detail) {
  let node = state.nodes.get(id);
  if (!node) {
    // A spawned worker: research -> research#2
    const base = id.split("#")[0];
    const spec = state.spec.nodes.find((n) => n.id === base);
    if (!spec) return;
    const index = Number(id.split("#")[1]);
    node = { ...spec, base, index, status: "idle", detail: "", x: SPINE_X, y: 0 };
    state.nodes.set(id, node);
    layout();
  }
  node.status = status;
  if (detail !== undefined) node.detail = detail;
  const group = el.nodes.querySelector(`[data-id="${cssEscape(id)}"]`);
  if (group) {
    group.setAttribute("class", `node is-${node.base} is-${node.status}`);
    const texts = group.querySelectorAll("text");
    texts[0].textContent = node.index === null ? node.label : `${node.label} ${node.index + 1}`;
    texts[2].textContent = short(node.detail, 22);
  }
  if (node.base === "research") {
    state.activeWorkers = status === "running" ? state.activeWorkers + 1 : Math.max(0, state.activeWorkers - 1);
    state.maxWorkers = Math.max(state.maxWorkers, state.activeWorkers);
  }
  updateStats();
}

/** Send a glowing packet along one edge, and flash its label. */
function flowAlong(sourceId, targetId, preview) {
  const key = `${sourceId}->${targetId}`;
  const edge =
    state.edges.get(key) ??
    state.edges.get(`${sourceId.split("#")[0]}->${targetId.split("#")[0]}`);
  if (!edge) return;

  edge.line.classList.add("is-active");
  edge.label.classList.add("is-visible");
  if (edge.kind === "loop") edge.line.classList.add("is-loop");

  if (supportsOffsetPath) {
    const group = svg("g", { class: "packet" });
    const dot = svg("circle", { cx: 0, cy: 0, r: 4 });
    group.append(dot);
    if (preview) {
      const text = svg("text", { x: 8, y: -8 });
      text.textContent = short(preview, 30);
      group.append(text);
    }
    group.style.offsetPath = `path("${edge.d}")`;
    group.style.offsetRotate = "0deg";
    el.packets.append(group);
    setTimeout(() => group.remove(), 1600);
  }

  clearTimeout(edge.timer);
  edge.timer = setTimeout(() => {
    edge.line.classList.remove("is-active");
    if (edge.kind === "flow") edge.line.classList.add("is-done");
    edge.label.classList.remove("is-visible");
  }, 1400);
}

// -------------------------------------------------------------------- stats
function updateStats() {
  const elapsed = state.startedAt ? (Date.now() - state.startedAt) / 1000 : 0;
  const stats = [
    ["elapsed", `${elapsed.toFixed(1)}s`],
    ["cycle", String(state.cycle + 1)],
    ["workers", `${state.activeWorkers} live / ${state.maxWorkers} peak`],
    ["searches", String(state.searches)],
    ["chars", String(state.reportText.length)],
  ];
  el.stats.innerHTML = stats
    .map(([key, value]) => `<span class="stat">${key} <b>${escapeHtml(value)}</b></span>`)
    .join("");
}

// ------------------------------------------------------------------- report
function paintReport() {
  const html = renderMarkdown(state.reportText);
  el.report.innerHTML = state.streaming ? `${html}<span class="caret"></span>` : html;
  el.report.scrollTop = el.report.scrollHeight;
}

function appendDelta(text) {
  state.reportText += text;
  if (state.renderTimer) return; // coalesce; a fast stream would repaint per token
  state.renderTimer = setTimeout(() => {
    state.renderTimer = null;
    paintReport();
    updateStats();
  }, 120);
}

function setReportStatus(text, cls) {
  el.reportStatus.textContent = text;
  el.reportStatus.className = `status-pill ${cls}`;
}

// --------------------------------------------------------------------- run
function reset() {
  state.cycle = 0;
  state.reportText = "";
  state.activeWorkers = 0;
  state.maxWorkers = 0;
  state.searches = 0;
  state.startedAt = Date.now();
  state.streaming = true;

  for (const node of state.nodes.values()) {
    node.status = "idle";
    node.detail = "";
  }
  for (const edge of state.edges.values()) {
    edge.line.classList.remove("is-active", "is-done", "is-loop");
    edge.label.classList.remove("is-visible");
  }
  el.packets.replaceChildren();
  el.report.innerHTML = "";
  setReportStatus("writing", "is-running");
  hideBanner();
  updateStats();
}

function showBanner(message) {
  el.banner.textContent = message;
  el.banner.hidden = false;
}
const hideBanner = () => {
  el.banner.hidden = true;
};

function handle(event) {
  switch (event.kind) {
    case "run_started": {
      if (event.graph) loadSpec(event.graph);
      reset();
      logLine("run_started", `topic: ${event.topic}`);
      break;
    }
    case "node_activated":
      setNodeState(event.node, "running", event.detail || "");
      logLine("node", `${event.label ?? event.node} — ${event.detail || ""}`);
      break;
    case "node_finished":
      setNodeState(event.node, "done", event.summary || "");
      break;
    case "edge_flow":
      flowAlong(event.source, event.target, event.preview);
      logLine("edge", `${event.source} → ${event.target} (${event.label})`);
      break;
    case "report_delta":
      appendDelta(event.text);
      break;
    case "report_done": {
      state.streaming = false;
      state.reportText = event.markdown || state.reportText;
      paintReport();
      state.cycle = event.review_cycles ?? state.cycle;
      state.searches = event.search_calls ?? state.searches;
      setReportStatus("done", "is-done");
      logLine("report_done", `saved to ${event.report_path}`, "ok");
      updateStats();
      break;
    }
    case "log":
      logLine("log", event.message);
      break;
    case "run_failed":
      state.streaming = false;
      setReportStatus("failed", "is-failed");
      showBanner(event.error || "The run failed.");
      logLine("run_failed", event.error, "error");
      break;
    case "done":
      if (state.streaming) {
        state.streaming = false;
        setReportStatus("done", "is-done");
        paintReport();
      }
      break;
    default:
      break;
  }
}

async function startRun(topic) {
  reset();
  el.run.disabled = true;
  el.stop.hidden = false;
  el.topic.disabled = true;
  state.controller = new AbortController();

  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
      signal: state.controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`Server responded ${response.status}.`);
    }

    // SSE over POST: parse the framed stream off the fetch body.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          try {
            handle(JSON.parse(line.slice(5).trim()));
          } catch (error) {
            logLine("parse_error", String(error), "warn");
          }
        }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      showBanner(error.message);
      logLine("error", String(error), "error");
    }
  } finally {
    el.run.disabled = false;
    el.stop.hidden = true;
    el.topic.disabled = false;
    state.controller = null;
    if (state.streaming) {
      state.streaming = false;
      paintReport();
    }
  }
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const topic = el.topic.value.trim();
  if (topic.length < 3) return;
  startRun(topic);
});

el.stop.addEventListener("click", () => state.controller?.abort());

// ------------------------------------------------------------------- boot
(async function boot() {
  try {
    const [health, spec] = await Promise.all([
      fetch("/api/health").then((r) => r.json()),
      fetch("/api/graph").then((r) => r.json()),
    ]);
    loadSpec(spec);
    updateStats();
    logLine("ready", `${health.model} · search: ${health.search_provider} · ${health.concurrency} workers`);

    if (health.status !== "ok") {
      showBanner(
        "The server is missing API keys. Copy .env.example to .env, add GROQ_API_KEY " +
          "(and TAVILY_API_KEY for web search), then restart the server."
      );
    }
    el.topic.focus();
  } catch (error) {
    showBanner(`Could not reach the server: ${error.message}`);
  }
})();
