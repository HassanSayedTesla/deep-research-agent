"""Declarative description of the agent graph.

This is the single source of truth for the pipeline topology. The API serves it
to the browser, which renders it as an SVG flow chart, and the CLI uses it to
explain a run. Keeping it as data (rather than hard-coding it in JavaScript)
means the UI can never drift out of sync with the workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LANE_PLANNER = 0
LANE_RESEARCH = 1
LANE_WRITER = 2
LANE_CRITIC = 3

LANE_LABELS = {
    LANE_PLANNER: "Plan",
    LANE_RESEARCH: "Research",
    LANE_WRITER: "Write",
    LANE_CRITIC: "Critique",
}


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """A single agent (or a lane of interchangeable agents) in the graph.

    `spawns=True` means the frontend renders one node per runtime instance,
    e.g. `research#0`, `research#1`, for a fan-out step.
    """

    id: str
    label: str
    agent: str
    lane: int
    kind: str = "agent"
    description: str = ""
    spawns: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """A directed hand-off between two nodes."""

    source: str
    target: str
    label: str
    kind: str = "flow"  # flow | loop | accept

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        id="planner",
        label="Planner",
        agent="QuestionAgent",
        lane=LANE_PLANNER,
        description="Turns the topic into a short list of research questions.",
    ),
    NodeSpec(
        id="research",
        label="Researcher",
        agent="AnswerAgent",
        lane=LANE_RESEARCH,
        kind="worker",
        spawns=True,
        description="Answers one question, using the web search tool. One per question.",
    ),
    NodeSpec(
        id="writer",
        label="Writer",
        agent="ReportAgent",
        lane=LANE_WRITER,
        description="Fuses every finding into a markdown report, streamed token by token.",
    ),
    NodeSpec(
        id="critic",
        label="Critic",
        agent="ReviewAgent",
        lane=LANE_CRITIC,
        description="Judges the draft: accept it, or send feedback back to the planner.",
    ),
)

EDGES: tuple[EdgeSpec, ...] = (
    EdgeSpec("planner", "research", "questions"),
    EdgeSpec("research", "writer", "findings"),
    EdgeSpec("writer", "critic", "draft"),
    EdgeSpec("critic", "planner", "feedback", kind="loop"),
    EdgeSpec("critic", "writer", "accepted", kind="accept"),
)

NODE_BY_ID = {node.id: node for node in NODES}


def graph_payload() -> dict[str, object]:
    """JSON-serialisable graph description handed to the frontend."""
    return {
        "nodes": [node.to_dict() for node in NODES],
        "edges": [edge.to_dict() for edge in EDGES],
        "lanes": [{"index": index, "label": label} for index, label in sorted(LANE_LABELS.items())],
    }


def worker_node_id(index: int) -> str:
    """Id of the nth runtime researcher instance (`research` spawns per question)."""
    return f"{NODE_BY_ID['research'].id}#{index}"


def parse_node_id(node_id: str) -> tuple[str, int | None]:
    """Split a node id such as `research#2` into its base id and instance index."""
    if "#" in node_id:
        base, _, raw_index = node_id.partition("#")
        return base, int(raw_index)
    return node_id, None
