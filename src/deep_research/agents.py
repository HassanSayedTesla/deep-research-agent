"""The agents.

Only one of the four pipeline stages is an `Agent` in LlamaIndex's sense. The
researcher needs tool calling, so it is a `FunctionAgent` with the web search
tool attached — exactly the pattern the course teaches. The other three stages
are single-turn LLM calls: giving them an agent wrapper would add a tool-calling
loop they never use, and the writer specifically needs a raw token stream to
push to the browser, which an agent hides.

    Planner   structured LLM call   -> list of questions
    Researcher FunctionAgent        -> one answer per question, web search tool
    Writer    streaming LLM call    -> markdown draft, token by token
    Critic    structured LLM call   -> accept, or specific feedback
"""

from __future__ import annotations

from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from .prompts import RESEARCHER_PROMPT
from .tools.web_search import WebSearcher, build_search_tools

RESEARCHER_NAME = "AnswerAgent"
RESEARCHER_DESCRIPTION = (
    "Answers one specific research question, using web search when the answer "
    "needs facts outside its own knowledge."
)


def build_research_agent(llm: LLM, searcher: WebSearcher) -> FunctionAgent:
    """A tool-using agent that answers exactly one question per run."""
    return FunctionAgent(
        name=RESEARCHER_NAME,
        description=RESEARCHER_DESCRIPTION,
        llm=llm,
        tools=build_search_tools(searcher),
        system_prompt=RESEARCHER_PROMPT,
        can_handoff_to=[],
        verbose=False,
    )


def agent_tool_names(agent: Any) -> list[str]:
    """Names of the tools an agent can call, for run metadata."""
    tools: Any = getattr(agent, "tools", None) or {}
    if isinstance(tools, dict):
        return sorted(tools)
    return sorted(str(getattr(tool, "__name__", tool)) for tool in tools)
