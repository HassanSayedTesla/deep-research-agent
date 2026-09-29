"""Structured outputs.

Handing a model a schema beats asking it for "five questions, one per line" and
praying. The planner and the critic both return these models, which means a
malformed reply fails loudly in one place instead of silently degrading the
report further downstream.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """The planner's output: the questions this round will answer."""

    questions: list[str] = Field(
        default_factory=list,
        description="Independent research questions, each answerable on its own.",
    )


class ReviewVerdict(BaseModel):
    """The critic's output: ship it, or say exactly what is missing."""

    acceptable: bool = Field(description="True when the draft is comprehensive and grounded.")
    feedback: str = Field(
        default="",
        description="Actionable gaps to research next. Empty when acceptable is true.",
    )
