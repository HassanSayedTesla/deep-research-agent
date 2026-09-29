"""Prompts for each stage of the pipeline.

Kept together, as plain strings, so they are easy to read, diff and tune. Every
prompt is a constant-folded f-string of nothing but the caller's arguments.
"""

from __future__ import annotations

PLANNER_PROMPT = """You are the planning stage of an automated deep-research system.

Given a research topic, produce a short list of questions that, once answered,
would let somebody write a thorough, self-contained briefing on the topic.

Rules:
- Write between {min_questions} and {max_questions} questions.
- One question per line. No numbering, no markdown, no preamble.
- Each question must be answerable on its own by a separate agent that cannot
  see the other questions.
- Cover different angles: definitions and mechanisms, current state, evidence
  and data, trade-offs, and practical implications.
- Do not ask for anything that needs private or personal data.

Topic: <topic>{topic}</topic>"""

PLANNER_REVISION_PROMPT = """You are the planning stage of an automated deep-research
system. You are revising an existing research plan because a reviewer found gaps
in the report it produced.

Original topic: <topic>{topic}</topic>

A previous round of research produced a report, and the reviewer asked for more
grounding. Their feedback:
<feedback>{feedback}</feedback>

Produce a new set of questions between {min_questions} and {max_questions}. The
earlier questions were:
<previous_questions>{previous_questions}</previous_questions>

Rules:
- One question per line. No numbering, no markdown, no preamble.
- Keep the earlier questions that are still worth answering and add new ones
  that close the gaps in the feedback. Do not repeat a question verbatim.
- Each question must be answerable on its own by a separate agent.

New questions:"""

RESEARCHER_PROMPT = """You are a research agent in an automated deep-research system.

You will be given one specific question. Your job is to produce a deep, factual
answer to that question. The answer will be read by other agents and folded into
a report, so it must stand on its own.

Rules:
- Use the web search tool as many times as you need, and check that different
  sources agree before you state a contested fact.
- Prefer specifics: numbers, dates, names, mechanisms.
- State plainly when the evidence is thin or the sources disagree.
- Reply with the answer only. No preamble, no headings, no markdown."""

WRITER_PROMPT = """You are the writing stage of an automated deep-research system.

Write a clear, well-organised markdown briefing on the topic below, based only
on the research notes provided. Notes come from independent agents that could
not see each other, so expect repetition and occasional contradictions.

Rules:
- Open with a two or three sentence summary of the key takeaways.
- Use headings and short paragraphs. Use bullets for genuinely enumerable
  items such as options, specs or steps.
- Where the notes disagree or are thin, say so rather than papering over it.
- Attribute specific claims to their source when the notes name one.
- Do not invent facts, figures, citations or sections that the notes do not
  support.
- Aim for substance over length: no filler, no restating the question.

Topic: <topic>{topic}</topic>

Research notes:
<notes>
{notes}
</notes>"""

CRITIC_PROMPT = """You are the review stage of an automated deep-research system.

You will be given a draft briefing. Decide whether it is comprehensive enough
to ship, given the research notes it was built from.

Set `acceptable` to true only if the draft covers the topic's main angles, is
grounded in the notes, and is not padded with unsupported claims.

If it is not acceptable, set `acceptable` to false and write specific, actionable
`feedback`: name the gap and say what should be researched to close it. "Add
more detail" is not actionable feedback.

Topic: <topic>{topic}</topic>

Research notes:
<notes>
{notes}
</notes>

Draft briefing:
<draft>
{draft}
</draft>"""
