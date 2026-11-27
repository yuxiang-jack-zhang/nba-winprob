"""Evals: the real catch-up agent answering questions about the seeded evening.

Opt-in, since every case calls the Claude API and costs money:
    NBA_RUN_LLM_EVALS=1 uv run pytest tests/test_agent_evals.py -v
Grading is deterministic: each answer must contain the facts the tools hold (regexes)
and must not contain claims they contradict. The ground truth is pinned by
tests/test_agent_tools.py, so a failure here is the agent's, not the data's.
"""

from __future__ import annotations

import os
import re
from typing import Any

import psycopg
import pytest

from nba_winprob.agent.catchup import ask
from nba_winprob.agent.notify import DISCORD_LIMIT
from nba_winprob.config import get_settings
from tests import agent_scenario as sc

pytestmark = pytest.mark.skipif(
    os.environ.get("NBA_RUN_LLM_EVALS") != "1",
    reason="calls the Claude API; set NBA_RUN_LLM_EVALS=1",
)

PNL = re.escape(f"{abs(sc.PNL):.2f}")  # "42.57"

# (question, must match every one, must match none, tool that must be called)
CASES = [
    (
        "Catch me up.",
        [PNL, r"\bNY\b", r"\b200\b", r"BOS|Celtics|Boston"],
        [r"\bup \$?" + PNL],  # it is a loss
        "run_status",
    ),
    (
        "Am I up or down tonight?",
        [PNL, r"down|loss|lost|-\$?" + PNL],
        [r"\bup \$?" + PNL],
        "run_status",
    ),
    ("Why did the bot buy NY?", [r"12[-\u2013]2", r"timeout"], [], "orders"),
    ("How's the Knicks game going?", [r"72", r"70", r"4:12", r"Q3|3rd|third"], [], "games"),
    ("Did anything go wrong with orders?", [r"reject"], [], "orders"),
    (
        "What is the bot holding in the Lakers game?",
        [r"no |not |isn't|doesn't"],
        [r"LAL \d+"],
        "games",
    ),
]


@pytest.fixture
def agent() -> object:
    import anthropic

    key = get_settings().anthropic_api_key or None
    return anthropic.Anthropic(api_key=key).beta.messages


@pytest.mark.parametrize(("question", "must", "must_not", "tool"), CASES, ids=[c[0] for c in CASES])
def test_catchup_answer(
    db: psycopg.Connection,
    agent: object,
    question: str,
    must: list[str],
    must_not: list[str],
    tool: str,
) -> None:
    run = sc.seed(db)
    answer = ask(agent, db, run, question, now=lambda: sc.NOW)
    print(f"\nQ: {question}\nA: {answer.text}\ntools: {answer.tool_calls}")
    assert answer.stop_reason == "end_turn"
    assert tool in answer.tool_calls
    missing = [p for p in must if not re.search(p, answer.text, re.IGNORECASE)]
    wrong = [p for p in must_not if re.search(p, answer.text, re.IGNORECASE)]
    assert not missing and not wrong, f"missing {missing}, contradicted {wrong}"
    assert len(answer.text) <= DISCORD_LIMIT  # fits one Discord message


def test_a_follow_up_uses_the_conversation(db: psycopg.Connection, agent: object) -> None:
    run = sc.seed(db)
    history: list[dict[str, Any]] = []
    ask(agent, db, run, "Am I up or down?", now=lambda: sc.NOW, history=history)
    answer = ask(agent, db, run, "Which game hurt most?", now=lambda: sc.NOW, history=history)
    print(f"\nfollow-up A: {answer.text}")
    assert answer.stop_reason == "end_turn"
    assert re.search(r"DEN|Denver|BOS|Boston|Celtics", answer.text)
    assert re.search(re.escape(f"{45 + sc.FEE_DEN:.2f}"), answer.text)  # the DEN loss, 47.21
