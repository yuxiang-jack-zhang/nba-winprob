"""The catch-up loop with a scripted client: no network, real Postgres for the tools."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

import psycopg

from nba_winprob.agent.catchup import ask
from tests import agent_scenario as sc


def tool_use(id_: str, name: str, **input_: Any) -> NS:
    return NS(type="tool_use", id=id_, name=name, input=input_)


def reply(stop: str, *blocks: NS) -> NS:
    return NS(stop_reason=stop, content=list(blocks))


class Scripted:
    """Returns canned responses in order and keeps every request for inspection."""

    def __init__(self, *responses: NS) -> None:
        self.responses, self.requests = list(responses), []

    def create(self, **kwargs: Any) -> NS:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_tool_results_go_back_in_one_message_and_the_text_comes_out(db: psycopg.Connection) -> None:
    run = sc.seed(db)
    api = Scripted(
        reply(
            "tool_use",
            NS(type="thinking", thinking=""),
            tool_use("a", "run_status"),
            tool_use("b", "recent_plays", game_id=sc.NYK, limit=1),
        ),
        reply("end_turn", NS(type="text", text="Down $42.57. Holding 200 NY.")),
    )
    answer = ask(api, db, run, "Catch me up", now=lambda: sc.NOW)
    assert (answer.text, answer.stop_reason) == ("Down $42.57. Holding 200 NY.", "end_turn")
    assert answer.tool_calls == ["run_status", "recent_plays"]
    second = api.requests[1]["messages"]
    assert second[1]["content"][0].type == "thinking"  # assistant turn sent back whole
    a, b = second[2]["content"]
    assert (a["tool_use_id"], b["tool_use_id"], a["is_error"]) == ("a", "b", False)
    assert json.loads(a["content"])["pnl"] == sc.PNL
    assert json.loads(b["content"])[0]["text"] == "NY full timeout"


def test_a_bad_tool_call_is_reported_back_as_an_error(db: psycopg.Connection) -> None:
    run = sc.seed(db)
    api = Scripted(
        reply("tool_use", tool_use("a", "recent_plays", game_id="LAL", limit=None)),
        reply("end_turn", NS(type="text", text="No Lakers game tonight.")),
    )
    assert ask(api, db, run, "Lakers?", now=lambda: sc.NOW).text == "No Lakers game tonight."
    [result] = api.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and "no game LAL" in result["content"]


def test_the_loop_gives_up_after_max_turns(db: psycopg.Connection) -> None:
    run = sc.seed(db)
    api = Scripted(*[reply("tool_use", tool_use(str(i), "games")) for i in range(3)])
    answer = ask(api, db, run, "?", now=lambda: sc.NOW, max_turns=3)
    assert answer.stop_reason == "max_turns" and answer.tool_calls == ["games"] * 3


def test_a_refusal_is_reported_not_raised(db: psycopg.Connection) -> None:
    run = sc.seed(db)
    answer = ask(Scripted(reply("refusal")), db, run, "?", now=lambda: sc.NOW)
    assert answer.stop_reason == "refusal" and "declined" in answer.text


def test_requests_offer_only_the_read_only_tools(db: psycopg.Connection) -> None:
    run = sc.seed(db)
    api = Scripted(reply("end_turn", NS(type="text", text="ok")))
    ask(api, db, run, "?", now=lambda: sc.NOW)
    [request] = api.requests
    assert [t["name"] for t in request["tools"]] == [
        "run_status",
        "games",
        "orders",
        "recent_plays",
    ]
    assert request["model"] == "claude-opus-5-5"
