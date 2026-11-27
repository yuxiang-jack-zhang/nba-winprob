"""Discord push updates: each fill, rejection, recap and run stop is reported once."""

from __future__ import annotations

import psycopg

from nba_winprob.agent.notify import Notifier, chunks, money
from nba_winprob.db import live_store
from tests import agent_scenario as sc

HALF_HOUR = 30 * 60


def test_the_evening_is_pushed_once_in_order(db: psycopg.Connection) -> None:
    notifier = Notifier(db, now=sc.T0)  # watching before anything happened
    run = sc.seed(db)
    messages, digest_due = notifier.poll(sc.T0 + HALF_HOUR)
    assert messages == [
        f"Bought 150 DEN @ 0.30 in DEN @ BOS (fee ${sc.FEE_DEN:.2f}). Why: {sc.DEN_REASON}",
        f"Bought 200 NY @ 0.40 in MIA @ NY (fee ${sc.FEE_NY:.2f}). Why: {sc.NY_REASON}",
        "Order rejected: buy 50 NY in MIA @ NY: order at ts 1795003016 is before the last event",
        "Final: DEN 104, BOS 110. Bot P&L on this game: -$47.21.",
    ]
    assert digest_due  # the run is live, NY is playing, and half an hour has passed
    assert notifier.poll(sc.T0 + HALF_HOUR + 60) == ([], False)

    live_store.stop_run(db, run, "feed exhausted")
    assert notifier.poll(sc.T0 + 3 * HALF_HOUR) == (
        [f"Run {run} (run_reversal_v1, paper) stopped: feed exhausted. P&L {money(sc.PNL)}."],
        False,  # no digest for a stopped run
    )
    assert notifier.poll(sc.T0 + 4 * HALF_HOUR) == ([], False)


def test_a_restart_does_not_replay_the_evening(db: psycopg.Connection) -> None:
    sc.seed(db)
    assert Notifier(db, now=sc.NOW).poll(sc.NOW + 1) == ([], False)


def test_no_digest_without_a_live_game(db: psycopg.Connection) -> None:
    sc.seed(db)
    db.execute("update game_state set status = 'final'")
    notifier = Notifier(db, now=sc.NOW)
    assert notifier.poll(sc.NOW + 2 * HALF_HOUR)[1] is False


def test_money_and_chunks() -> None:
    assert (money(3), money(-47.211), money(None)) == ("+$3.00", "-$47.21", "n/a (no price yet)")
    assert chunks("a\nb\nc", limit=4) == ["a\nb", "c"]
    assert chunks("x" * 5, limit=2) == ["xx", "xx", "x"]
    assert chunks("") == []
