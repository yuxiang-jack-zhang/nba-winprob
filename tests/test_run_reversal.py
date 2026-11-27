"""RunReversal v1 against synthetic play and quote sequences."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from nba_winprob.engine.events import Event, Fill, Order, Play, Quote, Side
from nba_winprob.strategies.run_reversal import RunReversal, RunReversalParams

T0 = 100_000.0  # wallclock of tip-off; plays default to ts = T0 + game seconds elapsed
G = "g"


def period_clock(e: float) -> tuple[int, float]:
    if e < 2880:
        period = min(int(e // 720) + 1, 4)
        return period, period * 720 - e
    period = 5 + int((e - 2880) // 300)
    return period, 2880 + (period - 4) * 300 - e


class Game:
    """Builds one game's plays and quotes, keeping the score."""

    def __init__(self, home: int, away: int) -> None:
        self.score_ = {"home": home, "away": away}

    def play(
        self,
        e: float,
        kind: str = "other",
        team: Side | None = None,
        points: int = 0,
        ts: float | None = None,
    ) -> Play:
        if team is not None:
            self.score_[team] += points
        period, clock = period_clock(e)
        return Play(
            ts=T0 + e if ts is None else ts,
            game_id=G,
            seq=0,
            period=period,
            clock=clock,
            home_score=self.score_["home"],
            away_score=self.score_["away"],
            team=team,
            kind=kind,  # type: ignore[arg-type]
            points=points,
            text="",
        )

    def score(self, e: float, team: Side, points: int) -> Play:
        return self.play(e, "score", team, points)

    def timeout(self, e: float, team: Side, ts: float | None = None) -> Play:
        return self.play(e, "timeout", team, ts=ts)

    @staticmethod
    def quote(ts: float, bid: float, ask: float) -> Quote:
        return Quote(ts=ts, game_id=G, ticker="T", yes_bid=bid, yes_ask=ask, volume=0)


class Harness:
    def __init__(self) -> None:
        self.s = RunReversal()
        self.pos: defaultdict[tuple[str, Side], int] = defaultdict(int)

    def send(self, *events: Event) -> list[Order]:
        return [o for e in events for o in self.s.on_event(e, self.pos)]

    def fill(self, order: Order, price: float) -> None:
        self.pos[(order.game_id, order.team)] += order.contracts
        self.s.on_fill(Fill(order=order, ts=order.ts + 30, price=price, fee=0.0))


def other(side: Side) -> Side:
    return "away" if side == "home" else "home"


def build(
    h: Harness,
    end_e: float = 1709,  # Q3 7:31
    runner: Side = "away",
    runner_pts: tuple[int, ...] = (3, 3, 2, 2, 2),
    lead: int = 5,  # the run-on team's lead after the run
    quote: tuple[float, float] | None = (0.40, 0.42),
    extra: tuple[Play, ...] = (),
) -> tuple[Game, Play]:
    """Send a run of ``runner_pts`` against 2 points, 40 s apart, ending at ``end_e``.

    Everything is sent except the run's last basket, which is returned unsent. The
    run-on team ends on 61. ``extra`` plays go just before the last basket.
    """
    run_on = other(runner)
    final = {run_on: 61, runner: 61 - lead}
    g = Game(home=final["home"], away=final["away"])
    g.score_[run_on] -= 2
    g.score_[runner] -= sum(runner_pts)
    if quote is not None:
        h.send(g.quote(T0 + end_e - 230, *quote))
    h.send(g.score(end_e - 200, run_on, 2))
    n = len(runner_pts)
    for i, pts in enumerate(runner_pts[:-1]):
        h.send(g.score(end_e - 40 * (n - 1 - i), runner, pts))
    h.send(*extra)
    return g, g.score(end_e, runner, runner_pts[-1])


def enter(h: Harness, **kw: object) -> tuple[Game, Order]:
    g, last = build(h, **kw)  # type: ignore[arg-type]
    [entry] = h.send(last)
    return g, entry


# --- trigger ---------------------------------------------------------------


def test_entry_order_reason_meta_and_size() -> None:
    h = Harness()
    _, entry = enter(h)
    assert (entry.team, entry.contracts, entry.ts) == ("home", 238, T0 + 1709)  # 100 // 0.42
    assert entry.reason == (
        "Away on a 12-2 run over 4:00, margin 5 (home 61-56), 7:31 left in Q3; buying home"
    )
    assert entry.meta == {
        "kind": "entry",
        "runner": "away",
        "run_net": 10,
        "runner_points": 12,
        "opponent_points": 2,
        "margin": 5,
        "game_seconds_left": 1171,
        "period": 3,
        "half": 2,
        "timeout_in_run_window": False,
        "price": 0.42,
    }


def test_buying_away_pays_one_minus_bid() -> None:
    h = Harness()
    _, entry = enter(h, runner="home")
    assert (entry.team, entry.contracts) == ("away", 166)  # 100 // (1 - 0.40)
    assert entry.reason.startswith("Home on a 12-2 run")
    assert "(home 56-61)" in entry.reason


def test_net_9_does_not_fire() -> None:
    h = Harness()
    _, last = build(h, runner_pts=(3, 3, 3, 2))  # nets 1, 4, 7, 9
    assert h.send(last) == []


def sliding_run(h: Harness, runner_pts: tuple[int, ...]) -> list[Order]:
    """Opponent's 8 points slide out of the window on the runner's last basket,
    so the net jumps from below 10 straight to sum(runner_pts) + 2."""
    g = Game(home=60, away=50)
    h.send(g.quote(T0, 0.40, 0.42))
    h.send(g.score(1000, "home", 3), g.score(1001, "home", 3), g.score(1002, "home", 2))
    for i, pts in enumerate(runner_pts):
        h.send(g.score(1010 + 20 * i, "away", pts))
    return h.send(g.score(1243, "away", 2))


def test_net_15_fires() -> None:
    assert len(sliding_run(Harness(), (3, 2, 2, 2, 2, 2))) == 1


def test_net_16_does_not_fire() -> None:
    assert sliding_run(Harness(), (3, 3, 2, 2, 2, 2)) == []


def test_runner_need_not_be_the_scorer() -> None:
    h = Harness()
    assert sliding_run(h, (3, 3, 2, 2, 2, 2)) == []  # away net 16
    g = Game(home=68, away=66)
    [entry] = h.send(g.score(1244, "home", 1))  # home free throw: away net 15
    assert (entry.team, entry.meta["runner"], entry.meta["run_net"]) == ("home", "away", 15)


@pytest.mark.parametrize(("lead", "fires"), [(9, True), (-9, True), (10, False), (-10, False)])
def test_margin_must_be_within_9(lead: int, fires: bool) -> None:
    h = Harness()
    _, last = build(h, lead=lead)
    assert len(h.send(last)) == int(fires)


@pytest.mark.parametrize(
    ("end_e", "fires"),
    [
        (2759, True),  # Q4 2:01
        (2760, False),  # Q4 2:00
        (3059, True),  # OT1 2:01
        (3060, False),  # OT1 2:00
    ],
)
def test_no_entries_in_the_last_two_minutes(end_e: float, fires: bool) -> None:
    h = Harness()
    _, last = build(h, end_e=end_e)
    assert len(h.send(last)) == int(fires)


def test_skips_without_a_quote_and_enters_on_a_later_basket() -> None:
    h = Harness()
    g, last = build(h, quote=None)
    assert h.send(last) == []
    assert h.s.skips == 1
    h.send(g.quote(T0 + 1710, 0.40, 0.42))
    [entry] = h.send(g.score(1712, "away", 2))  # net 12, still a trigger
    assert entry.meta["run_net"] == 12


def test_rearms_only_once_the_run_net_falls_below_5() -> None:
    h = Harness()
    g, entry = enter(h)
    h.fill(entry, 0.42)
    [exit_] = h.send(g.quote(T0 + 1710, 0.46, 0.48))
    h.fill(exit_, 0.46)
    # Window holds everything since 1509 throughout. Away net after each play:
    assert h.send(g.score(1712, "away", 2)) == []  # 12, margin 3: no position, but disarmed
    h.send(g.score(1715, "home", 3), g.score(1718, "home", 3))  # 9, 6
    h.send(g.score(1721, "home", 1))  # 5: not below 5, stays disarmed
    h.send(g.score(1724, "away", 3))  # 8
    assert h.send(g.score(1727, "away", 2)) == []  # 10, margin 5
    h.send(g.score(1730, "home", 3), g.score(1733, "home", 3))  # 7, 4: re-armed
    h.send(g.score(1736, "away", 3))  # 7
    [again] = h.send(g.score(1739, "away", 3))  # 10, margin 5
    assert again.team == "home"


def test_no_second_entry_while_pending_or_open() -> None:
    h = Harness()
    g, entry = enter(h)
    h.send(g.score(1712, "home", 3), g.score(1715, "home", 3))  # away net 4: re-armed
    h.send(g.score(1718, "away", 3))
    assert h.send(g.score(1721, "away", 3)) == []  # net 10, margin 5, entry unfilled
    h.fill(entry, 0.42)
    h.send(g.score(1724, "home", 3), g.score(1727, "home", 3), g.score(1730, "away", 3))
    assert h.send(g.score(1733, "away", 3)) == []  # net 10, margin 5, position open


# --- exits -----------------------------------------------------------------


def test_take_profit_sells_everything_held() -> None:
    h = Harness()
    g, entry = enter(h)
    h.fill(entry, 0.42)
    h.pos[(G, "home")] = 200  # e.g. a partial fill: sell what is actually held
    assert h.send(g.quote(T0 + 1740, 0.45, 0.47)) == []  # mid 0.46: +4c
    [exit_] = h.send(g.quote(T0 + 1800, 0.46, 0.48))  # mid 0.47: +5c
    assert (exit_.team, exit_.contracts) == ("home", -200)
    assert exit_.reason == "take profit: mid 0.47 vs entry 0.42"
    assert exit_.meta["kind"] == "take_profit"
    assert exit_.meta["mid"] == pytest.approx(0.47)
    assert exit_.meta["entry_price"] == 0.42
    for key in ("run_net", "runner_points", "opponent_points", "margin", "period", "half"):
        assert exit_.meta[key] == entry.meta[key]  # the trade's defining facts ride along
    assert {"game_seconds_left", "exit_game_seconds_left", "timeout_by_run_on_team"} <= set(
        exit_.meta
    )


def test_take_profit_on_away_uses_the_away_mid() -> None:
    h = Harness()
    g, entry = enter(h, runner="home")
    h.fill(entry, 0.60)
    assert h.send(g.quote(T0 + 1740, 0.35, 0.37)) == []  # away mid 0.64
    [exit_] = h.send(g.quote(T0 + 1800, 0.34, 0.36))  # away mid 0.65
    assert (exit_.team, exit_.contracts) == ("away", -166)


def test_time_stop_after_six_minutes_of_game_clock() -> None:
    h = Harness()
    g, entry = enter(h)
    h.fill(entry, 0.42)
    assert h.send(g.play(1709 + 359)) == []
    [exit_] = h.send(g.play(1709 + 360))
    assert exit_.meta["kind"] == "time_stop"
    assert exit_.reason == "time stop: 6:00 of game clock since entry; mid 0.41 vs entry 0.42"


def test_late_game_exit_at_two_minutes_left() -> None:
    h = Harness()
    g, entry = enter(h, end_e=2600)  # Q4 4:40
    h.fill(entry, 0.42)
    assert h.send(g.play(2759)) == []
    [exit_] = h.send(g.play(2760))
    assert exit_.meta["kind"] == "late_game"
    assert exit_.reason == "late game: 2:00 left in Q4; mid 0.41 vs entry 0.42"


def test_hold_variant_never_exits_but_keeps_the_entry_cutoff() -> None:
    h = Harness()
    h.s = RunReversal(
        RunReversalParams(take_profit=math.inf, max_hold_s=math.inf, exit_seconds_left=-math.inf)
    )
    g, entry = enter(h, end_e=2600)
    h.fill(entry, 0.42)
    assert h.send(g.quote(T0 + 2700, 0.90, 0.92), g.play(2700 + 360), g.play(2879)) == []


def test_no_exit_before_the_entry_fills_and_no_exit_spam() -> None:
    h = Harness()
    g, entry = enter(h)
    assert h.send(g.quote(T0 + 1740, 0.60, 0.62)) == []  # entry not filled yet
    h.fill(entry, 0.42)
    [exit_] = h.send(g.quote(T0 + 1800, 0.60, 0.62))
    assert h.send(g.quote(T0 + 1860, 0.61, 0.63), g.play(2100), g.play(2770)) == []
    h.fill(exit_, 0.60)
    assert h.send(g.quote(T0 + 1900, 0.61, 0.63), g.play(2780)) == []


# --- timeouts --------------------------------------------------------------


@pytest.mark.parametrize(
    ("team", "e", "ts_after_trigger", "in_run_window", "by_run_on_team"),
    [
        ("home", 1700, None, True, True),  # run-on team, inside the run window
        ("away", 1700, None, False, False),  # the runner's own timeout
        ("home", 1469, None, False, False),  # exactly 4:00 before the trigger: outside
        ("home", 1712, 30, False, True),  # 30 s wallclock after the trigger
        ("home", 1712, 31, False, False),
    ],
)
def test_timeout_flags(
    team: Side, e: float, ts_after_trigger: float | None, in_run_window: bool, by_run_on_team: bool
) -> None:
    h = Harness()
    g = Game(0, 0)  # only builds the timeout play; scores in it are not read
    before = (g.timeout(e, team),) if ts_after_trigger is None else ()
    g, last = build(h, extra=before)
    [entry] = h.send(last)
    if ts_after_trigger is not None:
        h.send(g.timeout(e, team, ts=T0 + 1709 + ts_after_trigger))
    h.fill(entry, 0.42)
    [exit_] = h.send(g.play(1709 + 360))
    assert entry.meta["timeout_in_run_window"] is in_run_window
    assert exit_.meta["timeout_by_run_on_team"] is by_run_on_team
