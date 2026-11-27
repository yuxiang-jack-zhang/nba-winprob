"""RunReversal v1: buy the team just run on in a close game, betting the price overreacted.

Implements "Pre-registered spec: RunReversal v1" in
docs/superpowers/specs/2026-11-26-intragame-trading-design.md. The numbers are the
spec's and are not to be tuned on results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nba_winprob.engine.events import Event, Fill, Order, Play, Positions, Quote, Settle, Side
from nba_winprob.features.ingame import RunTracker, other_side

# Absorbs float error so a move of exactly 5 cents (0.47 - 0.42) counts as 5 cents.
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class RunReversalParams:
    run_window_s: float = 240  # the run is measured over the trailing 4:00 of game clock
    min_run_net: int = 10
    max_run_net: int = 15
    max_margin: int = 9
    min_seconds_left: float = 120  # no entries at or under 2:00 left
    stake: float = 100  # dollars per entry
    rearm_below: int = 5
    take_profit: float = 0.05
    max_hold_s: float = 360  # game seconds
    timeout_after_s: float = 30  # wallclock seconds after the trigger
    exit_seconds_left: float = 120  # flatten at 2:00 left; -inf holds to settlement


V1 = RunReversalParams()


@dataclass(slots=True)
class _Trade:
    team: Side
    elapsed: float  # game seconds elapsed at the trigger play
    ts: float  # wallclock of the trigger play
    meta: dict[str, float | int | bool | str]
    price: float | None = None  # entry fill price; None while the entry is pending
    exiting: bool = False


@dataclass(slots=True)
class _Game:
    tracker: RunTracker
    quote: Quote | None = None
    armed: dict[Side, bool] = field(default_factory=lambda: {"home": True, "away": True})
    trade: _Trade | None = None  # pending or open: at most one per game


class RunReversal:
    name = "run_reversal_v1"

    def __init__(self, params: RunReversalParams = V1) -> None:
        self.params = params
        self.skips = 0  # triggers not entered for want of a usable quote
        self._games: dict[str, _Game] = {}

    def on_event(self, event: Event, positions: Positions) -> list[Order]:
        if isinstance(event, Settle):
            self._games.pop(event.game_id, None)
            return []
        g = self._games.get(event.game_id)
        if g is None:
            g = self._games[event.game_id] = _Game(RunTracker(self.params.run_window_s))
        if isinstance(event, Quote):
            g.quote = event
            return self._exit(g, event.ts, event.game_id, positions)
        g.tracker.update(event)
        for side in ("home", "away"):
            if g.tracker.run_net(side) < self.params.rearm_below:
                g.armed[side] = True
        return self._exit(g, event.ts, event.game_id, positions) or self._entry(g, event)

    def on_fill(self, fill: Fill) -> None:
        g = self._games.get(fill.order.game_id)
        if g is None or g.trade is None:  # e.g. a fill reported after the game settled
            return
        if fill.order.contracts > 0:
            g.trade.price = fill.price
        else:
            g.trade = None

    def _entry(self, g: _Game, play: Play) -> list[Order]:
        p, t = self.params, g.tracker
        if play.kind != "score":
            return []
        # The runner is whichever side is up over the window, not necessarily the scorer:
        # the spec defines the run purely by net points, which also moves as old baskets
        # slide out of the window.
        runner: Side = "home" if t.run_net("home") > 0 else "away"
        net = t.run_net(runner)
        if not (
            p.min_run_net <= net <= p.max_run_net
            and abs(t.margin) <= p.max_margin
            and t.seconds_left > p.min_seconds_left
            and g.armed[runner]
            and g.trade is None
        ):
            return []
        run_on = other_side(runner)
        price = None if g.quote is None else _ask(g.quote, run_on)
        if price is None or not 0 < price < 1:
            # Not disarmed: a later basket that still meets the trigger may enter.
            self.skips += 1
            return []
        g.armed[runner] = False
        meta: dict[str, float | int | bool | str] = {
            "kind": "entry",
            "runner": runner,
            "run_net": net,
            "runner_points": t.points(runner),
            "opponent_points": t.points(run_on),
            "margin": t.margin,
            "game_seconds_left": t.seconds_left,
            "period": t.period,
            "half": _half(t.period),
            # Only the run-window half of the spec's timeout split is knowable now;
            # the exit order carries the full flag as timeout_by_run_on_team.
            "timeout_in_run_window": t.timeout_called(
                run_on, game_window=(t.elapsed - p.run_window_s, t.elapsed)
            ),
            "price": price,
        }
        g.trade = _Trade(run_on, t.elapsed, play.ts, meta)
        reason = (
            f"{runner.capitalize()} on a {t.points(runner)}-{t.points(run_on)} run over "
            f"{_mmss(p.run_window_s)}, margin {abs(t.margin)} "
            f"(home {t.home_score}-{t.away_score}), {_mmss(t.clock)} left in "
            f"{_period(t.period)}; buying {run_on}"
        )
        contracts = math.floor(p.stake / price)
        return [Order(play.ts, play.game_id, run_on, contracts, reason, meta)]

    def _exit(self, g: _Game, ts: float, game_id: str, positions: Positions) -> list[Order]:
        p, t, trade = self.params, g.tracker, g.trade
        if trade is None or trade.price is None or trade.exiting:
            return []
        held = positions.get((game_id, trade.team), 0)
        if held <= 0:
            return []
        assert g.quote is not None  # an entry needs a quote
        mid = _mid(g.quote, trade.team)
        held_s = t.elapsed - trade.elapsed
        if mid - trade.price >= p.take_profit - _EPS:
            kind, why = "take_profit", "take profit:"
        elif held_s >= p.max_hold_s:
            kind, why = "time_stop", f"time stop: {_mmss(held_s)} of game clock since entry;"
        elif t.seconds_left <= p.exit_seconds_left:
            kind, why = "late_game", f"late game: {_mmss(t.clock)} left in {_period(t.period)};"
        else:
            return []
        trade.exiting = True
        meta = trade.meta | {
            "kind": kind,
            "entry_price": trade.price,
            "mid": mid,
            "held_game_seconds": held_s,
            "exit_game_seconds_left": t.seconds_left,
            "exit_margin": t.margin,
            # Spec split: a run-on team timeout within the run window or within 30 s
            # (wallclock, after the trigger play's ts) after the trigger.
            "timeout_by_run_on_team": t.timeout_called(
                trade.team,
                game_window=(trade.elapsed - p.run_window_s, trade.elapsed),
                ts_window=(trade.ts, trade.ts + p.timeout_after_s),
            ),
        }
        reason = f"{why} mid {mid:.2f} vs entry {trade.price:.2f}"
        return [Order(ts, game_id, trade.team, -held, reason, meta)]


def _ask(q: Quote, team: Side) -> float:
    """What buying ``team``'s contract costs: YES at the ask, or NO at 1 - bid."""
    return q.yes_ask if team == "home" else 1 - q.yes_bid


def _mid(q: Quote, team: Side) -> float:
    mid = (q.yes_bid + q.yes_ask) / 2
    return mid if team == "home" else 1 - mid


def _half(period: int) -> int | str:
    return 1 if period <= 2 else 2 if period <= 4 else "OT"


def _period(period: int) -> str:
    return f"Q{period}" if period <= 4 else f"OT{period - 4}"


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
