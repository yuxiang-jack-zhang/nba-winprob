"""Replay games through a strategy and a SimBroker; tabulate the round trips."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nba_winprob.engine.broker import SimBroker
from nba_winprob.engine.events import Event, Side, Strategy, sort_key


@dataclass(frozen=True)
class BacktestResult:
    broker: SimBroker  # fills, cancellations, cash
    trades: pd.DataFrame  # one row per round trip
    games: pd.Index  # every game replayed, traded or not

    @property
    def game_pnl(self) -> pd.Series:
        """P&L per game in dollars, 0 for games with no trades."""
        return self.trades.groupby("game_id")["pnl"].sum().reindex(self.games, fill_value=0.0)


def run_backtest(
    strategy: Strategy, games: Iterable[list[Event]], latency_s: float
) -> BacktestResult:
    """Each event goes to the broker first (fills reach ``on_fill``), then to the strategy."""
    broker = SimBroker(latency_s)
    game_ids = []
    for events in games:
        if not events:
            continue
        game_id = events[0].game_id
        game_ids.append(game_id)
        for event in sorted(events, key=sort_key):
            for fill in broker.process(event):
                strategy.on_fill(fill)
            for order in strategy.on_event(event, broker.positions):
                broker.submit(order)
        if any(g == game_id for g, _ in broker.positions):
            raise ValueError(f"game {game_id} has open positions but no settlement")
    return BacktestResult(broker, _round_trips(broker), pd.Index(game_ids, name="game_id"))


def _round_trips(broker: SimBroker) -> pd.DataFrame:
    """A round trip runs from a flat position in (game, team) back to flat or settlement.

    It carries the opening order's ``reason`` and ``meta`` (the decision being evaluated)
    and the ``exit_reason`` of whatever flattened it, plus the closing order's meta as
    ``exit_<key>`` (facts only known at exit, such as a timeout after the trigger).
    ``staked`` is dollars paid for contracts, before fees, matching ``papertrade``'s stake.
    """
    trips: dict[tuple[str, Side], dict[str, Any]] = {}
    rows = []
    for fill in broker.fills:
        o = fill.order
        key = (o.game_id, o.team)
        if key not in trips:  # the broker rejects sells from flat, so this is a buy
            trips[key] = {
                **o.meta,
                "game_id": o.game_id,
                "team": o.team,
                "entry_ts": fill.ts,
                "reason": o.reason,
                "held": 0,
                "contracts": 0,
                "staked": 0.0,
                "fees": 0.0,
                "pnl": 0.0,
            }
        trip = trips[key]
        trip["held"] += o.contracts
        trip["contracts"] += max(o.contracts, 0)
        trip["staked"] += fill.price * max(o.contracts, 0)
        trip["fees"] += fill.fee
        trip["pnl"] -= fill.price * o.contracts + fill.fee
        if trip["held"] == 0:
            exit_meta = {f"exit_{k}": v for k, v in o.meta.items()}
            rows.append(
                {**trips.pop(key), **exit_meta, "exit_ts": fill.ts, "exit_reason": o.reason}
            )
    # Settlement is a game's last event, so it follows every fill of that game.
    for settle, team, held in broker.payouts:
        trip = trips.pop((settle.game_id, team))
        trip["pnl"] += held if (team == "home") == settle.home_won else 0
        rows.append({**trip, "exit_ts": settle.ts, "exit_reason": "settle"})
    columns = ["game_id", "team", "entry_ts", "exit_ts", "contracts", "staked", "fees", "pnl"]
    columns += ["reason", "exit_reason"]
    frame = pd.DataFrame(rows)
    meta = [c for c in frame.columns if c not in columns and c != "held"]
    return frame.reindex(columns=columns + meta)
