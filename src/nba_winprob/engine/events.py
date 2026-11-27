"""The contract between data, strategies and brokers, shared by backtest and live trading.

A game is a time-ordered stream of events. A strategy sees one event at a time and
returns orders; a broker fills them. Backtest and live differ only in where events
come from (cached history vs live polling) and which broker fills the orders, so a
strategy that was backtested is the same code that trades.

Prices are for the home team's YES contract on Kalshi, in dollars in (0, 1). Buying
``team="home"`` buys YES at the ask; buying ``team="away"`` buys NO at ``1 - bid``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

Side = Literal["home", "away"]
PlayKind = Literal["score", "timeout", "end_period", "end_game", "other"]


@dataclass(frozen=True, slots=True)
class Play:
    """One play-by-play row. ``ts`` is when it happened (ESPN wallclock, unix seconds)."""

    ts: float
    game_id: str
    seq: int  # ESPN sequenceNumber; orders plays that share a timestamp
    period: int  # 1-4 regulation, 5+ overtime
    clock: float  # seconds left in the period
    home_score: int  # after this play
    away_score: int
    team: Side | None  # who made the play (scored, called the timeout), if anyone
    kind: PlayKind
    points: int  # points scored on this play; 0 unless kind == "score"
    text: str


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book for the home team's YES contract.

    Historical: the close of a one-minute Kalshi candle, ``ts`` = end of the minute.
    Live: the book when polled, ``ts`` = receipt time.
    """

    ts: float
    game_id: str
    ticker: str
    yes_bid: float
    yes_ask: float
    volume: float  # contracts traded in the bar (historical) or 0 (live snapshot)


@dataclass(frozen=True, slots=True)
class Settle:
    """The market resolved: every open position pays $1 per contract or nothing."""

    ts: float
    game_id: str
    home_won: bool


Event = Play | Quote | Settle

_RANK = {Play: 0, Quote: 1, Settle: 2}


def sort_key(event: Event) -> tuple[float, int, int]:
    """Replay order: by time; at equal times plays first, then quotes, then settlement.

    Plays before quotes at the same second means a strategy reacting to a play can
    never trade at a price stamped at that same moment: the broker fills only at a
    quote strictly after the order's time plus latency.
    """
    return (event.ts, _RANK[type(event)], event.seq if isinstance(event, Play) else 0)


@dataclass(frozen=True, slots=True)
class Order:
    """Buy (``contracts > 0``) or sell (``< 0``) the contract that pays if ``team`` wins.

    ``reason`` is shown to people: the catch-up agent explains trades from it.
    ``meta`` holds the facts behind the decision (run size, timeout, ...) so results
    can be split by them without parsing ``reason``.
    """

    ts: float  # when the strategy decided; filled no earlier than ts + broker latency
    game_id: str
    team: Side
    contracts: int
    reason: str
    meta: Mapping[str, float | int | bool | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fill:
    order: Order
    ts: float
    price: float  # dollars per contract of the side traded
    fee: float  # dollars, total for the fill


Positions = Mapping[tuple[str, Side], int]  # (game_id, team) -> contracts held


class Strategy(Protocol):
    """Sees every event in order and returns the orders it wants placed.

    The broker reports each of the strategy's fills through ``on_fill`` before the
    next event, so a strategy can key exits off its actual entry price.
    """

    name: str

    def on_event(self, event: Event, positions: Positions) -> list[Order]: ...

    def on_fill(self, fill: Fill) -> None: ...


def kalshi_taker_fee(price: float, contracts: int) -> float:
    """Kalshi's taker fee: 0.07 x C x P x (1 - P), rounded up to the next cent per order."""
    raw = 0.07 * abs(contracts) * price * (1 - price)
    return -(-round(raw * 10_000) // 100) / 100  # ceil to a cent without float drift
