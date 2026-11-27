"""Kalshi demo broker: places real orders on the demo exchange to rehearse the order path.

Same price rule as ``SimBroker``: buying home pays the YES ask, buying away pays the NO
price ``1 - bid``, sells mirror that. On Kalshi's single YES book that is ``bid`` at the ask
for buy-home / sell-away and ``ask`` at the bid for buy-away / sell-home, sent
immediate-or-cancel so an order fills now or not at all.

The demo exchange lists the same NBA tickers as production but has its own thin book, so
an order priced off production quotes may not fill. Its P&L means nothing.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from nba_winprob.brokers.kalshi_client import BookSide, KalshiClient, KalshiError
from nba_winprob.engine.events import Event, Fill, Order, Quote, Settle, Side, kalshi_taker_fee


class DemoBroker:
    """Learns each game's ticker from the quotes it sees; polls fills at most every
    ``poll_every_s`` seconds."""

    def __init__(
        self,
        client: KalshiClient,
        poll_every_s: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client, self.poll_every_s, self.clock = client, poll_every_s, clock
        self.positions: dict[tuple[str, Side], int] = {}
        self.fills: list[Fill] = []
        self.rejected: list[tuple[Order, str]] = []
        self._quotes: dict[str, Quote] = {}  # game_id -> latest quote
        self._orders: dict[str, Order] = {}  # exchange order_id -> our order
        self._seen: set[str] = set()  # fill_ids already reported
        self._started = int(clock())  # fills before this are not ours
        self._last_poll = float("-inf")

    def submit(self, order: Order) -> None:
        quote = self._quotes.get(order.game_id)
        if quote is None:
            self.rejected.append((order, "no quote yet"))
            return
        side: BookSide = "bid" if (order.team == "home") == (order.contracts > 0) else "ask"
        price = quote.yes_ask if side == "bid" else quote.yes_bid  # YES-book price
        if not 0 < price < 1:
            self.rejected.append((order, f"no {side} price to cross: {price}"))
            return
        try:
            resp = self.client.create_order(
                quote.ticker,
                side,
                abs(order.contracts),
                price,
                client_order_id=str(uuid.uuid4()),
                reduce_only=order.contracts < 0,  # a sell never opens the opposite side
            )
        except KalshiError as err:
            self.rejected.append((order, str(err)))
            return
        self._orders[resp["order_id"]] = order

    def process(self, event: Event) -> list[Fill]:
        if isinstance(event, Quote):
            self._quotes[event.game_id] = event
        elif isinstance(event, Settle):
            teams: tuple[Side, Side] = ("home", "away")
            for team in teams:
                self.positions.pop((event.game_id, team), None)
        if not self._orders or self.clock() - self._last_poll < self.poll_every_s:
            return []
        self._last_poll = self.clock()
        new = []
        for raw in self.client.get_fills(min_ts=self._started):
            fill_id = raw["fill_id"]
            if fill_id in self._seen or raw["order_id"] not in self._orders:
                continue
            self._seen.add(fill_id)
            new.append(self._fill(self._orders[raw["order_id"]], raw))
        new.sort(key=lambda f: f.ts)
        self.fills += new
        return new

    def _fill(self, order: Order, raw: dict[str, Any]) -> Fill:
        """The exchange fill as a contract ``Fill`` for the count actually filled."""
        count = round(float(raw["count_fp"]))
        signed = count if order.contracts > 0 else -count
        price = float(raw["yes_price_dollars" if order.team == "home" else "no_price_dollars"])
        fee = raw.get("fee_cost")
        if "ts" in raw:
            ts = float(raw["ts"])
        else:
            ts = datetime.fromisoformat(raw["created_time"]).timestamp()
        fill = Fill(
            dataclasses.replace(order, contracts=signed),
            ts,
            price,
            float(fee) if fee is not None else kalshi_taker_fee(price, count),
        )
        key = (order.game_id, order.team)
        held = self.positions.get(key, 0) + signed
        if held:
            self.positions[key] = held
        else:
            self.positions.pop(key, None)
        return fill
