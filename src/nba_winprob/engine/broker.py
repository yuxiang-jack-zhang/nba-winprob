"""Simulated broker: fills orders at later quotes, charges Kalshi fees, settles at $1/$0."""

from __future__ import annotations

from nba_winprob.engine.events import Event, Fill, Order, Quote, Settle, Side, kalshi_taker_fee


class SimBroker:
    """Fills each order at the first quote for its game with ``quote.ts >= order.ts + latency_s``
    and ``quote.ts > order.ts``.

    The strict inequality matters at ``latency_s == 0``: a quote stamped at the same second
    as the decision (including the very quote the strategy reacted to, or one that sorts
    after it at that second) never fills it. Every price is a price the strategy could not
    yet have seen when it decided.
    """

    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s
        self.cash = 0.0
        self.positions: dict[tuple[str, Side], int] = {}
        self.fills: list[Fill] = []
        self.cancelled: list[Order] = []
        self.payouts: list[tuple[Settle, Side, int]] = []  # (settlement, team, contracts held)
        self._pending: dict[str, list[Order]] = {}
        self._now = float("-inf")

    def submit(self, order: Order) -> None:
        if order.ts < self._now:
            raise ValueError(f"order at {order.ts} is stamped before the current event {self._now}")
        self._pending.setdefault(order.game_id, []).append(order)

    def process(self, event: Event) -> list[Fill]:
        """Advance to ``event``; return the fills it caused, in submission order."""
        self._now = event.ts
        if isinstance(event, Quote):
            return self._fill(event)
        if isinstance(event, Settle):
            self._settle(event)
        return []

    def _fill(self, quote: Quote) -> list[Fill]:
        waiting, fills = [], []
        for order in self._pending.pop(quote.game_id, []):
            if quote.ts > order.ts and quote.ts >= order.ts + self.latency_s:
                fills.append(self._execute(order, quote))
            else:
                waiting.append(order)
        if waiting:
            self._pending[quote.game_id] = waiting
        return fills

    def _execute(self, order: Order, quote: Quote) -> Fill:
        key = (order.game_id, order.team)
        held = self.positions.get(key, 0)
        buying = order.contracts > 0
        if order.team == "home":
            price = quote.yes_ask if buying else quote.yes_bid
        else:
            price = 1 - quote.yes_bid if buying else 1 - quote.yes_ask
        if buying and not 0 < price < 1:
            raise ValueError(f"cannot buy {order.team} at price {price}: {order}")
        if -order.contracts > held:
            raise ValueError(f"cannot sell {-order.contracts} {order.team}, holding {held}")
        fill = Fill(order, quote.ts, price, kalshi_taker_fee(price, order.contracts))
        self.cash -= price * order.contracts + fill.fee
        if held + order.contracts:
            self.positions[key] = held + order.contracts
        else:
            del self.positions[key]
        self.fills.append(fill)
        return fill

    def _settle(self, settle: Settle) -> None:
        self.cancelled += self._pending.pop(settle.game_id, [])
        winner: Side = "home" if settle.home_won else "away"
        teams: tuple[Side, Side] = ("home", "away")
        for team in teams:
            held = self.positions.pop((settle.game_id, team), 0)
            if held:
                self.cash += held if team == winner else 0
                self.payouts.append((settle, team, held))
