"""SimBroker, the replay loop and backtest metrics, on hand-built event sequences."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_winprob.engine.backtest import run_backtest
from nba_winprob.engine.broker import SimBroker
from nba_winprob.engine.events import (
    Event,
    Fill,
    Order,
    Play,
    Positions,
    Quote,
    Settle,
    Side,
    kalshi_taker_fee,
)
from nba_winprob.engine.metrics import summary, summary_by


def play(ts: float, game: str = "G", seq: int = 0) -> Play:
    return Play(ts, game, seq, 1, 600.0, 0, 0, None, "other", 0, "")


def quote(ts: float, bid: float, ask: float, game: str = "G") -> Quote:
    return Quote(ts, game, "T", bid, ask, 0.0)


def order(
    ts: float, team: Side, contracts: int, game: str = "G", reason: str = "r", **meta: bool
) -> Order:
    return Order(ts, game, team, contracts, reason, meta)


class Scripted:
    """Places preset orders when it sees given events; logs every callback in order."""

    name = "scripted"

    def __init__(self, plan: dict[Event, list[tuple[Side, int, str, dict[str, bool]]]]) -> None:
        self.plan = plan
        self.log: list[tuple[str, Event | Fill]] = []

    def on_event(self, event: Event, positions: Positions) -> list[Order]:
        self.log.append(("event", event))
        return [
            Order(event.ts, event.game_id, team, n, reason, meta)
            for team, n, reason, meta in self.plan.get(event, [])
        ]

    def on_fill(self, fill: Fill) -> None:
        self.log.append(("fill", fill))


def buy(team: Side, n: int, reason: str = "entry", **meta: bool) -> list:
    return [(team, n, reason, meta)]


# --- lookahead and latency -------------------------------------------------------


def test_fills_at_first_quote_at_or_after_latency() -> None:
    trigger = play(100)
    events: list[Event] = [
        trigger,
        quote(100, 0.38, 0.40),  # same second as the decision
        quote(120, 0.43, 0.45),  # inside the 30 s latency
        quote(130, 0.48, 0.50),  # exactly order.ts + latency: first eligible
        quote(160, 0.53, 0.55),
        Settle(200, "G", True),
    ]
    strategy = Scripted({trigger: buy("home", 10)})
    result = run_backtest(strategy, [events], latency_s=30)
    [fill] = result.broker.fills
    assert (fill.ts, fill.price) == (130, 0.50)


@pytest.mark.parametrize("decider", ["play", "quote"])
def test_zero_latency_never_fills_at_same_timestamp(decider: str) -> None:
    trigger: Event = play(100) if decider == "play" else quote(100, 0.30, 0.32)
    events: list[Event] = [
        trigger,
        quote(100, 0.38, 0.40),  # sorts after the trigger but is stamped at the same time
        quote(160, 0.48, 0.50),
        Settle(200, "G", True),
    ]
    result = run_backtest(Scripted({trigger: buy("home", 10)}), [events], latency_s=0)
    [fill] = result.broker.fills
    assert (fill.ts, fill.price) == (160, 0.50)


def test_ignores_quotes_for_other_games() -> None:
    broker = SimBroker(latency_s=0)
    broker.process(play(0))
    broker.submit(order(0, "home", 5))
    assert broker.process(quote(10, 0.4, 0.5, game="OTHER")) == []
    [fill] = broker.process(quote(20, 0.4, 0.5))
    assert fill.ts == 20


def test_rejects_order_stamped_before_the_current_event() -> None:
    broker = SimBroker(latency_s=30)
    broker.process(play(100))
    with pytest.raises(ValueError, match="before"):
        broker.submit(order(50, "home", 5))


# --- prices, fees, settlement ------------------------------------------------------


def filled(broker: SimBroker, o: Order, bid: float, ask: float) -> Fill:
    broker.submit(o)
    [fill] = broker.process(quote(o.ts + 1, bid, ask, game=o.game_id))
    return fill


@pytest.mark.parametrize(
    ("team", "contracts", "price"),
    [("home", 10, 0.44), ("away", 10, 0.60), ("home", -10, 0.40), ("away", -10, 0.56)],
)
def test_fill_prices(team: Side, contracts: int, price: float) -> None:
    broker = SimBroker(latency_s=0)
    if contracts < 0:
        filled(broker, order(0, team, 10), 0.10, 0.20)
    fill = filled(broker, order(10, team, contracts), bid=0.40, ask=0.44)
    assert fill.price == pytest.approx(price)
    assert broker.positions.get(("G", team), 0) == 10 + min(contracts, 0)


def test_fee_charged_on_every_fill() -> None:
    broker = SimBroker(latency_s=0)
    entry = filled(broker, order(0, "home", 10), bid=0.40, ask=0.44)
    assert entry.fee == kalshi_taker_fee(0.44, 10) == 0.18  # 0.07*10*.44*.56 = .1725 -> .18
    assert broker.cash == pytest.approx(-(4.40 + 0.18))
    exit_ = filled(broker, order(10, "home", -10), bid=0.50, ask=0.52)
    assert exit_.fee == 0.18  # 0.07*10*.5*.5 = .175 -> .18
    assert broker.cash == pytest.approx(-4.58 + 5.00 - 0.18)


@pytest.mark.parametrize("home_won", [True, False])
def test_settlement_pays_the_winning_side(home_won: bool) -> None:
    broker = SimBroker(latency_s=0)
    filled(broker, order(0, "home", 10), bid=0.40, ask=0.44)
    filled(broker, order(10, "away", 3), bid=0.40, ask=0.44)  # away at .60, fee .06
    broker.process(Settle(100, "G", home_won))
    payout = 10.0 if home_won else 3.0
    assert broker.cash == pytest.approx(-(4.40 + 0.18) - (1.80 + 0.06) + payout)
    assert broker.positions == {}


def test_pending_orders_cancelled_at_settle() -> None:
    broker = SimBroker(latency_s=30)
    broker.process(play(100))
    late = order(100, "home", 10)
    broker.submit(late)
    broker.process(Settle(110, "G", True))
    assert broker.cancelled == [late]
    assert broker.process(quote(200, 0.4, 0.5)) == []
    assert broker.fills == [] and broker.cash == 0.0


def test_oversell_rejected() -> None:
    broker = SimBroker(latency_s=0)
    filled(broker, order(0, "home", 5), bid=0.40, ask=0.44)
    broker.submit(order(10, "home", -6))
    with pytest.raises(ValueError, match="sell 6"):
        broker.process(quote(11, 0.40, 0.44))


@pytest.mark.parametrize(("team", "bid", "ask"), [("home", 0.90, 1.0), ("away", 0.0, 0.05)])
def test_buy_at_price_outside_unit_interval_rejected(team: Side, bid: float, ask: float) -> None:
    broker = SimBroker(latency_s=0)
    broker.submit(order(0, team, 5))
    with pytest.raises(ValueError, match="price"):
        broker.process(quote(1, bid, ask))


# --- replay loop ------------------------------------------------------------------


def test_on_fill_delivered_before_next_on_event() -> None:
    trigger, fills_it, after = play(0), quote(60, 0.40, 0.44), play(70)
    events: list[Event] = [after, fills_it, trigger, Settle(100, "G", True)]  # unsorted
    strategy = Scripted({trigger: buy("home", 5)})
    run_backtest(strategy, [events], latency_s=0)
    kinds = [(kind, getattr(x, "ts", None)) for kind, x in strategy.log]
    assert kinds == [("event", 0), ("fill", 60), ("event", 60), ("event", 70), ("event", 100)]


def test_open_position_without_settlement_is_an_error() -> None:
    trigger = play(0)
    events: list[Event] = [trigger, quote(60, 0.40, 0.44)]
    with pytest.raises(ValueError, match="no settlement"):
        run_backtest(Scripted({trigger: buy("home", 5)}), [events], latency_s=0)


def test_end_to_end_pnl_by_hand() -> None:
    a_entry, a_exit = play(0, "A"), quote(120, 0.45, 0.47, "A")
    b_entry = play(0, "B")
    games: list[list[Event]] = [
        [
            a_entry,
            quote(60, 0.38, 0.40, "A"),  # buy 20 home at .40, fee .34 (0.336)
            a_exit,
            quote(180, 0.50, 0.52, "A"),  # sell 20 at bid .50, fee .35
            Settle(300, "A", False),
        ],
        [
            b_entry,
            quote(60, 0.70, 0.72, "B"),  # buy 10 away at 1 - .70 = .30, fee .15 (0.147)
            Settle(300, "B", False),  # away wins: +$10
        ],
        [play(0, "C"), quote(60, 0.5, 0.52, "C"), Settle(300, "C", True)],
    ]
    strategy = Scripted(
        {
            a_entry: buy("home", 20, timeout=True),
            a_exit: [("home", -20, "exit", {})],
            b_entry: buy("away", 10, timeout=False),
        }
    )
    result = run_backtest(strategy, games, latency_s=30)

    a_pnl = 20 * (0.50 - 0.40) - 0.34 - 0.35  # 1.31
    b_pnl = 10 - 3.00 - 0.15  # 6.85
    assert result.game_pnl.to_dict() == pytest.approx({"A": a_pnl, "B": b_pnl, "C": 0.0})
    assert result.broker.cash == pytest.approx(a_pnl + b_pnl)

    trades = result.trades.set_index("game_id")
    assert trades["exit_reason"].to_dict() == {"A": "exit", "B": "settle"}
    assert trades["reason"].to_dict() == {"A": "entry", "B": "entry"}
    assert trades["timeout"].to_dict() == {"A": True, "B": False}
    assert trades["staked"].to_dict() == pytest.approx({"A": 8.0, "B": 3.0})
    assert trades.loc["A", ["entry_ts", "exit_ts"]].tolist() == [60, 180]

    s = summary(result.trades, result.games, seed=0)
    assert (s["games"], s["trades"], s["win_rate"]) == (3, 2, 1.0)
    assert s["total_pnl"] == pytest.approx(8.16)
    assert s["pnl_per_trade"] == pytest.approx(4.08)
    assert s["roi"] == pytest.approx(8.16 / 11.0)

    by = summary_by(result.trades, result.games, "timeout", seed=0)
    assert by.loc[True, "total_pnl"] == pytest.approx(a_pnl)
    assert by.loc[False, "total_pnl"] == pytest.approx(b_pnl)
    assert (by["games"] == 3).all()


def test_round_trip_spans_adds_and_partial_exits() -> None:
    broker_events: list[Event] = [
        e1 := play(0),
        e2 := quote(10, 0.40, 0.44),  # fills 10 at .44; decide to add 5
        e3 := quote(20, 0.40, 0.50),  # fills 5 at .50; decide to sell 8
        quote(30, 0.60, 0.62),  # sells 8 at .60; 7 left to settle
        Settle(40, "G", True),
    ]
    strategy = Scripted(
        {e1: buy("home", 10), e2: buy("home", 5, "add"), e3: [("home", -8, "trim", {})]}
    )
    result = run_backtest(strategy, [broker_events], latency_s=0)
    [trip] = result.trades.to_dict("records")
    assert trip["reason"] == "entry" and trip["exit_reason"] == "settle"
    assert trip["contracts"] == 15
    assert trip["staked"] == pytest.approx(4.40 + 2.50)
    assert trip["pnl"] == pytest.approx(result.broker.cash)


# --- bootstrap ----------------------------------------------------------------------


def trades_frame(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": [f"g{i}" for i in range(len(pnls))], "pnl": pnls, "staked": 1.0}
    )


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    pnls = [((i * 37) % 11) - 5.0 for i in range(60)]
    trades = trades_frame(pnls)
    games = pd.Index([f"g{i}" for i in range(80)])
    first, again = summary(trades, games, seed=7), summary(trades, games, seed=7)
    assert (first["ci_low"], first["ci_high"]) == (again["ci_low"], again["ci_high"])
    assert first["ci_low"] < first["total_pnl"] < first["ci_high"]
    other = summary(trades, games, seed=8)
    assert (other["ci_low"], other["ci_high"]) != (first["ci_low"], first["ci_high"])


def test_bootstrap_counts_games_without_trades() -> None:
    games = pd.Index([f"g{i}" for i in range(100)])
    s = summary(trades_frame([10.0]), games, seed=0)
    # One winning game in 100: about a third of resamples miss it entirely.
    assert s["games"] == 100
    assert s["ci_low"] == 0.0 and s["ci_high"] >= 10.0
