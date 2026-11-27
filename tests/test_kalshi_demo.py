"""Kalshi demo client and broker against a fake transport: no network, no real key."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from nba_winprob.brokers.demo import DemoBroker
from nba_winprob.brokers.kalshi_client import DEMO_BASE_URL, KalshiClient, KalshiError, sign
from nba_winprob.config import Settings
from nba_winprob.engine.events import Order, Play, Quote, Settle, Side, kalshi_taker_fee

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
NOW = 1_764_000_000.0


class FakeExchange:
    """Records every request and answers like the demo API would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.fills: list[dict[str, Any]] = []
        self.order_error: str | None = None
        self.fill_nothing = False  # the IOC order finds no liquidity
        self.get_error: str | None = None  # GETs fail, e.g. the network drops

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> Any:
        self.calls.append((method, url, dict(headers), json.loads(body) if body else None))
        if method == "POST":
            if self.order_error:
                raise KalshiError(self.order_error)
            filled = "0.00" if self.fill_nothing else self.orders[-1].get("count", "1")
            return {"order_id": f"oid{len(self.orders) - 1}", "fill_count": filled}
        if self.get_error:
            raise KalshiError(self.get_error)
        return {"fills": self.fills, "cursor": ""}

    @property
    def orders(self) -> list[dict[str, Any]]:
        return [body for method, _, _, body in self.calls if method == "POST"]

    @property
    def fill_polls(self) -> int:
        return sum(1 for method, url, _, _ in self.calls if "/portfolio/fills" in url)


class Clock:
    def __init__(self) -> None:
        self.t = NOW

    def __call__(self) -> float:
        return self.t


def client(exchange: FakeExchange, key: Any = RSA_KEY) -> KalshiClient:
    return KalshiClient("key-id", key, transport=exchange, clock=lambda: NOW)


def quote(bid: float = 0.40, ask: float = 0.45, game: str = "G") -> Quote:
    return Quote(NOW, game, f"KXNBAGAME-{game}", bid, ask, 0.0)


def order(team: Side, contracts: int, game: str = "G") -> Order:
    return Order(NOW, game, team, contracts, "test")


def raw_fill(oid: str, count: str, yes: str, fee: str | None = "0.02", fid: str = "f0") -> Any:
    fill = {
        "fill_id": fid,
        "order_id": oid,
        "count_fp": count,
        "yes_price_dollars": yes,
        "no_price_dollars": f"{1 - float(yes):.4f}",
        "ts": int(NOW) + 1,
    }
    if fee is not None:
        fill["fee_cost"] = fee
    return fill


# --- signing -------------------------------------------------------------------------------


def test_rsa_signature_verifies_over_timestamp_method_path_without_query() -> None:
    exchange = FakeExchange()
    client(exchange).request("GET", "/portfolio/fills", {"limit": 5})
    _, url, headers, _ = exchange.calls[0]
    assert url == f"{DEMO_BASE_URL}/portfolio/fills?limit=5"
    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(int(NOW * 1000))  # milliseconds
    message = f"{int(NOW * 1000)}GET/trade-api/v2/portfolio/fills".encode()
    # Salt length pinned to the digest length (32), as Kalshi documents; raises if wrong.
    RSA_KEY.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )


def test_sign_strips_the_query_string() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    signature = sign(key, "1700000000000", "GET", "/trade-api/v2/portfolio/fills?limit=5")
    message = b"1700000000000GET/trade-api/v2/portfolio/fills"
    key.public_key().verify(base64.b64decode(signature), message)


def test_ed25519_signature_verifies() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    exchange = FakeExchange()
    client(exchange, key).request("POST", "/portfolio/events/orders", body={"x": 1})
    _, _, headers, body = exchange.calls[0]
    assert body == {"x": 1}
    message = f"{int(NOW * 1000)}POST/trade-api/v2/portfolio/events/orders".encode()
    key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message)


# --- host guard ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://external-api.kalshi.com/trade-api/v2",
        "https://api.elections.kalshi.com/trade-api/v2",
        "https://demo-api.kalshi.co.evil.com/trade-api/v2",
    ],
)
def test_host_guard_refuses_non_demo_hosts(url: str) -> None:
    with pytest.raises(ValueError, match="non-demo"):
        KalshiClient("k", RSA_KEY, url, transport=FakeExchange())


def test_host_guard_accepts_both_demo_hosts_and_needs_explicit_opt_in_for_live() -> None:
    KalshiClient("k", RSA_KEY, "https://demo-api.kalshi.co/trade-api/v2")
    KalshiClient("k", RSA_KEY, DEMO_BASE_URL)
    live = "https://external-api.kalshi.com/trade-api/v2"
    assert KalshiClient("k", RSA_KEY, live, allow_live=True).base_url == live


def test_demo_keys_come_from_unprefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_DEMO_KEY_ID", "abc")
    monkeypatch.setenv("KALSHI_DEMO_PRIVATE_KEY_PATH", "/keys/demo.pem")
    settings = Settings(_env_file=None)
    assert settings.kalshi_demo_key_id == "abc"
    assert str(settings.kalshi_demo_private_key_path) == "/keys/demo.pem"


# --- orders --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("team", "contracts", "side", "price", "reduce_only"),
    [
        ("home", 10, "bid", "0.4500", False),  # buy YES at the ask
        ("away", 10, "ask", "0.4000", False),  # buy NO at 1 - bid = 0.60
        ("home", -10, "ask", "0.4000", True),  # sell YES at the bid
        ("away", -10, "bid", "0.4500", True),  # sell NO at 1 - ask = 0.55
    ],
)
def test_order_body(team: Side, contracts: int, side: str, price: str, reduce_only: bool) -> None:
    exchange = FakeExchange()
    broker = DemoBroker(client(exchange), poll_every_s=5)
    broker.process(quote(0.40, 0.45))
    broker.submit(order(team, contracts))
    body = exchange.orders[0]
    assert exchange.calls[0][1] == f"{DEMO_BASE_URL}/portfolio/events/orders"
    assert body["ticker"] == "KXNBAGAME-G"
    assert (body["side"], body["price"], body["count"]) == (side, price, "10")
    assert body["time_in_force"] == "immediate_or_cancel"
    assert body["reduce_only"] is reduce_only
    assert broker.rejected == []


def test_rejects_without_a_quote_for_that_game() -> None:
    exchange = FakeExchange()
    broker = DemoBroker(client(exchange), poll_every_s=5)
    broker.process(quote(game="OTHER"))
    with pytest.raises(ValueError, match="no quote yet"):
        broker.submit(order("home", 5))
    assert exchange.calls == []
    assert broker.rejected == [(order("home", 5), "no quote yet")]


def test_exchange_error_is_a_rejection() -> None:
    exchange = FakeExchange()
    exchange.order_error = "HTTP 400 invalid_order"
    broker = DemoBroker(client(exchange), poll_every_s=5)
    broker.process(quote())
    with pytest.raises(ValueError, match="invalid_order"):  # the runner records it rejected
        broker.submit(order("home", 5))
    assert broker.rejected == [(order("home", 5), "HTTP 400 invalid_order")]


def test_an_ioc_order_that_fills_nothing_is_a_rejection() -> None:
    exchange = FakeExchange()
    exchange.fill_nothing = True
    broker = DemoBroker(client(exchange), poll_every_s=5)
    broker.process(quote())
    with pytest.raises(ValueError, match="filled nothing"):
        broker.submit(order("home", 5))


def test_a_failed_fills_poll_is_retried_not_raised() -> None:
    exchange, clock = FakeExchange(), Clock()
    broker = DemoBroker(client(exchange), poll_every_s=5, clock=clock)
    broker.process(quote())
    broker.submit(order("home", 10))
    exchange.get_error = "GET fills: URLError timed out"
    assert broker.process(quote()) == []
    exchange.get_error = None
    exchange.fills = [raw_fill("oid0", "10.00", "0.4500")]
    clock.t += 5
    assert [f.order.contracts for f in broker.process(quote())] == [10]


def test_transport_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request

    from nba_winprob.brokers.kalshi_client import urllib_transport

    def down(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    with pytest.raises(KalshiError, match="URLError"):
        urllib_transport("GET", f"{DEMO_BASE_URL}/portfolio/fills", {}, None)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(KalshiError, match="TimeoutError"):
        urllib_transport("GET", f"{DEMO_BASE_URL}/portfolio/fills", {}, None)


def test_client_order_ids_are_unique() -> None:
    exchange = FakeExchange()
    broker = DemoBroker(client(exchange), poll_every_s=5)
    broker.process(quote())
    for _ in range(20):
        broker.submit(order("home", 1))
    ids = [body["client_order_id"] for body in exchange.orders]
    assert len(set(ids)) == 20 and all(ids)


# --- fills ---------------------------------------------------------------------------------


def test_fills_polled_at_most_every_n_seconds_and_reported_once() -> None:
    exchange, clock = FakeExchange(), Clock()
    broker = DemoBroker(client(exchange), poll_every_s=5, clock=clock)
    broker.process(quote())
    assert exchange.fill_polls == 0  # nothing submitted, nothing to poll for
    broker.submit(order("home", 10))
    exchange.fills = [raw_fill("oid0", "10.00", "0.4500")]
    assert len(broker.process(quote())) == 1
    clock.t += 4.9
    assert broker.process(quote()) == []
    assert exchange.fill_polls == 1
    clock.t += 0.1
    assert broker.process(quote()) == []  # polled again, same fill not reported twice
    assert exchange.fill_polls == 2
    assert f"min_ts={int(NOW)}" in exchange.calls[-1][1]


def test_fills_converted_per_side_with_exchange_or_computed_fee() -> None:
    exchange = FakeExchange()
    broker = DemoBroker(client(exchange), poll_every_s=0)
    broker.process(quote())
    broker.submit(order("home", 10))  # oid0
    broker.submit(order("away", 8))  # oid1
    exchange.fills = [
        raw_fill("oid0", "6.00", "0.4500", fee="0.05", fid="a"),  # partial IOC fill
        raw_fill("oid1", "8.00", "0.4000", fee=None, fid="b"),
        raw_fill("someone-else", "3.00", "0.5000", fid="c"),  # not our order: ignored
    ]
    home, away = broker.process(Play(NOW, "G", 1, 1, 600.0, 0, 0, None, "other", 0, ""))
    assert (home.order.contracts, home.price, home.fee, home.ts) == (6, 0.45, 0.05, NOW + 1)
    assert (away.order.contracts, away.price) == (8, 0.60)  # NO price
    assert away.fee == kalshi_taker_fee(0.60, 8)
    assert broker.positions == {("G", "home"): 6, ("G", "away"): 8}

    broker.submit(order("home", -6))  # oid2
    exchange.fills.append(raw_fill("oid2", "6.00", "0.4000", fid="d"))
    (sold,) = broker.process(quote())
    assert (sold.order.contracts, sold.price) == (-6, 0.40)
    assert broker.positions == {("G", "away"): 8}


def test_settle_clears_that_games_positions_only() -> None:
    exchange = FakeExchange()
    broker = DemoBroker(client(exchange), poll_every_s=0)
    broker.process(quote(game="G"))
    broker.process(quote(game="H"))
    broker.submit(order("home", 4, "G"))
    broker.submit(order("away", 2, "H"))
    exchange.fills = [
        raw_fill("oid0", "4.00", "0.45", fid="a"),
        raw_fill("oid1", "2.00", "0.4", fid="b"),
    ]
    broker.process(quote(game="G"))
    broker.process(Settle(NOW + 10, "G", home_won=True))
    assert broker.positions == {("H", "away"): 2}


def test_get_fills_follows_the_cursor() -> None:
    pages = iter(
        [{"fills": [{"fill_id": "1"}], "cursor": "c1"}, {"fills": [{"fill_id": "2"}], "cursor": ""}]
    )
    urls: list[str] = []

    def transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> Any:
        urls.append(url)
        return next(pages)

    fills = KalshiClient("k", RSA_KEY, transport=transport).get_fills(min_ts=7)
    assert [f["fill_id"] for f in fills] == ["1", "2"]
    assert "cursor=c1" in urls[1] and "min_ts=7" in urls[1]
