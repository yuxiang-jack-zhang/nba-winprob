"""Authenticated Kalshi Trade API client: signs requests, places and reads orders and fills.

Signing (docs.kalshi.com/getting_started/api_keys): headers ``KALSHI-ACCESS-KEY`` (key id),
``KALSHI-ACCESS-TIMESTAMP`` (unix ms) and ``KALSHI-ACCESS-SIGNATURE``, the base64 signature of
``timestamp + METHOD + path``, where path is the full path from ``/trade-api/v2`` without the
query string. Ed25519 keys sign the text directly; RSA keys use RSA-PSS, SHA-256, MGF1-SHA-256,
salt length = digest length.

Orders use the V2 endpoints (the legacy ``/portfolio/orders`` writes now return 410): one YES
book, ``side`` ``bid`` = buy YES and ``ask`` = sell YES (= buy NO at ``1 - price``), prices and
counts as fixed-point strings in dollars and contracts.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from nba_winprob.config import Settings

DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"
# Both demo hosts Kalshi documents (docs.kalshi.com/getting_started/api_environments).
DEMO_HOSTS = frozenset({"external-api.demo.kalshi.co", "demo-api.kalshi.co"})

BookSide = Literal["bid", "ask"]
PrivateKey = RSAPrivateKey | Ed25519PrivateKey
# (method, url, headers, body) -> decoded JSON response
Transport = Callable[[str, str, Mapping[str, str], bytes | None], Any]


class KalshiError(RuntimeError):
    """The exchange answered with an HTTP error; the message carries its status and body."""


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> Any:
    # No retries: a resent POST could place a second order once the first has closed.
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        raise KalshiError(f"{method} {url}: HTTP {err.code} {err.read().decode()[:500]}") from err


def sign(key: PrivateKey, timestamp_ms: str, method: str, path: str) -> str:
    """Base64 signature of ``timestamp_ms + method + path`` (path without its query string)."""
    message = (timestamp_ms + method + path.split("?")[0]).encode()
    if isinstance(key, Ed25519PrivateKey):
        signature = key.sign(message)
    else:
        pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)
        signature = key.sign(message, pss, hashes.SHA256())
    return base64.b64encode(signature).decode()


def load_private_key(path: Path) -> PrivateKey:
    key = serialization.load_pem_private_key(path.expanduser().read_bytes(), password=None)
    if not isinstance(key, RSAPrivateKey | Ed25519PrivateKey):
        raise ValueError(f"{path}: Kalshi keys are RSA or Ed25519, got {type(key).__name__}")
    return key


class KalshiClient:
    """A signed-request client. Refuses any host but Kalshi's demo unless ``allow_live``."""

    def __init__(
        self,
        key_id: str,
        private_key: PrivateKey,
        base_url: str = DEMO_BASE_URL,
        *,
        allow_live: bool = False,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if urllib.parse.urlsplit(base_url).hostname not in DEMO_HOSTS and not allow_live:
            raise ValueError(f"refusing non-demo Kalshi host {base_url!r} without allow_live")
        self.key_id, self.private_key = key_id, private_key
        self.base_url = base_url.rstrip("/")
        self.transport, self.clock = transport, clock

    @classmethod
    def demo(cls, settings: Settings) -> KalshiClient:
        """The demo client, keyed by ``KALSHI_DEMO_KEY_ID`` / ``KALSHI_DEMO_PRIVATE_KEY_PATH``."""
        if not settings.kalshi_demo_key_id or settings.kalshi_demo_private_key_path is None:
            raise ValueError("set KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH in .env")
        key = load_private_key(settings.kalshi_demo_private_key_path)
        return cls(settings.kalshi_demo_key_id, key)

    def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, str | int] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        """``method`` ``path`` (relative to the base URL, e.g. ``/portfolio/fills``), signed."""
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        timestamp = str(int(self.clock() * 1000))
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": sign(
                self.private_key, timestamp, method, urllib.parse.urlsplit(url).path
            ),
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode() if body is not None else None
        return self.transport(method, url, headers, data)

    def create_order(
        self,
        ticker: str,
        side: BookSide,
        count: int,
        price: float,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Immediate-or-cancel limit order on the YES book; returns ``order_id``, fill counts."""
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "count": str(count),
            "price": f"{price:.4f}",
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
            "reduce_only": reduce_only,  # only valid with IOC, which this always is
        }
        return self.request("POST", "/portfolio/events/orders", body=body)

    def get_fills(self, min_ts: int | None = None) -> list[dict[str, Any]]:
        """Every fill (after unix second ``min_ts``, if given), following the cursor to the end."""
        fills: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, str | int] = {"limit": 1000}
            if min_ts is not None:
                params["min_ts"] = min_ts
            if cursor:
                params["cursor"] = cursor
            page = self.request("GET", "/portfolio/fills", params)
            fills += page["fills"]
            cursor = page.get("cursor") or ""
            if not cursor or not page["fills"]:
                return fills

    def get_order(self, order_id: str) -> dict[str, Any]:
        """The order; ``status`` is ``resting``, ``canceled`` or ``executed``."""
        return self.request("GET", f"/portfolio/orders/{order_id}")["order"]

    def cancel_order(self, order_id: str, ticker: str) -> dict[str, Any]:
        # market_ticker routes the cancel to the order's shard; without it a cancel can 404.
        path = f"/portfolio/events/orders/{order_id}"
        return self.request("DELETE", path, {"market_ticker": ticker})
