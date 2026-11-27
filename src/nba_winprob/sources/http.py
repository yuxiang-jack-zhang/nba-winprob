"""GET a public JSON endpoint, with retries. No auth: market data and ESPN only."""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def get_json(url: str, tries: int = 4) -> Any:
    """Fetch and decode ``url``; retries transient errors, raises a 404 at once."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return json.loads(raw)
        except urllib.error.HTTPError as err:
            if err.code == 404 or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def cached_json(path: Path, url: str, key: str) -> Any:
    """``url``'s JSON, cached gzipped at ``path``.

    A response whose ``key`` is missing or empty is neither cached nor trusted from
    the cache: it raises, so the next run fetches it again.
    """
    if path.exists():
        data = json.loads(gzip.decompress(path.read_bytes()))
        if data.get(key):
            return data
    data = get_json(url)
    if not data.get(key):
        raise ValueError(f"no {key} at {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # rename is atomic: an interrupted run leaves no half file
    tmp.write_bytes(gzip.compress(json.dumps(data).encode()))
    tmp.replace(path)
    return data
