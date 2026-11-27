"""Download what the Kalshi paper-trade backtest needs, into data/papertrade/ (gitignored).

    python papertrade/fetch.py espn     # games, box scores, shot profiles, 2021-22 -> today
    python papertrade/fetch.py kalshi   # settled NBA game markets + the last pre-tip quote

Both are resumable: every game is cached as one small JSON file, so a re-run only
fetches what is missing. Public endpoints only; no account, no orders.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "papertrade"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FIRST_DAY = dt.date(2021, 10, 1)
# ESPN season types kept: 2 regular season, 3 playoffs, 5 play-in. Preseason and All-Star are out.
SEASON_TYPES = {2, 3, 5}
# ESPN -> Kalshi team codes, where they differ.
ESPN_TO_KALSHI = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}


def get(url: str, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return json.loads(raw)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- ESPN


def scoreboard_day(day: dt.date) -> list[dict]:
    path = ROOT / "days" / f"{day:%Y%m%d}.json"
    # Today's and yesterday's slates can still change; older days are final.
    if path.exists() and day < dt.date.today() - dt.timedelta(days=1):
        return json.loads(path.read_text())
    games = []
    for event in get(f"{ESPN}/scoreboard?dates={day:%Y%m%d}").get("events", []):
        comp = event["competitions"][0]
        if event["season"]["type"] not in SEASON_TYPES:
            continue
        if not comp["status"]["type"].get("completed"):
            continue
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        games.append(
            {
                "game_id": event["id"],
                "tip_utc": event["date"],
                "season": event["season"]["year"],
                "season_type": event["season"]["type"],
                "home": sides["home"]["team"]["abbreviation"],
                "away": sides["away"]["team"]["abbreviation"],
                "home_id": sides["home"]["team"]["id"],
                "away_id": sides["away"]["team"]["id"],
                "home_pts": int(sides["home"]["score"]),
                "away_pts": int(sides["away"]["score"]),
                "note": (comp.get("notes") or [{}])[0].get("headline", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(games))
    return games


def shot_profile(plays: list[dict]) -> dict[str, dict[str, float]]:
    """Per-team field-goal profile, the inputs to the project's shot features.

    ESPN gives x in feet across the court (hoop at 25) and y in feet out from the
    hoop. Zones are by distance, not the NBA's named zones the Kaggle data used, so
    these are the same features in spirit, not value-for-value.
    """
    out: dict[str, dict[str, float]] = {}
    for play in plays:
        if not play.get("shootingPlay") or "Free Throw" in play.get("type", {}).get("text", ""):
            continue
        coord, team = play.get("coordinate") or {}, play.get("team", {}).get("id")
        x, y = coord.get("x"), coord.get("y")
        if team is None or x is None or y is None or abs(x) > 60 or abs(y) > 100:
            continue
        dist = math.hypot(x - 25, y)
        is_three = "three point" in play.get("text", "").lower()
        row = out.setdefault(
            team,
            dict.fromkeys(("n", "made", "sum_x", "sum_y", "sum_dist", "paint", "mid", "long"), 0.0),
        )
        row["n"] += 1
        row["made"] += bool(play.get("scoringPlay"))
        row["sum_x"] += x
        row["sum_y"] += y
        row["sum_dist"] += dist
        row["long" if is_three else "paint" if dist < 8 else "mid"] += 1
    return out


def box_score(game_id: str) -> dict:
    path = ROOT / "games" / f"{game_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    summary = get(f"{ESPN}/summary?event={game_id}")
    players = []
    for team in summary.get("boxscore", {}).get("players", []):
        for group in team.get("statistics", []):
            minutes_idx = group["labels"].index("MIN") if "MIN" in group.get("labels", []) else 0
            for ath in group.get("athletes", []):
                stats = ath.get("stats") or []
                raw = stats[minutes_idx] if len(stats) > minutes_idx else "0"
                players.append(
                    {
                        "team_id": team["team"]["id"],
                        "player_id": ath["athlete"]["id"],
                        "name": ath["athlete"]["displayName"],
                        "minutes": 0 if ath.get("didNotPlay") or not raw.isdigit() else int(raw),
                    }
                )
    record = {
        "game_id": game_id,
        "players": players,
        "shots": shot_profile(summary.get("plays", [])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


def fetch_espn() -> None:
    days = [FIRST_DAY + dt.timedelta(d) for d in range((dt.date.today() - FIRST_DAY).days + 1)]
    days = [d for d in days if d.month not in (7, 8, 9)]
    with ThreadPoolExecutor(8) as pool:
        games = [g for day_games in pool.map(scoreboard_day, days) for g in day_games]
    print(f"{len(games)} completed games", flush=True)

    def safe_box(game: dict) -> dict | None:
        try:
            return box_score(game["game_id"])
        except Exception as err:  # one bad game must not lose the other 7,000
            print(f"  skip {game['game_id']}: {err}", flush=True)
            return None

    with ThreadPoolExecutor(8) as pool:
        boxes = [b for b in pool.map(safe_box, games) if b]
    print(f"{len(boxes)} box scores", flush=True)

    pd.DataFrame(games).to_parquet(ROOT / "games.parquet", index=False)
    pd.DataFrame([{"game_id": b["game_id"], **p} for b in boxes for p in b["players"]]).to_parquet(
        ROOT / "players.parquet", index=False
    )
    pd.DataFrame(
        [{"game_id": b["game_id"], "team_id": t, **s} for b in boxes for t, s in b["shots"].items()]
    ).to_parquet(ROOT / "shots.parquet", index=False)


# ----------------------------------------------------------------------------- Kalshi


def parse_ticker(ticker: str) -> tuple[dt.date, str, str, str]:
    """``KXNBAGAME-26JUN13NYKSAS-SAS`` -> (2026-06-13, away "NYK", home "SAS", yes team "SAS")."""
    _, middle, yes_team = ticker.split("-")
    day = dt.datetime.strptime(middle[:7], "%y%b%d").date()
    matchup = middle[7:]
    if matchup.startswith(yes_team):
        away, home = yes_team, matchup[len(yes_team) :]
    elif matchup.endswith(yes_team):
        away, home = matchup[: -len(yes_team)], yes_team
    else:
        raise ValueError(f"cannot split {ticker}")
    return day, away, home, yes_team


def settled_markets() -> pd.DataFrame:
    rows = []
    for path, historical in (("/historical/markets?", True), ("/markets?status=settled&", False)):
        cursor = ""
        while True:
            page = get(f"{KALSHI}{path}series_ticker=KXNBAGAME&limit=1000&cursor={cursor}")
            for m in page.get("markets", []):
                if m.get("result") not in ("yes", "no"):
                    continue
                try:
                    day, away, home, yes_team = parse_ticker(m["ticker"])
                except ValueError:
                    continue
                if yes_team != home:  # one market per game: the home team's
                    continue
                rows.append(
                    {
                        "ticker": m["ticker"],
                        "day": day,
                        "away_k": away,
                        "home_k": home,
                        "home_won": m["result"] == "yes",
                        "historical": historical,
                    }
                )
            cursor = page.get("cursor") or ""
            if not cursor or not page.get("markets"):
                break
    return pd.DataFrame(rows).drop_duplicates("ticker")


def pretip_quote(ticker: str, historical: bool, tip_ts: int) -> dict:
    """The last one-minute quote at or before tip-off, for the home team's YES side."""
    path = ROOT / "quotes" / f"{ticker}.json"
    if path.exists() and (cached := json.loads(path.read_text())).get("bid") is not None:
        return cached  # an empty cached quote is retried, not trusted
    base = f"/historical/markets/{ticker}" if historical else f"/series/KXNBAGAME/markets/{ticker}"
    url = f"{KALSHI}{base}/candlesticks?start_ts={tip_ts - 5400}&end_ts={tip_ts}&period_interval=1"
    try:
        candles = get(url).get("candlesticks", [])
    except urllib.error.HTTPError:
        candles = []
    quote: dict = {"ticker": ticker, "bid": None, "ask": None}

    def price(side: dict) -> float | None:
        value = side.get("close", side.get("close_dollars"))
        return float(value) if value is not None else None

    for candle in reversed(candles):
        bid, ask = price(candle.get("yes_bid", {})), price(candle.get("yes_ask", {}))
        if bid is not None and ask is not None and 0 < bid <= ask < 1:
            quote = {"ticker": ticker, "bid": bid, "ask": ask, "quote_ts": candle["end_period_ts"]}
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quote))
    return quote


def fetch_kalshi() -> None:
    markets = settled_markets()
    games = pd.read_parquet(ROOT / "games.parquet")
    tip = pd.to_datetime(games["tip_utc"], utc=True)
    games["day"] = tip.dt.tz_convert("America/New_York").dt.date
    games["tip_ts"] = (tip - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")  # unit-independent
    games["home_k"] = games["home"].replace(ESPN_TO_KALSHI)
    games["away_k"] = games["away"].replace(ESPN_TO_KALSHI)

    merged = markets.merge(
        games[["game_id", "day", "home_k", "away_k", "tip_ts"]], on=["day", "home_k", "away_k"]
    )
    print(f"{len(markets)} settled Kalshi games, {len(merged)} matched to ESPN", flush=True)

    with ThreadPoolExecutor(6) as pool:
        quotes = list(
            pool.map(lambda r: pretip_quote(r.ticker, r.historical, r.tip_ts), merged.itertuples())
        )
    out = merged.merge(pd.DataFrame(quotes), on="ticker").dropna(subset=["bid", "ask"])
    print(f"{len(out)} with a usable pre-tip quote", flush=True)
    out.to_parquet(ROOT / "kalshi.parquet", index=False)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    {"espn": fetch_espn, "kalshi": fetch_kalshi}[sys.argv[1]]()
