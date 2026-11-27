"""ESPN play-by-play: the raw game summary, cached, and parsed into ``Play`` events;
and, for live trading, the day's scoreboard and uncached summaries.

ESPN lists plays in game-clock order, but within one stoppage (fouls, free throws,
subs, reviews) that order is often wrong, and plays entered late (a correction, an
and-one free throw) carry stale ``homeScore``/``awayScore`` for many plays after.
``wallclock`` gets the real order, so plays are put in replay order (``sort_key``:
wallclock, then sequenceNumber) and the score is re-accumulated from ``scoreValue``
in that order. The accumulated final must equal ESPN's final score, or the game is
rejected (history only: live, the header score runs ahead of the plays fetched).
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from nba_winprob.engine.events import Play, PlayKind, Side
from nba_winprob.sources.http import cached_json, get_json

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
MAX_WALLCLOCK_JUMP = 3000  # seconds; the longest real gap seen is ~45 minutes
# ESPN season types kept: 2 regular season, 3 playoffs, 5 play-in. Preseason and All-Star are out.
SEASON_TYPES = {2, 3, 5}


@dataclass(frozen=True, slots=True)
class Game:
    """One game on ESPN's scoreboard."""

    game_id: str
    tip_utc: str  # ISO 8601, e.g. "2026-11-28T00:00Z"
    home: str  # ESPN abbreviation
    away: str
    state: Literal["pre", "in", "post"]
    home_won: bool | None  # set once the game is final; None before, or if never played


def parse_scoreboard(response: dict[str, Any]) -> list[Game]:
    """A scoreboard response's regular season, play-in and playoff games."""
    games = []
    for event in response.get("events", []):
        if event["season"]["type"] not in SEASON_TYPES:
            continue
        comp = event["competitions"][0]
        status = comp["status"]["type"]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        games.append(
            Game(
                game_id=event["id"],
                tip_utc=event["date"],
                home=sides["home"]["team"]["abbreviation"],
                away=sides["away"]["team"]["abbreviation"],
                state=status["state"],
                # A postponed game is "post" but not completed, and has no winner.
                home_won=bool(sides["home"]["winner"]) if status["completed"] else None,
            )
        )
    return games


def scoreboard(day: dt.date) -> list[Game]:
    """The day's games, fetched now (uncached: status changes)."""
    return parse_scoreboard(get_json(f"{ESPN}/scoreboard?dates={day:%Y%m%d}"))


def live_summary(game_id: str) -> dict[str, Any]:
    """A game's summary fetched now, uncached: in progress it changes every play."""
    return get_json(f"{ESPN}/summary?event={game_id}")


def summary(game_id: str, cache_dir: Path) -> dict[str, Any]:
    """ESPN's full game summary (plays, box score, header), cached raw per game."""
    return cached_json(
        cache_dir / f"{game_id}.json.gz", f"{ESPN}/summary?event={game_id}", key="plays"
    )


def parse_clock(display: str) -> float:
    """Seconds left in the period: ``"5:22"`` -> 322.0, ``"38.2"`` -> 38.2."""
    minutes, _, seconds = display.rpartition(":")
    return int(minutes or 0) * 60 + float(seconds)


def _kind(play: dict[str, Any]) -> PlayKind:
    text = play["type"]["text"]
    if play["scoringPlay"]:
        return "score"
    # Only real timeouts: a coach's challenge is "other" even when it costs one.
    if "Timeout" in text:
        return "timeout"
    if text == "End Period":
        return "end_period"
    if text == "End Game":
        return "end_game"
    return "other"


def parse_plays(summary: dict[str, Any], game_id: str, check_final: bool = True) -> list[Play]:
    """Every play as a ``Play``, in replay order, scores accumulated in that order.

    ``check_final=False`` is for games in progress, whose header score can be ahead of
    the plays in the same response.
    """
    competitors = summary["header"]["competitions"][0]["competitors"]
    sides: dict[str, Side] = {c["id"]: c["homeAway"] for c in competitors}
    final = {c["homeAway"]: int(c["score"]) for c in competitors}
    tip = datetime.fromisoformat(summary["header"]["competitions"][0]["date"]).timestamp()

    stamped = [
        (datetime.fromisoformat(p["wallclock"]).timestamp(), int(p["sequenceNumber"]), p)
        for p in summary["plays"]
    ]
    # Some games have wallclocks shifted by whole hours or a day (a date-rollover bug,
    # DST changeovers), for the whole game or from some play on. Real stoppages and
    # misordered plays stay well under these bounds.
    if not tip - 600 <= stamped[0][0] <= tip + MAX_WALLCLOCK_JUMP:
        raise ValueError(f"game {game_id}: first play {stamped[0][0] - tip:+.0f}s from tip")
    for (before, _, _), (after, seq, _) in itertools.pairwise(stamped):
        if abs(after - before) > MAX_WALLCLOCK_JUMP:
            raise ValueError(f"game {game_id} play {seq}: wallclock jumps {after - before:+.0f}s")
    raw = sorted(stamped, key=lambda t: t[:2])
    plays, score = [], {"home": 0, "away": 0}
    for ts, seq, p in raw:
        kind, team = _kind(p), sides.get(p.get("team", {}).get("id"))
        points = p["scoreValue"] if kind == "score" else 0
        if points:
            if team is None:
                raise ValueError(f"game {game_id} play {seq}: points with no team")
            score[team] += points
        plays.append(
            Play(
                ts=ts,
                game_id=game_id,
                seq=seq,
                period=p["period"]["number"],
                clock=parse_clock(p["clock"]["displayValue"]),
                home_score=score["home"],
                away_score=score["away"],
                team=team,
                kind=kind,
                points=points,
                text=p.get("text", p["type"]["text"]),  # ejections can lack text
            )
        )
    if check_final and score != final:
        raise ValueError(f"game {game_id}: plays sum to {score}, ESPN final is {final}")
    return plays
