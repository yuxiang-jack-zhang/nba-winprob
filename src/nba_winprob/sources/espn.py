"""ESPN play-by-play: the raw game summary, cached, and parsed into ``Play`` events.

ESPN lists plays in game-clock order, but within one stoppage (fouls, free throws,
subs, reviews) that order is often wrong, and plays entered late (a correction, an
and-one free throw) carry stale ``homeScore``/``awayScore`` for many plays after.
``wallclock`` gets the real order, so plays are put in replay order (``sort_key``:
wallclock, then sequenceNumber) and the score is re-accumulated from ``scoreValue``
in that order. The accumulated final must equal ESPN's final score, or the game is
rejected.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

from nba_winprob.engine.events import Play, PlayKind, Side
from nba_winprob.sources.http import cached_json

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
MAX_WALLCLOCK_JUMP = 3000  # seconds; the longest real gap seen is ~45 minutes


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


def parse_plays(summary: dict[str, Any], game_id: str) -> list[Play]:
    """Every play as a ``Play``, in replay order, scores accumulated in that order."""
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
    if score != final:
        raise ValueError(f"game {game_id}: plays sum to {score}, ESPN final is {final}")
    return plays
