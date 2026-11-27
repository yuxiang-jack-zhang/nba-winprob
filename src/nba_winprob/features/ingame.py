"""In-game features from the play stream: game clock, trailing scoring runs, timeouts."""

from __future__ import annotations

from collections import deque

from nba_winprob.engine.events import Play, Side

REGULATION_PERIODS = 4
PERIOD_S = 720
OVERTIME_S = 300


def game_seconds_elapsed(period: int, clock: float) -> float:
    """Game seconds since tip-off, continuous across periods and into overtime."""
    if period <= REGULATION_PERIODS:
        return period * PERIOD_S - clock
    return REGULATION_PERIODS * PERIOD_S + (period - REGULATION_PERIODS) * OVERTIME_S - clock


def game_seconds_left(period: int, clock: float) -> float:
    """Seconds left in regulation; in overtime, seconds left in the current OT period
    (the game can always extend, so that is all that is known)."""
    if period <= REGULATION_PERIODS:
        return (REGULATION_PERIODS - period) * PERIOD_S + clock
    return clock


def other_side(side: Side) -> Side:
    return "away" if side == "home" else "home"


class RunTracker:
    """Scoring over a trailing game-clock window, and team timeouts, for one game.

    Feed it the game's plays in order with ``update``. Windows are half-open,
    ``(start, end]``: the scoring window is ``(now - window_s, now]`` in game seconds
    elapsed, so a basket exactly ``window_s`` before the latest play has dropped out
    and one at the latest play's clock counts.
    """

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.period = 1
        self.clock: float = PERIOD_S
        self.home_score = 0
        self.away_score = 0
        self._baskets: deque[tuple[float, Side, int]] = deque()  # (elapsed, team, points)
        self._timeouts: list[tuple[float, float, Side]] = []  # (elapsed, ts, team)

    def update(self, play: Play) -> None:
        self.period, self.clock = play.period, play.clock
        self.home_score, self.away_score = play.home_score, play.away_score
        now = self.elapsed
        if play.kind == "score" and play.team is not None:
            self._baskets.append((now, play.team, play.points))
        elif play.kind == "timeout" and play.team is not None:  # team None: official timeout
            self._timeouts.append((now, play.ts, play.team))
        while self._baskets and self._baskets[0][0] <= now - self.window_s:
            self._baskets.popleft()

    @property
    def elapsed(self) -> float:
        return game_seconds_elapsed(self.period, self.clock)

    @property
    def seconds_left(self) -> float:
        return game_seconds_left(self.period, self.clock)

    @property
    def margin(self) -> int:
        """Home minus away, after the latest play."""
        return self.home_score - self.away_score

    def points(self, side: Side) -> int:
        """Points ``side`` scored in the trailing window."""
        return sum(p for _, team, p in self._baskets if team == side)

    def run_net(self, side: Side) -> int:
        """``side``'s points minus its opponent's, over the trailing window."""
        return self.points(side) - self.points(other_side(side))

    def timeout_called(
        self,
        side: Side,
        game_window: tuple[float, float] | None = None,
        ts_window: tuple[float, float] | None = None,
    ) -> bool:
        """Whether ``side`` called a timeout in either window, each ``(start, end]``:
        ``game_window`` in game seconds elapsed, ``ts_window`` in wallclock ts."""
        return any(
            team == side
            and (
                (game_window is not None and game_window[0] < e <= game_window[1])
                or (ts_window is not None and ts_window[0] < ts <= ts_window[1])
            )
            for e, ts, team in self._timeouts
        )
