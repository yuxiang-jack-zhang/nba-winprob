"""Shared fixtures.

The baseline tables are stored in Git LFS. Without ``git lfs`` installed a clone
gets small pointer files instead of CSVs, so every fixture that needs real data
checks for that and skips with an actionable message rather than failing oddly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_winprob.config import Settings, get_settings
from nba_winprob.data.pregame_features import load_base_tables

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an unfetched Git LFS pointer rather than the real file."""
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


def require_baseline(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} is missing; see README (Git LFS).")
    if is_lfs_pointer(path):
        pytest.skip(f"{path.name} is an unfetched Git LFS pointer; run `git lfs pull`.")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def base_tables(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cleaned ``(games, team_games)`` pair."""
    games_path = settings.baseline_dir / "games.csv"
    team_games_path = settings.baseline_dir / "team_games.csv"
    require_baseline(games_path)
    require_baseline(team_games_path)
    return load_base_tables(games_path, team_games_path)


@pytest.fixture(scope="session")
def baseline_master(settings: Settings) -> pd.DataFrame:
    """The committed ``pregame_master.csv`` the port must reproduce."""
    path = settings.baseline_dir / "pregame_master.csv"
    require_baseline(path)
    return pd.read_csv(path, low_memory=False)
