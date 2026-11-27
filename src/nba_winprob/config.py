"""Settings and paths.

Every path in this project comes from here. The notebooks this package replaces
each hardcoded a different absolute ``ROOT`` on a different laptop, so nothing
ran anywhere but the machine it was written on.

Relative paths are resolved against the project root -- the nearest ancestor of
the working directory containing ``pyproject.toml`` -- so commands work from any
subdirectory of a checkout, and fall back to the working directory otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def project_root() -> Path:
    """The checkout root, or the working directory if there is no ``pyproject.toml`` above it."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return cwd


class Settings(BaseSettings):
    """Configuration, overridable by ``NBA_*`` environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="NBA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    archive_path: Path = Path("data/datasets.zip")

    # Matches docker-compose.yml; override with NBA_DATABASE_URL.
    database_url: str = "postgresql://nba:nba@localhost:5433/nba"

    seed: int = 0
    test_season: int = 2023
    ingame_min_season: int = 2017

    elo_k: float = 20.0
    elo_init: float = 1500.0

    @field_validator("data_dir", "artifact_dir", "archive_path", mode="after")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value if value.is_absolute() else (project_root() / value).resolve()

    # Baseline tables, tracked in Git LFS. These are the inputs the pregame models
    # train from, so the repo reproduces its own numbers without the Kaggle archive.
    @property
    def baseline_dir(self) -> Path:
        return self.data_dir / "baseline"

    # Extracted straight from the Kaggle archive; one CSV per dataset per season.
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    # Combined and cached intermediates (Parquet): per-dataset season concatenations
    # and the possession-state table.
    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    # Tables rebuilt by this package, diffed against baseline_dir to prove the port.
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        """Create every directory this package writes to."""
        for path in (
            self.baseline_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifact_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so paths stay stable within a run."""
    return Settings()
