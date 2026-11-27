"""Saving and loading versioned model artifacts.

Every trained model goes to ``artifacts/<version>/`` with its metrics and the
feature columns it expects, so a served prediction can be traced back to the
exact model that produced it -- which is what lets the prediction log record a
model version rather than just a number.

Nothing here uses :mod:`pickle` on a whole estimator. ``joblib`` is used for
scikit-learn objects and ``state_dict`` for torch, so an artifact never carries
executable state that has to be trusted on load. The original project pickled a
dict containing a live model; that artifact is unloadable now and was not kept.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from nba_winprob.models.metrics import ClassificationMetrics

METADATA_FILENAME = "metadata.json"
SKLEARN_MODEL_FILENAME = "model.joblib"
TORCH_MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.joblib"


def git_revision() -> str | None:
    """The current commit, or ``None`` outside a git checkout.

    Recorded so a metrics table can be tied to the code that produced it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


@dataclass
class ArtifactMetadata:
    """What a served prediction needs to describe the model behind it."""

    version: str
    model: str
    variant: str
    seed: int
    test_season: int
    feature_cols: list[str]
    metrics: dict[str, ClassificationMetrics]
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    git_revision: str | None = field(default_factory=git_revision)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=str)


def version_name(model: str, variant: str, seed: int) -> str:
    """A stable, readable version: same inputs give the same name."""
    return f"pregame-{model}-{variant}-s{seed}"


def artifact_dir(root: Path, version: str) -> Path:
    return root / version


def save_metadata(directory: Path, metadata: ArtifactMetadata) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILENAME
    path.write_text(metadata.to_json() + "\n", encoding="utf-8")
    return path


def load_metadata(directory: Path) -> ArtifactMetadata:
    payload = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    return ArtifactMetadata(**payload)


def save_sklearn_model(directory: Path, model: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SKLEARN_MODEL_FILENAME
    joblib.dump(model, path)
    return path


def load_sklearn_model(directory: Path) -> Any:
    return joblib.load(directory / SKLEARN_MODEL_FILENAME)


def save_torch_model(directory: Path, model: Any, scaler: object) -> tuple[Path, Path]:
    """Save the MLP's weights and the scaler that its inputs must pass through.

    The scaler is part of the model as far as serving is concerned: feeding raw
    features to these weights produces silently wrong probabilities.
    """
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / TORCH_MODEL_FILENAME
    scaler_path = directory / SCALER_FILENAME
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def load_torch_model(directory: Path, model: Any) -> tuple[Any, Any]:
    """Load weights into ``model`` (built with the same config) and its scaler."""
    import torch

    state = torch.load(directory / TORCH_MODEL_FILENAME, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, joblib.load(directory / SCALER_FILENAME)
