"""torch and xgboost must survive in one process.

On macOS they load separate LLVM OpenMP runtimes and the combination crashes
hard -- SIGSEGV or ``OMP: Error #179`` -- rather than raising. Two guards keep it
working: importing xgboost before torch (``nba_winprob.models.__init__``) and
pinning torch to one thread (``train_mlp``).

A crash would take the whole pytest process down, so this runs in a subprocess
and checks the exit status. See ``nba_winprob/models/__init__.py``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = """
import numpy as np, pandas as pd
from nba_winprob.models import pregame as P

rng = np.random.default_rng(0)
n = 400
frame = pd.DataFrame({
    "gameId": np.arange(n),
    "homeWin": (rng.random(n) > 0.45).astype(float),
    "season": np.repeat([2017, 2018, 2019, 2020], n // 4),
    "a_diff": rng.normal(size=n),
    "b_diff": rng.normal(size=n),
    "elo_diff": rng.normal(size=n),
})
cols = ["a_diff", "b_diff", "elo_diff"]

grouped = P.prepare_grouped_split(frame, cols, test_season=2020)
P.train_logreg(grouped)
P.train_mlp(grouped, P.MlpConfig(max_epochs=3), seed=0)
print("torch done", flush=True)

chrono = P.prepare_chronological_split(frame, cols, test_season=2020)
P.train_xgboost(chrono)
print("xgboost done", flush=True)
"""


def test_torch_then_xgboost_in_one_process() -> None:
    pytest.importorskip("xgboost")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        "training torch then xgboost in one process failed "
        f"(exit {result.returncode}); OpenMP guards may have regressed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert "xgboost done" in result.stdout
