"""Models, metrics and plots.

**Import-order shim (macOS).** PyTorch ships its own copy of the LLVM OpenMP
runtime in ``torch/lib/libomp.dylib``, while the xgboost wheel links
``@rpath/libomp.dylib`` and resolves it to the Homebrew one. Loading torch first
and xgboost second puts two OpenMP runtimes in the process and segfaults
immediately on the first ``fit``; the reverse order is fine, because whichever
loads first wins and the second reuses it.

Importing xgboost here -- before any submodule gets a chance to import torch --
makes the order deterministic for anything that touches this package, since
Python runs this file before ``nba_winprob.models.pregame``. The import is
optional: a missing or unloadable xgboost (no OpenMP installed) must not stop the
logistic and MLP models, or the service that serves them, from working.

This ordering is **necessary but not sufficient**. Training torch with a
multi-threaded pool still corrupts a later xgboost ``fit`` even in the right
order, so :func:`nba_winprob.models.pregame.train_mlp` also pins torch to a
single thread. Measured on this machine: torch-then-xgboost segfaults;
xgboost-then-torch with a multi-threaded torch pool segfaults;
xgboost-then-torch with one torch thread survives. ``nthread=1`` on the xgboost
side and ``KMP_DUPLICATE_LIB_OK`` do not help.

Linux, where CI and the deployed image run, uses a single libgomp and has no
such conflict. ``tests/test_openmp_coexistence.py`` guards this.
"""

from __future__ import annotations

try:  # noqa: SIM105 - contextlib.suppress would hide which import failed
    import xgboost as _xgboost  # noqa: F401
except Exception:  # pragma: no cover - depends on the local OpenMP install
    # Left deliberately broad: a missing OpenMP runtime surfaces as OSError or
    # xgboost's own XGBoostError, which we cannot catch without importing it.
    pass
