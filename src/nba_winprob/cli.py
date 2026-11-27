"""Command line interface.

``nba-winprob build-features`` rebuilds the pregame tables from the base tables;
``nba-winprob train-pregame`` trains a model and writes a versioned artifact with
its metrics. Both are reproducible: same inputs, same seed, same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from nba_winprob.config import get_settings
from nba_winprob.data.pregame_features import (
    FeatureVariant,
    build_pregame_master,
    load_base_tables,
)
from nba_winprob.models import artifacts
from nba_winprob.models.metrics import ClassificationMetrics
from nba_winprob.models.pregame import (
    MlpConfig,
    mlp_feature_columns,
    prepare_chronological_split,
    prepare_grouped_split,
    train_logreg,
    train_mlp,
    train_xgboost,
    xgb_feature_columns,
    xgb_params,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="NBA win-probability: feature builds, model training and the database.",
)

VariantOption = Annotated[
    str,
    typer.Option(
        "--variant",
        help=(
            "'notebook' reproduces the class project; "
            "'corrected' fixes the rest and home/away-split bugs."
        ),
    ),
]

PREGAME_MODELS = ("logreg", "mlp", "xgboost")


def _resolve_variant(variant: str) -> FeatureVariant:
    if variant not in ("notebook", "corrected"):
        raise typer.BadParameter("variant must be 'notebook' or 'corrected'")
    return "notebook" if variant == "notebook" else "corrected"


def _load_master(variant: FeatureVariant) -> pd.DataFrame:
    """The pregame feature table, rebuilt from the base tables rather than read.

    Rebuilding keeps training honest: the committed ``pregame_master.csv`` is a
    baseline to diff against, not an input whose provenance is assumed.
    """
    settings = get_settings()
    games, team_games = load_base_tables(
        settings.baseline_dir / "games.csv", settings.baseline_dir / "team_games.csv"
    )
    _, master = build_pregame_master(
        games, team_games, k=settings.elo_k, init=settings.elo_init, variant=variant
    )
    return master


@app.command("build-features")
def build_features(
    variant: VariantOption = "notebook",
    compare: Annotated[
        bool, typer.Option(help="Diff the result against the committed baseline.")
    ] = True,
) -> None:
    """Rebuild ``pregame_model`` and ``pregame_master`` into ``data/processed/``."""
    settings = get_settings()
    settings.ensure_dirs()
    resolved = _resolve_variant(variant)

    games, team_games = load_base_tables(
        settings.baseline_dir / "games.csv", settings.baseline_dir / "team_games.csv"
    )
    model, master = build_pregame_master(
        games, team_games, k=settings.elo_k, init=settings.elo_init, variant=resolved
    )

    suffix = "" if resolved == "notebook" else f"_{resolved}"
    model_path = settings.processed_dir / f"pregame_model{suffix}.csv"
    master_path = settings.processed_dir / f"pregame_master{suffix}.csv"
    model.to_csv(model_path, index=False)
    master.to_csv(master_path, index=False)
    typer.echo(f"wrote {model_path} ({model.shape[0]} rows, {model.shape[1]} cols)")
    typer.echo(f"wrote {master_path} ({master.shape[0]} rows, {master.shape[1]} cols)")

    if compare and resolved == "notebook":
        _compare_to_baseline(master, settings.baseline_dir / "pregame_master.csv")


def _compare_to_baseline(master: pd.DataFrame, baseline_path: Path) -> None:
    baseline = pd.read_csv(baseline_path, low_memory=False)
    if list(master.columns) != list(baseline.columns):
        typer.secho("columns differ from the baseline", fg=typer.colors.RED)
        raise typer.Exit(1)

    got = master.sort_values("gameId").reset_index(drop=True)
    want = baseline.sort_values("gameId").reset_index(drop=True)
    if len(got) != len(want):
        typer.secho(f"row count differs: {len(got)} vs {len(want)}", fg=typer.colors.RED)
        raise typer.Exit(1)

    worst_col, worst = "", 0.0
    for col in want.columns:
        delta = float((got[col].astype(float) - want[col].astype(float)).abs().max())
        if delta > worst:
            worst_col, worst = col, delta
    if worst > 1e-9:
        typer.secho(f"values differ, worst: {worst_col} by {worst:g}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(
        f"matches baseline exactly ({len(want)} rows x {len(want.columns)} cols)",
        fg=typer.colors.GREEN,
    )


@app.command("train-pregame")
def train_pregame(
    model: Annotated[str, typer.Option("--model", help="logreg, mlp, xgboost, or all")] = "all",
    variant: VariantOption = "notebook",
    seed: Annotated[int, typer.Option(help="Seed for torch and the MLP's batch order.")] = 0,
) -> None:
    """Train pregame models and write versioned artifacts with their metrics."""
    settings = get_settings()
    settings.ensure_dirs()
    resolved = _resolve_variant(variant)
    requested = PREGAME_MODELS if model == "all" else (model,)
    for name in requested:
        if name not in PREGAME_MODELS:
            raise typer.BadParameter(
                f"unknown model {name!r}; choose from {PREGAME_MODELS} or 'all'"
            )

    master = _load_master(resolved)
    typer.echo(f"features: {resolved} variant, {master.shape[0]} games")

    for name in requested:
        version = artifacts.version_name(name, resolved, seed)
        directory = artifacts.artifact_dir(settings.artifact_dir, version)
        metrics, params, feature_cols, test_season = _train_one(
            name, master, seed=seed, directory=directory
        )
        artifacts.save_metadata(
            directory,
            artifacts.ArtifactMetadata(
                version=version,
                model=name,
                variant=resolved,
                seed=seed,
                test_season=test_season,
                feature_cols=feature_cols,
                metrics=metrics,
                params=params,
            ),
        )
        test = metrics["test"]
        typer.echo(
            f"{name:8s} test: log_loss={test['log_loss']:.5f} acc={test['accuracy']:.5f} "
            f"auc={test['roc_auc']:.5f} brier={test['brier']:.5f} ece={test['ece']:.5f}"
        )
        typer.echo(f"         -> {directory}")


def _train_one(
    name: str, master: pd.DataFrame, seed: int, directory: Path
) -> tuple[dict[str, ClassificationMetrics], dict[str, object], list[str], int]:
    """Train one model, save it, and return its metrics and parameters."""
    if name == "xgboost":
        cols = xgb_feature_columns(master)
        data = prepare_chronological_split(master, cols)
        model, metrics = train_xgboost(data)
        artifacts.save_sklearn_model(directory, model)
        return dict(metrics), dict(xgb_params(data.y_train)), cols, data.test_season

    cols = mlp_feature_columns(master)
    data = prepare_grouped_split(master, cols)
    if name == "logreg":
        pipeline, metrics = train_logreg(data)
        artifacts.save_sklearn_model(directory, pipeline)
        params: dict[str, object] = {
            "penalty": "l2 (default)",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
        }
        return dict(metrics), params, cols, data.test_season

    config = MlpConfig()
    net, scaler, history, metrics = train_mlp(data, config=config, seed=seed)
    artifacts.save_torch_model(directory, net, scaler)
    params = {
        "hidden_sizes": list(config.hidden_sizes),
        "activation": config.activation,
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "epochs_run": history.epochs[-1] if history.epochs else 0,
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
    }
    return dict(metrics), params, cols, data.test_season


db_app = typer.Typer(no_args_is_help=True, help="Postgres: migrations and the base-table ETL.")
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending schema migrations to NBA_DATABASE_URL."""
    from nba_winprob.db import migrate, store

    with store.connect(get_settings()) as conn:
        applied = migrate.apply_migrations(conn)
    typer.echo(f"applied {', '.join(applied)}" if applied else "schema up to date")


@db_app.command("load")
def db_load() -> None:
    """Load the baseline tables, then build and store both feature variants from the DB."""
    from nba_winprob.db import store

    with store.connect(get_settings()) as conn:
        counts = store.load_baseline(conn, get_settings())
    typer.echo(", ".join(f"{k}={v}" for k, v in counts.items()))


@app.command("show-metrics")
def show_metrics() -> None:
    """Print a table of every artifact's test metrics, newest config last."""
    settings = get_settings()
    rows = []
    for metadata_path in sorted(settings.artifact_dir.glob(f"*/{artifacts.METADATA_FILENAME}")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        test = payload["metrics"]["test"]
        rows.append(
            {
                "version": payload["version"],
                "model": payload["model"],
                "variant": payload["variant"],
                **{
                    k: round(test[k], 5)
                    for k in ("log_loss", "accuracy", "roc_auc", "brier", "ece")
                },
            }
        )
    if not rows:
        typer.echo("no artifacts yet; run `nba-winprob train-pregame`")
        return
    typer.echo(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
