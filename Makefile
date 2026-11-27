.PHONY: help install check lint type test features train db-up db-down db clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the venv and install everything, including dev tools
	uv sync --all-groups

lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

type:  ## mypy over src/
	uv run mypy

test:  ## pytest (DB tests skip without `make db-up`; slow tests need the Kaggle archive)
	uv run pytest -q -m "not slow"

check: lint type test  ## Everything CI runs

features:  ## Rebuild the feature tables and diff against the committed baseline
	uv run nba-winprob build-features

train:  ## Train every pregame model on both feature variants
	uv run nba-winprob train-pregame --variant notebook
	uv run nba-winprob train-pregame --variant corrected
	uv run nba-winprob show-metrics

db-up:  ## Start local Postgres (docker compose, port 5433)
	docker compose up -d --wait

db-down:  ## Stop local Postgres (keeps its volume)
	docker compose down

db: db-up  ## Migrate and load the baseline + features into local Postgres
	uv run nba-winprob db migrate
	uv run nba-winprob db load

clean:  ## Remove caches and build output (leaves data/ and artifacts/ alone)
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
