# In-game Kalshi trading — design

2026-11-26. Status: approach approved (B); this document is the plan of record.

## Intent

Turn the repo into a trading system Jack uses on Kalshi's NBA game markets *during*
games. First prove a strategy on paper; then let it place real orders in a live game.
The first strategy comes from Jack's manual trading: in close games, a big run by one
team tends to be followed by a counter-run, and timeouts are a good sign. More
strategies should plug in later.

The question being tested is **not** "do scoring runs reverse?" (largely a coin-flip
question) but **"does the Kalshi price overreact to runs?"** — measured directly as
what a trader would have made after crossing the spread and paying fees.

A catch-up agent (LLM with tools) answers "what is my bot holding, why, and how are
the games going?" for when Jack is away from the game.

## Goals, non-goals, anti-goals

- **Goals:** in-game history for every matched game; an honest backtest of Jack's
  rule exactly as he described it; one engine whose strategy code runs unchanged in
  backtest, paper and live; then a model that learns what the market gets wrong.
- **Non-goals (now):** player-level models; sub-minute historical data; other sports.
- **Anti-goals:** a backtest tuned until it looks profitable and presented as an
  edge; live orders without hard limits and a kill switch; any secret (Kalshi key,
  webhook) in the public repo.

## Approaches considered

- **A. Research scripts first** (beside `papertrade/`): fastest answer, but the live
  trader would be a rewrite, and rewrites are where backtest and live diverge.
- **B. One event-driven engine for backtest and live** — **chosen by Jack.** Events
  (plays, quotes, settlement) flow to a strategy, which emits orders, which a broker
  fills: simulated (backtest), Kalshi demo (paper), Kalshi (live).
- **C. An existing framework** (NautilusTrader, Backtrader): dropped; none model
  $1/$0 binary contracts or Kalshi fees.

## Architecture

```
 sources/espn.py ──Play──┐                      ┌─ SimBroker      (backtest: candles, latency, fees)
                         ├─▶ replay / live loop ─▶ Strategy ─Order─▶ DemoBroker  (paper: Kalshi demo API)
 sources/kalshi.py─Quote─┘   (sort_key order)    on_event()        └─ KalshiBroker(live: risk limits, kill switch)
                                                                         │
                                                         orders, fills, positions ─▶ Postgres ─▶ catch-up agent
```

The contract is `src/nba_winprob/engine/events.py` (`Play`, `Quote`, `Settle`,
`Order`, `Fill`, `Strategy`, `sort_key`, `kalshi_taker_fee`). Everything else builds
to it.

**Lookahead rule.** A strategy decides at an event's `ts`. The broker fills at the
first quote with `ts >= order.ts + latency`, strictly later than the deciding event
(plays sort before quotes at equal times). Historical quotes are one-minute candle
closes, so the earliest possible fill is the next minute's close — already after the
market reacted. This is conservative, and deliberately so.

## Phases

1. **History + engine + first strategy** (parallel):
   - *Data:* ESPN play-by-play and Kalshi one-minute candles for the 1,612 Kalshi
     games already matched in `data/papertrade/kalshi.parquet` (Apr 2025 – Nov 2026),
     cached per game under `data/ingame/` (gitignored), parsed into events.
   - *Engine:* `SimBroker`, the replay loop, P&L and bootstrap intervals by game.
   - *Strategy:* run and timeout detection; `RunReversal` as pre-registered below.
2. **Results:** run the pre-registered spec once; report everything in RESULTS.md,
   including a price-reaction study (average mid move after triggers) that needs no
   strategy at all.
3. **Model:** predict the final outcome (and the k-minute price move) from the
   market mid plus in-game features (run size, timeout, margin, time left,
   pregame price). Walk-forward by season. The question is whether features add
   anything *beyond the market price*; if not, that is the result.
4. **Paper live:** ESPN live polling + Kalshi demo exchange; orders/fills/positions in
   Postgres (migration 0002). Needs Jack's demo API key.
5. **Live:** real orders. Off by default; per-order and per-day dollar caps; kill
   switch; every order logged with its reason.
6. **Catch-up agent:** Claude with read-only tools over positions, trades (with
   reasons) and live game state.

## Pre-registered spec: `RunReversal` v1

Fixed before any in-game data is examined. Not to be tuned on the results; variants
below are reported in full alongside, never instead.

- **Run:** over the trailing 4:00 of game clock, the runner's points minus the
  opponent's points. Game clock is continuous across periods.
- **Trigger:** run net is 10–15, *and* the game margin is within 9 points
  (|home − away| ≤ 9) after the run, *and* more than 2:00 of game time remains
  (regulation; no entries in overtime's final 2:00 either).
- **Entry:** buy the team that was run on, ~$100 at the ask (contracts =
  floor(100 / price)). One open position per game; the trigger re-arms once the run
  net falls below 5.
- **Exit:** first of (a) the mid moved 5¢ in our favour from the entry fill, (b) 6:00
  of game clock since entry, (c) 2:00 of game time left. Sell at the bid. Positions
  still open at settlement are settled at $1/$0.
- **Execution:** latency 30 s (ESPN's live feed lags the broadcast); Kalshi taker fee
  on every fill; fills at the candle close, no depth.
- **Reported splits:** timeout by the run-on team within the run window or within
  30 s after the trigger (Jack's "timeouts are good signs") vs none; regular season
  vs playoffs; first vs second half. Sensitivity: latency 0 / 30 / 60 / 120 s; hold to
  settlement instead of exit rules.
- **Significance:** 95% bootstrap interval of total P&L, resampling games.

## Constraints

- Public repo: keys only in `.env` (gitignored); `.env.example` holds placeholders.
- ESPN and Kalshi market data are public; fetchers cache every response so a re-run
  only fetches what is missing, and send a generic User-Agent.
- No orders on any exchange before phase 4, and no real-money orders before Jack
  explicitly turns them on in phase 5.

## Left to the builder

File layout inside each module, cache formats, test fixtures, plotting.
