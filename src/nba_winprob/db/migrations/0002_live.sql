-- Live paper trading (phase 4): runs, the events they saw, their orders and fills,
-- and the latest state of each game. Everything is append-only except game_state.
--
-- game_id here is an ESPN event id (text). It is unrelated to the Kaggle-era
-- games table, so there is no foreign key to it.
-- ts columns are unix seconds (double precision), as in engine/events.py.

create table runs (
    id          bigserial primary key,
    strategy    text not null,
    broker      text not null check (broker in ('paper', 'demo', 'live')),
    params      jsonb not null default '{}' check (jsonb_typeof(params) = 'object'),
    started_at  timestamptz not null default now(),
    stopped_at  timestamptz,
    stop_reason text,
    check ((stopped_at is null) = (stop_reason is null))
);

-- Every play, quote and settlement a run saw, for audit and replay. payload holds
-- the event's dataclass fields; seq is arrival order (one sequence across runs).
create table events (
    seq     bigserial primary key,
    run_id  bigint not null references runs (id),
    ts      double precision not null,  -- receipt time
    game_id text not null,
    kind    text not null check (kind in ('play', 'quote', 'settle')),
    payload jsonb not null check (jsonb_typeof(payload) = 'object')
);

create index events_run_game_seq_idx on events (run_id, game_id, seq desc);

create table orders (
    id        bigserial primary key,
    run_id    bigint not null references runs (id),
    ts        double precision not null,
    game_id   text not null,
    team      text not null check (team in ('home', 'away')),
    contracts integer not null check (contracts <> 0),  -- > 0 buy, < 0 sell
    reason    text not null,
    meta      jsonb not null default '{}' check (jsonb_typeof(meta) = 'object'),
    status    text not null default 'pending'
              check (status in ('pending', 'filled', 'cancelled', 'rejected')),
    error     text
);

create index orders_run_ts_idx on orders (run_id, ts desc, id desc);

create table fills (
    id        bigserial primary key,
    order_id  bigint not null references orders (id),
    ts        double precision not null,
    price     double precision not null check (price between 0 and 1),
    fee       double precision not null check (fee >= 0),
    contracts integer not null check (contracts <> 0)
);

create index fills_order_idx on fills (order_id);

-- Latest score, clock and book per game: what the catch-up agent reads for "how is
-- the game going". Columns stay null until the first play or quote fills them.
create table game_state (
    game_id    text primary key,
    home       text not null,  -- team abbreviations
    away       text not null,
    home_score integer check (home_score >= 0),
    away_score integer check (away_score >= 0),
    period     integer check (period >= 1),
    clock      double precision check (clock >= 0),  -- seconds left in the period
    status     text not null default 'scheduled' check (status in ('scheduled', 'live', 'final')),
    last_play  text,
    yes_bid    double precision check (yes_bid between 0 and 1),
    yes_ask    double precision check (yes_ask between 0 and 1),
    updated_at timestamptz not null default now()
);
