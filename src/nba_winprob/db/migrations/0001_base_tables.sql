-- Games, per-team shot profiles, and the pregame feature store.
-- Column names are snake_case versions of the CSV headers in data/baseline/.

create table games (
    game_id      bigint primary key,
    season       smallint not null check (season between 1946 and 2100),
    game_date    date,     -- null for two 2023 games, as in the source
    home_team_id bigint not null,
    away_team_id bigint not null,
    home_win     boolean,  -- null until the game is played
    check (home_team_id <> away_team_id)
);

create index games_season_date_idx on games (season, game_date);

-- One row per team per game. Who is home, the season, the date and the result
-- live only in games; the loader checks the CSV's copies agree before dropping them.
create table team_games (
    game_id           bigint not null references games (game_id) on delete cascade,
    team_id           bigint not null,
    -- the game's own box-score shot profile (never a pregame feature)
    num_shots         double precision not null,
    avg_shot_distance double precision not null,
    total_made        double precision not null,
    fg_pct            double precision not null,
    avg_loc_x         double precision not null,
    avg_loc_y         double precision not null,
    shots_in_paint    double precision not null,
    shots_midrange    double precision not null,
    shots_long        double precision not null,
    -- lagged profiles from strictly earlier games; null when there is no history
    avg_loc_x_roll5          double precision,
    avg_loc_x_season         double precision,
    avg_loc_y_roll5          double precision,
    avg_loc_y_season         double precision,
    avg_shot_distance_roll5  double precision,
    avg_shot_distance_season double precision,
    fg_pct_roll5             double precision,
    fg_pct_season            double precision,
    num_shots_roll5          double precision,
    num_shots_season         double precision,
    shots_in_paint_roll5     double precision,
    shots_in_paint_season    double precision,
    shots_long_roll5         double precision,
    shots_long_season        double precision,
    shots_midrange_roll5     double precision,
    shots_midrange_season    double precision,
    total_made_roll5         double precision,
    total_made_season        double precision,
    primary key (game_id, team_id)
);

-- The model inputs for one game under one feature variant. JSONB rather than 76
-- columns: the set differs by variant and model, and serving selects columns by
-- the list saved with each model artifact.
create table pregame_features (
    game_id  bigint not null references games (game_id) on delete cascade,
    variant  text not null check (variant in ('notebook', 'corrected')),
    features jsonb not null check (jsonb_typeof(features) = 'object'),
    built_at timestamptz not null default now(),
    primary key (game_id, variant)
);
