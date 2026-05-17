# Architecture

Phase 1 stub. See `README.md` for the full architectural reference.

## Cadence Model (Phase 1 assumption)

- **Data frequency:** daily prices
- **Decision frequency:** weekly rebalance (per `config/settings.yaml`)
- **Evaluation window:** 30 days forward per decision (per `config/settings.yaml`)

## Module Boundaries

`schemas/` has zero internal dependencies. All other layers depend on `schemas/`
and never on each other transitively. Specifically:

- `ingestion/` must not score or rank.
- `features/` must not call LLMs or hit the network.
- `strategies/` must not approve decisions or call LLMs.
- `agents/` must not touch market data or execute trades.
- `human_review/` must not mutate raw data.

## Provider Seams

Two abstractions are reserved for later phases:

- `MarketDataProvider` (Phase 2): swappable data source.
- `ReasoningProvider` (Phase 8): swappable AI backend; default is `NullProvider`.

Neither is implemented in Phase 1.
