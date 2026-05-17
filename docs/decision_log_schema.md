# Decision Log Schema

Phase 1 stub. Authoritative schemas live in
`src/ai_investment_workflow/schemas/`.

## Records

- `Recommendation` — system-produced, structured AI output (Phase 8).
- `HumanDecision` — human review action recorded against a recommendation.
- `DecisionRecord` — persisted final decision joining the two by ID.
- `PerformanceRecord` — outcome tracked over the evaluation window.

## Persistence (MVP)

Append-only JSONL files under `data/processed/decisions/`, keyed by
`decision_id`. Migrating to Postgres is a Future Enhancement.

## ID Conventions

- `recommendation_id`: `rec_{YYYYMMDD}_{ASSET}` — unique per trading day.
- `decision_id`: `dec_{YYYYMMDD}_{ASSET}`.
