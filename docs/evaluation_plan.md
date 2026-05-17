# Evaluation Plan

Phase 1 stub. Implemented in Phase 6.

## Metrics

- Asset return over evaluation window
- Benchmark (SPY) return over same window
- Excess return (asset − benchmark)
- Max drawdown
- Hit rate (fraction of approved decisions with positive excess return)
- Approval / rejection / watchlist / override rates
- Performance by strategy
- Override impact (decisions where human action differed from system action)

## Outcome Labels

Per `PerformanceRecord.outcome_label`:
- `outperformed` — excess return > 0
- `underperformed` — excess return < 0
- `neutral` — excess return ≈ 0
- `pending` — evaluation window not yet complete
