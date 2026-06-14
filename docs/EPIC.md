# EPIC: AI Investment Workflow Engine — Phase 1 to Full Implementation

## The problem

Most "AI for investing" projects collapse into a black-box trading bot: opaque
predictions, no audit trail, no human in the loop, and an implicit claim of alpha that
doesn't survive contact with real markets. That framing is both weak and unsafe.

Real investment workflows look nothing like that. Data is incomplete, signals
conflict, risk constraints bind, and human judgment stays central. Decisions must be
**explainable**, **auditable**, and **evaluated over time** — including evaluation of
the *humans'* decisions, not just the model's.

## The vision

Build credible **decision-support infrastructure** that mirrors a real research desk:
generate candidates deterministically, ground AI reasoning in retrieved facts, route
every recommendation through mandatory human review, persist an auditable decision
log, and diagnose outcomes against a benchmark. The AI is a **structured reasoning
layer**, never an autonomous trader.

## Guiding principles (enforced end to end)

1. **Deterministic baseline before AI.** Data → features → strategies → simulation →
   evaluation all work with zero external dependencies before any LLM appears.
2. **Schema-first contracts.** Layers communicate only through frozen,
   `extra="forbid"` Pydantic models — `CompanySnapshot`, `FeatureSet`,
   `StrategySignal`, `ContextPacket`, `Recommendation`, `HumanDecision`,
   `DecisionRecord`, `PerformanceRecord`.
3. **Grounding, no fabrication.** Retrieval packets, agent claims, and diagnostics all
   trace to real upstream facts; missing inputs degrade to empty states.
4. **Human authority is absolute.** Recommendations are advisory; only an explicit
   human decision is final. No layer executes a trade.
5. **Offline & reproducible.** The full test suite runs with no network, no LLM SDK,
   and no Streamlit; artifact rebuilds are byte-identical.
6. **Heavy/optional deps isolated.** The Anthropic SDK and Streamlit are both imported
   lazily and are not required by the core package or tests.

---

## Phase-by-phase delivery

### Phase 1 — Repository scaffold
**Objective:** A clean, testable skeleton.
**Delivered:** Package under `src/ai_investment_workflow/` with placeholder layers
(`ingestion`, `features`, `strategies`, `risk`, `simulation`, `evaluation`, `rag`,
`agents`, `human_review`, `app`, `utils`, `schemas`), `config/*.yaml`, and
path/config/logging utilities, plus import tests. Paths honor a `DATA_DIR` override;
scripts run with `PYTHONPATH=src`.

### Phase 2 — Data ingestion
**Objective:** Clean, time-indexed market data.
**Delivered:** A provider interface with an offline `FixtureProvider` (deterministic
bundled data) and a live `yfinance` provider, plus normalization and schema
validation. *Quirk:* the bundled fixture covers 2025-01…2025-04, so downstream
scripts are seeded from prices/features parquet directly rather than the
date-windowed ingestion path.

### Phase 3 — Feature engineering
**Objective:** Reusable deterministic signals.
**Delivered:** Momentum, recent-return, volatility, and valuation features plus
per-asset `CompanySnapshot` construction. Pure pandas/numpy; no scoring, no LLM.

### Phase 4 — Strategy engine & risk rules
**Objective:** Repeatable candidate ranking.
**Delivered:** Value and momentum strategies behind a base `Strategy` interface,
composite scoring, cross-sectional ranking, **strategy-conflict detection**, and
configurable risk-rule flags (`risk/rules.py`). Output: `StrategySignal` per asset.
Strategies rank; they never approve.

### Phase 5 — Simulation & benchmarking
**Objective:** Turn rankings into a tracked paper portfolio.
**Delivered:** Portfolio state, weekly rebalance, SPY benchmark comparison, return and
drawdown calculation, and rankings-history construction.

### Phase 6 — Evaluation diagnostics (baseline)
**Objective:** Benchmark-relative outcome measurement.
**Delivered:** Forward-window return/drawdown helpers, per-decision `PerformanceRecord`s
(keyed by `decision_id`), portfolio diagnostics (total/excess return, hit rate, max
drawdown), and per-strategy attribution. This is the last layer of the deterministic
baseline.

### Phase 7 — Retrieval (RAG) layer
**Objective:** Ground the AI in facts, not vibes.
**Delivered:** One `ContextPacket` per top-N candidate, assembled from named
`ContextSource`s (snapshot, signal, performance, notes). The merge enforces
provenance: every populated field traces to exactly one source, identity is checked,
and **nothing is fabricated**. Deterministic embeddings + a vector store, with
byte-identical JSONL output. Strict guardrail: no LLM/network imports anywhere in the
layer.

### Phase 8 — AI reasoning layer
**Objective:** Role-specific analysis, grounded and structured.
**Delivered:** A `ReasoningProvider` protocol with a deterministic offline
`StubProvider` default and an optional Anthropic provider (SDK imported **lazily**,
gated behind `settings.reasoning_provider`, using `claude-opus-4-8` + adaptive
thinking + structured outputs). Analyst / Strategy / Risk agents reason only over a
`ContextPacket`; an orchestrator merges them into one schema-validated
`Recommendation`. **Every rationale/risk item must cite a packet field** —
`assert_grounded` rejects any claim that doesn't. No trade execution; output always
routes to human review.

### Phase 9 — Human review layer
**Objective:** Make human approval mandatory and auditable.
**Delivered:** A `DecisionSource` seam (deterministic `ScriptedDecisionSource`
default; optional stdin `InteractiveDecisionSource`), `apply_decision` merging a
`Recommendation` + `HumanDecision` into a `DecisionRecord` (with identity checks), and
an **append-only, deduped, byte-deterministic** decision log. A resumable
`ReviewQueue` skips already-decided items. The system action is never auto-applied —
nothing is recorded without an explicit human action.

### Phase 10 — Decision & override evaluation diagnostics
**Objective:** Evaluate the *decisions*, not just the model.
**Delivered (additively to Phase 6):** `diagnose_decisions` joins the decision log to
performance records **strictly on `decision_id`** and answers the questions that
justify the whole system — approval/override rates, hit rate, **approved-vs-rejected
excess return (the counterfactual)**, override impact, and performance by conviction
tier. Decisions without an outcome yet are *pending*: counted in coverage/rates,
excluded from outcome metrics, never fabricated. Output: deterministic
`decision_diagnostics.json`.

### Phase 11 — Lightweight dashboard
**Objective:** A human-facing review surface — built last, per the philosophy.
**Delivered:** A read-only Streamlit UI split into a **pure, offline-testable core**
(`data` loaders, `views` view models, `actions` recorder) and a **thin Streamlit
shell**. Streamlit is an optional dependency — the package and full suite import and
run without it; importing the `app` package never loads Streamlit. The UI shows ranked
candidates (scores, conflict, AI action + reasons/risks), evaluation metrics, and
decision history, with action buttons whose **only write path is through the
human-review layer**. The launcher resolves the package via absolute imports +
`PYTHONPATH` so `streamlit run` works whether or not the package is installed.

---

## Cross-cutting architecture

- **Two swappable "provider seams"** — `ReasoningProvider` (Phase 8) and
  `DecisionSource` (Phase 9) — keep the offline default deterministic while allowing a
  real LLM or interactive reviewer to plug in behind the same protocol. The dashboard
  reuses both rather than reimplementing logic.
- **Single source of truth per fact.** Provenance is enforced in retrieval, citations
  in reasoning, identity joins in review and diagnostics — the same discipline
  repeated at every layer.
- **Append-only, reproducible artifacts** under `data/processed/`
  (`context_packets.jsonl`, `recommendations.jsonl`, `decision_log.jsonl`,
  `performance_records.parquet`, `decision_diagnostics.json`).

## Definition of done — status

- **All 11 working phases implemented.** End-to-end flow runs on fixture data:
  ingestion → features → strategies/risk → simulation → evaluation → retrieval →
  agents → human review → decision diagnostics → dashboard.
- **Full test suite: 274 tests passing offline** — with neither the Anthropic SDK nor
  Streamlit installed.
- **Hard rules upheld throughout:** no trade execution, mandatory human review,
  structured/validated outputs, grounded reasoning, graceful degradation,
  deterministic rebuilds.
- **Out of scope (intentional):** live trading, brokerage integration, intraday/options,
  real-time news, autonomous execution, any claim of alpha.

> This is the strongest version of the project not because it's the most complex, but
> because it cleanly proves the full workflow from data ingestion to human-reviewed,
> benchmark-evaluated decisions.
