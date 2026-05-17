

# AI Investment Workflow Engine

> Human-in-the-loop AI decision infrastructure for investment research.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Status](https://img.shields.io/badge/Status-MVP%20Design-orange)
![Project Type](https://img.shields.io/badge/Type-AI%20Decision%20Infrastructure-purple)
![Domain](https://img.shields.io/badge/Domain-Finance%20%2B%20AI-green)

---

## Overview

The **AI Investment Workflow Engine** is a human-in-the-loop system for supporting investment research and portfolio decision workflows.

It combines:

- Structured financial data
- Deterministic strategy scoring
- Retrieval-grounded context
- Multi-agent analysis
- Human review
- Decision logging
- Benchmark-relative evaluation

This is **not** an autonomous trading bot.  
This is **not** a black-box prediction engine.  
This is **not** a claim to generate alpha.

The goal is to build credible AI decision infrastructure that mirrors how real investment workflows operate: candidate generation, signal review, contextual research, human approval, decision memory, and performance diagnostics.

---

## Table of Contents

- [Core Principle](#core-principle)
- [Why This Project Exists](#why-this-project-exists)
- [System Architecture](#system-architecture)
- [Design Principles](#design-principles)
- [MVP Scope](#mvp-scope)
- [Repository Structure](#repository-structure)
- [Core Layers](#core-layers)
- [Interface Contracts](#interface-contracts)
- [Development Roadmap](#development-roadmap)
- [Build Rules for Claude Code](#build-rules-for-claude-code)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Example Workflow](#example-workflow)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Resume Framing](#resume-framing)

---

## Core Principle

The system does **not** replace the analyst, investor, or portfolio manager.

It supports the human by doing four things:

| Function | Description |
|---|---|
| Surface candidates | Finds potentially interesting investment candidates |
| Structure reasoning | Organizes evidence, risks, and strategy fit |
| Enforce consistency | Applies repeatable strategy and risk logic |
| Compress information | Produces concise decision-ready summaries |

Final authority remains with the human reviewer.

The human can:

- Approve
- Reject
- Watchlist
- Override sizing
- Add notes
- Flag for later review

A recommendation is not treated as approved until the human review layer records an explicit decision.

---

## Why This Project Exists

Most AI finance projects are framed as trading bots or black-box predictors. That framing is usually weak.

Real investment workflows are different:

- Data is incomplete
- Markets are noisy
- Context matters
- Strategy signals conflict
- Risk constraints matter
- Human judgment remains central
- Decisions must be auditable
- Outcomes must be evaluated over time

This project treats AI as a **structured reasoning layer** inside a broader decision-support system.

The AI does not make autonomous trades. It receives grounded context, generates structured analysis, and passes recommendations to a human review layer.

---

## System Architecture

```text
Data Layer
   ↓
Feature & Strategy Layer
   ↓
Retrieval Layer
   ↓
AI Reasoning Layer
   ↓
Human Review Layer
   ↓
Decision Log / Memory Layer
   ↓
Simulation & Evaluation Layer
````

### Architecture Summary

| Layer                         | Purpose                                      | Output                               |
| ----------------------------- | -------------------------------------------- | ------------------------------------ |
| Data Layer                    | Ingest and normalize financial data          | Clean market data, company snapshots |
| Feature & Strategy Layer      | Generate deterministic signals               | Strategy scores, rankings            |
| Retrieval Layer               | Build grounded context packets               | Context for AI agents                |
| AI Reasoning Layer            | Analyze candidates with role-specific agents | Structured recommendations           |
| Human Review Layer            | Capture human decisions                      | Approvals, rejections, overrides     |
| Decision Log / Memory Layer   | Persist decisions and outcomes               | Auditable decision history           |
| Simulation & Evaluation Layer | Track performance and diagnostics            | Benchmark-relative reports           |

---

## Design Principles

### 1. Deterministic Baseline First

Build the deterministic pipeline before adding AI.

Correct build order:

```text
Data ingestion
→ Feature generation
→ Strategy ranking
→ Simulation
→ Evaluation
→ Retrieval
→ Agents
→ Human review
→ Dashboard
```

Do **not** build the agent layer before the baseline works.

---

### 2. Separation of Concerns

Each system layer has a distinct responsibility.

| Component    | Should Do                         | Should Not Do            |
| ------------ | --------------------------------- | ------------------------ |
| Ingestion    | Pull and normalize data           | Score investments        |
| Features     | Generate reusable signals         | Call LLMs                |
| Strategies   | Rank candidates deterministically | Approve decisions        |
| RAG          | Build context packets             | Invent missing facts     |
| Agents       | Produce structured analysis       | Execute trades           |
| Human Review | Approve, reject, override         | Mutate raw data          |
| Evaluation   | Diagnose outcomes                 | Generate recommendations |

---

### 3. Structured Interfaces

All major layers should communicate through explicit schema objects.

Core objects:

* `CompanySnapshot`
* `FeatureSet`
* `StrategySignal`
* `ContextPacket`
* `Recommendation`
* `HumanDecision`
* `DecisionRecord`
* `PerformanceRecord`

Preferred implementation:

* Pydantic models
* Typed function signatures
* Schema validation
* Small, testable modules

---

### 4. Constrained AI Usage

AI is used only inside the reasoning layer.

The AI layer must:

* Receive structured context
* Produce structured output
* Validate outputs against schemas
* Identify uncertainty and risks
* Never execute trades
* Never bypass human review

---

### 5. Evaluation Is a First-Class Component

The system should evaluate both recommendations and human decisions.

Key questions:

* Did approved recommendations outperform rejected ideas?
* Did high-conviction recommendations perform better?
* Did human overrides improve or hurt outcomes?
* Which strategy generated the strongest candidates?
* Where did the workflow fail?

---

## MVP Scope

### Included in V1

| Area         | MVP Requirement                             |
| ------------ | ------------------------------------------- |
| Universe     | 25 to 50 stocks                             |
| Data         | Public market data                          |
| Strategies   | Value and momentum                          |
| Ranking      | Composite scoring and conflict detection    |
| Retrieval    | Minimal context packet                      |
| Agents       | Analyst, strategy, and risk agents          |
| Human Review | Approve, reject, watchlist, override        |
| Memory       | Persistent decision log                     |
| Simulation   | Simple paper portfolio                      |
| Benchmark    | S&P 500 / SPY comparison                    |
| Evaluation   | Return, drawdown, hit rate, override impact |

### Excluded from V1

* Live trading
* Brokerage integration
* Intraday trading
* Options strategies
* Real-time news ingestion
* Complex execution modeling
* Full production UI
* Autonomous execution
* Advanced portfolio optimization
* Claims of alpha generation

---

## Repository Structure

```text
ai-investment-workflow-engine/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml
│
├── config/
│   ├── settings.yaml
│   ├── universe.yaml
│   ├── strategies.yaml
│   └── risk_rules.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── embeddings/
│
├── docs/
│   ├── architecture.md
│   ├── mvp_scope.md
│   ├── evaluation_plan.md
│   ├── decision_log_schema.md
│   └── screenshots/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_strategy_prototyping.ipynb
│   ├── 03_rag_experiments.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── reports/
│   ├── weekly/
│   ├── benchmark_comparisons/
│   └── examples/
│
├── src/
│   └── ai_investment_workflow/
│       ├── __init__.py
│       ├── schemas/
│       ├── ingestion/
│       ├── features/
│       ├── strategies/
│       ├── rag/
│       ├── agents/
│       ├── risk/
│       ├── simulation/
│       ├── evaluation/
│       ├── human_review/
│       ├── app/
│       └── utils/
│
├── tests/
│   ├── test_schemas.py
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_strategies.py
│   ├── test_rag.py
│   ├── test_agents.py
│   ├── test_simulation.py
│   └── test_evaluation.py
│
└── scripts/
    ├── run_ingestion.py
    ├── build_features.py
    ├── run_strategies.py
    ├── build_embeddings.py
    ├── generate_recommendations.py
    ├── run_simulation.py
    └── evaluate_system.py
```

---

## Core Layers

<details>
<summary><strong>1. Data Layer</strong></summary>

The data layer ingests and normalizes public financial data.

### Inputs

* Historical price data
* Basic fundamentals
* Sector classifications
* Benchmark data
* Optional macro indicators

### Outputs

* Clean time-indexed data
* Normalized asset-level tables
* Company snapshots
* Strategy-ready inputs

### Rule

The data layer should not contain strategy or AI logic.

</details>

---

<details>
<summary><strong>2. Feature and Strategy Layer</strong></summary>

This layer applies deterministic financial logic.

### Initial Strategies

* Value
* Momentum

### Optional Later Strategies

* Quality
* Low volatility
* Sector-relative ranking
* Macro overlay

### Outputs

* Feature values
* Strategy scores
* Composite score
* Candidate ranking
* Strategy conflict flags

### Example Strategy Signal

```json
{
  "asset_id": "AAPL",
  "timestamp": "2026-04-01",
  "strategy_scores": {
    "value": -0.4,
    "momentum": 0.8
  },
  "composite_score": 0.35,
  "rank": 7,
  "strategy_conflict": true
}
```

</details>

---

<details>
<summary><strong>3. Retrieval Layer</strong></summary>

The retrieval layer builds a grounded context packet for each candidate.

### Context Packet Contents

* Company snapshot
* Strategy scores
* Recent performance summary
* Risk flags
* Prior decisions
* Human notes
* Optional research notes

### Responsibilities

* Chunking
* Embedding generation
* Vector store persistence
* Top-k retrieval
* Context packet construction

### Rule

The AI reasoning layer should only reason over context packets.

</details>

---

<details>
<summary><strong>4. AI Reasoning Layer</strong></summary>

The AI reasoning layer uses role-specific agents.

### Initial Agents

| Agent          | Responsibility                                           |
| -------------- | -------------------------------------------------------- |
| Analyst Agent  | Summarizes strengths, weaknesses, and recent changes     |
| Strategy Agent | Evaluates strategy fit and signal conflict               |
| Risk Agent     | Reviews volatility, liquidity, exposure, and constraints |

### Final Recommendation Example

```json
{
  "asset_id": "AAPL",
  "timestamp": "2026-04-01",
  "action": "WATCHLIST",
  "conviction": 0.68,
  "rationale": [
    "Strong momentum signal",
    "Positive recent relative performance",
    "No hard risk rule violation"
  ],
  "risks": [
    "Elevated valuation",
    "Potential sector concentration",
    "Strategy conflict between value and momentum"
  ],
  "suggested_next_step": "Review manually before approval"
}
```

</details>

---

<details>
<summary><strong>5. Human Review Layer</strong></summary>

Every recommendation must pass through human review.

### Human Actions

* `APPROVE`
* `REJECT`
* `WATCHLIST`
* `OVERRIDE`
* `NEEDS_REVIEW`

### Human Decision Example

```json
{
  "asset_id": "AAPL",
  "recommendation_id": "rec_2026_04_01_AAPL",
  "human_action": "WATCHLIST",
  "override": true,
  "human_notes": "Momentum is strong, but valuation risk is too high before earnings.",
  "review_status": "completed",
  "reviewed_at": "2026-04-01T15:30:00"
}
```

</details>

---

<details>
<summary><strong>6. Decision Log / Memory Layer</strong></summary>

The decision log stores every reviewed recommendation.

### Stored Fields

* Asset ID
* Timestamp
* Strategy scores
* System recommendation
* System conviction
* System rationale
* System risk flags
* Human action
* Human notes
* Override status
* Final decision
* Later performance

### Example Decision Record

```json
{
  "decision_id": "dec_2026_04_01_AAPL",
  "asset_id": "AAPL",
  "timestamp": "2026-04-01",
  "system_action": "BUY",
  "system_conviction": 0.72,
  "human_action": "WATCHLIST",
  "override": true,
  "human_notes": "Strong momentum, but valuation and earnings risk are too high.",
  "final_status": "watchlisted"
}
```

</details>

---

<details>
<summary><strong>7. Simulation and Evaluation Layer</strong></summary>

The simulation and evaluation layer tracks outcomes.

### Metrics

* Return
* Benchmark-relative return
* Hit rate
* Drawdown
* Approval rate
* Override rate
* Watchlist conversion rate
* Performance by strategy
* Performance by recommendation tier
* Approved vs rejected performance
* Human override impact

### Example Performance Record

```json
{
  "decision_id": "dec_2026_04_01_AAPL",
  "asset_id": "AAPL",
  "evaluation_start": "2026-04-01",
  "evaluation_end": "2026-05-01",
  "asset_return": 0.042,
  "benchmark_return": 0.018,
  "excess_return": 0.024,
  "max_drawdown": -0.031,
  "outcome_label": "outperformed"
}
```

</details>

---

## Interface Contracts

The system should use explicit schemas between layers.

### Core Objects

| Object              | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `CompanySnapshot`   | Compact view of an asset at a point in time |
| `FeatureSet`        | Computed features for an asset              |
| `StrategySignal`    | Strategy scores, rank, and conflict flag    |
| `ContextPacket`     | Grounded input for AI reasoning             |
| `Recommendation`    | Structured AI-generated recommendation      |
| `HumanDecision`     | Human review action                         |
| `DecisionRecord`    | Persisted final decision                    |
| `PerformanceRecord` | Later outcome tracking                      |

### Example: `ContextPacket`

```json
{
  "asset_id": "AAPL",
  "timestamp": "2026-04-01",
  "company_snapshot": {
    "sector": "Technology",
    "price": 182.34
  },
  "strategy_signal": {
    "composite_score": 0.35,
    "rank": 7,
    "strategy_conflict": true
  },
  "recent_performance": {
    "return_3m": 0.12
  },
  "risk_flags": {
    "high_volatility": false
  },
  "prior_decisions": []
}
```

---

## Development Roadmap

### Phase 1: Repository Scaffold

* [ ] Create base project structure
* [ ] Add `README.md`
* [ ] Add `requirements.txt`
* [ ] Add `.env.example`
* [ ] Add `.gitignore`
* [ ] Add `pyproject.toml`
* [ ] Add placeholder modules
* [ ] Add import tests

### Phase 2: Data Ingestion

* [ ] Pull historical price data
* [ ] Save raw data
* [ ] Normalize market data
* [ ] Add schema checks
* [ ] Add ingestion tests

### Phase 3: Feature Engineering

* [ ] Generate momentum features
* [ ] Generate recent return features
* [ ] Generate volatility features
* [ ] Add basic valuation metrics
* [ ] Generate company snapshots
* [ ] Add feature tests

### Phase 4: Strategy Engine

* [ ] Implement base strategy interface
* [ ] Implement value strategy
* [ ] Implement momentum strategy
* [ ] Add composite ranking
* [ ] Add strategy conflict detection
* [ ] Add strategy tests

### Phase 5: Simulation and Benchmarking

* [ ] Build paper portfolio state
* [ ] Add weekly rebalance logic
* [ ] Add benchmark comparison
* [ ] Calculate returns
* [ ] Calculate drawdown
* [ ] Add simulation tests

### Phase 6: Retrieval Layer

* [ ] Define context packet schema
* [ ] Add chunking
* [ ] Add embedding generation
* [ ] Add vector store persistence
* [ ] Add top-k retrieval
* [ ] Add prompt/context builder

### Phase 7: AI Reasoning Layer

* [ ] Add Analyst Agent
* [ ] Add Strategy Agent
* [ ] Add Risk Agent
* [ ] Add orchestration
* [ ] Validate structured outputs
* [ ] Add agent tests

### Phase 8: Human Review

* [ ] Add review queue
* [ ] Add approve action
* [ ] Add reject action
* [ ] Add watchlist action
* [ ] Add override handling
* [ ] Persist decision log

### Phase 9: Evaluation Diagnostics

* [ ] Track approval rate
* [ ] Track override rate
* [ ] Track hit rate
* [ ] Track return vs benchmark
* [ ] Track performance by strategy
* [ ] Track human override impact
* [ ] Generate diagnostics report

### Phase 10: Lightweight Interface

* [ ] Add Streamlit dashboard
* [ ] Display ranked candidates
* [ ] Display AI recommendations
* [ ] Display top reasons and risks
* [ ] Add human action buttons
* [ ] Display decision history
* [ ] Display evaluation metrics

---

## Build Rules for Claude Code

When using Claude Code to build this project:

1. Read `README.md` first.
2. Treat this README as the source of truth.
3. Do not build the UI first.
4. Do not build agents before the deterministic baseline works.
5. Do not create fake complexity.
6. Do not overbuild the MVP.
7. Keep modules small and testable.
8. Use explicit schemas between layers.
9. Add tests as each layer is implemented.
10. Prefer deterministic logic before AI reasoning.
11. Keep AI outputs structured and validated.
12. Never allow the AI layer to execute trades.
13. Treat human review as mandatory.
14. Treat evaluation as a first-class system component.
15. Keep implementation readable and PEP8-compliant.
16. Avoid large monolithic files.
17. Use clear docstrings.
18. Keep notebooks exploratory only.
19. Move reusable logic into `src/`.
20. Do not claim the system generates alpha.

---

## Configuration

### `config/universe.yaml`

```yaml
universe:
  - AAPL
  - MSFT
  - NVDA
  - AMZN
  - GOOGL
```

### `config/strategies.yaml`

```yaml
strategies:
  value:
    enabled: true
    weight: 0.5

  momentum:
    enabled: true
    weight: 0.5
```

### `config/risk_rules.yaml`

```yaml
risk_rules:
  max_single_position_weight: 0.05
  max_sector_weight: 0.30
  max_volatility_threshold: 0.40
  min_liquidity_threshold: 1000000
```

### `config/settings.yaml`

```yaml
settings:
  rebalance_frequency: weekly
  benchmark: SPY
  evaluation_window_days: 30
  top_n_candidates: 10
  require_human_approval: true
```

---

## Running the Project

### Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Deterministic Pipeline

```bash
python scripts/run_ingestion.py
python scripts/build_features.py
python scripts/run_strategies.py
python scripts/run_simulation.py
python scripts/evaluate_system.py
```

### Later AI Workflow

```bash
python scripts/build_embeddings.py
python scripts/generate_recommendations.py
python scripts/evaluate_system.py
```

---

## Testing

Run all tests:

```bash
pytest
```

Suggested test categories:

| Test Type          | Purpose                                   |
| ------------------ | ----------------------------------------- |
| Import tests       | Verify package structure                  |
| Schema tests       | Validate interface contracts              |
| Ingestion tests    | Verify data loading and normalization     |
| Feature tests      | Validate deterministic feature generation |
| Strategy tests     | Validate scoring and ranking              |
| Simulation tests   | Validate portfolio logic                  |
| Evaluation tests   | Validate performance metrics              |
| RAG tests          | Validate retrieval and context packets    |
| Agent tests        | Validate structured AI outputs            |
| Human review tests | Validate decision logging                 |

---

## Example Workflow

A typical weekly review cycle:

```text
1. Pull market data
2. Normalize data
3. Generate features
4. Score value and momentum strategies
5. Rank candidates
6. Build context packets
7. Generate AI recommendations
8. Queue recommendations for human review
9. Human approves, rejects, watchlists, or overrides
10. Store final decisions
11. Simulate portfolio impact
12. Evaluate performance against benchmark
13. Feed decision history into future retrieval
```

---

## Example Candidate Review

```text
Ticker: AAPL
Sector: Technology
Composite Rank: 7
Value Score: -0.40
Momentum Score: 0.80
Strategy Conflict: Yes

AI Recommendation: WATCHLIST
Conviction: 0.68

Top Reasons:
1. Strong recent momentum
2. Positive relative performance
3. No hard risk rule violation

Top Risks:
1. Elevated valuation
2. Strategy conflict between value and momentum
3. Potential sector concentration

Human Action:
WATCHLIST

Human Notes:
Momentum is strong, but valuation risk is too high before earnings.
```

---

## Limitations

This project uses public data and simplified assumptions.

Known limitations:

* Public data only
* Simplified execution assumptions
* No live trading
* No guarantee of alpha
* No brokerage integration
* No real-time decisioning
* Limited initial universe
* Simplified portfolio construction
* AI outputs require validation
* Human review is required
* Backtests may not reflect real-world execution
* Public fundamentals may be delayed or incomplete

These constraints are intentional.

The purpose is to demonstrate credible investment workflow infrastructure, not to overclaim trading performance.

---

## Future Enhancements

Potential future improvements:

* Add quality strategy
* Add macro regime classification
* Add sector-relative ranking
* Add SEC filing retrieval
* Add earnings transcript summaries
* Add analyst note ingestion
* Add richer decision memory
* Add override impact analysis
* Add Streamlit dashboard
* Add FastAPI backend
* Add Postgres persistence
* Add scheduled weekly workflow
* Add multi-period performance attribution
* Add model comparison across agent prompts
* Add confidence calibration
* Add audit trail exports

---

## What This Project Demonstrates

This project demonstrates the ability to:

* Design layered data systems
* Build deterministic financial pipelines
* Define clean interfaces between system layers
* Integrate structured data with AI reasoning
* Use retrieval to ground LLM outputs
* Build constrained multi-agent workflows
* Preserve human judgment in high-stakes decisions
* Track decisions over time
* Evaluate system performance against benchmarks
* Build feedback-driven decision infrastructure

This is stronger than a simple AI trading bot because it reflects how real investment workflows operate.

---

## Resume Framing

### Full Version

> Designed and implemented a human-in-the-loop AI investment workflow engine integrating structured financial data, deterministic strategy scoring, retrieval-grounded context, multi-agent analysis, human review, decision logging, and benchmark-relative evaluation to support explainable investment decision workflows.

### Short Version

> Built AI decision infrastructure for investment workflows, combining financial data pipelines, strategy scoring, RAG-based context retrieval, multi-agent analysis, human approval, and benchmark-relative performance evaluation.

---

## Build Philosophy

Build the deterministic baseline first.

Then add retrieval.

Then add agents.

Then add human review.

Then add evaluation diagnostics.

Then add the dashboard.

Do not invert this sequence.

The strongest version of this project is not the most complex version.

The strongest version is the one that cleanly proves the full workflow from data ingestion to human-reviewed decision logging and benchmark-relative evaluation.

```
```
