"""Phase 4 risk-rule engine tests.

The risk module must be a deterministic boolean engine. These tests
verify each rule against synthetic inputs and check that the module
itself does not import any disallowed layer (agents, rag, human_review,
simulation, app).
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

from ai_investment_workflow import risk
from ai_investment_workflow.risk import RiskConfig, evaluate_risk
from ai_investment_workflow.schemas import CompanySnapshot, FeatureSet


AS_OF = date(2025, 6, 1)


def _snapshot(asset_id: str = "TST", price: float = 100.0) -> CompanySnapshot:
    return CompanySnapshot(
        asset_id=asset_id, timestamp=AS_OF, sector="Unknown", price=price
    )


def _cfg() -> RiskConfig:
    return RiskConfig(
        max_volatility_threshold=0.40,
        near_52w_high_threshold=0.02,
        near_52w_low_threshold=0.05,
        extended_from_52w_low_threshold=0.50,
    )


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def test_high_volatility_triggers_when_threshold_exceeded() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"volatility_21d": 0.55, "distance_from_52w_high": -0.20, "distance_from_52w_low": 0.30},
        config=_cfg(),
    )
    assert flags["high_volatility"] is True


def test_high_volatility_uses_max_of_21d_and_63d() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"volatility_21d": 0.10, "volatility_63d": 0.50},
        config=_cfg(),
    )
    assert flags["high_volatility"] is True


def test_high_volatility_false_when_below_threshold() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"volatility_21d": 0.20, "volatility_63d": 0.30},
        config=_cfg(),
    )
    assert flags["high_volatility"] is False


def test_near_52w_high_triggers_when_close_to_high() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"distance_from_52w_high": -0.005, "distance_from_52w_low": 0.40},
        config=_cfg(),
    )
    assert flags["near_52w_high"] is True


def test_near_52w_high_false_when_far_from_high() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"distance_from_52w_high": -0.20, "distance_from_52w_low": 0.40},
        config=_cfg(),
    )
    assert flags["near_52w_high"] is False


def test_near_52w_low_triggers_when_close_to_low() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"distance_from_52w_high": -0.30, "distance_from_52w_low": 0.01},
        config=_cfg(),
    )
    assert flags["near_52w_low"] is True
    assert flags["extended_from_52w_low"] is False


def test_extended_from_52w_low_triggers_when_far_above_low() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"distance_from_52w_high": 0.0, "distance_from_52w_low": 0.80},
        config=_cfg(),
    )
    assert flags["extended_from_52w_low"] is True
    assert flags["near_52w_low"] is False


def test_liquidity_unavailable_is_always_flagged() -> None:
    flags = evaluate_risk(_snapshot(), {}, config=_cfg())
    assert flags["liquidity_unavailable"] is True


# ---------------------------------------------------------------------------
# Missing-input handling
# ---------------------------------------------------------------------------


def test_missing_volatility_sets_missing_input_flag() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"distance_from_52w_high": -0.10, "distance_from_52w_low": 0.20},
        config=_cfg(),
    )
    assert flags["high_volatility"] is False
    assert flags["high_volatility_missing_input"] is True


def test_missing_52w_high_distance_sets_missing_input_flag() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"volatility_21d": 0.20, "distance_from_52w_low": 0.20},
        config=_cfg(),
    )
    assert flags["near_52w_high"] is False
    assert flags["near_52w_high_missing_input"] is True


def test_missing_52w_low_distance_sets_missing_input_flag() -> None:
    flags = evaluate_risk(
        _snapshot(),
        {"volatility_21d": 0.20, "distance_from_52w_high": -0.10},
        config=_cfg(),
    )
    assert flags["near_52w_low"] is False
    assert flags["extended_from_52w_low"] is False
    assert flags["near_52w_low_missing_input"] is True


def test_empty_features_does_not_crash() -> None:
    flags = evaluate_risk(_snapshot(), {}, config=_cfg())
    # All boolean, all defined
    assert all(isinstance(v, bool) for v in flags.values())


# ---------------------------------------------------------------------------
# FeatureSet acceptance + determinism
# ---------------------------------------------------------------------------


def test_accepts_feature_set_object() -> None:
    fs = FeatureSet(
        asset_id="TST",
        timestamp=AS_OF,
        features={
            "volatility_21d": 0.45,
            "distance_from_52w_high": -0.01,
            "distance_from_52w_low": 0.55,
        },
    )
    flags = evaluate_risk(_snapshot(), fs, config=_cfg())
    assert flags["high_volatility"] is True
    assert flags["near_52w_high"] is True
    assert flags["extended_from_52w_low"] is True


def test_evaluate_risk_is_deterministic() -> None:
    feats = {
        "volatility_21d": 0.42,
        "distance_from_52w_high": -0.03,
        "distance_from_52w_low": 0.60,
    }
    a = evaluate_risk(_snapshot(), feats, config=_cfg())
    b = evaluate_risk(_snapshot(), feats, config=_cfg())
    assert a == b


# ---------------------------------------------------------------------------
# Architectural guardrails
# ---------------------------------------------------------------------------


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
            if node.level:
                # Relative imports: resolve roughly against the package path.
                rel = node.module or ""
                found.add(f"..{rel}")
    return found


def test_risk_module_does_not_import_forbidden_layers() -> None:
    risk_dir = Path(risk.__file__).parent
    forbidden = ("agents", "rag", "human_review", "simulation", "app")
    for py in risk_dir.rglob("*.py"):
        imports = _module_imports(py)
        for imp in imports:
            for fb in forbidden:
                assert f"ai_investment_workflow.{fb}" not in imp, (
                    f"{py.name} imports forbidden layer: {imp}"
                )


def test_risk_module_does_not_import_llm_sdks() -> None:
    risk_dir = Path(risk.__file__).parent
    forbidden_pkgs = ("anthropic", "openai", "langchain", "litellm")
    for py in risk_dir.rglob("*.py"):
        imports = _module_imports(py)
        for imp in imports:
            for pkg in forbidden_pkgs:
                assert not imp.startswith(pkg), (
                    f"{py.name} imports LLM SDK: {imp}"
                )
