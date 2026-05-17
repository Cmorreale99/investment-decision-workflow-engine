"""Every subpackage must import cleanly."""

from __future__ import annotations

import importlib

SUBPACKAGES = [
    "ai_investment_workflow",
    "ai_investment_workflow.schemas",
    "ai_investment_workflow.ingestion",
    "ai_investment_workflow.features",
    "ai_investment_workflow.strategies",
    "ai_investment_workflow.rag",
    "ai_investment_workflow.agents",
    "ai_investment_workflow.risk",
    "ai_investment_workflow.simulation",
    "ai_investment_workflow.evaluation",
    "ai_investment_workflow.human_review",
    "ai_investment_workflow.app",
    "ai_investment_workflow.utils",
]


def test_all_subpackages_import() -> None:
    for name in SUBPACKAGES:
        importlib.import_module(name)


def test_schemas_namespace_exports() -> None:
    schemas = importlib.import_module("ai_investment_workflow.schemas")
    for symbol in (
        "CompanySnapshot",
        "ContextPacket",
        "DecisionRecord",
        "FeatureSet",
        "HumanAction",
        "HumanDecision",
        "OutcomeLabel",
        "PerformanceRecord",
        "Recommendation",
        "StrategySignal",
        "SystemAction",
    ):
        assert hasattr(schemas, symbol), f"schemas missing export: {symbol}"
