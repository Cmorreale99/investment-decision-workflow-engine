"""Project and data paths resolve correctly relative to the repo root."""

from __future__ import annotations

from ai_investment_workflow.utils import (
    config_dir,
    data_dir,
    embeddings_dir,
    interim_dir,
    processed_dir,
    project_root,
    raw_dir,
)


def test_project_root_contains_readme() -> None:
    root = project_root()
    assert root.is_dir()
    assert (root / "README.md").is_file()
    assert (root / "pyproject.toml").is_file()


def test_config_dir_exists_and_has_yamls() -> None:
    cdir = config_dir()
    assert cdir.is_dir()
    for name in (
        "settings.yaml",
        "universe.yaml",
        "strategies.yaml",
        "risk_rules.yaml",
    ):
        assert (cdir / name).is_file(), f"missing config: {name}"


def test_data_subdirs_exist() -> None:
    for getter in (raw_dir, interim_dir, processed_dir, embeddings_dir):
        path = getter()
        assert path.is_dir(), f"missing data dir: {path}"


def test_data_dir_honors_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path


def test_data_dir_default_under_project_root(monkeypatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert data_dir() == project_root() / "data"
