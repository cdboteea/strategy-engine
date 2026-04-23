"""Load and validate strategy YAMLs from disk."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

from .schema import Strategy
from ..config import REGISTRY_DIR


class RegistryError(Exception):
    """Raised when a registry YAML fails to load or validate."""


def _read_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RegistryError(f"{path}: YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise RegistryError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def load_one(path: Path | str) -> Strategy:
    """Load and validate a single strategy YAML."""
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"{path}: file not found")
    data = _read_yaml(path)
    try:
        strat = Strategy.model_validate(data)
    except Exception as e:
        raise RegistryError(f"{path}: validation failed: {e}") from e
    # Filename must match id
    expected_fname = f"{strat.id}.yaml"
    if path.name != expected_fname:
        raise RegistryError(
            f"{path}: filename mismatch — id is {strat.id!r} but filename is {path.name!r}. "
            f"Expected {expected_fname!r}."
        )
    return strat


def iter_yaml_files(root: Path | None = None) -> list[Path]:
    root = root or REGISTRY_DIR
    if not root.exists():
        return []
    return sorted(root.rglob("*.yaml"))


def load_all(root: Path | None = None) -> list[Strategy]:
    """Load every strategy under the registry root. Fails fast on first error."""
    return [load_one(p) for p in iter_yaml_files(root)]


def validate_all(root: Path | None = None) -> tuple[list[Strategy], list[tuple[Path, str]]]:
    """
    Validate every strategy. Returns (ok_strategies, errors).
    Does NOT raise — collects all issues for reporting.
    """
    ok: list[Strategy] = []
    errors: list[tuple[Path, str]] = []
    ids_seen: dict[str, Path] = {}
    for p in iter_yaml_files(root):
        try:
            strat = load_one(p)
        except RegistryError as e:
            errors.append((p, str(e)))
            continue
        if strat.id in ids_seen:
            errors.append((p, f"duplicate id {strat.id!r} already defined in {ids_seen[strat.id]}"))
            continue
        ids_seen[strat.id] = p
        ok.append(strat)
    return ok, errors
