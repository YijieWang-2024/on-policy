"""Helpers for saving and reusing run configuration metadata."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def namespace_to_dict(args: Namespace) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in vars(args).items()}


def save_run_config(args: Namespace, run_dir: str | Path, save_dir: str | Path | None,
                    *, num_agents: int | None = None) -> None:
    """Write reproducibility metadata to the run root and model directory."""
    run_dir = Path(run_dir)
    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_agents": num_agents,
        "all_args": namespace_to_dict(args),
    }

    targets = [run_dir / CONFIG_FILENAME]
    if save_dir is not None:
        targets.append(Path(save_dir) / CONFIG_FILENAME)

    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def load_model_config(model_dir: str | Path | None) -> dict[str, Any]:
    """Load config metadata from a model directory or its parent run directory."""
    if not model_dir:
        return {}
    model_dir = Path(model_dir)
    for candidate in (model_dir / CONFIG_FILENAME, model_dir.parent / CONFIG_FILENAME):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("all_args", payload)
    return {}


def explicit_option_names(argv: list[str]) -> set[str]:
    """Return CLI option names explicitly provided as --foo or --foo=bar."""
    names = set()
    for token in argv:
        if token.startswith("--"):
            names.add(token[2:].split("=", 1)[0].replace("-", "_"))
    return names


def apply_saved_args(args: Namespace, saved_args: dict[str, Any],
                     explicit_names: set[str], *, skip: set[str] | None = None) -> None:
    """Apply saved args unless the current CLI explicitly set the option."""
    skip = skip or set()
    for key, value in saved_args.items():
        if key not in explicit_names and key not in skip:
            setattr(args, key, value)
