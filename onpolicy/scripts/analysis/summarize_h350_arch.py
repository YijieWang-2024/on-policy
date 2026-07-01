"""Summarize held-out Mean/Flat/Set MEC evaluations across seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _derived(payload: dict) -> dict[str, float]:
    metrics = payload["metrics"]
    frozen = payload.get("freeze_hap_metrics")
    horizon = float(payload["episode_horizon"])
    total_workload = metrics["A"] + metrics["U"]
    result = {
        "cost_per_slot": metrics["train"] / horizon,
        "accept_rate": metrics["A"] / max(total_workload, 1e-12),
        "w1": metrics["w1"],
        "queue_share": metrics["q"] / max(metrics["train"], 1e-12),
        "overflow_share": metrics["ovf"] / max(metrics["train"], 1e-12),
    }
    if frozen is not None:
        result["freeze_hap_delta"] = (
            frozen["train"] / max(metrics["train"], 1e-12) - 1.0
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="training_logs")
    parser.add_argument("--prefix", default="v6_h350")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--expected_seeds",
        type=int,
        default=3,
        help="required completed seeds per architecture",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    pattern = re.compile(
        rf"^{re.escape(args.prefix)}_(mean|flat|set)_seed(\d+)"
        r"\.heldout\.json$"
    )
    grouped: dict[str, list[dict[str, float]]] = {
        "mean": [],
        "flat": [],
        "set": [],
    }
    runs = []
    for path in sorted(log_dir.glob(f"{args.prefix}_*_seed*.heldout.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        metrics = _derived(payload)
        architecture, seed = match.group(1), int(match.group(2))
        grouped[architecture].append(metrics)
        runs.append(
            {
                "architecture": architecture,
                "seed": seed,
                **metrics,
            }
        )

    if not runs:
        raise FileNotFoundError(
            f"no held-out evaluations found for prefix {args.prefix}"
        )

    summary = {}
    for architecture, rows in grouped.items():
        if not rows:
            continue
        summary[architecture] = {}
        for key in rows[0]:
            values = np.asarray([row[key] for row in rows], dtype=float)
            summary[architecture][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    gates = {
        "complete_three_architecture_matrix": all(
            len(grouped[architecture]) == args.expected_seeds
            for architecture in ("mean", "flat", "set")
        ),
        "all_metrics_finite": all(
            np.isfinite(value)
            for run in runs
            for key, value in run.items()
            if key not in {"architecture", "seed"}
        ),
    }
    set_rows = grouped["set"]
    if set_rows:
        set_costs = np.asarray(
            [row["cost_per_slot"] for row in set_rows], dtype=float
        )
        set_accept = np.asarray(
            [row["accept_rate"] for row in set_rows], dtype=float
        )
        set_overflow = np.asarray(
            [row["overflow_share"] for row in set_rows], dtype=float
        )
        set_freeze = np.asarray(
            [row["freeze_hap_delta"] for row in set_rows], dtype=float
        )
        set_w1 = np.asarray([row["w1"] for row in set_rows], dtype=float)
        gates.update(
            {
                "set_cost_seed_cv_below_10pct": (
                    float(np.std(set_costs, ddof=1))
                    / max(float(np.mean(set_costs)), 1e-12)
                    <= 0.10
                    if len(set_costs) > 1
                    else False
                ),
                "set_min_accept_rate_at_least_60pct": (
                    float(np.min(set_accept)) >= 0.60
                ),
                "set_max_overflow_share_below_5pct": (
                    float(np.max(set_overflow)) <= 0.05
                ),
                "set_hap_freeze_positive_in_two_seeds": (
                    int(np.sum(set_freeze > 0.0)) >= 2
                ),
                "set_mean_hap_freeze_at_least_5pct": (
                    float(np.mean(set_freeze)) >= 0.05
                ),
            }
        )
        heuristic_path = log_dir / f"{args.prefix}.heuristic.heldout.json"
        if heuristic_path.exists():
            heuristic = json.loads(
                heuristic_path.read_text(encoding="utf-8-sig")
            )
            hover_w1 = heuristic.get("baseline_w1", {}).get("hover")
            if hover_w1 is not None:
                gates["set_mean_spatial_cost_beats_hover_by_10pct"] = (
                    float(np.mean(set_w1)) <= 0.90 * float(hover_w1)
                )

    reconstruction_ready = bool(gates) and all(gates.values())
    result = {
        "prefix": args.prefix,
        "runs": runs,
        "summary": summary,
        "reconstruction_readiness": {
            "ready": reconstruction_ready,
            "gates": gates,
            "interpretation": (
                "All phase-1 stability gates passed; reconstruction/Sinkhorn/"
                "PPG implementation may start."
                if reconstruction_ready
                else "Keep phase 1 active and inspect failed gates before "
                "adding reconstruction."
            ),
        },
    }
    output = (
        Path(args.output)
        if args.output
        else log_dir / f"{args.prefix}.summary.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("architecture  cost/slot       accept          W1       HAP-freeze")
    for architecture in ("mean", "flat", "set"):
        if architecture not in summary:
            continue
        row = summary[architecture]
        print(
            f"{architecture:>12}  "
            f"{row['cost_per_slot']['mean']:.4f}±"
            f"{row['cost_per_slot']['std']:.4f}  "
            f"{100 * row['accept_rate']['mean']:.1f}±"
            f"{100 * row['accept_rate']['std']:.1f}%  "
            f"{row['w1']['mean']:.1f}±{row['w1']['std']:.1f}  "
            f"{100 * row['freeze_hap_delta']['mean']:.1f}±"
            f"{100 * row['freeze_hap_delta']['std']:.1f}%"
        )
    print("\nreconstruction readiness:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"  => {'READY' if reconstruction_ready else 'NOT READY'}")


if __name__ == "__main__":
    main()
