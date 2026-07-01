"""Plot reset-perm flat-descriptor dimension sweep validation curves."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SLOT_COUNT = 350.0
SMOOTH_WINDOW = 3

GROUPS = {
    16: ["resetperm1500_flatdesc_dim16_flatcrit_seed1.log"],
    32: ["resetperm1500_flatdesc_dim32_flatcrit_seed1.log"],
    64: ["resetperm1500_flatdesc_dim64_flatcrit_seed1.log"],
    128: ["resetperm1500_flatdesc_dim128_flatcrit_seed1.log"],
    256: [
        "resetperm1500_flat_descriptor_flatcrit_seed1.log",
        "resetperm1500_flat_descriptor_flatcrit_seed2.log",
        "resetperm1500_flat_descriptor_flatcrit_seed3.log",
    ],
}

STEP_RE = re.compile(r"total num timesteps\s+(\d+)/")
EVAL_RE = re.compile(r"eval average episode rewards of agent:\s+(-?\d+(?:\.\d+)?)")
BEST_RE = re.compile(
    r"new best fixed-validation reward\s+-?\d+(?:\.\d+)?\s+at\s+(\d+)\s+steps"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_eval_costs(log_path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    last_step: int | None = None
    pending_eval_index: int | None = None

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if step_match := STEP_RE.search(line):
            last_step = int(step_match.group(1))
            pending_eval_index = None
            continue

        if eval_match := EVAL_RE.search(line):
            if last_step is None:
                continue
            reward = float(eval_match.group(1))
            points.append((last_step, -reward / SLOT_COUNT))
            pending_eval_index = len(points) - 1
            continue

        if best_match := BEST_RE.search(line):
            if pending_eval_index is not None:
                step = int(best_match.group(1))
                _, cost = points[pending_eval_index]
                points[pending_eval_index] = (step, cost)
                pending_eval_index = None

    by_step: dict[int, float] = {}
    for step, cost in points:
        by_step[step] = cost
    return sorted(by_step.items())


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.copy()
    width = min(window, int(values.size))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(padded, kernel, mode="valid")


def seed_from_name(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else "?"


def aggregate(points_by_run: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(points_by_run) == 1:
        steps = np.array([step for step, _ in points_by_run[0]], dtype=float)
        costs = np.array([cost for _, cost in points_by_run[0]], dtype=float)
        return steps, smooth(costs), None

    common_steps = sorted(set.intersection(*(set(step for step, _ in run) for run in points_by_run)))
    if not common_steps:
        raise RuntimeError("No common eval steps for multi-seed aggregate.")

    smoothed_runs = []
    for run in points_by_run:
        by_step = dict(run)
        costs = np.array([by_step[step] for step in common_steps], dtype=float)
        smoothed_runs.append(smooth(costs))

    stacked = np.vstack(smoothed_runs)
    return (
        np.array(common_steps, dtype=float),
        stacked.mean(axis=0),
        stacked.std(axis=0, ddof=0),
    )


def main() -> None:
    root = repo_root()
    log_dir = root / "training_logs"
    out_dir = root / "eval_outputs" / "resetperm_flat_descriptor_dim_sweep_1500k"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    colors = plt.get_cmap("tab10")
    csv_rows: list[dict[str, str | int | float]] = []
    summary_rows: list[dict[str, str | int | float]] = []

    for index, (dim, logs) in enumerate(GROUPS.items()):
        points_by_run = []
        used_logs = []
        for log_name in logs:
            log_path = log_dir / log_name
            if not log_path.exists():
                print(f"missing: {log_path}")
                continue
            points = parse_eval_costs(log_path)
            if not points:
                print(f"no eval points: {log_path}")
                continue
            points_by_run.append(points)
            used_logs.append(log_name)

            raw_steps = np.array([step for step, _ in points], dtype=float)
            raw_costs = np.array([cost for _, cost in points], dtype=float)
            smoothed = smooth(raw_costs)
            seed = seed_from_name(log_name)
            for step, cost, smoothed_cost in zip(raw_steps, raw_costs, smoothed):
                csv_rows.append(
                    {
                        "dim": dim,
                        "seed": seed,
                        "step": int(step),
                        "cost_per_slot": float(cost),
                        "smoothed_cost_per_slot": float(smoothed_cost),
                    }
                )

        if not points_by_run:
            continue

        steps, mean_cost, std_cost = aggregate(points_by_run)
        color = colors(index)
        label = f"dim={dim}"
        if len(points_by_run) > 1:
            label += f" mean +/- std (n={len(points_by_run)})"
        ax.plot(steps / 1_000_000, mean_cost, label=label, color=color, linewidth=2.2)
        if std_cost is not None:
            ax.fill_between(
                steps / 1_000_000,
                mean_cost - std_cost,
                mean_cost + std_cost,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        summary_rows.append(
            {
                "dim": dim,
                "runs": len(points_by_run),
                "last_step": int(steps[-1]),
                "last_smoothed_cost_per_slot": float(mean_cost[-1]),
                "best_smoothed_cost_per_slot": float(np.min(mean_cost)),
                "logs": ";".join(used_logs),
            }
        )

    ax.set_title("Reset-perm flat descriptor dimension sweep")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Fixed-validation cost / slot (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png_path = out_dir / "flat_descriptor_dim_sweep_validation_cost.png"
    pdf_path = out_dir / "flat_descriptor_dim_sweep_validation_cost.pdf"
    csv_path = out_dir / "flat_descriptor_dim_sweep_validation_cost.csv"
    summary_path = out_dir / "flat_descriptor_dim_sweep_summary.csv"

    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dim", "seed", "step", "cost_per_slot", "smoothed_cost_per_slot"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dim",
                "runs",
                "last_step",
                "last_smoothed_cost_per_slot",
                "best_smoothed_cost_per_slot",
                "logs",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
