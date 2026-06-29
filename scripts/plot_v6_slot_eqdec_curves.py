"""Plot smoothed SetRec/slot-EqDec training curves from stdout logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUPS = [
    (
        "MeanPool Slot+Flat",
        [
            "diag1500_slot_eqdec_set_slot_flatcrit_seed1",
            "next1500_slot_eqdec_set_slot_flatcrit_seed2",
            "next1500_slot_eqdec_set_slot_flatcrit_seed3",
        ],
    ),
    (
        "LatentSlots Slot+Flat",
        ["next1500_slot_eqdec_latent_slot_flatcrit_seed1"],
    ),
    (
        "MeanPool Slot+Set detached",
        ["diag1500_slot_eqdec_set_slot_setcrit_seed1"],
    ),
    (
        "MeanPool Slot+Set separate",
        ["next1500_slot_eqdec_set_slot_setcrit_separate_seed1"],
    ),
    (
        "MeanPool Slot+Set shared-grad",
        ["next1500_slot_eqdec_set_slot_setcrit_sharedgrad_seed1"],
    ),
]


STEP_RE = re.compile(r"total num timesteps\s+(\d+)/")
TRAIN_RE = re.compile(r"average episode rewards is\s+(-?\d+(?:\.\d+)?)")
EVAL_RE = re.compile(
    r"eval average episode rewards of agent:\s+(-?\d+(?:\.\d+)?)"
)
BEST_RE = re.compile(
    r"new best fixed-validation reward\s+-?\d+(?:\.\d+)?\s+at\s+(\d+)\s+steps"
)


@dataclass
class Curve:
    experiment: str
    train_steps: np.ndarray
    train_rewards: np.ndarray
    eval_steps: np.ndarray
    eval_rewards: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_log(path: Path, experiment: str) -> Curve:
    train: list[tuple[int, float]] = []
    evals: list[tuple[int, float]] = []
    pending_step: int | None = None
    pending_reward: float | None = None
    last_step: int | None = None
    pending_eval_index: int | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
            last_step = step
            if pending_reward is not None:
                train.append((step, pending_reward))
                pending_reward = None
                pending_step = None
            else:
                pending_step = step
            continue

        train_match = TRAIN_RE.search(line)
        if train_match:
            reward = float(train_match.group(1))
            if pending_step is not None:
                train.append((pending_step, reward))
                pending_step = None
                pending_reward = None
            else:
                pending_reward = reward
            continue

        eval_match = EVAL_RE.search(line)
        if eval_match and last_step is not None:
            reward = float(eval_match.group(1))
            evals.append((last_step, reward))
            pending_eval_index = len(evals) - 1
            continue

        best_match = BEST_RE.search(line)
        if best_match and pending_eval_index is not None:
            step = int(best_match.group(1))
            _, reward = evals[pending_eval_index]
            evals[pending_eval_index] = (step, reward)
            pending_eval_index = None

    train = dedupe_steps(train)
    evals = dedupe_steps(evals)
    return Curve(
        experiment=experiment,
        train_steps=np.array([x for x, _ in train], dtype=float),
        train_rewards=np.array([y for _, y in train], dtype=float),
        eval_steps=np.array([x for x, _ in evals], dtype=float),
        eval_rewards=np.array([y for _, y in evals], dtype=float),
    )


def dedupe_steps(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items())


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
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


def aggregate(
    curves: list[Curve],
    *,
    attr_steps: str,
    attr_values: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    available = [
        curve
        for curve in curves
        if getattr(curve, attr_steps).size > 0
        and getattr(curve, attr_values).size > 0
    ]
    if not available:
        return np.array([]), np.array([]), None, []

    if len(available) == 1:
        steps = getattr(available[0], attr_steps)
        values = smooth(getattr(available[0], attr_values), window)
        return steps, values, None, [available[0].experiment]

    lo = max(float(getattr(curve, attr_steps).min()) for curve in available)
    hi = min(float(getattr(curve, attr_steps).max()) for curve in available)
    grid = sorted(
        {
            float(step)
            for curve in available
            for step in getattr(curve, attr_steps)
            if lo <= float(step) <= hi
        }
    )
    steps = np.array(grid, dtype=float)
    values = []
    for curve in available:
        curve_steps = getattr(curve, attr_steps)
        curve_values = smooth(getattr(curve, attr_values), window)
        values.append(np.interp(steps, curve_steps, curve_values))
    stacked = np.vstack(values)
    return (
        steps,
        stacked.mean(axis=0),
        stacked.std(axis=0, ddof=0),
        [curve.experiment for curve in available],
    )


def plot_curves(
    log_dir: Path,
    output_dir: Path,
    *,
    window: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    parsed: dict[str, Curve] = {}
    missing: list[str] = []

    for _, experiments in GROUPS:
        for experiment in experiments:
            if experiment in parsed:
                continue
            path = log_dir / f"{experiment}.log"
            if path.exists():
                parsed[experiment] = parse_log(path, experiment)
            else:
                missing.append(experiment)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_train, ax_eval = axes
    csv_rows: list[dict[str, str | float]] = []

    for idx, (label, experiments) in enumerate(GROUPS):
        curves = [parsed[name] for name in experiments if name in parsed]
        if not curves:
            continue
        color = colors(idx)
        train_steps, train_mean, train_std, used = aggregate(
            curves,
            attr_steps="train_steps",
            attr_values="train_rewards",
            window=window,
        )
        if train_steps.size:
            x = train_steps / 1_000_000.0
            suffix = f" ({len(used)} runs)" if len(used) > 1 else ""
            ax_train.plot(
                x,
                train_mean,
                label=f"{label}{suffix}",
                color=color,
                linewidth=2.2,
            )
            if train_std is not None:
                ax_train.fill_between(
                    x,
                    train_mean - train_std,
                    train_mean + train_std,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            for step, mean, std in zip(
                train_steps,
                train_mean,
                train_std if train_std is not None else np.zeros_like(train_mean),
            ):
                csv_rows.append(
                    {
                        "panel": "train",
                        "group": label,
                        "step": int(step),
                        "mean_reward": float(mean),
                        "std_reward": float(std),
                        "runs": ";".join(used),
                    }
                )

        eval_steps, eval_mean, eval_std, eval_used = aggregate(
            curves,
            attr_steps="eval_steps",
            attr_values="eval_rewards",
            window=1,
        )
        if eval_steps.size:
            x = eval_steps / 1_000_000.0
            ax_eval.plot(
                x,
                eval_mean,
                label=label,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                alpha=0.9,
            )
            if eval_std is not None:
                ax_eval.fill_between(
                    x,
                    eval_mean - eval_std,
                    eval_mean + eval_std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            for step, mean, std in zip(
                eval_steps,
                eval_mean,
                eval_std if eval_std is not None else np.zeros_like(eval_mean),
            ):
                csv_rows.append(
                    {
                        "panel": "eval",
                        "group": label,
                        "step": int(step),
                        "mean_reward": float(mean),
                        "std_reward": float(std),
                        "runs": ";".join(eval_used),
                    }
                )

    ax_train.set_title(
        "V6 HAP load-bearing: slot-level EqDec training curves",
        fontsize=15,
        pad=12,
    )
    ax_train.set_ylabel("Smoothed train reward (higher is better)")
    ax_eval.set_ylabel("Validation reward")
    ax_eval.set_xlabel("Environment steps (millions)")
    for ax in axes:
        ax.grid(True, alpha=0.24, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_train.legend(loc="lower right", fontsize=9, frameon=True)
    ax_eval.legend(loc="lower right", fontsize=8, frameon=True, ncol=2)
    note = (
        f"Training curves use centered moving average, window={window} log "
        "points. Shaded bands are std across available seeds; single-run "
        "groups have no band."
    )
    if missing:
        note += " Missing logs: " + ", ".join(missing)
    fig.text(0.01, 0.01, note, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    png_path = output_dir / "slot_eqdec_training_curves.png"
    pdf_path = output_dir / "slot_eqdec_training_curves.pdf"
    csv_path = output_dir / "slot_eqdec_training_curves.csv"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "panel",
                "group",
                "step",
                "mean_reward",
                "std_reward",
                "runs",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=root / "training_logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "eval_outputs" / "slot_eqdec_next_1500k",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered moving-average window in logged training points.",
    )
    args = parser.parse_args()
    png_path, pdf_path = plot_curves(
        args.log_dir,
        args.output_dir,
        window=args.smooth_window,
    )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
