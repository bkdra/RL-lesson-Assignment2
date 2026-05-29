from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_curve(csv_path: Path) -> tuple[list[int], list[float]]:
    episodes: list[int] = []
    rewards: list[float] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not contain a CSV header")

        if "episode" not in reader.fieldnames or "reward" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain 'episode' and 'reward' columns, got {reader.fieldnames}"
            )

        for row in reader:
            try:
                episodes.append(int(float(row["episode"])))
                rewards.append(float(row["reward"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {row}") from exc

    if not episodes:
        raise ValueError(f"{csv_path} does not contain any training rows")

    return episodes, rewards


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values[:]

    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(index + 1, window)
        smoothed.append(running_sum / current_window)
    return smoothed


def concatenate_training_curves(csv_paths: list[Path]) -> tuple[list[int], list[float], list[int]]:
    combined_episodes: list[int] = []
    combined_rewards: list[float] = []
    boundaries: list[int] = []
    episode_offset = 0

    for csv_path in csv_paths:
        episodes, rewards = load_training_curve(csv_path)
        for episode, reward in zip(episodes, rewards):
            combined_episodes.append(episode + episode_offset)
            combined_rewards.append(reward)
        if episodes:
            episode_offset += episodes[-1]
            boundaries.append(combined_episodes[-1])

    return combined_episodes, combined_rewards, boundaries


def plot_curves(csv_paths: list[Path], output_path: Path, smooth_window: int) -> None:
    episodes, rewards, boundaries = concatenate_training_curves(csv_paths)
    plotted_rewards = moving_average(rewards, smooth_window)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, plotted_rewards, linewidth=1.6, label=("Reward (smoothed)" if smooth_window>1 else "Reward"))

    # Linear trend fitted to the smoothed series
    try:
        coeffs = np.polyfit(np.array(episodes, dtype=float), np.array(plotted_rewards, dtype=float), 1)
        trend = np.polyval(coeffs, np.array(episodes, dtype=float))
        plt.plot(episodes, trend, linestyle="--", color="k", linewidth=1.2, label="Trend (linear)")
    except Exception:
        pass

    # Mark boundaries between concatenated runs
    for b in boundaries[:-1]:
        plt.axvline(b, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if smooth_window > 1:
        plt.title(f"Training Curve (concatenated) — moving average window={smooth_window}")
    else:
        plt.title("Training Curve (concatenated)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    resolved_paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            resolved_paths.extend(sorted(path.glob("*.csv")))
        else:
            resolved_paths.append(path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    return unique_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward-vs-episode training curves from one or more CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more CSV files, or directories containing CSV files",
    )
    parser.add_argument(
        "--output",
        default="training_curves.png",
        help="Path to save the combined plot image",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window for smoothing rewards",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = resolve_input_paths(args.inputs)

    if not csv_paths:
        raise FileNotFoundError("No CSV files were found in the provided inputs")

    output_path = Path(args.output).expanduser().resolve()
    plot_curves(csv_paths, output_path, smooth_window=max(1, args.smooth_window))

    print(f"Saved plot to {output_path}")
    for csv_path in csv_paths:
        print(f"Included {csv_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()