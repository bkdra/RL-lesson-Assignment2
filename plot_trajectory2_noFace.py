from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}, got {data.shape[1]}")
    return data[:, :3]


def set_equal_3d_aspect(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    spans = maxs - mins
    radius = max(spans.max() / 2.0, 1e-9)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def find_nearest_xy_point(points: np.ndarray, target: np.ndarray) -> tuple[int, np.ndarray, float]:
    distances = np.linalg.norm(points[:, :2] - target[:2], axis=1)
    index = int(np.argmin(distances))
    return index, points[index], float(distances[index])


def plot_projection(
    ax,
    points: np.ndarray,
    x_idx: int,
    y_idx: int,
    title: str,
    xlabel: str,
    ylabel: str,
    compare_point: np.ndarray | None = None,
    compare_label: str = "compare point",
) -> None:
    ax.plot(points[:, x_idx], points[:, y_idx], color="#1f77b4", linewidth=2)
    ax.scatter(points[0, x_idx], points[0, y_idx], color="#2ca02c", s=45, label="start")
    ax.scatter(points[-1, x_idx], points[-1, y_idx], color="#d62728", s=45, label="end")

    if compare_point is not None:
        nearest_index, nearest_point, _ = find_nearest_xy_point(points, compare_point)
        z_diff = float(compare_point[2] - nearest_point[2])
        ax.scatter(compare_point[x_idx], compare_point[y_idx], color="#ff7f0e", s=65, label=compare_label)
        ax.scatter(nearest_point[x_idx], nearest_point[y_idx], color="#9467bd", s=45, label="nearest traj point")
        ax.plot(
            [compare_point[x_idx], nearest_point[x_idx]],
            [compare_point[y_idx], nearest_point[y_idx]],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.annotate(
            f"Δz={z_diff:.6f}",
            xy=(compare_point[x_idx], compare_point[y_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            color="#ff7f0e",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def build_figure(points: np.ndarray, title: str, compare_point: np.ndarray | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)

    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", linewidth=2)
    ax3d.scatter(
        [points[0, 0], points[-1, 0]],
        [points[0, 1], points[-1, 1]],
        [points[0, 2], points[-1, 2]],
        color=["#2ca02c", "#d62728"],
        s=45,
        label="start/end",
    )

    if compare_point is not None:
        nearest_index, nearest_point, _ = find_nearest_xy_point(points, compare_point)
        z_diff = float(compare_point[2] - nearest_point[2])
        ax3d.scatter(
            [compare_point[0]],
            [compare_point[1]],
            [compare_point[2]],
            color="#ff7f0e",
            s=65,
            label="compare point",
        )
        ax3d.plot(
            [compare_point[0], nearest_point[0]],
            [compare_point[1], nearest_point[1]],
            [compare_point[2], nearest_point[2]],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax3d.text(
            compare_point[0],
            compare_point[1],
            compare_point[2],
            f"  Δz={z_diff:.6f}",
            color="#ff7f0e",
        )
    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    set_equal_3d_aspect(ax3d, points)
    ax3d.legend(loc="best")

    ax_xy = fig.add_subplot(1, 3, 2)
    plot_projection(
        ax_xy,
        points,
        0,
        1,
        "Top-down view (X-Y)",
        "X",
        "Y",
        compare_point=compare_point,
        compare_label="compare point",
    )

    ax_yz = fig.add_subplot(1, 3, 3)
    plot_projection(
        ax_yz,
        points,
        1,
        2,
        "Side view (Y-Z)",
        "Y",
        "Z",
        compare_point=compare_point,
        compare_label="compare point",
    )

    fig.suptitle(title)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the shape of trajectory2_noFace.txt")
    parser.add_argument(
        "input",
        nargs="?",
        default="trajectory2_noFace.txt",
        help="Path to the trajectory text file",
    )
    parser.add_argument(
        "--output",
        default="trajectory2_noFace.png",
        help="Where to save the plot image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving",
    )
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional comparison point to add to the plot and compare Z against the trajectory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    points = load_trajectory(input_path)
    compare_point = np.array(args.point, dtype=float) if args.point is not None else None
    figure = build_figure(points, input_path.name, compare_point=compare_point)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if compare_point is not None:
        nearest_index, nearest_point, xy_distance = find_nearest_xy_point(points, compare_point)
        z_diff = float(compare_point[2] - nearest_point[2])
        print(
            "Comparison point: "
            f"({compare_point[0]:.6f}, {compare_point[1]:.6f}, {compare_point[2]:.6f})"
        )
        print(
            "Nearest trajectory point: "
            f"index={nearest_index}, "
            f"({nearest_point[0]:.6f}, {nearest_point[1]:.6f}, {nearest_point[2]:.6f})"
        )
        print(f"XY distance to nearest point: {xy_distance:.6f}")
        print(f"Z difference (point - trajectory): {z_diff:.6f}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
