from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


ACTIVE_COLOR = "#d62728"
INACTIVE_COLOR = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read batch occlusion summary JSON and plot box center/range distributions."
    )
    parser.add_argument("--summary-json", type=str, required=True, help="Path to summary JSON.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <summary-parent>/<summary-stem>_plots.",
    )
    parser.add_argument("--xy-alpha", type=float, default=0.55)
    parser.add_argument("--hist-bins", type=int, default=30)
    parser.add_argument("--hexbin-gridsize", type=int, default=22)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_boxes(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for sample in summary["samples"]:
        sample_index = int(sample["sample_index"])
        query_key = int(sample["query_key"])
        for box_idx, box in enumerate(sample["boxes"]):
            center = np.asarray(box["center"], dtype=np.float32)
            size = np.asarray(box["size"], dtype=np.float32)
            yaw = float(box["yaw"])
            is_active = 1.0 if bool(box["active"]) else 0.0
            rng = float(np.sqrt(center[0] ** 2 + center[1] ** 2))
            rows.append(
                [
                    float(sample_index),
                    float(query_key),
                    float(box_idx),
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                    yaw,
                    is_active,
                    rng,
                ]
            )
    if not rows:
        raise RuntimeError("No boxes found in summary JSON.")
    arr = np.asarray(rows, dtype=np.float32)
    return {
        "sample_index": arr[:, 0],
        "query_key": arr[:, 1],
        "box_index": arr[:, 2],
        "center_x": arr[:, 3],
        "center_y": arr[:, 4],
        "center_z": arr[:, 5],
        "size_l": arr[:, 6],
        "size_w": arr[:, 7],
        "size_h": arr[:, 8],
        "yaw": arr[:, 9],
        "active": arr[:, 10] > 0.5,
        "range_xy": arr[:, 11],
    }


def masked(data: dict[str, np.ndarray], active: bool) -> dict[str, np.ndarray]:
    mask = data["active"] if active else ~data["active"]
    return {k: v[mask] for k, v in data.items()}


def plot_main_figure(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    xy_alpha: float,
    hist_bins: int,
) -> None:
    active = masked(data, active=True)
    inactive = masked(data, active=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(
        inactive["center_x"],
        inactive["center_y"],
        s=18,
        c=INACTIVE_COLOR,
        alpha=xy_alpha,
        linewidths=0.0,
        label=f"inactive ({inactive['center_x'].shape[0]})",
    )
    ax.scatter(
        active["center_x"],
        active["center_y"],
        s=24,
        c=ACTIVE_COLOR,
        alpha=min(0.9, xy_alpha + 0.15),
        linewidths=0.0,
        label=f"active ({active['center_x'].shape[0]})",
    )
    ax.set_title("center_x vs center_y")
    ax.set_xlabel("center_x (m)")
    ax.set_ylabel("center_y (m)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    ax.scatter(
        inactive["center_x"],
        inactive["center_z"],
        s=18,
        c=INACTIVE_COLOR,
        alpha=xy_alpha,
        linewidths=0.0,
        label="inactive",
    )
    ax.scatter(
        active["center_x"],
        active["center_z"],
        s=24,
        c=ACTIVE_COLOR,
        alpha=min(0.9, xy_alpha + 0.15),
        linewidths=0.0,
        label="active",
    )
    ax.set_title("center_x vs center_z")
    ax.set_xlabel("center_x (m)")
    ax.set_ylabel("center_z (m)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    bins = np.linspace(
        float(np.min(data["range_xy"])),
        float(np.max(data["range_xy"])) + 1e-6,
        int(hist_bins),
    )
    ax.hist(
        inactive["range_xy"],
        bins=bins,
        color=INACTIVE_COLOR,
        alpha=0.55,
        label="inactive",
        density=True,
    )
    ax.hist(
        active["range_xy"],
        bins=bins,
        color=ACTIVE_COLOR,
        alpha=0.55,
        label="active",
        density=True,
    )
    ax.axvline(float(np.mean(inactive["range_xy"])), color=INACTIVE_COLOR, linestyle="--", linewidth=1.4)
    ax.axvline(float(np.mean(active["range_xy"])), color=ACTIVE_COLOR, linestyle="--", linewidth=1.4)
    ax.set_title(r"range = sqrt(x^2 + y^2)")
    ax.set_xlabel("range_xy (m)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    labels = ["x", "y", "z", "range"]
    active_series = [
        active["center_x"],
        active["center_y"],
        active["center_z"],
        active["range_xy"],
    ]
    inactive_series = [
        inactive["center_x"],
        inactive["center_y"],
        inactive["center_z"],
        inactive["range_xy"],
    ]
    positions_inactive = np.arange(len(labels)) * 2.0 - 0.35
    positions_active = np.arange(len(labels)) * 2.0 + 0.35
    bp_inactive = ax.boxplot(
        inactive_series,
        positions=positions_inactive,
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    bp_active = ax.boxplot(
        active_series,
        positions=positions_active,
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    for patch in bp_inactive["boxes"]:
        patch.set(facecolor=INACTIVE_COLOR, alpha=0.5)
    for patch in bp_active["boxes"]:
        patch.set(facecolor=ACTIVE_COLOR, alpha=0.5)
    ax.set_xticks(np.arange(len(labels)) * 2.0)
    ax.set_xticklabels(labels)
    ax.set_title("active vs inactive distribution compare")
    ax.set_ylabel("value")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=INACTIVE_COLOR, alpha=0.5, label="inactive"),
            plt.Rectangle((0, 0), 1, 1, color=ACTIVE_COLOR, alpha=0.5, label="active"),
        ],
        frameon=False,
    )

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bev_compare(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    gridsize: int,
) -> None:
    active = masked(data, active=True)
    inactive = masked(data, active=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, subset, panel_title in [
        (axes[0], active, "active box centers (x-y)"),
        (axes[1], inactive, "inactive box centers (x-y)"),
    ]:
        hb = ax.hexbin(
            subset["center_x"],
            subset["center_y"],
            gridsize=int(gridsize),
            mincnt=1,
            cmap="viridis",
        )
        fig.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel("center_x (m)")
        ax.set_ylabel("center_y (m)")
        ax.set_title(panel_title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_stats(data: dict[str, np.ndarray]) -> dict[str, Any]:
    def stats_for(subset: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            "count": int(subset["center_x"].shape[0]),
            "center_x_mean": float(np.mean(subset["center_x"])),
            "center_y_mean": float(np.mean(subset["center_y"])),
            "center_z_mean": float(np.mean(subset["center_z"])),
            "range_xy_mean": float(np.mean(subset["range_xy"])),
            "range_xy_median": float(np.median(subset["range_xy"])),
            "range_xy_std": float(np.std(subset["range_xy"])),
        }

    active = masked(data, active=True)
    inactive = masked(data, active=False)
    return {
        "all": stats_for(data),
        "active": stats_for(active),
        "inactive": stats_for(inactive),
    }


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    summary = load_summary(summary_path)
    data = collect_boxes(summary)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else summary_path.parent / f"{summary_path.stem}_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"Occlusion Box Distribution\n"
        f"source={summary.get('source', 'unknown')} "
        f"box_count={summary.get('box_count', 'unknown')} "
        f"samples={len(summary.get('samples', []))}"
    )

    main_png = out_dir / "box_distribution_overview.png"
    bev_png = out_dir / "active_inactive_bev_compare.png"
    stats_json = out_dir / "distribution_stats.json"

    plot_main_figure(
        data=data,
        out_path=main_png,
        title=title,
        xy_alpha=float(args.xy_alpha),
        hist_bins=int(args.hist_bins),
    )
    plot_bev_compare(
        data=data,
        out_path=bev_png,
        title=title,
        gridsize=int(args.hexbin_gridsize),
    )
    stats = summarize_stats(data)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[INFO] summary_json={summary_path}")
    print(f"[INFO] total_boxes={data['center_x'].shape[0]}")
    print(f"[INFO] active_boxes={int(np.sum(data['active']))}")
    print(f"[INFO] inactive_boxes={int(np.sum(~data['active']))}")
    print(f"[INFO] saved_overview={main_png}")
    print(f"[INFO] saved_bev_compare={bev_png}")
    print(f"[INFO] saved_stats={stats_json}")


if __name__ == "__main__":
    main()
