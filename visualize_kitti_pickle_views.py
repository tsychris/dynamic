from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from kitti_dataloader import load_kitti_points, resolve_kitti_path

DEFAULT_QUERY_FILE = "/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle"
DEFAULT_KITTI_ROOT = "/TIEVNAS/KITTI"
DEFAULT_FALLBACK_ROOT = "/TIEVNAS/jyf/KITTI"
POINT_COLOR = "#1f77b4"


def load_pickle_records(pickle_path: str, split_index: int) -> tuple[dict[int, dict[str, Any]], str]:
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        if split_index < 0 or split_index >= len(obj):
            raise IndexError(f"split_index={split_index} is out of range for {pickle_path} with len={len(obj)}")
        records = obj[split_index]
        if not isinstance(records, dict):
            raise TypeError(f"Expected dict at list[{split_index}] in {pickle_path}, got {type(records)}")
        return records, "evaluation"

    if isinstance(obj, dict):
        return obj, "query"

    raise TypeError(f"Unsupported pickle structure in {pickle_path}: {type(obj)}")


def select_record_key(records: dict[int, dict[str, Any]], sample_index: int, record_key: int | None) -> tuple[int, int]:
    keys = sorted(records.keys())
    if record_key is not None:
        if record_key not in records:
            raise KeyError(f"record_key={record_key} is not present in the pickle.")
        return int(record_key), int(keys.index(record_key))

    if sample_index < 0 or sample_index >= len(keys):
        raise IndexError(f"sample_index={sample_index} is outside range [0, {len(keys) - 1}]")
    return int(keys[sample_index]), int(sample_index)


def resolve_record_path(record: dict[str, Any], kitti_root: str, fallback_root: str) -> str:
    for field in ("query_submap", "submap_path", "query"):
        raw_path = record.get(field)
        if raw_path is not None:
            return resolve_kitti_path(str(raw_path), kitti_root=kitti_root, fallback_root=fallback_root)
    raise KeyError("Record does not contain query_submap/submap_path/query.")


def maybe_subsample_points(points: np.ndarray, num_points: int | None) -> np.ndarray:
    if num_points is None or num_points <= 0 or points.shape[0] <= num_points:
        return points.astype(np.float32, copy=True)
    idx = np.linspace(0, points.shape[0] - 1, int(num_points), dtype=np.int64)
    return points[idx].astype(np.float32, copy=False)


def compute_axis_limits(points_xyz: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    margin = 0.08 * span + 0.8
    low = mins - margin
    high = maxs + margin
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1])), (float(low[2]), float(high[2]))


def scatter_plane(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    axis_x: int,
    xlabel: str,
    ylabel: str,
    title: str,
    point_size: float,
    aspect: float | str = "equal",
    y_tick_scale: float = 1.0,
) -> None:
    ax.scatter(
        x_values,
        y_values,
        s=point_size,
        c=POINT_COLOR,
        alpha=0.85,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(aspect, adjustable="box")
    if y_tick_scale != 1.0:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / y_tick_scale:.1f}"))
    ax.grid(alpha=0.2, linewidth=0.5)


def save_figure(
    points_xyz: np.ndarray,
    source_label: str,
    point_path: str,
    record_key: int,
    sample_index: int,
    out_path: Path,
    point_size: float,
    z_exaggeration: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)

    xlim, ylim, zlim = compute_axis_limits(points_xyz)
    z_scaled = points_xyz[:, 2] * float(z_exaggeration)
    zlim_scaled = (zlim[0] * float(z_exaggeration), zlim[1] * float(z_exaggeration))
    z_title_suffix = "" if np.isclose(z_exaggeration, 1.0) else f" (z x{float(z_exaggeration):.1f})"

    scatter_plane(
        axes[0],
        x_values=points_xyz[:, 0],
        y_values=points_xyz[:, 1],
        axis_x=0,
        xlabel="x (m)",
        ylabel="y (m)",
        title="XY View",
        point_size=point_size,
    )
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)

    scatter_plane(
        axes[1],
        x_values=points_xyz[:, 0],
        y_values=z_scaled,
        axis_x=0,
        xlabel="x (m)",
        ylabel="z (m)",
        title=f"XZ View{z_title_suffix}",
        point_size=point_size,
        y_tick_scale=float(z_exaggeration),
    )
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*zlim_scaled)

    scatter_plane(
        axes[2],
        x_values=points_xyz[:, 1],
        y_values=z_scaled,
        axis_x=1,
        xlabel="y (m)",
        ylabel="z (m)",
        title=f"YZ View{z_title_suffix}",
        point_size=point_size,
        y_tick_scale=float(z_exaggeration),
    )
    axes[2].set_xlim(*ylim)
    axes[2].set_ylim(*zlim_scaled)

    fig.suptitle(
        f"KITTI Pointcloud Projections\nsource={source_label} key={record_key} sample_index={sample_index}"
    )
    fig.text(
        0.5,
        0.01,
        f"path={point_path}\npoints={points_xyz.shape[0]}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize XY/XZ/YZ views for a KITTI pointcloud resolved from a pickle record."
    )
    parser.add_argument("--pickle-file", type=str, default=DEFAULT_QUERY_FILE, help="Path to query/evaluation pickle.")
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="List index for evaluation-style pickles. Ignored for plain dict query pickles.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset-local index used when --record-key is not provided.",
    )
    parser.add_argument("--record-key", type=int, default=None, help="Original record key from the pickle.")
    parser.add_argument("--kitti-root", type=str, default=DEFAULT_KITTI_ROOT, help="Primary KITTI root.")
    parser.add_argument("--fallback-root", type=str, default=DEFAULT_FALLBACK_ROOT, help="Fallback KITTI root.")
    parser.add_argument(
        "--vis-num-points",
        type=int,
        default=None,
        help="Number of points to visualize. Use <=0 or omit to keep the full pointcloud.",
    )
    parser.add_argument("--point-size", type=float, default=1.2, help="Scatter marker size.")
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=4.0,
        help="Visual exaggeration factor for the z axis in XZ and YZ views. Use 1.0 for no exaggeration.",
    )
    parser.add_argument("--out-path", type=str, default=None, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pickle_path = str(Path(args.pickle_file).expanduser().resolve())
    records, pickle_type = load_pickle_records(pickle_path=pickle_path, split_index=int(args.split_index))
    record_key, sample_index = select_record_key(
        records=records,
        sample_index=int(args.sample_index),
        record_key=args.record_key,
    )
    record = records[record_key]
    point_path = resolve_record_path(
        record=record,
        kitti_root=str(args.kitti_root),
        fallback_root=str(args.fallback_root),
    )

    raw_points = load_kitti_points(point_path, use_intensity=False)
    vis_points = maybe_subsample_points(raw_points[:, :3], num_points=args.vis_num_points)

    output_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path is not None
        else (
            Path(__file__).resolve().parent
            / "vis_pointcloud_views"
            / Path(pickle_path).stem
            / f"key_{record_key:06d}.png"
        )
    )

    save_figure(
        points_xyz=vis_points,
        source_label=f"{Path(pickle_path).name} ({pickle_type})",
        point_path=point_path,
        record_key=record_key,
        sample_index=sample_index,
        out_path=output_path,
        point_size=float(args.point_size),
        z_exaggeration=float(args.z_exaggeration),
    )

    print(f"[INFO] pickle={pickle_path}")
    print(f"[INFO] pickle_type={pickle_type}")
    print(f"[INFO] record_key={record_key} sample_index={sample_index}")
    print(f"[INFO] point_path={point_path}")
    print(f"[INFO] raw_points={raw_points.shape[0]} vis_points={vis_points.shape[0]}")
    print(f"[INFO] z_exaggeration={float(args.z_exaggeration):.2f}")
    print(f"[INFO] saved_figure={output_path}")


if __name__ == "__main__":
    main()
