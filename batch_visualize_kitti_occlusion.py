from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from visualize_kitti_occlusion import (
    BOX_COLOR,
    DROP_COLOR,
    INSERT_COLOR,
    KEEP_COLOR,
    KITTIPointCloudQueryDataset,
    AdversarialOcclusionGenerator,
    Line2D,
    build_occlusion_cases,
    compute_axis_limits,
    downsample_for_plot,
    draw_box_bev,
    draw_box_xz,
    infer_generator_config,
    load_checkpoint_file,
    load_kitti_points,
    matplotlib,
    maybe_subsample_points,
    parse_box_count_list,
    resolve_data_args_for_random_init,
    resolve_device,
    resolve_model_args,
    resolve_record_path,
    sample_or_pad_points,
    scatter_xy,
    scatter_xz,
    select_query_key,
)
from lpr_models import build_descriptor_model

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DEFAULT_OUT_DIR = Path("/media/autolab/tsy/dynamic/vis_occlusion_batch")


@dataclass
class SampleResult:
    sample_index: int
    query_key: int
    point_path: str
    vis_points_xyz: np.ndarray
    case: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch visualize KITTI generator outputs over a range of sample indices."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to adversarial checkpoint.")
    parser.add_argument("--query-file", type=str, default=None, help="Override query pickle used by the checkpoint.")
    parser.add_argument("--kitti-root", type=str, default=None, help="Override KITTI root used by the checkpoint.")
    parser.add_argument("--fallback-root", type=str, default=None, help="Override fallback KITTI root.")
    parser.add_argument("--sample-start", type=int, default=1, help="Inclusive dataset-local start index.")
    parser.add_argument("--sample-stop", type=int, default=1000, help="Inclusive dataset-local stop index.")
    parser.add_argument("--sample-step", type=int, default=10, help="Step size for sampled indices.")
    parser.add_argument("--box-counts", type=str, default="3", help="Comma-separated active box counts.")
    parser.add_argument(
        "--samples-per-figure",
        type=int,
        default=6,
        help="Number of samples packed into one PNG.",
    )
    parser.add_argument(
        "--vis-num-points",
        type=int,
        default=None,
        help="Number of points to visualize. Default uses model num_points. Use <=0 for the full scan.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=12000,
        help="Cap for rendered points per category in each subplot.",
    )
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=4.0,
        help="Visual exaggeration factor for z in X-Z views.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-object-insertion", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--prefix",
        type=str,
        default="kitti_boxscan",
        help="Prefix for PNG and JSON outputs.",
    )
    return parser.parse_args()


def build_descriptor_from_checkpoint(
    ckpt: dict[str, Any],
    ckpt_args: dict[str, Any],
    model_num_points: int,
    use_intensity: bool,
    device: torch.device,
) -> torch.nn.Module | None:
    if "descriptor" not in ckpt:
        return None
    descriptor = build_descriptor_model(
        arch=str(ckpt_args.get("descriptor_arch", "pointnetvlad")),
        num_points=model_num_points,
        emb_dim=int(ckpt_args.get("emb_dim", 256)),
        in_channels=4 if use_intensity else 3,
    )
    descriptor.load_state_dict(ckpt["descriptor"], strict=True)
    descriptor.to(device)
    descriptor.eval()
    return descriptor


def load_generator_and_dataset(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    torch.device,
    AdversarialOcclusionGenerator,
    torch.nn.Module | None,
    KITTIPointCloudQueryDataset,
    int,
    bool,
    float,
    float,
]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = load_checkpoint_file(checkpoint_path)
    ckpt_args = ckpt.get("args", {})
    if "generator" not in ckpt:
        raise KeyError(f"Checkpoint does not contain a generator state dict: {checkpoint_path}")

    feature_dim, num_boxes = infer_generator_config(ckpt["generator"])
    model_num_points = int(ckpt_args.get("num_points", 4096))
    use_intensity = bool(ckpt_args.get("use_intensity", False))
    point_weight = float(ckpt_args.get("point_weight", 1.0))
    geom_weight = float(ckpt_args.get("geom_weight", 2.0))
    query_file, kitti_root, fallback_root = resolve_model_args(
        ckpt_args=ckpt_args,
        query_file=args.query_file,
        kitti_root=args.kitti_root,
        fallback_root=args.fallback_root,
    )

    dataset = KITTIPointCloudQueryDataset(
        query_filepath=query_file,
        kitti_root=kitti_root,
        fallback_root=fallback_root,
        num_points=model_num_points,
        use_intensity=use_intensity,
        random_sample=False,
        prefer_cached=True,
    )

    descriptor = build_descriptor_from_checkpoint(
        ckpt=ckpt,
        ckpt_args=ckpt_args,
        model_num_points=model_num_points,
        use_intensity=use_intensity,
        device=device,
    )

    generator = AdversarialOcclusionGenerator(
        num_boxes=num_boxes,
        feature_dim=feature_dim,
        points_per_box=int(ckpt_args.get("points_per_box", 64)),
        temperature=float(ckpt_args.get("temperature", 0.2)),
        point_weight=point_weight,
        geom_weight=geom_weight,
    )
    generator.load_state_dict(ckpt["generator"], strict=True)
    generator.to(device)
    generator.eval()

    return (
        ckpt,
        ckpt_args,
        device,
        generator,
        descriptor,
        dataset,
        model_num_points,
        use_intensity,
        point_weight,
        geom_weight,
    )


def build_sample_result(
    dataset: KITTIPointCloudQueryDataset,
    generator: AdversarialOcclusionGenerator,
    descriptor: torch.nn.Module | None,
    sample_index: int,
    active_box_count: int,
    device: torch.device,
    use_intensity: bool,
    model_num_points: int,
    vis_num_points: int | None,
    use_object_insertion: bool,
) -> SampleResult:
    query_key = select_query_key(dataset=dataset, sample_index=sample_index, query_key=None)
    record = dataset.queries[query_key]
    point_path = resolve_record_path(dataset=dataset, record=record)

    raw_points = load_kitti_points(point_path, use_intensity=use_intensity)
    vis_points = maybe_subsample_points(raw_points, num_points=vis_num_points)
    model_points = sample_or_pad_points(raw_points, num_points=model_num_points, random_sample=False).astype(
        np.float32
    )

    cases = build_occlusion_cases(
        generator=generator,
        descriptor=descriptor,
        vis_points=vis_points,
        model_points=model_points,
        active_box_counts=[active_box_count],
        device=device,
        use_object_insertion=use_object_insertion,
    )
    return SampleResult(
        sample_index=int(sample_index),
        query_key=int(query_key),
        point_path=point_path,
        vis_points_xyz=vis_points[:, :3].astype(np.float32, copy=False),
        case=cases[0],
    )


def render_group_figure(
    group_results: Sequence[SampleResult],
    out_path: Path,
    title: str,
    max_plot_points: int,
    z_exaggeration: float,
) -> None:
    n = len(group_results)
    rows = int(math.ceil(n / 2.0))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.8 * rows), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for ax in axes.reshape(-1):
        ax.axis("off")

    z_scale = float(z_exaggeration)
    z_title_suffix = "" if np.isclose(z_scale, 1.0) else f" z x{z_scale:.1f}"

    for idx, result in enumerate(group_results):
        row = idx // 2
        col0 = 2 * (idx % 2)
        bev_ax = axes[row, col0]
        xz_ax = axes[row, col0 + 1]
        bev_ax.axis("on")
        xz_ax.axis("on")

        case = result.case
        xlim, ylim, zlim = compute_axis_limits(result.vis_points_xyz)
        zlim_scaled = (zlim[0] * z_scale, zlim[1] * z_scale)

        kept_plot = downsample_for_plot(case.kept_points_xyz, max_points=max_plot_points)
        dropped_plot = downsample_for_plot(case.dropped_points_xyz, max_points=max_plot_points)
        inserted_plot = (
            None
            if case.inserted_points_xyz is None
            else downsample_for_plot(case.inserted_points_xyz, max_points=max_plot_points)
        )

        scatter_xy(bev_ax, kept_plot, KEEP_COLOR, "kept", size=1.2)
        scatter_xy(bev_ax, dropped_plot, DROP_COLOR, "removed", size=1.8)
        if inserted_plot is not None:
            scatter_xy(bev_ax, inserted_plot, INSERT_COLOR, "inserted", size=1.8)
        for center, size, yaw, is_active in zip(case.centers, case.sizes, case.yaws, case.active_box_mask):
            if not bool(is_active):
                continue
            draw_box_bev(bev_ax, center=center, size=size, yaw=float(yaw), color=BOX_COLOR)
        bev_ax.set_title(
            f"BEV idx={result.sample_index} key={result.query_key}\n"
            f"occ={case.actual_occluded_fraction:.3f}"
            + (f" cos={case.cosine_similarity:.3f}" if case.cosine_similarity is not None else "")
        )
        bev_ax.set_xlim(*xlim)
        bev_ax.set_ylim(*ylim)
        bev_ax.set_xlabel("x (m)")
        bev_ax.set_ylabel("y (m)")
        bev_ax.set_aspect("equal", adjustable="box")
        bev_ax.grid(alpha=0.2, linewidth=0.5)

        scatter_xz(xz_ax, kept_plot, KEEP_COLOR, "kept", size=1.2, z_scale=z_scale)
        scatter_xz(xz_ax, dropped_plot, DROP_COLOR, "removed", size=1.8, z_scale=z_scale)
        if inserted_plot is not None:
            scatter_xz(xz_ax, inserted_plot, INSERT_COLOR, "inserted", size=1.8, z_scale=z_scale)
        for center, size, yaw, is_active in zip(case.centers, case.sizes, case.yaws, case.active_box_mask):
            if not bool(is_active):
                continue
            draw_box_xz(xz_ax, center=center, size=size, yaw=float(yaw), color=BOX_COLOR, z_scale=z_scale)
        xz_ax.set_title(
            f"X-Z{z_title_suffix}\n"
            f"active={int(case.active_box_count)} removed={case.dropped_points_xyz.shape[0]}"
        )
        xz_ax.set_xlim(*xlim)
        xz_ax.set_ylim(*zlim_scaled)
        xz_ax.set_xlabel("x (m)")
        xz_ax.set_ylabel("z (m)")
        xz_ax.set_aspect("equal", adjustable="box")
        if not np.isclose(z_scale, 1.0):
            xz_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / z_scale:.1f}"))
        xz_ax.grid(alpha=0.2, linewidth=0.5)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=KEEP_COLOR, markersize=6, label="kept/clean"),
        Line2D([], [], marker="o", linestyle="", color=DROP_COLOR, markersize=6, label="removed"),
        Line2D([], [], marker="o", linestyle="", color=INSERT_COLOR, markersize=6, label="inserted"),
        Line2D([], [], color=BOX_COLOR, linewidth=1.8, label="active box"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sample_to_json_dict(result: SampleResult) -> dict[str, Any]:
    case = result.case
    return {
        "sample_index": int(result.sample_index),
        "query_key": int(result.query_key),
        "point_path": result.point_path,
        "visualized_points": int(result.vis_points_xyz.shape[0]),
        "active_box_count": int(case.active_box_count),
        "active_box_indices": [int(i) for i, flag in enumerate(case.active_box_mask.tolist()) if flag],
        "actual_occluded_fraction": float(case.actual_occluded_fraction),
        "num_removed_points": int(case.dropped_points_xyz.shape[0]),
        "num_kept_points": int(case.kept_points_xyz.shape[0]),
        "num_inserted_points": int(0 if case.inserted_points_xyz is None else case.inserted_points_xyz.shape[0]),
        "cosine_similarity": None if case.cosine_similarity is None else float(case.cosine_similarity),
        "boxes": [
            {
                "center": [float(v) for v in center.tolist()],
                "size": [float(v) for v in size.tolist()],
                "yaw": float(yaw),
                "active": bool(is_active),
            }
            for center, size, yaw, is_active in zip(case.centers, case.sizes, case.yaws, case.active_box_mask)
        ],
    }


def main() -> None:
    args = parse_args()
    (
        ckpt,
        ckpt_args,
        device,
        generator,
        descriptor,
        dataset,
        model_num_points,
        use_intensity,
        point_weight,
        geom_weight,
    ) = load_generator_and_dataset(args)

    if args.samples_per_figure < 1:
        raise ValueError("--samples-per-figure must be >= 1")

    active_box_counts = parse_box_count_list(args.box_counts, max_boxes=generator.num_boxes)
    if len(active_box_counts) != 1:
        raise ValueError("This batch script expects exactly one box-count value, e.g. --box-counts 3")
    active_box_count = int(active_box_counts[0])

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = list(range(int(args.sample_start), int(args.sample_stop) + 1, int(args.sample_step)))
    sample_indices = [idx for idx in sample_indices if 0 <= idx < len(dataset)]
    if not sample_indices:
        raise RuntimeError("No valid sample indices remain after range filtering.")

    vis_num_points = args.vis_num_points
    if vis_num_points is not None and vis_num_points <= 0:
        vis_num_points = None
    if vis_num_points is None:
        vis_num_points = model_num_points

    source_label = Path(args.checkpoint).name
    use_object_insertion = not args.no_object_insertion

    summary: dict[str, Any] = {
        "source": source_label,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "query_file": dataset.query_filepath,
        "kitti_root": dataset.kitti_root,
        "fallback_root": dataset.fallback_root,
        "sample_start": int(args.sample_start),
        "sample_stop": int(args.sample_stop),
        "sample_step": int(args.sample_step),
        "sample_indices": [int(v) for v in sample_indices],
        "box_count": active_box_count,
        "samples_per_figure": int(args.samples_per_figure),
        "model_num_points": int(model_num_points),
        "vis_num_points": int(vis_num_points),
        "use_intensity": bool(use_intensity),
        "use_object_insertion": bool(use_object_insertion),
        "point_weight": float(point_weight),
        "geom_weight": float(geom_weight),
        "z_exaggeration": float(args.z_exaggeration),
        "samples": [],
        "figures": [],
    }

    group_results: list[SampleResult] = []
    figure_idx = 0
    for sample_index in sample_indices:
        result = build_sample_result(
            dataset=dataset,
            generator=generator,
            descriptor=descriptor,
            sample_index=sample_index,
            active_box_count=active_box_count,
            device=device,
            use_intensity=use_intensity,
            model_num_points=model_num_points,
            vis_num_points=vis_num_points,
            use_object_insertion=use_object_insertion,
        )
        summary["samples"].append(sample_to_json_dict(result))
        group_results.append(result)
        print(
            f"[INFO] sample_index={result.sample_index} query_key={result.query_key} "
            f"occluded={result.case.actual_occluded_fraction:.3f}"
        )

        if len(group_results) == args.samples_per_figure:
            figure_idx += 1
            out_path = out_dir / f"{args.prefix}_group_{figure_idx:03d}.png"
            render_group_figure(
                group_results=group_results,
                out_path=out_path,
                title=(
                    f"KITTI Generator Box Scan\n"
                    f"source={source_label} active_boxes={active_box_count} "
                    f"indices={group_results[0].sample_index}-{group_results[-1].sample_index}"
                ),
                max_plot_points=int(args.max_plot_points),
                z_exaggeration=float(args.z_exaggeration),
            )
            summary["figures"].append(
                {
                    "path": str(out_path),
                    "sample_indices": [int(item.sample_index) for item in group_results],
                    "query_keys": [int(item.query_key) for item in group_results],
                }
            )
            print(f"[INFO] saved_figure={out_path}")
            group_results = []

    if group_results:
        figure_idx += 1
        out_path = out_dir / f"{args.prefix}_group_{figure_idx:03d}.png"
        render_group_figure(
            group_results=group_results,
            out_path=out_path,
            title=(
                f"KITTI Generator Box Scan\n"
                f"source={source_label} active_boxes={active_box_count} "
                f"indices={group_results[0].sample_index}-{group_results[-1].sample_index}"
            ),
            max_plot_points=int(args.max_plot_points),
            z_exaggeration=float(args.z_exaggeration),
        )
        summary["figures"].append(
            {
                "path": str(out_path),
                "sample_indices": [int(item.sample_index) for item in group_results],
                "query_keys": [int(item.query_key) for item in group_results],
            }
        )
        print(f"[INFO] saved_figure={out_path}")

    json_path = out_dir / f"{args.prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] saved_summary={json_path}")


if __name__ == "__main__":
    main()
