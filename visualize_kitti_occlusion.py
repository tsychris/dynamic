from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from kitti_dataloader import KITTIPointCloudQueryDataset, load_kitti_points, sample_or_pad_points
from lpr_models import build_descriptor_model
from occlusion_generator import (
    AdversarialOcclusionGenerator,
    _box_shadow_scores_per_box,
    _fuse_active_box_scores,
    _pack_active_inserted_points,
    _straight_through_threshold,
    apply_hard_drop_and_insert,
    sample_inserted_points,
)

KEEP_COLOR = "#c8c8c8"
DROP_COLOR = "#d62728"
INSERT_COLOR = "#2ca02c"
BOX_COLOR = "#00bcd4"
DEFAULT_QUERY_FILE = "/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle"
DEFAULT_KITTI_ROOT = "/TIEVNAS/KITTI"
DEFAULT_FALLBACK_ROOT = "/TIEVNAS/jyf/KITTI"


@dataclass
class OcclusionCase:
    active_box_count: int
    actual_occluded_fraction: float
    kept_points_xyz: np.ndarray
    dropped_points_xyz: np.ndarray
    inserted_points_xyz: np.ndarray | None
    active_box_mask: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    yaws: np.ndarray
    cosine_similarity: float | None


def parse_box_count_list(raw: str, max_boxes: int) -> list[int]:
    counts: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if not (1 <= count <= max_boxes):
            raise ValueError(f"active box count must be in [1, {max_boxes}], got {count}")
        counts.append(count)
    if not counts:
        raise ValueError("At least one active box count is required.")
    return counts


def infer_generator_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    feature_dim = int(state_dict["point_mlp.4.weight"].shape[0])
    num_boxes = int(state_dict["box_head.weight"].shape[0] // 7)
    return feature_dim, num_boxes


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        raw_device = "cuda" if torch.cuda.is_available() else "cpu"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, falling back to CPU.")
        raw_device = "cpu"
    return torch.device(raw_device)


def default_checkpoint_path() -> Path:
    root = Path(__file__).resolve().parent
    search_dirs = [root / "checkpoints_adv", root / "checkpoints"]
    candidates: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.exists():
            candidates.extend(search_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError("Cannot find any checkpoint under dynamic/checkpoints_adv or dynamic/checkpoints.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint_path(raw_path: str | None) -> Path:
    if raw_path is None:
        return default_checkpoint_path()
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {raw_path}")


def resolve_model_args(
    ckpt_args: dict[str, Any],
    query_file: str | None,
    kitti_root: str | None,
    fallback_root: str | None,
) -> tuple[str, str, str]:
    resolved_query = query_file or ckpt_args.get("query_file")
    resolved_kitti_root = kitti_root or ckpt_args.get("kitti_root", "/TIEVNAS/KITTI")
    resolved_fallback_root = fallback_root or ckpt_args.get("fallback_root", "/TIEVNAS/jyf/KITTI")
    if resolved_query is None:
        raise ValueError("Checkpoint does not contain query_file, please pass --query-file explicitly.")
    return str(resolved_query), str(resolved_kitti_root), str(resolved_fallback_root)


def resolve_data_args_for_random_init(
    query_file: str | None,
    kitti_root: str | None,
    fallback_root: str | None,
) -> tuple[str, str, str]:
    return (
        str(query_file or DEFAULT_QUERY_FILE),
        str(kitti_root or DEFAULT_KITTI_ROOT),
        str(fallback_root or DEFAULT_FALLBACK_ROOT),
    )


def load_checkpoint_file(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def resolve_record_path(dataset: KITTIPointCloudQueryDataset, record: dict[str, Any]) -> str:
    if not (record.get("query_submap") or record.get("submap_path") or record.get("query")):
        raise KeyError("Record does not contain query_submap/submap_path/query.")
    return dataset._get_point_path(record)  # noqa: SLF001 - reuse dataset path resolution logic


def select_query_key(dataset: KITTIPointCloudQueryDataset, sample_index: int, query_key: int | None) -> int:
    if query_key is not None:
        if query_key not in dataset.queries:
            raise KeyError(f"query_key={query_key} is not present in the dataset.")
        return int(query_key)

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} is outside dataset range [0, {len(dataset) - 1}].")
    return int(dataset.keys[sample_index])


def maybe_subsample_points(points: np.ndarray, num_points: int | None) -> np.ndarray:
    if num_points is None or num_points <= 0 or points.shape[0] <= num_points:
        return points.astype(np.float32, copy=True)
    return sample_or_pad_points(points, num_points=num_points, random_sample=False).astype(np.float32, copy=False)


def prepare_tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(points, dtype=np.float32)).unsqueeze(0).to(device)


def _parse_xyz_triplet(values: Any, field_name: str, box_idx: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Box {box_idx} field '{field_name}' must have shape [3], got {arr.shape}.")
    return arr


def load_manual_boxes(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_boxes = obj.get("boxes") if isinstance(obj, dict) else obj
    if not isinstance(raw_boxes, list) or len(raw_boxes) == 0:
        raise ValueError("Manual box JSON must be a non-empty list or a dict with a non-empty 'boxes' list.")

    centers: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    yaws: list[float] = []
    for idx, raw_box in enumerate(raw_boxes):
        if not isinstance(raw_box, dict):
            raise ValueError(f"Box {idx} must be a JSON object.")
        centers.append(_parse_xyz_triplet(raw_box.get("center"), "center", idx))
        sizes.append(_parse_xyz_triplet(raw_box.get("size"), "size", idx))
        if np.any(sizes[-1] <= 0):
            raise ValueError(f"Box {idx} size must be positive, got {sizes[-1].tolist()}.")
        if "yaw" in raw_box:
            yaw = float(raw_box["yaw"])
        elif "yaw_deg" in raw_box:
            yaw = float(np.deg2rad(float(raw_box["yaw_deg"])))
        else:
            raise ValueError(f"Box {idx} must provide 'yaw' (radians) or 'yaw_deg' (degrees).")
        yaws.append(yaw)

    return (
        np.stack(centers, axis=0).astype(np.float32, copy=False),
        np.stack(sizes, axis=0).astype(np.float32, copy=False),
        np.asarray(yaws, dtype=np.float32),
    )


def make_prefix_active_mask(num_boxes: int, active_box_count: int, device: torch.device) -> torch.Tensor:
    if not (1 <= active_box_count <= num_boxes):
        raise ValueError(f"active_box_count must be in [1, {num_boxes}], got {active_box_count}")
    mask = torch.zeros((1, num_boxes), device=device, dtype=torch.bool)
    mask[0, :active_box_count] = True
    return mask


def build_manual_occlusion(
    points_tensor: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    active_box_mask: torch.Tensor,
    geom_weight: float,
    points_per_box: int,
    use_object_insertion: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    xyz = points_tensor[..., :3]
    per_box_geom_scores = _box_shadow_scores_per_box(
        points_xyz=xyz,
        centers=centers,
        sizes=sizes,
        yaws=yaws,
    )
    geom_scores = float(geom_weight) * _fuse_active_box_scores(
        per_box_scores=per_box_geom_scores,
        active_box_mask=active_box_mask,
    )
    geom_scores = geom_scores.clamp(0.0, 1.0)
    hard_mask, _, _ = _straight_through_threshold(soft_scores=geom_scores, threshold=0.5)

    inserted_points = None
    if use_object_insertion:
        box_surface_points = sample_inserted_points(
            centers=centers,
            sizes=sizes,
            yaws=yaws,
            points_per_box=points_per_box,
        )
        inserted_points = _pack_active_inserted_points(
            box_surface_points=box_surface_points,
            active_box_mask=active_box_mask,
        )
    return hard_mask, geom_scores, inserted_points


def compute_cosine_similarity_from_adv_points(
    descriptor: torch.nn.Module | None,
    clean_emb: torch.Tensor | None,
    adv_points: torch.Tensor,
) -> float | None:
    if descriptor is None or clean_emb is None:
        return None
    adv_emb = descriptor(adv_points)
    return float(F.cosine_similarity(clean_emb, adv_emb, dim=-1).item())


def box_corners_3d(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    half = 0.5 * size
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return local @ rot.T + center[None, :]


def draw_box_bev(ax: plt.Axes, center: np.ndarray, size: np.ndarray, yaw: float, color: str) -> None:
    corners = box_corners_3d(center=center, size=size, yaw=yaw)
    order = [0, 1, 2, 3, 0]
    ax.plot(corners[order, 0], corners[order, 1], color=color, linewidth=1.5, alpha=0.95)
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32) * float(size[0]) * 0.6
    ax.arrow(
        float(center[0]),
        float(center[1]),
        float(heading[0]),
        float(heading[1]),
        color=color,
        width=0.02,
        head_width=0.35,
        length_includes_head=True,
        alpha=0.85,
    )


def draw_box_xz(ax: plt.Axes, center: np.ndarray, size: np.ndarray, yaw: float, color: str, z_scale: float = 1.0) -> None:
    corners = box_corners_3d(center=center, size=size, yaw=yaw)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 2] * z_scale, corners[j, 2] * z_scale],
            color=color,
            linewidth=1.1,
            alpha=0.8,
        )


def scatter_xy(ax: plt.Axes, points_xyz: np.ndarray, color: str, label: str, size: float) -> None:
    if points_xyz.size == 0:
        return
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], s=size, c=color, alpha=0.85, linewidths=0.0, label=label)


def scatter_xz(ax: plt.Axes, points_xyz: np.ndarray, color: str, label: str, size: float, z_scale: float = 1.0) -> None:
    if points_xyz.size == 0:
        return
    ax.scatter(
        points_xyz[:, 0],
        points_xyz[:, 2] * z_scale,
        s=size,
        c=color,
        alpha=0.85,
        linewidths=0.0,
        label=label,
    )


def downsample_for_plot(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if points_xyz.size == 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    idx = np.linspace(0, points_xyz.shape[0] - 1, max_points, dtype=np.int64)
    return points_xyz[idx]


def compute_axis_limits(points_xyz: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    margin = 0.08 * span + 0.8
    low = mins - margin
    high = maxs + margin
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1])), (float(low[2]), float(high[2]))


def compute_descriptor_similarity(
    descriptor: torch.nn.Module | None,
    generator: AdversarialOcclusionGenerator,
    model_points: torch.Tensor,
    active_box_count: int,
    active_box_mask: torch.Tensor | None,
    use_object_insertion: bool,
) -> float | None:
    if descriptor is None:
        return None
    clean_emb = descriptor(model_points)
    active_box_counts = torch.tensor([active_box_count], device=model_points.device, dtype=torch.long)
    occl = generator(
        model_points,
        active_box_counts=active_box_counts,
        active_box_mask=active_box_mask,
        generate_insertion=use_object_insertion,
    )
    adv_points = apply_hard_drop_and_insert(
        points=model_points,
        hard_drop_mask=occl.hard_drop_mask,
        inserted_points_xyz=occl.inserted_points if use_object_insertion else None,
    )
    adv_emb = descriptor(adv_points)
    return float(F.cosine_similarity(clean_emb, adv_emb, dim=-1).item())


def build_occlusion_cases(
    generator: AdversarialOcclusionGenerator,
    descriptor: torch.nn.Module | None,
    vis_points: np.ndarray,
    model_points: np.ndarray,
    active_box_counts: Sequence[int],
    device: torch.device,
    use_object_insertion: bool,
) -> list[OcclusionCase]:
    vis_tensor = prepare_tensor(vis_points, device=device)
    model_tensor = prepare_tensor(model_points, device=device)
    cases: list[OcclusionCase] = []
    clean_emb = None
    same_vis_and_model_points = vis_points.shape == model_points.shape and np.array_equal(vis_points, model_points)

    with torch.inference_mode():
        if descriptor is not None:
            clean_emb = descriptor(model_tensor)
        for active_box_count in active_box_counts:
            active_box_tensor = torch.tensor([active_box_count], device=device, dtype=torch.long)
            occl = generator(vis_tensor, active_box_counts=active_box_tensor, generate_insertion=use_object_insertion)
            hard_mask = occl.hard_drop_mask[0] > 0.5
            active_mask = occl.active_box_mask[0].detach().cpu().numpy().astype(bool, copy=True)

            vis_xyz = vis_tensor[0, :, :3]
            kept_points_xyz = vis_xyz[~hard_mask].detach().cpu().numpy()
            dropped_points_xyz = vis_xyz[hard_mask].detach().cpu().numpy()

            inserted_points_xyz = None
            if use_object_insertion and occl.inserted_points is not None:
                num_insert = int(min(int(hard_mask.sum().item()), occl.inserted_points.shape[1]))
                inserted_points_xyz = occl.inserted_points[0, :num_insert].detach().cpu().numpy()

            if descriptor is None:
                cosine_similarity = None
            elif same_vis_and_model_points:
                adv_points = apply_hard_drop_and_insert(
                    points=model_tensor,
                    hard_drop_mask=occl.hard_drop_mask,
                    inserted_points_xyz=occl.inserted_points if use_object_insertion else None,
                )
                adv_emb = descriptor(adv_points)
                cosine_similarity = float(F.cosine_similarity(clean_emb, adv_emb, dim=-1).item())
            else:
                cosine_similarity = compute_descriptor_similarity(
                    descriptor=descriptor,
                    generator=generator,
                    model_points=model_tensor,
                    active_box_count=active_box_count,
                    active_box_mask=occl.active_box_mask,
                    use_object_insertion=use_object_insertion,
                )

            cases.append(
                OcclusionCase(
                    active_box_count=int(active_box_count),
                    actual_occluded_fraction=float(occl.hard_drop_mask[0].float().mean().item()),
                    kept_points_xyz=kept_points_xyz,
                    dropped_points_xyz=dropped_points_xyz,
                    inserted_points_xyz=inserted_points_xyz,
                    active_box_mask=active_mask,
                    centers=occl.centers[0].detach().cpu().numpy(),
                    sizes=occl.sizes[0].detach().cpu().numpy(),
                    yaws=occl.yaws[0].detach().cpu().numpy(),
                    cosine_similarity=cosine_similarity,
                )
            )
    return cases


def build_manual_occlusion_cases(
    descriptor: torch.nn.Module | None,
    vis_points: np.ndarray,
    model_points: np.ndarray,
    centers_np: np.ndarray,
    sizes_np: np.ndarray,
    yaws_np: np.ndarray,
    active_box_counts: Sequence[int],
    device: torch.device,
    geom_weight: float,
    points_per_box: int,
    use_object_insertion: bool,
) -> list[OcclusionCase]:
    vis_tensor = prepare_tensor(vis_points, device=device)
    model_tensor = prepare_tensor(model_points, device=device)
    centers = torch.from_numpy(np.asarray(centers_np, dtype=np.float32)).unsqueeze(0).to(device)
    sizes = torch.from_numpy(np.asarray(sizes_np, dtype=np.float32)).unsqueeze(0).to(device)
    yaws = torch.from_numpy(np.asarray(yaws_np, dtype=np.float32)).unsqueeze(0).to(device)
    num_boxes = int(centers.shape[1])
    cases: list[OcclusionCase] = []
    clean_emb = None
    same_vis_and_model_points = vis_points.shape == model_points.shape and np.array_equal(vis_points, model_points)

    with torch.inference_mode():
        if descriptor is not None:
            clean_emb = descriptor(model_tensor)
        for active_box_count in active_box_counts:
            active_box_mask = make_prefix_active_mask(
                num_boxes=num_boxes,
                active_box_count=int(active_box_count),
                device=device,
            )
            hard_mask, _, inserted_points = build_manual_occlusion(
                points_tensor=vis_tensor,
                centers=centers,
                sizes=sizes,
                yaws=yaws,
                active_box_mask=active_box_mask,
                geom_weight=geom_weight,
                points_per_box=points_per_box,
                use_object_insertion=use_object_insertion,
            )
            hard_mask_1d = hard_mask[0] > 0.5
            active_mask = active_box_mask[0].detach().cpu().numpy().astype(bool, copy=True)

            vis_xyz = vis_tensor[0, :, :3]
            kept_points_xyz = vis_xyz[~hard_mask_1d].detach().cpu().numpy()
            dropped_points_xyz = vis_xyz[hard_mask_1d].detach().cpu().numpy()

            inserted_points_xyz = None
            if use_object_insertion and inserted_points is not None:
                num_insert = int(min(int(hard_mask_1d.sum().item()), inserted_points.shape[1]))
                inserted_points_xyz = inserted_points[0, :num_insert].detach().cpu().numpy()

            if same_vis_and_model_points:
                adv_points = apply_hard_drop_and_insert(
                    points=model_tensor,
                    hard_drop_mask=hard_mask,
                    inserted_points_xyz=inserted_points if use_object_insertion else None,
                )
                cosine_similarity = compute_cosine_similarity_from_adv_points(
                    descriptor=descriptor,
                    clean_emb=clean_emb,
                    adv_points=adv_points,
                )
            else:
                model_hard_mask, _, model_inserted_points = build_manual_occlusion(
                    points_tensor=model_tensor,
                    centers=centers,
                    sizes=sizes,
                    yaws=yaws,
                    active_box_mask=active_box_mask,
                    geom_weight=geom_weight,
                    points_per_box=points_per_box,
                    use_object_insertion=use_object_insertion,
                )
                adv_points = apply_hard_drop_and_insert(
                    points=model_tensor,
                    hard_drop_mask=model_hard_mask,
                    inserted_points_xyz=model_inserted_points if use_object_insertion else None,
                )
                cosine_similarity = compute_cosine_similarity_from_adv_points(
                    descriptor=descriptor,
                    clean_emb=clean_emb,
                    adv_points=adv_points,
                )

            cases.append(
                OcclusionCase(
                    active_box_count=int(active_box_count),
                    actual_occluded_fraction=float(hard_mask[0].float().mean().item()),
                    kept_points_xyz=kept_points_xyz,
                    dropped_points_xyz=dropped_points_xyz,
                    inserted_points_xyz=inserted_points_xyz,
                    active_box_mask=active_mask,
                    centers=centers[0].detach().cpu().numpy(),
                    sizes=sizes[0].detach().cpu().numpy(),
                    yaws=yaws[0].detach().cpu().numpy(),
                    cosine_similarity=cosine_similarity,
                )
            )
    return cases


def save_visualization_figure(
    vis_points_xyz: np.ndarray,
    cases: Sequence[OcclusionCase],
    checkpoint_label: str,
    query_key: int,
    out_path: Path,
    max_plot_points: int,
    z_exaggeration: float,
) -> None:
    cols = len(cases) + 1
    fig, axes = plt.subplots(2, cols, figsize=(4.4 * cols, 8.6), constrained_layout=True)
    if cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    xlim, ylim, zlim = compute_axis_limits(vis_points_xyz)
    z_scale = float(z_exaggeration)
    zlim_scaled = (zlim[0] * z_scale, zlim[1] * z_scale)
    z_title_suffix = "" if np.isclose(z_scale, 1.0) else f" z x{z_scale:.1f}"
    clean_plot = downsample_for_plot(vis_points_xyz, max_points=max_plot_points)

    scatter_xy(axes[0, 0], clean_plot, KEEP_COLOR, "clean", size=1.5)
    axes[0, 0].set_title(f"Clean BEV\nquery_key={query_key} points={vis_points_xyz.shape[0]}")
    axes[0, 0].set_xlim(*xlim)
    axes[0, 0].set_ylim(*ylim)
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].grid(alpha=0.2, linewidth=0.5)

    scatter_xz(axes[1, 0], clean_plot, KEEP_COLOR, "clean", size=1.5, z_scale=z_scale)
    axes[1, 0].set_title(f"Clean X-Z{z_title_suffix}")
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*zlim_scaled)
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("z (m)")
    axes[1, 0].set_aspect("equal", adjustable="box")
    if not np.isclose(z_scale, 1.0):
        axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / z_scale:.1f}"))
    axes[1, 0].grid(alpha=0.2, linewidth=0.5)

    for col, case in enumerate(cases, start=1):
        bev_ax = axes[0, col]
        xz_ax = axes[1, col]

        kept_plot = downsample_for_plot(case.kept_points_xyz, max_points=max_plot_points)
        dropped_plot = downsample_for_plot(case.dropped_points_xyz, max_points=max_plot_points)
        inserted_plot = (
            None
            if case.inserted_points_xyz is None
            else downsample_for_plot(case.inserted_points_xyz, max_points=max_plot_points)
        )

        scatter_xy(bev_ax, kept_plot, KEEP_COLOR, "kept", size=1.4)
        scatter_xy(bev_ax, dropped_plot, DROP_COLOR, "removed", size=2.0)
        if inserted_plot is not None:
            scatter_xy(bev_ax, inserted_plot, INSERT_COLOR, "inserted", size=2.0)
        for center, size, yaw, is_active in zip(case.centers, case.sizes, case.yaws, case.active_box_mask):
            if not bool(is_active):
                continue
            draw_box_bev(bev_ax, center=center, size=size, yaw=float(yaw), color=BOX_COLOR)
        title = f"BEV boxes={case.active_box_count}\noccluded={case.actual_occluded_fraction:.3f}"
        if case.cosine_similarity is not None:
            title += f" cos={case.cosine_similarity:.3f}"
        bev_ax.set_title(title)
        bev_ax.set_xlim(*xlim)
        bev_ax.set_ylim(*ylim)
        bev_ax.set_xlabel("x (m)")
        bev_ax.set_ylabel("y (m)")
        bev_ax.set_aspect("equal", adjustable="box")
        bev_ax.grid(alpha=0.2, linewidth=0.5)

        scatter_xz(xz_ax, kept_plot, KEEP_COLOR, "kept", size=1.4, z_scale=z_scale)
        scatter_xz(xz_ax, dropped_plot, DROP_COLOR, "removed", size=2.0, z_scale=z_scale)
        if inserted_plot is not None:
            scatter_xz(xz_ax, inserted_plot, INSERT_COLOR, "inserted", size=2.0, z_scale=z_scale)
        for center, size, yaw, is_active in zip(case.centers, case.sizes, case.yaws, case.active_box_mask):
            if not bool(is_active):
                continue
            draw_box_xz(xz_ax, center=center, size=size, yaw=float(yaw), color=BOX_COLOR, z_scale=z_scale)
        xz_ax.set_title(f"X-Z boxes={case.active_box_count}{z_title_suffix}")
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
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"KITTI Occlusion Visualization\nsource={checkpoint_label}", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(
    out_path: Path,
    source_label: str,
    point_path: str,
    query_key: int,
    sample_index: int,
    vis_points_xyz: np.ndarray,
    model_num_points: int,
    cases: Sequence[OcclusionCase],
) -> None:
    summary = {
        "source": source_label,
        "point_path": point_path,
        "query_key": int(query_key),
        "sample_index": int(sample_index),
        "visualized_points": int(vis_points_xyz.shape[0]),
        "model_num_points": int(model_num_points),
        "cases": [],
    }
    for case in cases:
        summary["cases"].append(
            {
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
        )
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize learned KITTI adversarial occlusion boxes and removed points under different active-box counts."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to an adversarial training checkpoint.")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip checkpoint loading and use a randomly initialized generator for quick visualization sanity checks.",
    )
    parser.add_argument(
        "--manual-boxes-json",
        type=str,
        default=None,
        help="Path to JSON box specs. When set, visualization uses these boxes instead of model-predicted boxes.",
    )
    parser.add_argument("--query-file", type=str, default=None, help="Override query pickle used by the checkpoint.")
    parser.add_argument("--kitti-root", type=str, default=None, help="Override KITTI root used by the checkpoint.")
    parser.add_argument("--fallback-root", type=str, default=None, help="Override fallback KITTI root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset-local index used when --query-key is not set.")
    parser.add_argument("--query-key", type=int, default=None, help="Original query key from the pickle file.")
    parser.add_argument(
        "--box-counts",
        type=str,
        default=None,
        help="Comma-separated active occlusion box counts. Defaults to 1..min(num_boxes, 5).",
    )
    parser.add_argument(
        "--vis-num-points",
        type=int,
        default=None,
        help="Number of points to visualize. Default uses the checkpoint model num_points. Use <=0 for the full scan.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help="Cap for rendered points per category in each subplot to keep plotting responsive.",
    )
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=1.0,
        help="Visual exaggeration factor for z in X-Z views. Use 1.0 for no exaggeration.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device string such as auto, cpu, cuda, cuda:0.")
    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
        help="Model point count used in random-init mode. Ignored when a checkpoint is loaded.",
    )
    parser.add_argument(
        "--use-intensity",
        action="store_true",
        help="Use xyz+intensity when loading points in random-init mode. Ignored when checkpoint provides this setting.",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=10,
        help="Number of occlusion boxes in random-init mode.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
        help="Generator feature dimension in random-init mode.",
    )
    parser.add_argument("--points-per-box", type=int, default=64, help="points_per_box used by the generator.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generator temperature used during training.")
    parser.add_argument(
        "--point-weight",
        type=float,
        default=None,
        help="Weight for learned per-point logits. Default reads checkpoint args and falls back to 1.0.",
    )
    parser.add_argument(
        "--geom-weight",
        type=float,
        default=None,
        help="Weight for box-derived geometry score. Default reads checkpoint args and falls back to 2.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for active-box sampling, and for random-init weights when --random-init is set.",
    )
    parser.add_argument("--no-object-insertion", action="store_true", help="Disable inserted box-surface points.")
    parser.add_argument("--out-path", type=str, default=None, help="Output PNG path. A JSON summary is saved next to it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    use_manual_boxes = args.manual_boxes_json is not None
    ckpt: dict[str, Any] | None = None
    ckpt_args: dict[str, Any] = {}
    descriptor = None

    if args.random_init:
        torch.manual_seed(args.seed)
        checkpoint_path = None
        checkpoint_label = f"random-init seed={args.seed}"
        feature_dim = int(args.feature_dim)
        num_boxes = int(args.num_boxes)
        model_num_points = int(args.num_points)
        use_intensity = bool(args.use_intensity)
        point_weight = float(1.0 if args.point_weight is None else args.point_weight)
        geom_weight = float(2.0 if args.geom_weight is None else args.geom_weight)
        query_file, kitti_root, fallback_root = resolve_data_args_for_random_init(
            query_file=args.query_file,
            kitti_root=args.kitti_root,
            fallback_root=args.fallback_root,
        )
    elif args.checkpoint is not None or not use_manual_boxes:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        checkpoint_label = checkpoint_path.name
        ckpt = load_checkpoint_file(checkpoint_path)
        ckpt_args = ckpt.get("args", {})
        if "generator" not in ckpt:
            raise KeyError(f"Checkpoint does not contain a generator state dict: {checkpoint_path}")

        feature_dim, num_boxes = infer_generator_config(ckpt["generator"])
        model_num_points = int(ckpt_args.get("num_points", 4096))
        use_intensity = bool(ckpt_args.get("use_intensity", False))
        point_weight = float(ckpt_args.get("point_weight", 1.0) if args.point_weight is None else args.point_weight)
        geom_weight = float(ckpt_args.get("geom_weight", 2.0) if args.geom_weight is None else args.geom_weight)
        query_file, kitti_root, fallback_root = resolve_model_args(
            ckpt_args=ckpt_args,
            query_file=args.query_file,
            kitti_root=args.kitti_root,
            fallback_root=args.fallback_root,
        )
    else:
        checkpoint_path = None
        checkpoint_label = f"manual-boxes {Path(args.manual_boxes_json).name}"
        feature_dim = int(args.feature_dim)
        num_boxes = int(args.num_boxes)
        model_num_points = int(args.num_points)
        use_intensity = bool(args.use_intensity)
        point_weight = float(1.0 if args.point_weight is None else args.point_weight)
        geom_weight = float(2.0 if args.geom_weight is None else args.geom_weight)
        query_file, kitti_root, fallback_root = resolve_data_args_for_random_init(
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
    query_key = select_query_key(dataset=dataset, sample_index=args.sample_index, query_key=args.query_key)
    sample_index = int(dataset.keys.index(query_key))
    record = dataset.queries[query_key]
    point_path = resolve_record_path(dataset=dataset, record=record)

    raw_points = load_kitti_points(point_path, use_intensity=use_intensity)
    vis_num_points = model_num_points if args.vis_num_points is None else int(args.vis_num_points)
    vis_points = maybe_subsample_points(raw_points, num_points=vis_num_points)
    model_points = sample_or_pad_points(raw_points, num_points=model_num_points, random_sample=False).astype(np.float32)

    if ckpt is not None and "descriptor" in ckpt:
        descriptor = build_descriptor_model(
            arch=str(ckpt_args.get("descriptor_arch", "pointnetvlad")),
            num_points=model_num_points,
            emb_dim=int(ckpt_args.get("emb_dim", 256)),
            in_channels=4 if use_intensity else 3,
        )
        descriptor.load_state_dict(ckpt["descriptor"], strict=True)
        descriptor.to(device)
        descriptor.eval()

    generator = None
    manual_boxes = None
    if use_manual_boxes:
        manual_boxes = load_manual_boxes(args.manual_boxes_json)
        num_boxes = int(manual_boxes[0].shape[0])
        checkpoint_label = (
            f"{checkpoint_label} + manual-boxes {Path(args.manual_boxes_json).name}"
            if ckpt is not None or args.random_init
            else checkpoint_label
        )
    else:
        generator = AdversarialOcclusionGenerator(
            num_boxes=num_boxes,
            feature_dim=feature_dim,
            points_per_box=int(args.points_per_box),
            temperature=float(args.temperature),
            point_weight=point_weight,
            geom_weight=geom_weight,
        )
        if ckpt is not None:
            generator.load_state_dict(ckpt["generator"], strict=True)
        generator.to(device)
        generator.eval()

    raw_box_counts = args.box_counts
    if raw_box_counts is None:
        raw_box_counts = ",".join(str(v) for v in range(1, min(num_boxes, 5) + 1))
    active_box_counts = parse_box_count_list(raw_box_counts, max_boxes=num_boxes)

    if manual_boxes is not None:
        cases = build_manual_occlusion_cases(
            descriptor=descriptor,
            vis_points=vis_points,
            model_points=model_points,
            centers_np=manual_boxes[0],
            sizes_np=manual_boxes[1],
            yaws_np=manual_boxes[2],
            active_box_counts=active_box_counts,
            device=device,
            geom_weight=geom_weight,
            points_per_box=int(args.points_per_box),
            use_object_insertion=not args.no_object_insertion,
        )
    else:
        cases = build_occlusion_cases(
            generator=generator,
            descriptor=descriptor,
            vis_points=vis_points,
            model_points=model_points,
            active_box_counts=active_box_counts,
            device=device,
            use_object_insertion=not args.no_object_insertion,
        )

    output_tag = (
        f"manual_boxes_{Path(args.manual_boxes_json).stem}"
        if manual_boxes is not None
        else ("random_init" if checkpoint_path is None else checkpoint_path.stem)
    )
    default_out_path = (
        Path(__file__).resolve().parent
        / "vis_occlusion"
        / output_tag
        / f"query_{query_key:06d}_boxes.png"
    )
    out_path = Path(args.out_path).expanduser().resolve() if args.out_path is not None else default_out_path
    json_path = out_path.with_suffix(".json")
    if args.manual_boxes_json is not None:
        manual_boxes_path = Path(args.manual_boxes_json).expanduser().resolve()
        if manual_boxes_path == json_path:
            json_path = out_path.with_name(f"{out_path.stem}.summary.json")

    save_visualization_figure(
        vis_points_xyz=vis_points[:, :3],
        cases=cases,
        checkpoint_label=checkpoint_label,
        query_key=query_key,
        out_path=out_path,
        max_plot_points=args.max_plot_points,
        z_exaggeration=float(args.z_exaggeration),
    )
    save_summary_json(
        out_path=json_path,
        source_label=checkpoint_label,
        point_path=point_path,
        query_key=query_key,
        sample_index=sample_index,
        vis_points_xyz=vis_points[:, :3],
        model_num_points=model_num_points,
        cases=cases,
    )

    print(f"[INFO] source={checkpoint_label}")
    print(f"[INFO] query_key={query_key} sample_index={sample_index}")
    print(f"[INFO] point_path={point_path}")
    print(f"[INFO] vis_points={vis_points.shape[0]} model_points={model_num_points}")
    print(f"[INFO] point_weight={point_weight:.3f} geom_weight={geom_weight:.3f}")
    print(f"[INFO] z_exaggeration={float(args.z_exaggeration):.2f}")
    for case in cases:
        msg = (
            f"[INFO] active_boxes={case.active_box_count} "
            f"occluded={case.actual_occluded_fraction:.3f} "
            f"removed={case.dropped_points_xyz.shape[0]}"
        )
        if case.cosine_similarity is not None:
            msg += f" cos={case.cosine_similarity:.3f}"
        print(msg)
    print(f"[INFO] saved_figure={out_path}")
    print(f"[INFO] saved_summary={json_path}")


if __name__ == "__main__":
    main()
