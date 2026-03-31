from __future__ import annotations

import argparse
import io
import random
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from kitti_dataloader import KITTIPointCloudQueryDataset, make_kitti_query_collate_fn, resolve_kitti_path
from lpr_models import build_descriptor_model, embedding_consistency_loss
from occlusion_generator import (
    AdversarialOcclusionGenerator,
    apply_hard_drop_and_insert,
    apply_soft_drop,
)

VIS_PT_ROOT = Path("/media/autolab/tsy/dynamic/vis_pt")
VIS_PT_DISABLED_REASON: str | None = None


def add_tensorboard_custom_layout(writer: SummaryWriter) -> None:
    layout = {
        "Losses": {
            "Iter Place": [
                "Multiline",
                [
                    "train_iter/loss_place_clean",
                    "train_iter/loss_place_adv",
                    "train_iter/loss_consistency",
                ],
            ],
            "Iter Total": [
                "Multiline",
                [
                    "train_iter/loss_f",
                    "train_iter/loss_g",
                ],
            ],
            "Epoch Place": [
                "Multiline",
                [
                    "train_epoch/loss_place_clean",
                    "train_epoch/loss_place_adv",
                    "train_epoch/loss_consistency",
                ],
            ],
            "Epoch Total": [
                "Multiline",
                [
                    "train_epoch/loss_f",
                    "train_epoch/loss_g",
                ],
            ],
        },
        "Occlusion": {
            "Drop Ratio": [
                "Multiline",
                [
                    "train_iter/drop_ratio",
                    "train_iter/target_drop_ratio",
                ],
            ],
            "Epoch Drop Ratio": [
                "Multiline",
                [
                    "train_epoch/drop_ratio",
                    "train_epoch/target_drop_ratio",
                ],
            ],
        },
        "Regularization": {
            "Iter Regularization": [
                "Multiline",
                [
                    "train_iter/reg_size_prior",
                    "train_iter/reg_height_prior",
                    "train_iter/reg_range_prior",
                ],
            ],
            "Epoch Regularization": [
                "Multiline",
                [
                    "train_epoch/reg_size_prior",
                    "train_epoch/reg_height_prior",
                    "train_epoch/reg_range_prior",
                ],
            ],
        },
        "Boxes": {
            "Iter Center Mean": [
                "Multiline",
                [
                    "train_iter/box_center_x_mean",
                    "train_iter/box_center_y_mean",
                    "train_iter/box_center_z_mean",
                ],
            ],
            "Epoch Center Mean": [
                "Multiline",
                [
                    "train_epoch/box_center_x_mean",
                    "train_epoch/box_center_y_mean",
                    "train_epoch/box_center_z_mean",
                ],
            ],
            "Iter Size Mean": [
                "Multiline",
                [
                    "train_iter/box_size_l_mean",
                    "train_iter/box_size_w_mean",
                    "train_iter/box_size_h_mean",
                ],
            ],
            "Epoch Size Mean": [
                "Multiline",
                [
                    "train_epoch/box_size_l_mean",
                    "train_epoch/box_size_w_mean",
                    "train_epoch/box_size_h_mean",
                ],
            ],
            "Yaw": [
                "Multiline",
                [
                    "train_iter/box_yaw_mean",
                    "train_epoch/box_yaw_mean",
                ],
            ],
        },
    }
    writer.add_custom_scalars(layout)


def _log_box_param_scalars(
    writer: SummaryWriter,
    prefix: str,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    step: int,
) -> None:
    centers_cpu = centers.detach().cpu()
    sizes_cpu = sizes.detach().cpu()
    yaws_cpu = yaws.detach().cpu()

    if centers_cpu.numel() == 0:
        return

    center_axes = ("x", "y", "z")
    size_axes = ("l", "w", "h")

    for axis_idx, axis_name in enumerate(center_axes):
        writer.add_scalar(
            f"{prefix}/box_center_{axis_name}_mean",
            float(centers_cpu[..., axis_idx].mean().item()),
            step,
        )

    for axis_idx, axis_name in enumerate(size_axes):
        writer.add_scalar(
            f"{prefix}/box_size_{axis_name}_mean",
            float(sizes_cpu[..., axis_idx].mean().item()),
            step,
        )

    writer.add_scalar(f"{prefix}/box_yaw_mean", float(yaws_cpu.mean().item()), step)

    center_per_box = centers_cpu.mean(dim=0)
    size_per_box = sizes_cpu.mean(dim=0)
    yaw_per_box = yaws_cpu.mean(dim=0)
    for box_idx in range(center_per_box.shape[0]):
        box_prefix = f"{prefix}/box_{box_idx:02d}"
        for axis_idx, axis_name in enumerate(center_axes):
            writer.add_scalar(
                f"{box_prefix}/center_{axis_name}",
                float(center_per_box[box_idx, axis_idx].item()),
                step,
            )
        for axis_idx, axis_name in enumerate(size_axes):
            writer.add_scalar(
                f"{box_prefix}/size_{axis_name}",
                float(size_per_box[box_idx, axis_idx].item()),
                step,
            )
        writer.add_scalar(f"{box_prefix}/yaw", float(yaw_per_box[box_idx].item()), step)


def _log_box_param_histograms(
    writer: SummaryWriter,
    prefix: str,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    step: int,
) -> None:
    centers_cpu = centers.detach().cpu()
    sizes_cpu = sizes.detach().cpu()
    yaws_cpu = yaws.detach().cpu()

    if centers_cpu.numel() == 0:
        return

    center_axes = ("x", "y", "z")
    size_axes = ("l", "w", "h")

    for axis_idx, axis_name in enumerate(center_axes):
        writer.add_histogram(
            f"{prefix}/center_{axis_name}",
            centers_cpu[..., axis_idx].reshape(-1),
            step,
        )

    for axis_idx, axis_name in enumerate(size_axes):
        writer.add_histogram(
            f"{prefix}/size_{axis_name}",
            sizes_cpu[..., axis_idx].reshape(-1),
            step,
        )

    writer.add_histogram(f"{prefix}/yaw", yaws_cpu.reshape(-1), step)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def masked_batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    positives_mask: torch.Tensor,
    negatives_mask: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Batch-hard triplet with externally provided positive/negative masks.
    """
    dist = torch.cdist(embeddings, embeddings, p=2)
    eye = torch.eye(embeddings.shape[0], device=embeddings.device, dtype=torch.bool)
    positives_mask = positives_mask.to(embeddings.device) & (~eye)
    negatives_mask = negatives_mask.to(embeddings.device) & (~eye)

    pos_dist = torch.where(positives_mask, dist, torch.zeros_like(dist))
    hardest_pos = pos_dist.max(dim=1).values

    big = torch.full_like(dist, 1e6)
    neg_dist = torch.where(negatives_mask, dist, big)
    hardest_neg = neg_dist.min(dim=1).values

    valid = positives_mask.any(dim=1) & negatives_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


class KITTIPairBatchSampler(Sampler[List[int]]):
    """
    Build mini-batches with anchor-positive pairs so metric loss is always valid.
    """

    def __init__(
        self,
        dataset: KITTIPointCloudQueryDataset,
        batch_size: int = 32,
        num_batches_per_epoch: int | None = None,
    ) -> None:
        # PyTorch Sampler constructor signatures differ across versions.
        super().__init__(dataset)
        if batch_size < 4:
            raise ValueError("batch_size should be >= 4")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = (
            int(num_batches_per_epoch)
            if num_batches_per_epoch is not None
            else max(len(dataset) // self.batch_size, 1)
        )

        self.key_to_local = {k: i for i, k in enumerate(self.dataset.keys)}
        self.valid_anchor_keys: List[int] = []
        for k in self.dataset.keys:
            pos = [p for p in self.dataset.get_positives_ndx(k) if (p in self.key_to_local and p != k)]
            if len(pos) > 0:
                self.valid_anchor_keys.append(k)
        if len(self.valid_anchor_keys) == 0:
            raise RuntimeError("No valid anchor with positive pairs found in dataset.")

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            batch: List[int] = []
            pair_slots = self.batch_size // 2

            for _ in range(pair_slots):
                a_key = random.choice(self.valid_anchor_keys)
                pos_candidates = [p for p in self.dataset.get_positives_ndx(a_key) if (p in self.key_to_local and p != a_key)]
                if len(pos_candidates) == 0:
                    continue
                p_key = random.choice(pos_candidates)
                batch.append(self.key_to_local[a_key])
                batch.append(self.key_to_local[p_key])

            while len(batch) < self.batch_size:
                batch.append(random.randrange(len(self.dataset)))
            random.shuffle(batch)
            yield batch[: self.batch_size]


@dataclass
class TrainConfig:
    epochs: int = 20
    margin: float = 0.2
    adv_weight: float = 1.0
    consistency_weight: float = 0.2
    size_prior_weight: float = 0.1
    height_prior_weight: float = 0.05
    range_prior_weight: float = 0.05
    use_object_insertion: bool = True
    drop_ratio_set: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)


def sample_drop_ratios(batch: int, ratio_set: tuple[float, ...], device: torch.device) -> torch.Tensor:
    ratios = [ratio_set[random.randrange(len(ratio_set))] for _ in range(batch)]
    return torch.tensor(ratios, device=device, dtype=torch.float32)


def _resolve_record_asset(
    rec: dict,
    keys: tuple[str, ...],
    kitti_root: str,
    fallback_root: str,
) -> str | None:
    for key in keys:
        raw_path = rec.get(key)
        if raw_path is None:
            continue
        try:
            return resolve_kitti_path(raw_path, kitti_root=kitti_root, fallback_root=fallback_root)
        except FileNotFoundError:
            continue
    return None


def _infer_kitti_calib_path(point_path: str, kitti_root: str, fallback_root: str) -> str:
    point_parts = Path(point_path).parts
    if "sequences" not in point_parts:
        raise FileNotFoundError(f"Cannot infer KITTI calib path from point path: {point_path}")

    seq_idx = point_parts.index("sequences") + 1
    if seq_idx >= len(point_parts):
        raise FileNotFoundError(f"Cannot infer KITTI sequence id from point path: {point_path}")
    seq = point_parts[seq_idx]

    candidates = []
    for root in (kitti_root, fallback_root):
        root_path = Path(root)
        candidates.append(root_path / "data_odometry_calib" / "dataset" / "sequences" / seq / "calib.txt")
        candidates.append(root_path / "data_odometry_calib" / "dataset" / "sequences" / "calib.txt")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot resolve calib path for sequence {seq} from point path: {point_path}")


def _load_kitti_calib(calib_path: str) -> tuple[np.ndarray, np.ndarray]:
    p2 = None
    tr = None
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(":", 1)
            vals = np.fromstring(val, sep=" ")
            if key == "P2":
                p2 = vals.reshape(3, 4)
            if key in {"Tr", "Tr_velo_to_cam", "Tr:"}:
                tr = vals.reshape(3, 4)
    if p2 is None or tr is None:
        raise ValueError(f"Failed to parse P2/Tr from calib file: {calib_path}")
    return p2.astype(np.float32), tr.astype(np.float32)


def _project_points_to_front_depth(
    points_xyz: np.ndarray,
    p2: np.ndarray,
    tr: np.ndarray,
    img_w: int,
    img_h: int,
    out_w: int | None = None,
    out_h: int | None = None,
    max_depth_m: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected points shape [N, >=3], got {xyz.shape}")
    xyz = xyz[:, :3]

    dists = np.linalg.norm(xyz, axis=1)
    xyz = xyz[dists < max_depth_m]
    if xyz.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32), np.zeros((img_h, img_w), dtype=bool)

    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    tr4 = np.vstack([tr, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    cam_pts = (tr4 @ xyz_h.T).T

    mask_front = cam_pts[:, 2] > 0.1
    cam_pts = cam_pts[mask_front]
    if cam_pts.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32), np.zeros((img_h, img_w), dtype=bool)

    proj = (p2 @ cam_pts.T).T
    depth = cam_pts[:, 2]
    u = np.round(proj[:, 0] / np.maximum(proj[:, 2], 1e-8)).astype(np.int32)
    v = np.round(proj[:, 1] / np.maximum(proj[:, 2], 1e-8)).astype(np.int32)

    keep = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[keep]
    v = v[keep]
    depth = depth[keep]

    out_w = img_w if out_w is None else int(out_w)
    out_h = img_h if out_h is None else int(out_h)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid output size: {(out_w, out_h)}")
    if out_w != img_w:
        u = np.round(u * ((out_w - 1) / max(img_w - 1, 1))).astype(np.int32)
    if out_h != img_h:
        v = np.round(v * ((out_h - 1) / max(img_h - 1, 1))).astype(np.int32)

    order = np.argsort(depth)[::-1]
    u = u[order]
    v = v[order]
    depth = depth[order]

    depth_map = np.zeros((out_h, out_w), dtype=np.float32)
    valid_mask = np.zeros((out_h, out_w), dtype=bool)
    depth_map[v, u] = depth
    valid_mask[v, u] = True
    return depth_map, valid_mask


def _densify_depth_map(depth_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    inv_depth = np.zeros_like(depth_map, dtype=np.float32)
    inv_depth[valid_mask] = 1.0 / np.maximum(depth_map[valid_mask], 1e-6)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dense_inv = cv2.dilate(inv_depth, k5, iterations=1)
    dense_inv = cv2.dilate(dense_inv, k3, iterations=1)

    filled_mask = dense_inv > 0
    dense_depth = np.zeros_like(depth_map, dtype=np.float32)
    dense_depth[filled_mask] = 1.0 / np.maximum(dense_inv[filled_mask], 1e-6)
    return dense_depth, filled_mask


def _colorize_depth_map(depth_map: np.ndarray, valid_mask: np.ndarray, max_depth_m: float) -> np.ndarray:
    import cv2

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    clipped = np.clip(depth_map, 0.0, max_depth_m)
    normalized[valid_mask] = 1.0 - (clipped[valid_mask] / max_depth_m)

    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    out = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    out[valid_mask] = colored[valid_mask]
    return out


def _overlay_depth_on_image(
    rgb: np.ndarray,
    depth_color: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    depth_float = depth_color.astype(np.float32)
    mask3 = np.repeat(valid_mask[:, :, None], 3, axis=2)
    out[mask3] = (1.0 - alpha) * out[mask3] + alpha * depth_float[mask3]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _resize_vis_to_image(
    color_img: np.ndarray,
    valid_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    resized_color = cv2.resize(color_img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    resized_valid = cv2.resize(valid_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0
    return resized_color, resized_valid


def save_range_view_comparison(
    clean_points: torch.Tensor,
    adv_points: torch.Tensor,
    query_rec: dict,
    kitti_root: str,
    fallback_root: str,
    out_path: Path,
    title: str,
) -> bool:
    global VIS_PT_DISABLED_REASON

    if VIS_PT_DISABLED_REASON is not None:
        return False

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from PIL import Image
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local matplotlib/numpy build
        VIS_PT_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] vis-pt disabled: {VIS_PT_DISABLED_REASON}")
        return False

    try:
        point_path = _resolve_record_asset(
            query_rec,
            keys=("query_submap", "submap_path", "query"),
            kitti_root=kitti_root,
            fallback_root=fallback_root,
        )
        if point_path is None:
            raise FileNotFoundError("No query_submap/submap_path/query field found for vis-pt")

        image_path = _resolve_record_asset(
            query_rec,
            keys=("query_img", "img_path", "image_path"),
            kitti_root=kitti_root,
            fallback_root=fallback_root,
        )
        if image_path is None:
            raise FileNotFoundError("No query_img/img_path/image_path field found for vis-pt")

        calib_path = _resolve_record_asset(
            query_rec,
            keys=("calib_path", "query_calib"),
            kitti_root=kitti_root,
            fallback_root=fallback_root,
        )
        if calib_path is None:
            calib_path = _infer_kitti_calib_path(point_path, kitti_root=kitti_root, fallback_root=fallback_root)

        rgb = np.array(Image.open(image_path).convert("RGB"))
        img_h, img_w = rgb.shape[:2]
        p2, tr = _load_kitti_calib(calib_path)

        clean_xyz = clean_points.detach().cpu().numpy()[..., :3]
        adv_xyz = adv_points.detach().cpu().numpy()[..., :3]
        max_depth_m = 50.0
        render_w = max(img_w // 2, 320)
        render_h = max(img_h // 2, 96)

        clean_sparse, clean_sparse_valid = _project_points_to_front_depth(
            clean_xyz,
            p2=p2,
            tr=tr,
            img_w=img_w,
            img_h=img_h,
            out_w=render_w,
            out_h=render_h,
            max_depth_m=max_depth_m,
        )
        adv_sparse, adv_sparse_valid = _project_points_to_front_depth(
            adv_xyz,
            p2=p2,
            tr=tr,
            img_w=img_w,
            img_h=img_h,
            out_w=render_w,
            out_h=render_h,
            max_depth_m=max_depth_m,
        )

        clean_dense, clean_dense_valid = _densify_depth_map(clean_sparse, clean_sparse_valid)
        adv_dense, adv_dense_valid = _densify_depth_map(adv_sparse, adv_sparse_valid)

        clean_color = _colorize_depth_map(clean_dense, clean_dense_valid, max_depth_m=max_depth_m)
        adv_color = _colorize_depth_map(adv_dense, adv_dense_valid, max_depth_m=max_depth_m)
        clean_color, clean_dense_valid = _resize_vis_to_image(clean_color, clean_dense_valid, img_w=img_w, img_h=img_h)
        adv_color, adv_dense_valid = _resize_vis_to_image(adv_color, adv_dense_valid, img_w=img_w, img_h=img_h)
        clean_overlay = _overlay_depth_on_image(rgb, clean_color, clean_dense_valid)
        adv_overlay = _overlay_depth_on_image(rgb, adv_color, adv_dense_valid)

        frame_id = Path(point_path).stem
        clean_valid = int(clean_sparse_valid.sum())
        adv_valid = int(adv_sparse_valid.sum())
        clean_fill = 100.0 * float(clean_dense_valid.mean())
        adv_fill = 100.0 * float(adv_dense_valid.mean())
    except Exception as exc:  # pragma: no cover - depends on local KITTI assets
        VIS_PT_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] vis-pt disabled: {VIS_PT_DISABLED_REASON}")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].imshow(clean_color)
    axes[0].set_title(
        f"Clean Front Dense Depth\nsampled_px={clean_valid} filled={clean_fill:.1f}% frame={frame_id}"
    )
    axes[0].axis("off")

    axes[1].imshow(adv_color)
    axes[1].set_title(
        f"Adversarial Front Dense Depth\nsampled_px={adv_valid} filled={adv_fill:.1f}% frame={frame_id}"
    )
    axes[1].axis("off")

    axes[2].imshow(clean_overlay)
    axes[2].set_title("Clean Dense Depth Overlay")
    axes[2].axis("off")

    axes[3].imshow(adv_overlay)
    axes[3].set_title("Adversarial Dense Depth Overlay")
    axes[3].axis("off")

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - depends on local plotting backend/filesystem
        VIS_PT_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] vis-pt disabled: {VIS_PT_DISABLED_REASON}")
        plt.close(fig)
        return False
    plt.close(fig)
    return True


def train_one_epoch(
    loader: DataLoader,
    descriptor: nn.Module,
    generator: AdversarialOcclusionGenerator,
    opt_f: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    vis_dir: Path | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[Dict[str, float], int]:
    descriptor.train()
    generator.train()
    vis_saved = False
    epoch_box_centers: List[torch.Tensor] = []
    epoch_box_sizes: List[torch.Tensor] = []
    epoch_box_yaws: List[torch.Tensor] = []

    meter = {
        "loss_f": 0.0,
        "loss_g": 0.0,
        "loss_place_clean": 0.0,
        "loss_place_adv": 0.0,
        "loss_consistency": 0.0,
        "drop_ratio": 0.0,
        "target_drop_ratio": 0.0,
        "reg_size_prior": 0.0,
        "reg_height_prior": 0.0,
        "reg_range_prior": 0.0,
    }

    for batch_idx, (points, labels, positives_mask, negatives_mask) in enumerate(loader):
        points = points.to(device, non_blocking=True)
        positives_mask = positives_mask.to(device, non_blocking=True)
        negatives_mask = negatives_mask.to(device, non_blocking=True)
        drop_ratios = sample_drop_ratios(points.shape[0], cfg.drop_ratio_set, device)

        # (1) maximize_G L_place(f(G(x)))
        set_requires_grad(descriptor, False)
        set_requires_grad(generator, True)

        occl = generator(points, drop_ratios=drop_ratios, generate_insertion=False)
        adv_points_for_g = apply_soft_drop(points, occl.st_drop_mask)
        emb_adv = descriptor(adv_points_for_g)
        loss_place_adv = masked_batch_hard_triplet_loss(
            emb_adv,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            margin=cfg.margin,
        )
        reg_size_prior = occl.regularization["size_prior"]
        reg_height_prior = occl.regularization["height_prior"]
        reg_range_prior = occl.regularization["range_prior"]
        reg = (
            cfg.size_prior_weight * reg_size_prior
            + cfg.height_prior_weight * reg_height_prior
            + cfg.range_prior_weight * reg_range_prior
        )
        loss_g = -loss_place_adv + reg

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        # (2) minimize_f L_place(clean) + L_place(adv)
        set_requires_grad(descriptor, True)
        set_requires_grad(generator, False)
        with torch.no_grad():
            occl_detach = generator(
                points,
                drop_ratios=drop_ratios,
                generate_insertion=cfg.use_object_insertion,
            )
            adv_points_for_f = apply_hard_drop_and_insert(
                points=points,
                hard_drop_mask=occl_detach.hard_drop_mask,
                inserted_points_xyz=occl_detach.inserted_points if cfg.use_object_insertion else None,
            )

        if vis_dir is not None and not vis_saved:
            dataset = loader.dataset
            if not isinstance(dataset, KITTIPointCloudQueryDataset):
                raise TypeError(f"vis-pt expects KITTIPointCloudQueryDataset, got {type(dataset).__name__}")
            query_key = int(labels[0].item())
            vis_saved = save_range_view_comparison(
                clean_points=points[0],
                adv_points=adv_points_for_f[0],
                query_rec=dataset.queries[query_key],
                kitti_root=dataset.kitti_root,
                fallback_root=dataset.fallback_root,
                out_path=vis_dir / f"epoch_{epoch:03d}_step_{global_step:06d}.png",
                title=(
                    f"epoch={epoch} step={global_step} "
                    f"query_key={query_key} "
                    f"target_drop={float(drop_ratios[0].item()):.2f} "
                    f"actual_drop={float(occl_detach.hard_drop_mask[0].mean().item()):.2f}"
                ),
            )
            if VIS_PT_DISABLED_REASON is not None:
                vis_saved = True

        emb_clean = descriptor(points)
        emb_adv_detach = descriptor(adv_points_for_f)

        loss_clean = masked_batch_hard_triplet_loss(
            emb_clean,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            margin=cfg.margin,
        )
        loss_adv = masked_batch_hard_triplet_loss(
            emb_adv_detach,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            margin=cfg.margin,
        )
        loss_cons = embedding_consistency_loss(emb_clean, emb_adv_detach)
        loss_f = loss_clean + cfg.adv_weight * loss_adv + cfg.consistency_weight * loss_cons

        opt_f.zero_grad(set_to_none=True)
        loss_f.backward()
        opt_f.step()

        with torch.no_grad():
            meter["loss_f"] += float(loss_f.item())
            meter["loss_g"] += float(loss_g.item())
            meter["loss_place_clean"] += float(loss_clean.item())
            meter["loss_place_adv"] += float(loss_adv.item())
            meter["loss_consistency"] += float(loss_cons.item())
            meter["drop_ratio"] += float(occl_detach.hard_drop_mask.mean().item())
            meter["target_drop_ratio"] += float(drop_ratios.mean().item())
            meter["reg_size_prior"] += float(reg_size_prior.item())
            meter["reg_height_prior"] += float(reg_height_prior.item())
            meter["reg_range_prior"] += float(reg_range_prior.item())
            epoch_box_centers.append(occl_detach.centers.detach().cpu())
            epoch_box_sizes.append(occl_detach.sizes.detach().cpu())
            epoch_box_yaws.append(occl_detach.yaws.detach().cpu())

        if writer is not None:
            writer.add_scalar("train_iter/loss_f", float(loss_f.item()), global_step)
            writer.add_scalar("train_iter/loss_g", float(loss_g.item()), global_step)
            writer.add_scalar("train_iter/loss_place_clean", float(loss_clean.item()), global_step)
            writer.add_scalar("train_iter/loss_place_adv", float(loss_adv.item()), global_step)
            writer.add_scalar("train_iter/loss_consistency", float(loss_cons.item()), global_step)
            writer.add_scalar("train_iter/drop_ratio", float(occl_detach.hard_drop_mask.mean().item()), global_step)
            writer.add_scalar("train_iter/target_drop_ratio", float(drop_ratios.mean().item()), global_step)
            writer.add_scalar("train_iter/reg_size_prior", float(reg_size_prior.item()), global_step)
            writer.add_scalar("train_iter/reg_height_prior", float(reg_height_prior.item()), global_step)
            writer.add_scalar("train_iter/reg_range_prior", float(reg_range_prior.item()), global_step)
            writer.add_scalar("train_iter/lr_f", float(opt_f.param_groups[0]["lr"]), global_step)
            writer.add_scalar("train_iter/lr_g", float(opt_g.param_groups[0]["lr"]), global_step)
            _log_box_param_scalars(
                writer,
                prefix="train_iter",
                centers=occl_detach.centers,
                sizes=occl_detach.sizes,
                yaws=occl_detach.yaws,
                step=global_step,
            )

        global_step += 1

    num_iter = max(len(loader), 1)
    for k in meter:
        meter[k] /= num_iter

    if writer is not None and epoch_box_centers:
        epoch_centers = torch.cat(epoch_box_centers, dim=0)
        epoch_sizes = torch.cat(epoch_box_sizes, dim=0)
        epoch_yaws = torch.cat(epoch_box_yaws, dim=0)
        _log_box_param_scalars(
            writer,
            prefix="train_epoch",
            centers=epoch_centers,
            sizes=epoch_sizes,
            yaws=epoch_yaws,
            step=epoch,
        )
        _log_box_param_histograms(
            writer,
            prefix="train_epoch_hist/boxes",
            centers=epoch_centers,
            sizes=epoch_sizes,
            yaws=epoch_yaws,
            step=epoch,
        )

    return meter, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adversarial occlusion LPR directly on KITTI.")
    parser.add_argument(
        "--query-file",
        type=str,
        default="/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle",
    )
    parser.add_argument("--kitti-root", type=str, default="/TIEVNAS/KITTI")
    parser.add_argument("--fallback-root", type=str, default="/TIEVNAS/jyf/KITTI")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches-per-epoch", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--max-elems", type=int, default=None)
    parser.add_argument("--descriptor-arch", type=str, default="pointnetvlad")
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--use-intensity", action="store_true")
    parser.add_argument("--lr-f", type=float, default=1e-3)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--geom-weight", type=float, default=2.0)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--size-prior-weight", type=float, default=0.1)
    parser.add_argument("--height-prior-weight", type=float, default=0.05)
    parser.add_argument("--range-prior-weight", type=float, default=0.05)
    parser.add_argument("--no-object-insertion", action="store_true")
    parser.add_argument("--save-dir", type=str, default="/media/autolab/tsy/dynamic/checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--tb-logdir", type=str, default="/media/autolab/tsy/dynamic/tb_runs")
    parser.add_argument(
        "--vis-pt",
        action="store_true",
        help="Save one clean/adversarial KITTI front-depth comparison image per epoch.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    dataset = KITTIPointCloudQueryDataset(
        query_filepath=args.query_file,
        kitti_root=args.kitti_root,
        fallback_root=args.fallback_root,
        num_points=args.num_points,
        use_intensity=args.use_intensity,
        random_sample=True,
        max_elems=args.max_elems,
        prefer_cached=True,
    )
    print(f"[INFO] dataset_size={len(dataset)}")
    print(f"[INFO] bin_root={args.kitti_root} fallback_root={args.fallback_root}")

    sampler = KITTIPairBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_kitti_query_collate_fn(dataset),
    )

    in_channels = 4 if args.use_intensity else 3
    descriptor = build_descriptor_model(
        arch=args.descriptor_arch,
        num_points=args.num_points,
        emb_dim=args.emb_dim,
        in_channels=in_channels,
    ).to(device)
    generator = AdversarialOcclusionGenerator(
        num_boxes=10,
        points_per_box=64,
        temperature=0.2,
        point_weight=args.point_weight,
        geom_weight=args.geom_weight,
    ).to(device)

    opt_f = torch.optim.Adam(descriptor.parameters(), lr=args.lr_f)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)

    cfg = TrainConfig(
        epochs=args.epochs,
        margin=args.margin,
        adv_weight=args.adv_weight,
        consistency_weight=args.consistency_weight,
        size_prior_weight=args.size_prior_weight,
        height_prior_weight=args.height_prior_weight,
        range_prior_weight=args.range_prior_weight,
        use_object_insertion=not args.no_object_insertion,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_root = Path(args.tb_logdir)
    tb_root.mkdir(parents=True, exist_ok=True)
    start_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base = args.run_name
    if run_base is None:
        run_base = f"{args.descriptor_arch}_np{args.num_points}_bs{args.batch_size}"
    session_name = f"{run_base}_{start_time_tag}"
    writer = SummaryWriter(log_dir=str(tb_root / session_name))
    add_tensorboard_custom_layout(writer)
    writer.add_text("config/args", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    writer.add_text("config/device", str(device))
    writer.add_text("config/session_name", session_name)
    writer.add_text("config/start_time", start_time_tag)
    writer.add_scalar("meta/dataset_size", float(len(dataset)), 0)
    global_step = 0
    print(f"[INFO] tensorboard_logdir={tb_root / session_name}")
    vis_dir = None
    if args.vis_pt:
        vis_dir = VIS_PT_ROOT / session_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] vis_pt_dir={vis_dir}")

    try:
        for epoch in range(1, cfg.epochs + 1):
            stats, global_step = train_one_epoch(
                loader=loader,
                descriptor=descriptor,
                generator=generator,
                opt_f=opt_f,
                opt_g=opt_g,
                cfg=cfg,
                device=device,
                epoch=epoch,
                vis_dir=vis_dir,
                writer=writer,
                global_step=global_step,
            )
            print(
                f"[Epoch {epoch:03d}] "
                f"loss_f={stats['loss_f']:.4f} "
                f"loss_g={stats['loss_g']:.4f} "
                f"clean={stats['loss_place_clean']:.4f} "
                f"adv={stats['loss_place_adv']:.4f} "
                f"cons={stats['loss_consistency']:.4f} "
                f"drop={stats['drop_ratio']:.3f}"
            )

            writer.add_scalar("train_epoch/loss_f", stats["loss_f"], epoch)
            writer.add_scalar("train_epoch/loss_g", stats["loss_g"], epoch)
            writer.add_scalar("train_epoch/loss_place_clean", stats["loss_place_clean"], epoch)
            writer.add_scalar("train_epoch/loss_place_adv", stats["loss_place_adv"], epoch)
            writer.add_scalar("train_epoch/loss_consistency", stats["loss_consistency"], epoch)
            writer.add_scalar("train_epoch/drop_ratio", stats["drop_ratio"], epoch)
            writer.add_scalar("train_epoch/target_drop_ratio", stats["target_drop_ratio"], epoch)
            writer.add_scalar("train_epoch/reg_size_prior", stats["reg_size_prior"], epoch)
            writer.add_scalar("train_epoch/reg_height_prior", stats["reg_height_prior"], epoch)
            writer.add_scalar("train_epoch/reg_range_prior", stats["reg_range_prior"], epoch)
            writer.add_scalar("train_epoch/lr_f", float(opt_f.param_groups[0]["lr"]), epoch)
            writer.add_scalar("train_epoch/lr_g", float(opt_g.param_groups[0]["lr"]), epoch)
            writer.flush()

            if epoch % args.save_every == 0:
                ckpt = {
                    "epoch": epoch,
                    "descriptor": descriptor.state_dict(),
                    "generator": generator.state_dict(),
                    "opt_f": opt_f.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "args": vars(args),
                    "session_name": session_name,
                    "start_time": start_time_tag,
                }
                ckpt_path = save_dir / f"{session_name}_epoch_{epoch:03d}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"[INFO] saved checkpoint: {ckpt_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
