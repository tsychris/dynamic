from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _angle_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def sample_active_box_counts(
    batch: int,
    max_boxes: int,
    device: torch.device,
    min_boxes: int = 1,
) -> torch.Tensor:
    if max_boxes < 1:
        raise ValueError(f"max_boxes must be >= 1, got {max_boxes}")
    min_boxes = max(1, min(int(min_boxes), int(max_boxes)))
    return torch.randint(min_boxes, max_boxes + 1, (int(batch),), device=device, dtype=torch.long)


def _sample_active_box_mask(active_box_counts: torch.Tensor, num_boxes: int) -> torch.Tensor:
    """
    Randomly pick exactly k active boxes for each sample.
    """
    if num_boxes < 1:
        raise ValueError(f"num_boxes must be >= 1, got {num_boxes}")

    counts = active_box_counts.to(dtype=torch.long)
    counts = counts.clamp(1, num_boxes)
    bsz = int(counts.shape[0])
    active = torch.zeros((bsz, num_boxes), device=counts.device, dtype=torch.bool)

    for b in range(bsz):
        perm = torch.randperm(num_boxes, device=counts.device)
        active[b, perm[: int(counts[b].item())]] = True

    return active


def _straight_through_threshold(
    soft_scores: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward uses a hard thresholded mask, backward follows the soft scores.
    """
    soft = soft_scores.clamp(0.0, 1.0)
    hard = (soft >= float(threshold)).to(dtype=soft.dtype)
    st = hard - soft.detach() + soft
    return hard, st, soft


def _decode_box_params(
    raw_box: torch.Tensor,
    scene_extent_xyz: torch.Tensor,
    min_box_size: torch.Tensor,
    max_box_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.tanh(raw_box[..., 0:3]) * scene_extent_xyz
    sizes = min_box_size + torch.sigmoid(raw_box[..., 3:6]) * (max_box_size - min_box_size)
    yaws = torch.tanh(raw_box[..., 6]) * torch.pi
    return centers, sizes, yaws


def _box_shadow_scores_per_box(
    points_xyz: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    sharpness: float = 30.0,
) -> torch.Tensor:
    """
    Approximate LiDAR occlusion from inserted cuboids:
    1) points inside box are occluded;
    2) points in the box's angular shadow cone and farther than the box are occluded.
    """
    eps = 1e-6
    bsz, npts, _ = points_xyz.shape
    nbox = centers.shape[1]

    p = points_xyz[:, :, None, :]  # [B, N, 1, 3]
    c = centers[:, None, :, :]  # [B, 1, M, 3]
    rel = p - c

    # Rotate points into box local frame (around z axis).
    yaw = yaws[:, None, :]  # [B, 1, M]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x_local = cos_yaw * rel[..., 0] + sin_yaw * rel[..., 1]
    y_local = -sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]
    z_local = rel[..., 2]

    half = 0.5 * sizes[:, None, :, :]  # [B, 1, M, 3]
    inside_margin = 1.0 - torch.maximum(
        torch.maximum(torch.abs(x_local) / (half[..., 0] + eps), torch.abs(y_local) / (half[..., 1] + eps)),
        torch.abs(z_local) / (half[..., 2] + eps),
    )
    inside_score = torch.sigmoid(inside_margin * sharpness)

    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    rp = torch.sqrt(px * px + py * py + pz * pz + eps)
    rc = torch.sqrt(cx * cx + cy * cy + cz * cz + eps)

    theta_p = torch.atan2(py, px)
    theta_c = torch.atan2(cy, cx)
    phi_p = torch.atan2(pz, torch.sqrt(px * px + py * py + eps))
    phi_c = torch.atan2(cz, torch.sqrt(cx * cx + cy * cy + eps))

    # Angular half extents from box dimensions.
    lateral_radius = 0.5 * torch.sqrt(sizes[:, None, :, 0] ** 2 + sizes[:, None, :, 1] ** 2)
    half_theta = torch.atan2(lateral_radius, rc + eps)
    half_phi = torch.atan2(0.5 * sizes[:, None, :, 2], rc + eps)

    d_theta = torch.abs(_angle_wrap(theta_p - theta_c))
    d_phi = torch.abs(phi_p - phi_c)
    in_theta = torch.sigmoid((half_theta - d_theta) * sharpness)
    in_phi = torch.sigmoid((half_phi - d_phi) * sharpness)

    # Point should be behind the box along the ray to be shadowed.
    behind = torch.sigmoid((rp - (rc + 0.5 * sizes[:, None, :, 0])) * sharpness)
    shadow_score = in_theta * in_phi * behind

    if nbox == 0:
        return torch.zeros((bsz, npts, 0), device=points_xyz.device, dtype=points_xyz.dtype)
    return torch.maximum(inside_score, shadow_score)  # [B, N, M]


def _fuse_active_box_scores(
    per_box_scores: torch.Tensor,
    active_box_mask: torch.Tensor,
) -> torch.Tensor:
    if per_box_scores.ndim != 3:
        raise ValueError(f"Expected per_box_scores [B, N, M], got {tuple(per_box_scores.shape)}")
    if active_box_mask.ndim != 2:
        raise ValueError(f"Expected active_box_mask [B, M], got {tuple(active_box_mask.shape)}")

    masked_scores = per_box_scores * active_box_mask[:, None, :].to(dtype=per_box_scores.dtype)
    return masked_scores.max(dim=2).values


def _box_surface_template(points_per_box: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Deterministic points on unit cube surface in [-0.5, 0.5]^3.
    """
    idx = torch.arange(points_per_box, device=device, dtype=dtype)
    face_id = (idx.to(torch.long) % 6).to(torch.long)
    u = ((idx * 0.61803398875) % 1.0) - 0.5
    v = ((idx * 0.41421356237) % 1.0) - 0.5

    pts = torch.zeros(points_per_box, 3, device=device, dtype=dtype)
    # +/- X faces
    m = face_id == 0
    pts[m, 0] = 0.5
    pts[m, 1] = u[m]
    pts[m, 2] = v[m]
    m = face_id == 1
    pts[m, 0] = -0.5
    pts[m, 1] = u[m]
    pts[m, 2] = v[m]
    # +/- Y faces
    m = face_id == 2
    pts[m, 1] = 0.5
    pts[m, 0] = u[m]
    pts[m, 2] = v[m]
    m = face_id == 3
    pts[m, 1] = -0.5
    pts[m, 0] = u[m]
    pts[m, 2] = v[m]
    # +/- Z faces
    m = face_id == 4
    pts[m, 2] = 0.5
    pts[m, 0] = u[m]
    pts[m, 1] = v[m]
    m = face_id == 5
    pts[m, 2] = -0.5
    pts[m, 0] = u[m]
    pts[m, 1] = v[m]
    return pts


def sample_inserted_points(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    points_per_box: int,
) -> torch.Tensor:
    """
    Create synthetic LiDAR returns on box surfaces for object insertion.
    """
    bsz, nbox, _ = centers.shape
    template = _box_surface_template(points_per_box, centers.device, centers.dtype)  # [K, 3]
    local = template[None, None, :, :] * sizes[:, :, None, :]  # [B, M, K, 3]

    cos_y = torch.cos(yaws)[:, :, None]
    sin_y = torch.sin(yaws)[:, :, None]

    x = local[..., 0]
    y = local[..., 1]
    z = local[..., 2]
    rot_x = cos_y * x - sin_y * y
    rot_y = sin_y * x + cos_y * y
    rot_z = z

    world = torch.stack([rot_x, rot_y, rot_z], dim=-1) + centers[:, :, None, :]
    return world


def _pack_active_inserted_points(
    box_surface_points: torch.Tensor,
    active_box_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pack active box surface samples to the front so insertion only uses active boxes.
    """
    if box_surface_points.ndim != 4:
        raise ValueError(f"Expected box_surface_points [B, M, K, 3], got {tuple(box_surface_points.shape)}")

    bsz, nbox, points_per_box, _ = box_surface_points.shape
    packed = box_surface_points.new_zeros((bsz, nbox * points_per_box, 3))
    for b in range(bsz):
        active_idx = torch.nonzero(active_box_mask[b], as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            continue
        selected = box_surface_points[b, active_idx].reshape(-1, 3)
        packed[b, : selected.shape[0]] = selected
    return packed


def apply_soft_drop(points: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable drop used during generator update.
    """
    out = points.clone()
    out[..., 0:3] = points[..., 0:3] * (1.0 - drop_mask[..., None])
    if points.shape[-1] > 3:
        out[..., 3:] = points[..., 3:] * (1.0 - drop_mask[..., None])
    return out


def apply_hard_drop_and_insert(
    points: torch.Tensor,
    hard_drop_mask: torch.Tensor,
    inserted_points_xyz: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Non-differentiable composition used during descriptor update.
    """
    out = points.clone()
    bsz, _, channels = out.shape

    for b in range(bsz):
        drop_idx = torch.nonzero(hard_drop_mask[b] > 0.5, as_tuple=False).squeeze(-1)
        if drop_idx.numel() == 0:
            continue

        out[b, drop_idx] = 0.0
        if inserted_points_xyz is None:
            continue

        num_insert = min(drop_idx.numel(), inserted_points_xyz.shape[1])
        use_idx = drop_idx[:num_insert]
        out[b, use_idx, 0:3] = inserted_points_xyz[b, :num_insert]
        if channels > 3:
            out[b, use_idx, 3:] = 0.0

    return out


@dataclass
class OcclusionOutput:
    hard_drop_mask: torch.Tensor
    st_drop_mask: torch.Tensor
    soft_drop_mask: torch.Tensor
    scores: torch.Tensor
    geom_scores: torch.Tensor
    active_box_counts: torch.Tensor
    active_box_mask: torch.Tensor
    centers: torch.Tensor
    sizes: torch.Tensor
    yaws: torch.Tensor
    inserted_points: Optional[torch.Tensor]
    regularization: Dict[str, torch.Tensor]


class AdversarialOcclusionGenerator(nn.Module):
    """
    Generator for dynamic occlusion:
    - predicts vehicle-like cuboids;
    - randomly activates a subset of the predicted cuboids per sample;
    - removes every point geometrically occluded by the active cuboids.
    """

    def __init__(
        self,
        num_boxes: int = 3,
        feature_dim: int = 128,
        points_per_box: int = 64,
        temperature: float = 0.2,
        point_weight: float = 1.0,
        geom_weight: float = 2.0,
        scene_extent_xyz: tuple[float, float, float] = (40.0, 40.0, 3.0),
        min_box_size: tuple[float, float, float] = (2.5, 1.5, 1.2),
        max_box_size: tuple[float, float, float] = (6.5, 2.8, 3.0),
    ) -> None:
        super().__init__()
        self.num_boxes = num_boxes
        self.points_per_box = points_per_box
        self.temperature = temperature
        self.point_weight = point_weight
        self.geom_weight = geom_weight

        self.register_buffer("scene_extent_xyz", torch.tensor(scene_extent_xyz).float())
        self.register_buffer("min_box_size", torch.tensor(min_box_size).float())
        self.register_buffer("max_box_size", torch.tensor(max_box_size).float())
        self.register_buffer("nominal_car_size", torch.tensor([4.2, 1.8, 1.6]).float())

        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Linear(128, num_boxes * 7)
        self.point_head = nn.Sequential(
            nn.Linear(feature_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        active_box_counts: torch.Tensor | None = None,
        generate_insertion: bool = True,
    ) -> OcclusionOutput:
        """
        points: [B, N, C], first 3 dims are xyz.
        active_box_counts: [B], random count of active boxes in [1, num_boxes].
        """
        xyz = points[..., 0:3]
        bsz, _, _ = xyz.shape

        if active_box_counts is None:
            active_box_counts = sample_active_box_counts(
                batch=bsz,
                max_boxes=self.num_boxes,
                device=points.device,
            )
        elif active_box_counts.ndim == 0:
            active_box_counts = active_box_counts.unsqueeze(0).repeat(bsz)
        active_box_counts = active_box_counts.to(device=points.device, dtype=torch.long).clamp(1, self.num_boxes)

        active_box_fraction = active_box_counts.to(dtype=points.dtype) / float(self.num_boxes)
        active_box_mask = _sample_active_box_mask(active_box_counts=active_box_counts, num_boxes=self.num_boxes)

        point_feat = self.point_mlp(xyz)  # [B, N, F]
        global_feat = point_feat.max(dim=1).values  # [B, F]
        cond = self.global_mlp(torch.cat([global_feat, active_box_fraction[:, None]], dim=-1))  # [B, 128]

        raw_box = self.box_head(cond).view(bsz, self.num_boxes, 7)
        centers, sizes, yaws = _decode_box_params(
            raw_box=raw_box,
            scene_extent_xyz=self.scene_extent_xyz.to(points.dtype),
            min_box_size=self.min_box_size.to(points.dtype),
            max_box_size=self.max_box_size.to(points.dtype),
        )

        per_box_geom_scores = _box_shadow_scores_per_box(
            points_xyz=xyz,
            centers=centers,
            sizes=sizes,
            yaws=yaws,
        )
        geom_scores = self.geom_weight * _fuse_active_box_scores(
            per_box_scores=per_box_geom_scores,
            active_box_mask=active_box_mask,
        )
        geom_scores = geom_scores.clamp(0.0, 1.0)

        hard, st, soft = _straight_through_threshold(
            soft_scores=geom_scores,
            threshold=0.5,
        )

        inserted_points = None
        if generate_insertion:
            box_surface_points = sample_inserted_points(
                centers=centers,
                sizes=sizes,
                yaws=yaws,
                points_per_box=self.points_per_box,
            )
            inserted_points = _pack_active_inserted_points(
                box_surface_points=box_surface_points,
                active_box_mask=active_box_mask,
            )

        regularization = self._regularization(centers=centers, sizes=sizes)

        return OcclusionOutput(
            hard_drop_mask=hard,
            st_drop_mask=st,
            soft_drop_mask=soft,
            scores=geom_scores,
            geom_scores=geom_scores,
            active_box_counts=active_box_counts,
            active_box_mask=active_box_mask,
            centers=centers,
            sizes=sizes,
            yaws=yaws,
            inserted_points=inserted_points,
            regularization=regularization,
        )

    def _regularization(self, centers: torch.Tensor, sizes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Keep generated boxes in realistic dynamic-object ranges.
        """
        nominal = self.nominal_car_size.to(device=sizes.device, dtype=sizes.dtype).view(1, 1, 3).expand_as(sizes)
        size_prior = F.smooth_l1_loss(sizes, nominal)

        # Keep objects near drivable vertical band.
        height_prior = ((centers[..., 2] + 1.0) ** 2).mean()

        # Prevent centers drifting too far from typical sensor range.
        radial = torch.sqrt(centers[..., 0] ** 2 + centers[..., 1] ** 2 + 1e-6)
        range_prior = F.relu(radial - 45.0).pow(2).mean()

        return {
            "size_prior": size_prior,
            "height_prior": height_prior,
            "range_prior": range_prior,
        }
