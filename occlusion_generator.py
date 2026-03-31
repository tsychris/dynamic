from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _angle_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _hard_topk_mask(
    scores: torch.Tensor,
    drop_ratios: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project soft scores into an exact top-k hard mask for each sample.
    """
    bsz, npts = scores.shape
    soft = torch.sigmoid(scores / max(temperature, 1e-4))
    hard = torch.zeros_like(scores)

    for b in range(bsz):
        k = int(torch.round(drop_ratios[b] * npts).clamp(1, npts - 1).item())
        idx = torch.topk(scores[b], k=k, dim=-1).indices
        hard[b, idx] = 1.0

    # Straight-through estimator: forward sees hard, backward follows soft.
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


def _box_shadow_scores(
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

    per_box_score = torch.maximum(inside_score, shadow_score)  # [B, N, M]
    fused = per_box_score.max(dim=2).values  # [B, N]
    if nbox == 0:
        return torch.zeros((bsz, npts), device=points_xyz.device, dtype=points_xyz.dtype)
    return fused


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
    return world.reshape(bsz, nbox * points_per_box, 3)


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
    centers: torch.Tensor
    sizes: torch.Tensor
    yaws: torch.Tensor
    inserted_points: Optional[torch.Tensor]
    regularization: Dict[str, torch.Tensor]


class AdversarialOcclusionGenerator(nn.Module):
    """
    Generator for dynamic occlusion:
    - predicts vehicle-like cuboids;
    - computes occlusion scores from geometry + learned per-point scores;
    - enforces exact dropout ratio with hard top-k projection.
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
        drop_ratios: torch.Tensor,
        generate_insertion: bool = True,
    ) -> OcclusionOutput:
        """
        points: [B, N, C], first 3 dims are xyz.
        drop_ratios: [B], expected in [0, 1], typically sampled from {0.1..0.5}.
        """
        xyz = points[..., 0:3]
        bsz, npts, _ = xyz.shape

        if drop_ratios.ndim == 0:
            drop_ratios = drop_ratios.unsqueeze(0).repeat(bsz)
        drop_ratios = drop_ratios.to(points.device, dtype=points.dtype)
        drop_ratios = drop_ratios.clamp(0.01, 0.95)

        point_feat = self.point_mlp(xyz)  # [B, N, F]
        global_feat = point_feat.max(dim=1).values  # [B, F]
        cond = self.global_mlp(torch.cat([global_feat, drop_ratios[:, None]], dim=-1))  # [B, 128]

        raw_box = self.box_head(cond).view(bsz, self.num_boxes, 7)
        centers, sizes, yaws = _decode_box_params(
            raw_box=raw_box,
            scene_extent_xyz=self.scene_extent_xyz.to(points.dtype),
            min_box_size=self.min_box_size.to(points.dtype),
            max_box_size=self.max_box_size.to(points.dtype),
        )

        geom_scores = _box_shadow_scores(
            points_xyz=xyz,
            centers=centers,
            sizes=sizes,
            yaws=yaws,
        )

        cond_expand = cond[:, None, :].expand(-1, npts, -1)
        point_logits = self.point_head(torch.cat([point_feat, cond_expand], dim=-1)).squeeze(-1)
        scores = self.point_weight * point_logits + self.geom_weight * geom_scores

        hard, st, soft = _hard_topk_mask(
            scores=scores,
            drop_ratios=drop_ratios,
            temperature=self.temperature,
        )

        inserted_points = None
        if generate_insertion:
            inserted_points = sample_inserted_points(
                centers=centers,
                sizes=sizes,
                yaws=yaws,
                points_per_box=self.points_per_box,
            )

        regularization = self._regularization(centers=centers, sizes=sizes)

        return OcclusionOutput(
            hard_drop_mask=hard,
            st_drop_mask=st,
            soft_drop_mask=soft,
            scores=scores,
            geom_scores=geom_scores,
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
