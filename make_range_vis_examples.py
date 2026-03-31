from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_points(bin_path: str) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def load_kitti_calib(calib_path: str) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"Failed to parse P2/Tr from {calib_path}")
    return p2.astype(np.float32), tr.astype(np.float32)


def project_points_to_range_view(
    points_xyz: np.ndarray,
    height: int = 64,
    width: int = 1024,
    fov_up_deg: float = 2.0,
    fov_down_deg: float = -24.8,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected points shape [N, >=3], got {xyz.shape}")
    xyz = xyz[:, :3]

    ranges = np.linalg.norm(xyz, axis=1)
    valid = ranges > 1e-6
    xyz = xyz[valid]
    ranges = ranges[valid]
    if xyz.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / np.maximum(ranges, 1e-6), -1.0, 1.0))

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)

    proj_x = 0.5 * (1.0 - yaw / np.pi)
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    proj_x = np.floor(np.clip(proj_x * width, 0, width - 1)).astype(np.int32)
    proj_y = np.floor(np.clip(proj_y * height, 0, height - 1)).astype(np.int32)

    order = np.argsort(ranges)[::-1]
    range_view = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    range_view[proj_y[order], proj_x[order]] = ranges[order]
    valid_mask[proj_y[order], proj_x[order]] = True
    return range_view, valid_mask


def project_to_front_depth(
    points_xyz: np.ndarray,
    p2: np.ndarray,
    tr: np.ndarray,
    img_w: int,
    img_h: int,
    max_depth_m: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    dists = np.linalg.norm(xyz, axis=1)
    xyz = xyz[dists < max_depth_m]

    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    tr4 = np.vstack([tr, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    cam_pts = (tr4 @ xyz_h.T).T

    mask_front = cam_pts[:, 2] > 0.1
    cam_pts = cam_pts[mask_front]
    if cam_pts.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32), np.zeros((img_h, img_w), dtype=bool)

    proj = (p2 @ cam_pts.T).T
    z = cam_pts[:, 2]
    u = np.round(proj[:, 0] / np.maximum(proj[:, 2], 1e-8)).astype(np.int32)
    v = np.round(proj[:, 1] / np.maximum(proj[:, 2], 1e-8)).astype(np.int32)

    keep = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[keep]
    v = v[keep]
    z = z[keep]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    z = z[order]

    depth = np.zeros((img_h, img_w), dtype=np.float32)
    valid = np.zeros((img_h, img_w), dtype=bool)
    depth[v, u] = z
    valid[v, u] = True
    return depth, valid


def densify_depth(depth_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_depth = np.zeros_like(depth_map, dtype=np.float32)
    inv_depth[valid_mask] = 1.0 / np.maximum(depth_map[valid_mask], 1e-6)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    d_inv = cv2.dilate(inv_depth, k5, iterations=1)
    d_inv = cv2.dilate(d_inv, k3, iterations=1)

    filled = d_inv > 0
    dense_depth = np.zeros_like(depth_map, dtype=np.float32)
    dense_depth[filled] = 1.0 / np.maximum(d_inv[filled], 1e-6)
    return dense_depth, filled


def colorize_depth(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    max_depth_m: float,
    invert: bool = True,
) -> np.ndarray:
    normalized = np.zeros_like(depth_map, dtype=np.float32)
    clipped = np.clip(depth_map, 0.0, max_depth_m)
    normalized[valid_mask] = clipped[valid_mask] / max_depth_m
    if invert:
        normalized[valid_mask] = 1.0 - normalized[valid_mask]

    img_u8 = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    img_u8[valid_mask] = colored[valid_mask]
    return img_u8


def overlay_depth_on_image(rgb: np.ndarray, color_depth: np.ndarray, valid_mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    base = rgb.astype(np.float32)
    over = color_depth.astype(np.float32)
    out = base.copy()
    mask3 = np.repeat(valid_mask[:, :, None], 3, axis=2)
    out[mask3] = (1.0 - alpha) * base[mask3] + alpha * over[mask3]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def save_figure(
    rgb: np.ndarray,
    spherical_color: np.ndarray,
    front_sparse_color: np.ndarray,
    front_dense_color: np.ndarray,
    overlay: np.ndarray,
    out_path: Path,
    frame_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.reshape(-1)

    axes[0].imshow(spherical_color)
    axes[0].set_title("Current Spherical Range View")
    axes[0].axis("off")

    axes[1].imshow(front_sparse_color)
    axes[1].set_title("Reference-Style Front Sparse Depth")
    axes[1].axis("off")

    axes[2].imshow(front_dense_color)
    axes[2].set_title("Reference-Style Front Dense Depth")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Front Dense Depth Overlay on RGB")
    axes[3].axis("off")

    fig.suptitle(f"KITTI Frame {frame_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate current-vs-reference-style range/depth visualization examples.")
    parser.add_argument("--bin-path", required=True, type=str)
    parser.add_argument("--calib-path", required=True, type=str)
    parser.add_argument("--image-path", required=True, type=str)
    parser.add_argument("--out-path", required=True, type=str)
    parser.add_argument("--max-depth-m", type=float, default=50.0)
    args = parser.parse_args()

    points_xyz = load_points(args.bin_path)
    p2, tr = load_kitti_calib(args.calib_path)

    rgb = np.array(Image.open(args.image_path).convert("RGB"))
    img_h, img_w = rgb.shape[:2]

    spherical_depth, spherical_valid = project_points_to_range_view(points_xyz)
    front_sparse_depth, front_sparse_valid = project_to_front_depth(
        points_xyz=points_xyz,
        p2=p2,
        tr=tr,
        img_w=img_w,
        img_h=img_h,
        max_depth_m=args.max_depth_m,
    )
    front_dense_depth, front_dense_valid = densify_depth(front_sparse_depth, front_sparse_valid)

    spherical_color = colorize_depth(
        depth_map=spherical_depth,
        valid_mask=spherical_valid,
        max_depth_m=args.max_depth_m,
    )
    front_sparse_color = colorize_depth(
        depth_map=front_sparse_depth,
        valid_mask=front_sparse_valid,
        max_depth_m=args.max_depth_m,
    )
    front_dense_color = colorize_depth(
        depth_map=front_dense_depth,
        valid_mask=front_dense_valid,
        max_depth_m=args.max_depth_m,
    )
    overlay = overlay_depth_on_image(rgb, front_dense_color, front_dense_valid)

    frame_id = Path(args.bin_path).stem
    save_figure(
        rgb=rgb,
        spherical_color=spherical_color,
        front_sparse_color=front_sparse_color,
        front_dense_color=front_dense_color,
        overlay=overlay,
        out_path=Path(args.out_path),
        frame_id=frame_id,
    )
    print(args.out_path)


if __name__ == "__main__":
    main()
