from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kitti_dataloader import KITTIPointCloudQueryDataset, make_kitti_query_collate_fn
from lpr_models import build_descriptor_model
from train_kitti import KITTIPairBatchSampler, masked_batch_hard_triplet_loss


def train_one_epoch(
    loader: DataLoader,
    descriptor: nn.Module,
    optimizer: torch.optim.Optimizer,
    margin: float,
    device: torch.device,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[Dict[str, float], int]:
    descriptor.train()

    meter = {
        "loss_place": 0.0,
        "valid_batch": 0.0,
    }

    for points, _, positives_mask, negatives_mask in loader:
        points = points.to(device, non_blocking=True)
        positives_mask = positives_mask.to(device, non_blocking=True)
        negatives_mask = negatives_mask.to(device, non_blocking=True)

        embeddings = descriptor(points)
        loss_place = masked_batch_hard_triplet_loss(
            embeddings=embeddings,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            margin=margin,
        )
        valid_batch = float(loss_place.requires_grad)
        if not loss_place.requires_grad:
            # No valid positive/negative pair in this batch. Keep the step well-defined.
            loss_place = embeddings.sum() * 0.0

        optimizer.zero_grad(set_to_none=True)
        loss_place.backward()
        optimizer.step()

        meter["loss_place"] += float(loss_place.item())
        meter["valid_batch"] += valid_batch

        if writer is not None:
            writer.add_scalar("train_iter/loss_place", float(loss_place.item()), global_step)
            writer.add_scalar("train_iter/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            writer.add_scalar("train_iter/valid_batch", valid_batch, global_step)

        global_step += 1

    num_iter = max(len(loader), 1)
    for k in meter:
        meter[k] /= num_iter
    return meter, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNetVLAD baseline on KITTI.")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--save-dir", type=str, default="/media/autolab/tsy/dynamic/checkpoints_pnv")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--tb-logdir", type=str, default="/media/autolab/tsy/dynamic/tb_runs")
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
    optimizer = torch.optim.Adam(descriptor.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_root = Path(args.tb_logdir)
    tb_root.mkdir(parents=True, exist_ok=True)
    start_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base = args.run_name
    if run_base is None:
        run_base = f"pnv_np{args.num_points}_bs{args.batch_size}"
    session_name = f"{run_base}_{start_time_tag}"
    writer = SummaryWriter(log_dir=str(tb_root / session_name))
    writer.add_text("config/args", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    writer.add_text("config/device", str(device))
    writer.add_text("config/session_name", session_name)
    writer.add_text("config/start_time", start_time_tag)
    writer.add_scalar("meta/dataset_size", float(len(dataset)), 0)
    global_step = 0
    print(f"[INFO] tensorboard_logdir={tb_root / session_name}")

    try:
        for epoch in range(1, args.epochs + 1):
            stats, global_step = train_one_epoch(
                loader=loader,
                descriptor=descriptor,
                optimizer=optimizer,
                margin=args.margin,
                device=device,
                writer=writer,
                global_step=global_step,
            )
            print(f"[Epoch {epoch:03d}] loss_place={stats['loss_place']:.4f}")

            writer.add_scalar("train_epoch/loss_place", stats["loss_place"], epoch)
            writer.add_scalar("train_epoch/valid_batch", stats["valid_batch"], epoch)
            writer.add_scalar("train_epoch/lr", float(optimizer.param_groups[0]["lr"]), epoch)
            writer.flush()

            if epoch % args.save_every == 0:
                ckpt = {
                    "epoch": epoch,
                    "descriptor": descriptor.state_dict(),
                    "opt_f": optimizer.state_dict(),
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
