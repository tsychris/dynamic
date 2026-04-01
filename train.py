from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from lpr_models import batch_hard_triplet_loss, build_descriptor_model, embedding_consistency_loss
from occlusion_generator import (
    AdversarialOcclusionGenerator,
    apply_hard_drop_and_insert,
    apply_soft_drop,
    sample_active_box_counts,
)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class SyntheticPlaceDataset(Dataset):
    """
    Synthetic dataset only for smoke test.
    Replace with your real LPR dataset that returns:
      points: [N, C], labels: int
    """

    def __init__(self, num_places: int = 40, samples_per_place: int = 20, num_points: int = 1024) -> None:
        super().__init__()
        self.num_places = num_places
        self.samples_per_place = samples_per_place
        self.num_points = num_points
        self.total = num_places * samples_per_place

        g = torch.Generator().manual_seed(42)
        base = torch.randn(num_places, num_points, 3, generator=g) * 8.0
        # Add a place-specific centroid, so positives share coarse geometry.
        base_centers = torch.randn(num_places, 1, 3, generator=g) * 20.0
        base[:, :, 2] = base[:, :, 2] * 0.2
        self.base_clouds = base + base_centers

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = idx % self.num_places
        pts = self.base_clouds[label].clone()

        # Small SE(2) perturbation to mimic revisit.
        yaw = (torch.rand(1).item() - 0.5) * 0.4
        c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        trans = torch.tensor(
            [
                (torch.rand(1).item() - 0.5) * 1.0,
                (torch.rand(1).item() - 0.5) * 1.0,
                (torch.rand(1).item() - 0.5) * 0.2,
            ]
        )
        noise = torch.randn_like(pts) * 0.03
        pts = pts @ rot.T + trans + noise
        return pts, torch.tensor(label, dtype=torch.long)


class PKBatchSampler(Sampler[List[int]]):
    """
    Samples P classes and K samples per class for each batch.
    Ensures triplet positives and negatives exist in every mini-batch.
    """

    def __init__(
        self,
        labels: List[int],
        p_classes: int = 8,
        k_samples: int = 4,
        num_batches: int = 100,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.num_batches = num_batches

        self.label_to_indices: Dict[int, List[int]] = {}
        for i, lb in enumerate(labels):
            self.label_to_indices.setdefault(lb, []).append(i)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_labels = random.sample(self.unique_labels, k=min(self.p_classes, len(self.unique_labels)))
            batch: List[int] = []
            for lb in chosen_labels:
                pool = self.label_to_indices[lb]
                if len(pool) >= self.k_samples:
                    picked = random.sample(pool, self.k_samples)
                else:
                    picked = random.choices(pool, k=self.k_samples)
                batch.extend(picked)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


@dataclass
class TrainConfig:
    epochs: int = 8
    lr_f: float = 1e-3
    lr_g: float = 1e-3
    margin: float = 0.2
    adv_weight: float = 1.0
    consistency_weight: float = 0.2
    size_prior_weight: float = 0.1
    height_prior_weight: float = 0.05
    range_prior_weight: float = 0.05
    use_object_insertion: bool = True


def train_one_epoch(
    loader: DataLoader,
    descriptor: nn.Module,
    generator: AdversarialOcclusionGenerator,
    opt_f: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    descriptor.train()
    generator.train()

    meter = {
        "loss_f": 0.0,
        "loss_g": 0.0,
        "loss_place_clean": 0.0,
        "loss_place_adv": 0.0,
        "occluded_fraction": 0.0,
        "active_num_boxes": 0.0,
    }

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        active_box_counts = sample_active_box_counts(
            batch=labels.shape[0],
            max_boxes=generator.num_boxes,
            device=device,
        )

        # (1) maximize_G L_place(f(G(x)))
        set_requires_grad(descriptor, False)
        set_requires_grad(generator, True)

        occl = generator(points, active_box_counts=active_box_counts, generate_insertion=False)
        adv_points_for_g = apply_soft_drop(points, occl.st_drop_mask)
        emb_adv = descriptor(adv_points_for_g)
        loss_place_adv = batch_hard_triplet_loss(emb_adv, labels, margin=cfg.margin)

        reg = (
            cfg.size_prior_weight * occl.regularization["size_prior"]
            + cfg.height_prior_weight * occl.regularization["height_prior"]
            + cfg.range_prior_weight * occl.regularization["range_prior"]
        )
        loss_g = -loss_place_adv + reg

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        # (2) minimize_f L_place on clean + adversarial
        set_requires_grad(descriptor, True)
        set_requires_grad(generator, False)
        with torch.no_grad():
            occl_detach = generator(
                points,
                active_box_counts=active_box_counts,
                generate_insertion=cfg.use_object_insertion,
            )
            adv_points_for_f = apply_hard_drop_and_insert(
                points=points,
                hard_drop_mask=occl_detach.hard_drop_mask,
                inserted_points_xyz=occl_detach.inserted_points if cfg.use_object_insertion else None,
            )

        emb_clean = descriptor(points)
        emb_adv_detach = descriptor(adv_points_for_f)

        loss_clean = batch_hard_triplet_loss(emb_clean, labels, margin=cfg.margin)
        loss_adv = batch_hard_triplet_loss(emb_adv_detach, labels, margin=cfg.margin)
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
            meter["occluded_fraction"] += float(occl_detach.hard_drop_mask.mean().item())
            meter["active_num_boxes"] += float(occl_detach.active_box_counts.float().mean().item())

    num_iter = max(len(loader), 1)
    for k in meter:
        meter[k] /= num_iter
    return meter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial dynamic occlusion training for LPR")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-places", type=int, default=40)
    parser.add_argument("--samples-per-place", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--p-classes", type=int, default=8)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=60)
    parser.add_argument("--descriptor-arch", type=str, default="pointnetvlad")
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--lr-f", type=float, default=1e-3)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--size-prior-weight", type=float, default=0.1)
    parser.add_argument("--height-prior-weight", type=float, default=0.05)
    parser.add_argument("--range-prior-weight", type=float, default=0.05)
    parser.add_argument("--no-object-insertion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticPlaceDataset(
        num_places=args.num_places,
        samples_per_place=args.samples_per_place,
        num_points=args.num_points,
    )
    labels = [i % args.num_places for i in range(len(dataset))]
    sampler = PKBatchSampler(
        labels=labels,
        p_classes=args.p_classes,
        k_samples=args.k_samples,
        num_batches=args.num_batches,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    descriptor = build_descriptor_model(
        arch=args.descriptor_arch,
        num_points=args.num_points,
        emb_dim=args.emb_dim,
        in_channels=3,
    ).to(device)
    generator = AdversarialOcclusionGenerator(
        num_boxes=3,
        points_per_box=64,
        temperature=0.2,
        geom_weight=2.0,
    ).to(device)

    opt_f = torch.optim.Adam(descriptor.parameters(), lr=args.lr_f)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr_f=args.lr_f,
        lr_g=args.lr_g,
        margin=args.margin,
        adv_weight=args.adv_weight,
        consistency_weight=args.consistency_weight,
        size_prior_weight=args.size_prior_weight,
        height_prior_weight=args.height_prior_weight,
        range_prior_weight=args.range_prior_weight,
        use_object_insertion=not args.no_object_insertion,
    )

    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(
            loader=loader,
            descriptor=descriptor,
            generator=generator,
            opt_f=opt_f,
            opt_g=opt_g,
            cfg=cfg,
            device=device,
        )
        print(
            f"[Epoch {epoch:03d}] "
            f"loss_f={stats['loss_f']:.4f} "
            f"loss_g={stats['loss_g']:.4f} "
            f"clean={stats['loss_place_clean']:.4f} "
            f"adv={stats['loss_place_adv']:.4f} "
            f"occluded={stats['occluded_fraction']:.3f} "
            f"boxes={stats['active_num_boxes']:.2f}"
        )

    ckpt = {
        "descriptor": descriptor.state_dict(),
        "generator": generator.state_dict(),
        "args": vars(args),
    }
    ckpt_path = Path(__file__).resolve().parent / "dynamic_checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
