from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from kitti_dataloader import KITTIPointCloudQueryDataset
from lpr_models import build_descriptor_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI place-recognition recall.")
    parser.add_argument(
        "--query-file",
        type=str,
        default="/TIEVNAS/jyf/KITTI/kitti_vxp_test_queries_baseline_p10_n25_yaw.pickle",
    )
    parser.add_argument("--kitti-root", type=str, default="/TIEVNAS/KITTI")
    parser.add_argument("--fallback-root", type=str, default="/TIEVNAS/jyf/KITTI")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--descriptor-arch", type=str, default="pointnetvlad")
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--max-elems", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-intensity", action="store_true")
    parser.add_argument("--exclude-self", dest="exclude_self", action="store_true")
    parser.add_argument("--include-self", dest="exclude_self", action="store_false")
    parser.set_defaults(exclude_self=True)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_descriptor(
    ckpt_path: str | None,
    descriptor_arch: str,
    num_points: int,
    in_channels: int,
    emb_dim: int,
    device: torch.device,
):
    state_dict = None
    if ckpt_path is None:
        print("[WARN] --checkpoint is not set, evaluating with random initialized descriptor.")
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            saved_args = ckpt.get("args", {})
            descriptor_arch = saved_args.get("descriptor_arch", descriptor_arch)
            num_points = saved_args.get("num_points", num_points)
            emb_dim = saved_args.get("emb_dim", emb_dim)
            state_dict = ckpt["descriptor"] if "descriptor" in ckpt else ckpt
        else:
            state_dict = ckpt

    model = build_descriptor_model(
        arch=descriptor_arch,
        num_points=num_points,
        emb_dim=emb_dim,
        in_channels=in_channels,
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    model.eval()
    return model


def _extract_embeddings(
    dataset: KITTIPointCloudQueryDataset,
    model,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, List[int]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    key_order: List[int] = []
    feats = None
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Extract embeddings"):
            points = points.to(device, non_blocking=True)
            emb = model(points).detach().cpu().numpy().astype(np.float32)

            labels_np = labels.numpy().tolist()
            key_order.extend(int(x) for x in labels_np)
            if feats is None:
                feats = np.empty((len(dataset), emb.shape[1]), dtype=np.float32)
            start = len(key_order) - emb.shape[0]
            feats[start : start + emb.shape[0], :] = emb

    if feats is None:
        if hasattr(model, "proj"):
            feat_dim = model.proj[-1].out_features
        elif hasattr(model, "net_vlad"):
            feat_dim = model.net_vlad.output_dim
        else:
            raise ValueError("Cannot infer descriptor output dimension from model.")
        feats = np.empty((0, feat_dim), dtype=np.float32)
    return feats, key_order


def compute_recall(
    features: np.ndarray,
    dataset: KITTIPointCloudQueryDataset,
    key_order: List[int],
    topk: int = 25,
    exclude_self: bool = True,
    max_evals: int | None = None,
) -> Dict[str, object]:
    if len(features) == 0:
        raise ValueError("Empty feature set.")

    num_db = features.shape[0]
    topk = min(topk, num_db)
    key_to_local = {k: i for i, k in enumerate(key_order)}
    tree = KDTree(features)
    recall_count = np.zeros(topk, dtype=np.float64)
    top1_similarity: List[float] = []
    one_percent_retrieved = 0

    threshold = max(int(round(num_db / 100.0)), 1)
    num_evaluated = 0

    max_loop = len(key_order) if max_evals is None else min(len(key_order), int(max_evals))
    for i in tqdm(range(max_loop), desc="Compute Recall"):
        q_key = key_order[i]
        positives_key = dataset.get_positives_ndx(q_key)

        positives_local = []
        for pk in positives_key:
            if pk in key_to_local:
                local_idx = key_to_local[pk]
                if exclude_self and local_idx == i:
                    continue
                positives_local.append(local_idx)

        if len(positives_local) == 0:
            continue

        num_evaluated += 1
        query = features[i : i + 1]
        # query extra neighbors then remove self.
        search_k = min(num_db, topk + 1 if exclude_self else topk)
        _, idx = tree.query(query, k=search_k)
        pred = idx[0].tolist()

        if exclude_self:
            pred = [p for p in pred if p != i]
        pred = pred[:topk]

        pos_set = set(positives_local)
        for rank, p in enumerate(pred):
            if p in pos_set:
                if rank == 0:
                    top1_similarity.append(float(np.dot(features[i], features[p])))
                recall_count[rank] += 1.0
                break

        if len(set(pred[:threshold]).intersection(pos_set)) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        raise RuntimeError("No valid query for evaluation. Check query file or filtering.")

    recall = np.cumsum(recall_count) / float(num_evaluated)
    one_percent_recall = one_percent_retrieved / float(num_evaluated)
    return {
        "num_db": num_db,
        "num_evaluated": num_evaluated,
        "topk": topk,
        "recall": recall.tolist(),
        "recall_at_1": float(recall[0]),
        "recall_at_5": float(recall[min(4, topk - 1)]),
        "recall_at_10": float(recall[min(9, topk - 1)]),
        "one_percent_recall": float(one_percent_recall),
        "top1_similarity_mean": float(np.mean(top1_similarity)) if len(top1_similarity) > 0 else None,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = KITTIPointCloudQueryDataset(
        query_filepath=args.query_file,
        kitti_root=args.kitti_root,
        fallback_root=args.fallback_root,
        num_points=args.num_points,
        use_intensity=args.use_intensity,
        random_sample=False,
        prefer_cached=True,
        max_elems=args.max_elems,
    )
    print(f"[INFO] Dataset size: {len(dataset)}")

    in_channels = 4 if args.use_intensity else 3
    model = _load_descriptor(
        ckpt_path=args.checkpoint,
        descriptor_arch=args.descriptor_arch,
        num_points=args.num_points,
        in_channels=in_channels,
        emb_dim=args.emb_dim,
        device=device,
    )

    features, key_order = _extract_embeddings(
        dataset=dataset,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"[INFO] Extracted features: {features.shape}")

    result = compute_recall(
        features=features,
        dataset=dataset,
        key_order=key_order,
        topk=args.topk,
        exclude_self=args.exclude_self,
        max_evals=args.max_evals,
    )

    print(f"[RESULT] evaluated={result['num_evaluated']} db={result['num_db']}")
    print(f"[RESULT] Recall@1: {result['recall_at_1'] * 100:.2f}%")
    print(f"[RESULT] Recall@5: {result['recall_at_5'] * 100:.2f}%")
    print(f"[RESULT] Recall@10: {result['recall_at_10'] * 100:.2f}%")
    print(f"[RESULT] Recall@1%: {result['one_percent_recall'] * 100:.2f}%")
    for i, r in enumerate(result["recall"], start=1):
        print(f"[RESULT] Recall@{i}: {r * 100:.2f}%")

    if args.save_json is not None:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_file": args.query_file,
            "checkpoint": args.checkpoint,
            "kitti_root": args.kitti_root,
            "fallback_root": args.fallback_root,
            "num_points": args.num_points,
            "topk": args.topk,
            "exclude_self": args.exclude_self,
            "result": result,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Saved result json: {out_path}")


if __name__ == "__main__":
    main()
