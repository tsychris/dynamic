from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kitti_dataloader import load_kitti_points, resolve_kitti_path, sample_or_pad_points
from lpr_models import build_descriptor_model
from occlusion_generator import AdversarialOcclusionGenerator, apply_hard_drop_and_insert


def infer_generator_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    feature_dim = int(state_dict["point_mlp.4.weight"].shape[0])
    num_boxes = int(state_dict["box_head.weight"].shape[0] // 7)
    return feature_dim, num_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PointNetVLAD recall on KITTI with dirty query / dirty db.")
    parser.add_argument(
        "--query-file",
        type=str,
        default="/TIEVNAS/jyf/KITTI/kitti_00_evaluation_query_th20m_yaw_1runs.pickle",
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default="/TIEVNAS/jyf/KITTI/kitti_00_evaluation_database_th20m_yaw_1runs.pickle",
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--db-index", type=int, default=0)
    parser.add_argument("--kitti-root", type=str, default="/TIEVNAS/KITTI")
    parser.add_argument("--fallback-root", type=str, default="/TIEVNAS/jyf/KITTI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--max-query-elems", type=int, default=None)
    parser.add_argument("--max-db-elems", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-time-gap", type=float, default=10.0)
    parser.add_argument("--filter-same-frame", action="store_true")
    parser.add_argument("--require-db-earlier", action="store_true")
    parser.add_argument(
        "--dirty-generator-checkpoint",
        type=str,
        default=None,
        help="Checkpoint that provides generator weights for dirty query/db synthesis. Defaults to --checkpoint.",
    )
    parser.add_argument(
        "--query-dirty-active-boxes",
        type=int,
        default=None,
        help="Use a fixed number of active occlusion boxes per query. Defaults to random [1, num_boxes].",
    )
    parser.add_argument(
        "--db-dirty-active-boxes",
        type=int,
        default=None,
        help="Use a fixed number of active occlusion boxes per db item. Defaults to query setting.",
    )
    parser.add_argument(
        "--no-query-object-insertion",
        action="store_true",
        help="Disable synthetic inserted object points and keep only query dropout.",
    )
    parser.add_argument(
        "--no-db-object-insertion",
        action="store_true",
        help="Disable synthetic inserted object points and keep only db dropout.",
    )
    parser.add_argument(
        "--dirty-point-weight",
        type=float,
        default=None,
        help="Override generator point-logit weight at inference time. Default uses checkpoint setting.",
    )
    parser.add_argument(
        "--dirty-geom-weight",
        type=float,
        default=None,
        help="Override generator geometry weight at inference time. Default uses checkpoint setting.",
    )
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def load_descriptor_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    descriptor_arch = saved_args.get("descriptor_arch", "pointnetvlad")
    num_points = saved_args.get("num_points", 4096)
    emb_dim = saved_args.get("emb_dim", 256)
    use_intensity = saved_args.get("use_intensity", False)
    in_channels = 4 if use_intensity else 3

    model = build_descriptor_model(
        arch=descriptor_arch,
        num_points=num_points,
        emb_dim=emb_dim,
        in_channels=in_channels,
    ).to(device)
    model.load_state_dict(ckpt["descriptor"], strict=False)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return model, num_points, use_intensity, emb_dim


def load_generator_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    point_weight_override: float | None = None,
    geom_weight_override: float | None = None,
) -> tuple[AdversarialOcclusionGenerator, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "generator" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain generator weights. "
            "Pass --dirty-generator-checkpoint with an adversarial-training checkpoint."
        )

    saved_args = ckpt.get("args", {})
    feature_dim, num_boxes = infer_generator_config(ckpt["generator"])
    point_weight = float(saved_args.get("point_weight", 1.0) if point_weight_override is None else point_weight_override)
    geom_weight = float(saved_args.get("geom_weight", 2.0) if geom_weight_override is None else geom_weight_override)
    generator = AdversarialOcclusionGenerator(
        num_boxes=num_boxes,
        feature_dim=feature_dim,
        point_weight=point_weight,
        geom_weight=geom_weight,
    ).to(device)
    generator.load_state_dict(ckpt["generator"], strict=True)
    generator.eval()
    print(
        f"[INFO] Loaded dirty-query generator: {ckpt_path} "
        f"point_weight={point_weight:.3f} geom_weight={geom_weight:.3f}"
    )
    return generator, saved_args


def _load_pickle_records(pickle_path: str, split_index: int) -> tuple[Dict[int, dict], str]:
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        if split_index >= len(obj):
            raise IndexError(f"split_index={split_index} out of range for {pickle_path} with len={len(obj)}")
        records = obj[split_index]
        if not isinstance(records, dict):
            raise TypeError(f"Expected dict at list[{split_index}] in {pickle_path}, got {type(records)}")
        return records, "evaluation"

    if isinstance(obj, dict):
        return obj, "query"

    raise TypeError(f"Unsupported pickle structure in {pickle_path}: {type(obj)}")


def _normalize_submap_path(path: str) -> str:
    p = str(path)
    p = p.replace("/TIEVNAS/jyf/KITTI", "/TIEVNAS/KITTI")
    p = p.replace("/TIEVNAS/jinyuanfeng/KITTI", "/TIEVNAS/KITTI")
    return p


class KITTIPicklePointCloudDataset(Dataset):
    def __init__(
        self,
        pickle_path: str,
        split_index: int,
        kitti_root: str,
        fallback_root: str,
        num_points: int,
        use_intensity: bool,
        random_sample: bool,
        max_elems: int | None = None,
    ) -> None:
        super().__init__()
        self.records, self.pickle_type = _load_pickle_records(pickle_path, split_index)
        self.keys = sorted(self.records.keys())
        if max_elems is not None:
            self.keys = self.keys[: int(max_elems)]
        self.kitti_root = kitti_root
        self.fallback_root = fallback_root
        self.num_points = int(num_points)
        self.use_intensity = bool(use_intensity)
        self.random_sample = bool(random_sample)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        rec = self.records[key]
        pcd_path = resolve_kitti_path(
            rec["submap_path"],
            kitti_root=self.kitti_root,
            fallback_root=self.fallback_root,
        )
        points = load_kitti_points(pcd_path, use_intensity=self.use_intensity)
        points = sample_or_pad_points(points, num_points=self.num_points, random_sample=self.random_sample)
        return torch.from_numpy(points).float(), torch.tensor(int(key), dtype=torch.long)

    def get_record(self, idx: int) -> dict:
        return self.records[self.keys[idx]]

    def get_time(self, idx: int) -> float:
        return float(self.get_record(idx)["timestamp"])

    def get_submap_path(self, idx: int) -> str:
        return _normalize_submap_path(self.get_record(idx)["submap_path"])


def extract_embeddings(
    dataset: KITTIPicklePointCloudDataset,
    model,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dirty_generator: AdversarialOcclusionGenerator | None = None,
    dirty_active_boxes: int | None = None,
    dirty_object_insertion: bool = True,
    desc: str = "Extract embeddings",
) -> tuple[np.ndarray, List[int], Dict[str, float] | None]:
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
    dirty_stats = None
    dirty_occluded_acc = 0.0
    dirty_active_box_acc = 0.0
    dirty_batches = 0
    with torch.no_grad():
        for points, labels in tqdm(loader, desc=desc):
            points = points.to(device, non_blocking=True)
            model_input = points
            if dirty_generator is not None:
                active_box_counts = None
                if dirty_active_boxes is not None:
                    active_box_counts = torch.full(
                        (points.shape[0],),
                        int(dirty_active_boxes),
                        device=device,
                        dtype=torch.long,
                    )
                occl = dirty_generator(
                    points,
                    active_box_counts=active_box_counts,
                    generate_insertion=dirty_object_insertion,
                )
                model_input = apply_hard_drop_and_insert(
                    points=points,
                    hard_drop_mask=occl.hard_drop_mask,
                    inserted_points_xyz=occl.inserted_points if dirty_object_insertion else None,
                )
                dirty_occluded_acc += float(occl.hard_drop_mask.mean().item())
                dirty_active_box_acc += float(occl.active_box_counts.float().mean().item())
                dirty_batches += 1

            emb = model(model_input).detach().cpu().numpy().astype(np.float32)
            labels_np = labels.numpy().tolist()
            key_order.extend(int(x) for x in labels_np)
            if feats is None:
                feats = np.empty((len(dataset), emb.shape[1]), dtype=np.float32)
            start = len(key_order) - emb.shape[0]
            feats[start : start + emb.shape[0], :] = emb

    if feats is None:
        raise RuntimeError("No embedding extracted.")
    if dirty_generator is not None:
        dirty_stats = {
            "requested_active_boxes": None if dirty_active_boxes is None else int(dirty_active_boxes),
            "actual_active_boxes_mean": float(dirty_active_box_acc / max(dirty_batches, 1)),
            "actual_occluded_fraction_mean": float(dirty_occluded_acc / max(dirty_batches, 1)),
            "object_insertion": bool(dirty_object_insertion),
        }
    return feats, key_order, dirty_stats


def _get_true_neighbors(
    query_dataset: KITTIPicklePointCloudDataset,
    query_local_idx: int,
    db_local_key_to_idx: Dict[int, int],
    db_index: int,
) -> List[int]:
    q_rec = query_dataset.get_record(query_local_idx)

    if query_dataset.pickle_type == "evaluation":
        gt_keys = q_rec.get(db_index, [])
        return [db_local_key_to_idx[k] for k in gt_keys if k in db_local_key_to_idx]

    # Backward compatibility: self-retrieval query pickle with positives list/bitarray.
    pos = q_rec["positives"]
    if hasattr(pos, "search"):
        return [db_local_key_to_idx[k] for k in list(pos.search(1)) if k in db_local_key_to_idx]
    return [db_local_key_to_idx[k] for k in list(pos) if k in db_local_key_to_idx]


def compute_recall(
    db_features: np.ndarray,
    query_features: np.ndarray,
    query_dataset: KITTIPicklePointCloudDataset,
    db_dataset: KITTIPicklePointCloudDataset,
    query_key_order: List[int],
    db_key_order: List[int],
    topk: int,
    db_index: int,
    max_evals: int | None,
    min_time_gap: float,
    filter_same_frame: bool,
    require_db_earlier: bool,
) -> Dict[str, object]:
    num_db = db_features.shape[0]
    topk = min(topk, num_db)
    db_local_key_to_idx = {k: i for i, k in enumerate(db_key_order)}
    db_sq_norms = np.sum(db_features * db_features, axis=1)

    recall_count = np.zeros(topk, dtype=np.float64)
    top1_similarity: List[float] = []
    one_percent_retrieved = 0
    threshold = max(int(round(num_db / 100.0)), 1)
    num_evaluated = 0

    eval_count = len(query_key_order) if max_evals is None else min(len(query_key_order), int(max_evals))

    for q_local_idx in tqdm(range(eval_count), desc="Compute Recall"):
        true_neighbors = _get_true_neighbors(
            query_dataset=query_dataset,
            query_local_idx=q_local_idx,
            db_local_key_to_idx=db_local_key_to_idx,
            db_index=db_index,
        )
        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1
        q_feat = query_features[q_local_idx]
        q_sq_norm = float(np.dot(q_feat, q_feat))
        similarities = db_features @ q_feat
        distances = db_sq_norms - 2.0 * similarities + q_sq_norm
        indices = np.argsort(distances)

        q_time = query_dataset.get_time(q_local_idx)
        q_path = query_dataset.get_submap_path(q_local_idx)

        filtered = []
        for db_local_idx in indices.tolist():
            db_time = db_dataset.get_time(db_local_idx)
            db_path = db_dataset.get_submap_path(db_local_idx)

            if filter_same_frame and q_path == db_path:
                continue
            if require_db_earlier and not (q_time > db_time):
                continue
            if abs(q_time - db_time) <= min_time_gap:
                continue

            filtered.append(db_local_idx)
            if len(filtered) == topk:
                break

        pos_set = set(true_neighbors)
        for rank, db_local_idx in enumerate(filtered):
            if db_local_idx in pos_set:
                if rank == 0:
                    top1_similarity.append(float(similarities[db_local_idx]))
                recall_count[rank] += 1.0
                break

        if len(set(filtered[:threshold]).intersection(pos_set)) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        raise RuntimeError("No valid query for evaluation after GT lookup/filtering.")

    recall = np.cumsum(recall_count) / float(num_evaluated)
    one_percent_recall = one_percent_retrieved / float(num_evaluated)
    return {
        "num_db": num_db,
        "num_query": len(query_key_order),
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
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, num_points, use_intensity, emb_dim = load_descriptor_from_checkpoint(args.checkpoint, device)
    dirty_generator_ckpt = args.dirty_generator_checkpoint or args.checkpoint
    dirty_generator, dirty_saved_args = load_generator_from_checkpoint(
        dirty_generator_ckpt,
        device,
        point_weight_override=args.dirty_point_weight,
        geom_weight_override=args.dirty_geom_weight,
    )
    query_dirty_object_insertion = not args.no_query_object_insertion
    if not args.no_query_object_insertion:
        query_dirty_object_insertion = not bool(dirty_saved_args.get("no_object_insertion", False))
    db_dirty_object_insertion = not args.no_db_object_insertion
    if not args.no_db_object_insertion:
        db_dirty_object_insertion = not bool(dirty_saved_args.get("no_object_insertion", False))
    dirty_point_weight = float(
        dirty_saved_args.get("point_weight", 1.0) if args.dirty_point_weight is None else args.dirty_point_weight
    )
    dirty_geom_weight = float(
        dirty_saved_args.get("geom_weight", 2.0) if args.dirty_geom_weight is None else args.dirty_geom_weight
    )
    db_dirty_active_boxes = (
        int(args.db_dirty_active_boxes)
        if args.db_dirty_active_boxes is not None
        else args.query_dirty_active_boxes
    )
    print(
        f"[INFO] dirty_query=True dirty_db=True generator_ckpt={dirty_generator_ckpt} "
        f"query_active_boxes={args.query_dirty_active_boxes if args.query_dirty_active_boxes is not None else 'random'} "
        f"db_active_boxes={db_dirty_active_boxes if db_dirty_active_boxes is not None else 'random'} "
        f"query_object_insertion={query_dirty_object_insertion} "
        f"db_object_insertion={db_dirty_object_insertion} "
        f"point_weight={dirty_point_weight:.3f} geom_weight={dirty_geom_weight:.3f}"
    )

    query_dataset = KITTIPicklePointCloudDataset(
        pickle_path=args.query_file,
        split_index=args.query_index,
        kitti_root=args.kitti_root,
        fallback_root=args.fallback_root,
        num_points=num_points,
        use_intensity=use_intensity,
        random_sample=False,
        max_elems=args.max_query_elems,
    )
    db_dataset = KITTIPicklePointCloudDataset(
        pickle_path=args.db_file,
        split_index=args.db_index,
        kitti_root=args.kitti_root,
        fallback_root=args.fallback_root,
        num_points=num_points,
        use_intensity=use_intensity,
        random_sample=False,
        max_elems=args.max_db_elems,
    )
    print(f"[INFO] Query size: {len(query_dataset)}")
    print(f"[INFO] DB size: {len(db_dataset)}")

    query_features, query_key_order, dirty_stats = extract_embeddings(
        dataset=query_dataset,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        dirty_generator=dirty_generator,
        dirty_active_boxes=args.query_dirty_active_boxes,
        dirty_object_insertion=query_dirty_object_insertion,
        desc="Extract dirty query embeddings",
    )
    db_features, db_key_order, db_dirty_stats = extract_embeddings(
        dataset=db_dataset,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        dirty_generator=dirty_generator,
        dirty_active_boxes=db_dirty_active_boxes,
        dirty_object_insertion=db_dirty_object_insertion,
        desc="Extract dirty db embeddings",
    )
    print(f"[INFO] Query features: {query_features.shape}")
    print(f"[INFO] DB features: {db_features.shape}")
    if dirty_stats is not None:
        print(
            f"[INFO] dirty_query actual_active_boxes_mean={dirty_stats['actual_active_boxes_mean']:.4f} "
            f"actual_occluded_fraction_mean={dirty_stats['actual_occluded_fraction_mean']:.4f}"
        )
    if db_dirty_stats is not None:
        print(
            f"[INFO] dirty_db actual_active_boxes_mean={db_dirty_stats['actual_active_boxes_mean']:.4f} "
            f"actual_occluded_fraction_mean={db_dirty_stats['actual_occluded_fraction_mean']:.4f}"
        )

    result = compute_recall(
        db_features=db_features,
        query_features=query_features,
        query_dataset=query_dataset,
        db_dataset=db_dataset,
        query_key_order=query_key_order,
        db_key_order=db_key_order,
        topk=args.topk,
        db_index=args.db_index,
        max_evals=args.max_evals,
        min_time_gap=args.min_time_gap,
        filter_same_frame=args.filter_same_frame,
        require_db_earlier=args.require_db_earlier,
    )

    print(f"[RESULT] evaluated={result['num_evaluated']} query={result['num_query']} db={result['num_db']}")
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
            "db_file": args.db_file,
            "query_index": args.query_index,
            "db_index": args.db_index,
            "checkpoint": args.checkpoint,
            "dirty_generator_checkpoint": dirty_generator_ckpt,
            "dirty_point_weight": dirty_point_weight,
            "dirty_geom_weight": dirty_geom_weight,
            "num_points": num_points,
            "use_intensity": use_intensity,
            "emb_dim": emb_dim,
            "min_time_gap": args.min_time_gap,
            "filter_same_frame": args.filter_same_frame,
            "require_db_earlier": args.require_db_earlier,
            "dirty_query": True,
            "dirty_db": True,
            "dirty_query_stats": dirty_stats,
            "dirty_db_stats": db_dirty_stats,
            "result": result,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Saved result json: {out_path}")


if __name__ == "__main__":
    main()
