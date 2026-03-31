from __future__ import annotations

import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

try:
    from bitarray import bitarray
except ImportError:  # pragma: no cover
    bitarray = None


def _is_bitarray_obj(x: Any) -> bool:
    return bitarray is not None and isinstance(x, bitarray)


def _resolve_query_file_path(query_filepath: str, kitti_root: str) -> str:
    path = Path(query_filepath)
    if path.exists():
        return str(path)

    # Allow passing only file name, e.g. "kitti_vxp_training_queries_...pickle"
    candidate = Path(kitti_root) / path.name
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Cannot find query file: {query_filepath}")


def _candidate_paths_for_kitti(path_str: str, kitti_root: str, fallback_root: str) -> list[str]:
    path = str(path_str)
    candidates = [path]

    alias_roots = ["/TIEVNAS/KITTI", "/TIEVNAS/jinyuanfeng/KITTI", "/TIEVNAS/jyf/KITTI"]
    for alias in alias_roots:
        if alias in path:
            candidates.append(path.replace(alias, kitti_root))
            candidates.append(path.replace(alias, fallback_root))

    if not os.path.isabs(path):
        candidates.append(str(Path(kitti_root) / path))
        candidates.append(str(Path(fallback_root) / path))

    # keep order, remove duplicates
    uniq: list[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def resolve_kitti_path(path_str: str, kitti_root: str = "/TIEVNAS/jyf/KITTI", fallback_root: str = "/TIEVNAS/KITTI") -> str:
    for p in _candidate_paths_for_kitti(path_str, kitti_root=kitti_root, fallback_root=fallback_root):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot resolve path: {path_str}\n"
        f"tried kitti_root={kitti_root}, fallback_root={fallback_root}"
    )


def load_kitti_points(path: str, use_intensity: bool = False) -> np.ndarray:
    if path.endswith(".bin"):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.endswith(".npy"):
        pts = np.load(path)
        if pts.ndim != 2:
            raise ValueError(f"Unexpected point shape in {path}: {pts.shape}")
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
    else:
        raise ValueError(f"Unsupported pointcloud format: {path}")

    pts = pts.astype(np.float32, copy=False)
    if use_intensity:
        return pts[:, :4]
    return pts[:, :3]


def sample_or_pad_points(points: np.ndarray, num_points: int, random_sample: bool = True) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.zeros((num_points, points.shape[1]), dtype=np.float32)
    if n == num_points:
        return points
    if n > num_points:
        if random_sample:
            idx = np.random.choice(n, size=num_points, replace=False)
        else:
            idx = np.linspace(0, n - 1, num_points, dtype=np.int64)
        return points[idx]

    # n < num_points: repeat with replacement
    if random_sample:
        extra = np.random.choice(n, size=num_points - n, replace=True)
    else:
        extra = np.arange(num_points - n) % n
    idx = np.concatenate([np.arange(n), extra], axis=0)
    return points[idx]


class KITTIPointCloudQueryDataset(Dataset):
    """
    KITTI place-recognition dataset from query pickle.

    Default query files are expected under:
    - /TIEVNAS/jyf/KITTI
    Pointcloud files are auto-resolved with fallback to:
    - /TIEVNAS/KITTI
    """

    def __init__(
        self,
        query_filepath: str = "/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle",
        kitti_root: str = "/TIEVNAS/jyf/KITTI",
        fallback_root: str = "/TIEVNAS/KITTI",
        num_points: int = 4096,
        use_intensity: bool = False,
        random_sample: bool = True,
        transform=None,
        max_elems: Optional[int] = None,
        prefer_cached: bool = True,
    ) -> None:
        super().__init__()
        self.kitti_root = kitti_root
        self.fallback_root = fallback_root
        self.num_points = int(num_points)
        self.use_intensity = bool(use_intensity)
        self.random_sample = bool(random_sample)
        self.transform = transform

        resolved_query_file = _resolve_query_file_path(query_filepath=query_filepath, kitti_root=kitti_root)
        self.query_filepath = resolved_query_file
        self.queries = self._load_queries(self.query_filepath, prefer_cached=prefer_cached)

        self.keys = sorted(self.queries.keys())
        if max_elems is not None:
            self.keys = self.keys[: int(max_elems)]

    def _load_queries(self, query_filepath: str, prefer_cached: bool) -> Dict[int, Dict[str, Any]]:
        query_path = Path(query_filepath)
        cached_path = query_path.with_name(f"{query_path.stem}_cached{query_path.suffix}")

        use_path = query_path
        if prefer_cached and not query_path.stem.endswith("_cached") and cached_path.exists():
            use_path = cached_path

        with open(use_path, "rb") as f:
            queries = pickle.load(f)

        # Convert positives/negatives list -> bitarray and save cache.
        if bitarray is not None and len(queries) > 0:
            any_key = next(iter(queries.keys()))
            pos = queries[any_key].get("positives")
            if not _is_bitarray_obj(pos):
                qlen = len(queries)
                for k in queries:
                    pos_set = set(queries[k]["positives"])
                    neg_set = set(queries[k]["negatives"])
                    queries[k]["positives"] = bitarray([i in pos_set for i in range(qlen)])
                    queries[k]["negatives"] = bitarray([i in neg_set for i in range(qlen)])

                if not query_path.stem.endswith("_cached"):
                    try:
                        with open(cached_path, "wb") as f:
                            pickle.dump(queries, f)
                    except OSError:
                        pass
        return queries

    def __len__(self) -> int:
        return len(self.keys)

    def _get_point_path(self, rec: Dict[str, Any]) -> str:
        raw_path = rec.get("query_submap")
        if raw_path is None:
            raw_path = rec.get("submap_path")
        if raw_path is None:
            raw_path = rec.get("query")
        if raw_path is None:
            raise KeyError("No submap path field found in record. Expected one of query_submap/submap_path/query.")

        return resolve_kitti_path(
            path_str=raw_path,
            kitti_root=self.kitti_root,
            fallback_root=self.fallback_root,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        query_key = self.keys[idx]
        rec = self.queries[query_key]
        pcd_path = self._get_point_path(rec)

        points = load_kitti_points(pcd_path, use_intensity=self.use_intensity)
        points = sample_or_pad_points(points, num_points=self.num_points, random_sample=self.random_sample)

        if self.transform is not None:
            points = self.transform(points)

        if not torch.is_tensor(points):
            points = torch.from_numpy(np.asarray(points, dtype=np.float32))
        points = points.float()

        # label uses original query key (not local idx) to keep masks aligned with query dict.
        return points, torch.tensor(int(query_key), dtype=torch.long)

    def get_positives_ndx(self, query_key: int) -> list[int]:
        p = self.queries[int(query_key)]["positives"]
        if _is_bitarray_obj(p):
            return list(p.search(bitarray([True])))
        return list(p)

    def get_negatives_ndx(self, query_key: int) -> list[int]:
        n = self.queries[int(query_key)]["negatives"]
        if _is_bitarray_obj(n):
            return list(n.search(bitarray([True])))
        return list(n)

    def get_dataset_type(self) -> str:
        return "KITTIPointCloudQueryDataset"


def make_kitti_query_collate_fn(dataset: KITTIPointCloudQueryDataset):
    """
    Return:
      points: [B, N, C]
      labels: [B]
      positives_mask: [B, B]
      negatives_mask: [B, B]
    """

    def collate_fn(data_list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = default_collate([e[0] for e in data_list])
        labels = [int(e[1].item()) for e in data_list]
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        positives_mask = [[dataset.queries[q]["positives"][o] for o in labels] for q in labels]
        negatives_mask = [[dataset.queries[q]["negatives"][o] for o in labels] for q in labels]

        positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
        negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)
        return points, labels_tensor, positives_mask, negatives_mask

    return collate_fn


class KITTICsvPointCloudDataset(Dataset):
    """
    Simple frame-level KITTI dataset from csv (e.g. /TIEVNAS/jyf/KITTI/all_annotation.csv).
    """

    def __init__(
        self,
        csv_path: str = "/TIEVNAS/jyf/KITTI/all_annotation.csv",
        kitti_root: str = "/TIEVNAS/jyf/KITTI",
        fallback_root: str = "/TIEVNAS/KITTI",
        sequences: Optional[Iterable[int]] = None,
        num_points: int = 4096,
        use_intensity: bool = False,
        random_sample: bool = True,
        transform=None,
    ) -> None:
        super().__init__()
        import pandas as pd

        resolved_csv = _resolve_query_file_path(csv_path, kitti_root)
        self.df = pd.read_csv(resolved_csv)
        self.kitti_root = kitti_root
        self.fallback_root = fallback_root
        self.num_points = int(num_points)
        self.use_intensity = bool(use_intensity)
        self.random_sample = bool(random_sample)
        self.transform = transform

        if sequences is not None:
            seq_set = {f"{int(s):02d}" for s in sequences}
            seq_pat = re.compile(r"/sequences/(\d{2})/")
            keep = []
            for i, p in enumerate(self.df["submap_path"].tolist()):
                m = seq_pat.search(str(p))
                keep.append(m is not None and m.group(1) in seq_set)
            self.df = self.df[np.array(keep, dtype=bool)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        pcd_path = resolve_kitti_path(
            str(row["submap_path"]),
            kitti_root=self.kitti_root,
            fallback_root=self.fallback_root,
        )
        points = load_kitti_points(pcd_path, use_intensity=self.use_intensity)
        points = sample_or_pad_points(points, num_points=self.num_points, random_sample=self.random_sample)

        if self.transform is not None:
            points = self.transform(points)
        if not torch.is_tensor(points):
            points = torch.from_numpy(np.asarray(points, dtype=np.float32))
        points = points.float()
        label = torch.tensor(int(idx), dtype=torch.long)
        return points, label

    def get_dataset_type(self) -> str:
        return "KITTICsvPointCloudDataset"


def build_kitti_query_dataloader(
    query_filepath: str = "/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle",
    kitti_root: str = "/TIEVNAS/jyf/KITTI",
    fallback_root: str = "/TIEVNAS/KITTI",
    num_points: int = 4096,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    use_intensity: bool = False,
    random_sample: bool = True,
    transform=None,
    max_elems: Optional[int] = None,
    prefer_cached: bool = True,
) -> tuple[KITTIPointCloudQueryDataset, DataLoader]:
    dataset = KITTIPointCloudQueryDataset(
        query_filepath=query_filepath,
        kitti_root=kitti_root,
        fallback_root=fallback_root,
        num_points=num_points,
        use_intensity=use_intensity,
        random_sample=random_sample,
        transform=transform,
        max_elems=max_elems,
        prefer_cached=prefer_cached,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_kitti_query_collate_fn(dataset),
        drop_last=False,
    )
    return dataset, loader


if __name__ == "__main__":
    ds, dl = build_kitti_query_dataloader(
        query_filepath="/TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle",
        batch_size=4,
        num_workers=0,
        max_elems=64,
    )
    batch = next(iter(dl))
    points, labels, pos_mask, neg_mask = batch
    print(f"points={tuple(points.shape)} labels={tuple(labels.shape)}")
    print(f"positives_mask={tuple(pos_mask.shape)} negatives_mask={tuple(neg_mask.shape)}")
