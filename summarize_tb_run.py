from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util


DEFAULT_TAG_GROUPS = {
    "generator_core": [
        "train_epoch/loss_g",
        "train_epoch/loss_place_adv",
        "train_epoch/occluded_fraction",
        "train_epoch/active_num_boxes",
        "train_epoch/reg_size_prior",
        "train_epoch/reg_height_prior",
        "train_epoch/reg_range_prior",
    ],
    "descriptor_core": [
        "train_epoch/loss_f",
        "train_epoch/loss_place_clean",
        "train_epoch/loss_consistency",
    ],
    "box_means": [
        "train_epoch/box_center_x_mean",
        "train_epoch/box_center_y_mean",
        "train_epoch/box_center_z_mean",
        "train_epoch/box_size_l_mean",
        "train_epoch/box_size_w_mean",
        "train_epoch/box_size_h_mean",
        "train_epoch/box_yaw_mean",
    ],
    "generator_iter": [
        "train_iter/loss_g",
        "train_iter/loss_place_adv",
        "train_iter/occluded_fraction",
        "train_iter/active_num_boxes",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scalar trends from a TensorBoard run.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="TensorBoard run directory containing events.out.tfevents.*",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.3,
        help="Fraction of the series tail used for slope/trend estimation.",
    )
    parser.add_argument(
        "--iter-tail-points",
        type=int,
        default=200,
        help="Maximum number of iter-level points to use for tail trend estimation.",
    )
    parser.add_argument(
        "--show-tags",
        type=str,
        default=None,
        help="Optional comma-separated explicit tags to summarize. Defaults to built-in generator-focused groups.",
    )
    return parser.parse_args()


def resolve_event_file(run_dir: Path) -> Path:
    if run_dir.is_file():
        return run_dir
    candidates = sorted(run_dir.glob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No event file found under: {run_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_scalar(value) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        arr = tensor_util.make_ndarray(value.tensor)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    return None


def load_selected_scalars(event_file: Path, tags: set[str]) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {tag: [] for tag in tags}
    loader = event_file_loader.EventFileLoader(str(event_file))
    for event in loader.Load():
        if not event.summary.value:
            continue
        step = int(event.step)
        for value in event.summary.value:
            tag = value.tag
            if tag not in series:
                continue
            scalar = extract_scalar(value)
            if scalar is None or not np.isfinite(scalar):
                continue
            series[tag].append((step, scalar))
    return {tag: vals for tag, vals in series.items() if vals}


def calc_trend(points: List[Tuple[int, float]], tail_frac: float, max_tail_points: int | None = None) -> dict:
    steps = np.asarray([p[0] for p in points], dtype=np.float64)
    values = np.asarray([p[1] for p in points], dtype=np.float64)
    n = len(values)
    tail_n = max(int(round(n * tail_frac)), 3)
    tail_n = min(tail_n, n)
    if max_tail_points is not None:
        tail_n = min(tail_n, max_tail_points)
    tail_steps = steps[-tail_n:]
    tail_vals = values[-tail_n:]

    start = float(values[0])
    end = float(values[-1])
    delta = end - start
    rel_delta = delta / max(abs(start), 1e-8)

    if tail_n >= 2 and np.any(tail_steps != tail_steps[0]):
        slope = float(np.polyfit(tail_steps, tail_vals, 1)[0])
    else:
        slope = 0.0

    tail_delta = float(tail_vals[-1] - tail_vals[0]) if tail_n >= 2 else 0.0
    tail_mean = float(np.mean(tail_vals))
    tail_std = float(np.std(tail_vals))
    coeff_var = tail_std / max(abs(tail_mean), 1e-8)

    if abs(tail_delta) <= max(1e-6, 0.02 * max(abs(tail_mean), 1e-8)):
        trend = "flat"
    elif tail_delta > 0:
        trend = "up"
    else:
        trend = "down"

    stability = "stable" if coeff_var < 0.1 else ("moderate" if coeff_var < 0.3 else "noisy")
    return {
        "count": n,
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "start": start,
        "end": end,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "delta": delta,
        "rel_delta": float(rel_delta),
        "tail_n": int(tail_n),
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_delta": tail_delta,
        "tail_slope": slope,
        "trend": trend,
        "stability": stability,
    }


def format_metric_line(tag: str, stats: dict) -> str:
    return (
        f"{tag}\n"
        f"  steps={stats['start_step']}->{stats['end_step']} count={stats['count']}\n"
        f"  start={stats['start']:.6f} end={stats['end']:.6f} "
        f"delta={stats['delta']:+.6f} rel={100.0 * stats['rel_delta']:+.2f}%\n"
        f"  min={stats['min']:.6f} max={stats['max']:.6f}\n"
        f"  tail_mean={stats['tail_mean']:.6f} tail_std={stats['tail_std']:.6f} "
        f"tail_delta={stats['tail_delta']:+.6f} trend={stats['trend']} stability={stats['stability']}"
    )


def print_group(title: str, tags: Iterable[str], summaries: Dict[str, dict]) -> None:
    printed = False
    for tag in tags:
        if tag not in summaries:
            continue
        if not printed:
            print(f"\n[{title}]", flush=True)
            printed = True
        print(format_metric_line(tag, summaries[tag]), flush=True)


def verdict_lines(summaries: Dict[str, dict]) -> List[str]:
    lines: List[str] = []

    occ = summaries.get("train_epoch/occluded_fraction")
    adv = summaries.get("train_epoch/loss_place_adv")
    clean = summaries.get("train_epoch/loss_place_clean")
    boxes = summaries.get("train_epoch/active_num_boxes")
    reg_h = summaries.get("train_epoch/reg_height_prior")
    reg_r = summaries.get("train_epoch/reg_range_prior")
    reg_s = summaries.get("train_epoch/reg_size_prior")
    cz = summaries.get("train_epoch/box_center_z_mean")
    sh = summaries.get("train_epoch/box_size_h_mean")

    if occ is not None:
        lines.append(
            f"Occlusion strength: end={occ['end']:.4f}, trend={occ['trend']}, stability={occ['stability']}."
        )
    if boxes is not None:
        lines.append(
            f"Active boxes: end={boxes['end']:.4f}, trend={boxes['trend']}, stability={boxes['stability']}."
        )
    if adv is not None and clean is not None:
        gap = adv["end"] - clean["end"]
        if gap > 0:
            lines.append(
                f"Adversarial difficulty: adv clean-gap at end = {gap:+.4f}. Generator is still making harder samples than clean."
            )
        elif gap < 0:
            lines.append(
                f"Adversarial difficulty: adv clean-gap at end = {gap:+.4f}. Generator is not making samples harder than clean."
            )
        else:
            lines.append("Adversarial difficulty: adv and clean losses are nearly equal at the end.")

    reg_bits = []
    for name, obj in [
        ("size", reg_s),
        ("height", reg_h),
        ("range", reg_r),
    ]:
        if obj is not None:
            reg_bits.append(f"{name}={obj['end']:.4f}({obj['trend']})")
    if reg_bits:
        lines.append("Regularization: " + ", ".join(reg_bits) + ".")

    if cz is not None and sh is not None:
        bottom = cz["end"] - 0.5 * sh["end"]
        lines.append(
            f"Box geometry at end: center_z={cz['end']:.4f}, size_h={sh['end']:.4f}, implied bottom_z={bottom:.4f}."
        )

    return lines


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    event_file = resolve_event_file(run_dir)

    if args.show_tags is None:
        tags: List[str] = []
        for group_tags in DEFAULT_TAG_GROUPS.values():
            tags.extend(group_tags)
    else:
        tags = [tag.strip() for tag in args.show_tags.split(",") if tag.strip()]
    tag_set = set(tags)

    print(f"[INFO] event_file={event_file}", flush=True)
    print(f"[INFO] tail_frac={args.tail_frac}", flush=True)

    series = load_selected_scalars(event_file=event_file, tags=tag_set)
    if not series:
        raise RuntimeError("No selected scalar tags were found in the event file.")

    summaries: Dict[str, dict] = {}
    for tag, points in series.items():
        max_tail_points = args.iter_tail_points if tag.startswith("train_iter/") else None
        summaries[tag] = calc_trend(points, tail_frac=args.tail_frac, max_tail_points=max_tail_points)

    for group_name, group_tags in DEFAULT_TAG_GROUPS.items():
        print_group(group_name, group_tags, summaries)

    extra_tags = [tag for tag in tags if all(tag not in group for group in DEFAULT_TAG_GROUPS.values())]
    if extra_tags:
        print_group("extra", extra_tags, summaries)

    print("\n[Verdict]", flush=True)
    for line in verdict_lines(summaries):
        print(f"- {line}", flush=True)


if __name__ == "__main__":
    main()
