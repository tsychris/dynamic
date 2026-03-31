from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate dirty-query / clean-db recall sweeps.")
    parser.add_argument(
        "--inference-script",
        type=str,
        default="/media/autolab/tsy/dynamic/inference_pnv_dirtyquery.py",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
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
    parser.add_argument("--adv-checkpoint", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument(
        "--dirty-generator-checkpoint",
        type=str,
        default=None,
        help="Generator checkpoint used to synthesize dirty queries for all compared models. Defaults to adv checkpoint.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated dirty drop ratios.",
    )
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-query-elems", type=int, default=None)
    parser.add_argument("--max-db-elems", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-time-gap", type=float, default=10.0)
    parser.add_argument("--require-db-earlier", action="store_true")
    parser.add_argument("--filter-same-frame", action="store_true")
    parser.add_argument("--no-dirty-object-insertion", action="store_true")
    parser.add_argument(
        "--dirty-point-weight",
        type=float,
        default=None,
        help="Override generator point-logit weight for all batch runs. Defaults to checkpoint setting.",
    )
    parser.add_argument(
        "--dirty-geom-weight",
        type=float,
        default=None,
        help="Override generator geometry weight for all batch runs. Defaults to checkpoint setting.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/media/autolab/tsy/dynamic/dirtyquery_sweeps",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kitti00_dirtyquery",
        help="Filename prefix for all output json/csv files.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_ratios(raw: str) -> list[float]:
    ratios: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        ratio = float(token)
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"Dirty ratio must be in (0, 1), got {ratio}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("No valid dirty ratios parsed.")
    return ratios


def ratio_tag(ratio: float) -> str:
    return f"p{int(round(ratio * 100.0)):02d}"


def build_command(
    args: argparse.Namespace,
    checkpoint: str,
    dirty_generator_checkpoint: str,
    dirty_ratio: float,
    save_json: Path,
) -> list[str]:
    cmd = [
        args.python_bin,
        args.inference_script,
        "--query-file",
        args.query_file,
        "--db-file",
        args.db_file,
        "--query-index",
        str(args.query_index),
        "--db-index",
        str(args.db_index),
        "--kitti-root",
        args.kitti_root,
        "--fallback-root",
        args.fallback_root,
        "--checkpoint",
        checkpoint,
        "--dirty-generator-checkpoint",
        dirty_generator_checkpoint,
        "--topk",
        str(args.topk),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--min-time-gap",
        str(args.min_time_gap),
        "--dirty-drop-ratio",
        str(dirty_ratio),
        "--save-json",
        str(save_json),
    ]
    if args.require_db_earlier:
        cmd.append("--require-db-earlier")
    if args.filter_same_frame:
        cmd.append("--filter-same-frame")
    if args.no_dirty_object_insertion:
        cmd.append("--no-dirty-object-insertion")
    if args.dirty_point_weight is not None:
        cmd.extend(["--dirty-point-weight", str(args.dirty_point_weight)])
    if args.dirty_geom_weight is not None:
        cmd.extend(["--dirty-geom-weight", str(args.dirty_geom_weight)])
    if args.max_query_elems is not None:
        cmd.extend(["--max-query-elems", str(args.max_query_elems)])
    if args.max_db_elems is not None:
        cmd.extend(["--max-db-elems", str(args.max_db_elems)])
    if args.max_evals is not None:
        cmd.extend(["--max-evals", str(args.max_evals)])
    return cmd


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dirty_generator_checkpoint = args.dirty_generator_checkpoint or args.adv_checkpoint
    model_specs = [
        ("adv", args.adv_checkpoint),
        ("baseline", args.baseline_checkpoint),
    ]

    summary_rows: list[dict[str, object]] = []

    for dirty_ratio in ratios:
        for model_name, checkpoint in model_specs:
            out_json = save_dir / f"{args.prefix}_{model_name}_{ratio_tag(dirty_ratio)}.json"
            if args.skip_existing and out_json.exists():
                print(f"[INFO] Skip existing: {out_json}")
            else:
                cmd = build_command(
                    args=args,
                    checkpoint=checkpoint,
                    dirty_generator_checkpoint=dirty_generator_checkpoint,
                    dirty_ratio=dirty_ratio,
                    save_json=out_json,
                )
                print("[RUN]", " ".join(cmd))
                subprocess.run(cmd, check=True)

            payload = load_result(out_json)
            result = payload["result"]
            dirty_stats = payload.get("dirty_query_stats", {})
            row = {
                "model": model_name,
                "checkpoint": checkpoint,
                "dirty_generator_checkpoint": dirty_generator_checkpoint,
                "dirty_drop_ratio": dirty_ratio,
                "dirty_point_weight": payload.get("dirty_point_weight"),
                "dirty_geom_weight": payload.get("dirty_geom_weight"),
                "actual_drop_ratio_mean": dirty_stats.get("actual_drop_ratio_mean"),
                "object_insertion": dirty_stats.get("object_insertion"),
                "num_evaluated": result.get("num_evaluated"),
                "recall_at_1": result.get("recall_at_1"),
                "recall_at_5": result.get("recall_at_5"),
                "recall_at_10": result.get("recall_at_10"),
                "one_percent_recall": result.get("one_percent_recall"),
                "top1_similarity_mean": result.get("top1_similarity_mean"),
                "json_path": str(out_json),
            }
            summary_rows.append(row)
            print(
                f"[SUMMARY] model={model_name} ratio={dirty_ratio:.2f} "
                f"R@1={100.0 * float(row['recall_at_1']):.2f}% "
                f"R@5={100.0 * float(row['recall_at_5']):.2f}% "
                f"R@10={100.0 * float(row['recall_at_10']):.2f}%"
            )

    summary_json = save_dir / f"{args.prefix}_summary.json"
    summary_csv = save_dir / f"{args.prefix}_summary.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "checkpoint",
                "dirty_generator_checkpoint",
                "dirty_drop_ratio",
                "dirty_point_weight",
                "dirty_geom_weight",
                "actual_drop_ratio_mean",
                "object_insertion",
                "num_evaluated",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "one_percent_recall",
                "top1_similarity_mean",
                "json_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Saved summary json: {summary_json}")
    print(f"[INFO] Saved summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
