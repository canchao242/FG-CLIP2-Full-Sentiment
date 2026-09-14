"""Historical grouped benchmark, not fresh confirmation of Full-v5.

Five architectures x three paired random seeds. Validation selects checkpoints.
These test partitions were subsequently inspected during model development;
they must not be described as newly untouched. Results retain all seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus as core


ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE_ROOT = ROOT / "checkpoints" / "fgclip2_submission_v1_core_repeated"
DEFAULT_SPLIT_ROOT = ROOT / "data_submission_v1"
DEFAULT_SEEDS = [42, 123, 3407]
DEFAULT_EXPERIMENTS = ["global_only", "full", "no_token", "no_region", "no_cross"]

SUMMARY_METRICS = [
    "test_balanced_macro_f1",
    "test_balanced_acc",
    "test_zh_macro_f1",
    "test_en_macro_f1",
    "test_zh_acc",
    "test_en_acc",
    "best_balanced_macro_f1",
    "best_balanced_acc",
    "classifier_trainable_params",
    "end_to_end_total_params",
    "training_time_seconds",
]


def comma_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Supply at least one comma-separated value.")
    return values


def seed_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(values) < 2 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Supply at least two unique comma-separated seeds.")
    return values


def make_core_args(
    seed: int,
    seed_root: Path,
    split_root: Path,
    epochs: int | None,
    steps_per_epoch: int | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_feature_source="embedding",
        save_root=str(seed_root),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        zh_train_csv=str(split_root / "zh" / "train.csv"),
        zh_val_csv=str(split_root / "zh" / "val.csv"),
        zh_test_csv=str(split_root / "zh" / "test.csv"),
        en_train_csv=str(split_root / "en" / "train.csv"),
        en_val_csv=str(split_root / "en" / "val.csv"),
        en_test_csv=str(split_root / "en" / "test.csv"),
    )


def result_path(save_root: Path, seed: int, experiment: str) -> Path:
    return save_root / f"seed_{seed}" / experiment / "run_result.json"


def run_one(
    experiment: str,
    seed: int,
    save_root: Path,
    split_root: Path,
    epochs: int | None,
    steps_per_epoch: int | None,
    rerun: bool,
) -> dict:
    destination = result_path(save_root, seed, experiment)
    if destination.exists() and not rerun:
        print(f"resume: {destination}")
        return json.loads(destination.read_text(encoding="utf-8"))

    seed_root = save_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    arguments = make_core_args(seed, seed_root, split_root, epochs, steps_per_epoch)
    wall_start = time.perf_counter()
    result = core.run_experiment(experiment, arguments)
    result.update(
        {
            "seed": seed,
            "paired_seed": True,
            "split_version": "submission_v1",
            "run_wall_time_seconds": time.perf_counter() - wall_start,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def summarize(results: list[dict], save_root: Path) -> None:
    runs = pd.DataFrame(results).sort_values(["experiment", "seed"]).reset_index(drop=True)
    runs_path = save_root / "core_ablation_repeated_runs.csv"
    runs.to_csv(runs_path, index=False, encoding="utf-8-sig")

    rows = []
    structured: dict[str, object] = {
        "split_version": "submission_v1",
        "experiments": {},
    }
    for experiment, subset in runs.groupby("experiment", sort=False):
        experiment_json: dict[str, object] = {
            "seeds": [int(value) for value in subset["seed"]],
            "metrics": {},
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "mean_plus_minus_std": f"{mean:.4f} +/- {std:.4f}",
                    "n": len(values),
                }
            )
            experiment_json["metrics"][metric] = {
                "mean": mean,
                "std": std,
                "values": [float(value) for value in values],
            }
        structured["experiments"][experiment] = experiment_json

    paired = runs.pivot(index="seed", columns="experiment", values="test_balanced_macro_f1")
    if {"full", "no_cross"}.issubset(paired.columns):
        paired = paired.dropna(subset=["full", "no_cross"])
        delta = paired["no_cross"] - paired["full"]
        statistic, p_value = ttest_rel(paired["no_cross"], paired["full"])
        structured["paired_no_cross_minus_full"] = {
            "n": int(len(delta)),
            "values": [float(value) for value in delta],
            "mean": float(delta.mean()),
            "sample_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "paired_t_statistic": float(statistic),
            "two_sided_p_value": float(p_value),
            "interpretation_note": "With only three paired seeds, treat this test as descriptive, not conclusive.",
        }

    summary = pd.DataFrame(rows)
    summary.to_csv(save_root / "core_ablation_mean_std.csv", index=False, encoding="utf-8-sig")
    (save_root / "core_ablation_mean_std.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"runs: {runs_path}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=comma_list, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--seeds", type=seed_list, default=DEFAULT_SEEDS)
    parser.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    invalid = sorted(set(args.experiments) - set(core.ABLATION_EXPERIMENTS))
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}")
    required = [
        args.split_root / language / f"{split}.csv"
        for language in ("zh", "en")
        for split in ("train", "val", "test")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run create_leakage_safe_splits.py first. Missing: {missing}")

    args.save_root.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(args.experiments) * len(args.seeds)
    run_number = 0
    for experiment in args.experiments:
        for seed in args.seeds:
            run_number += 1
            print("\n" + "#" * 100)
            print(f"SUBMISSION CORE RUN {run_number}/{total} | experiment={experiment} | seed={seed}")
            print("#" * 100)
            results.append(
                run_one(
                    experiment=experiment,
                    seed=seed,
                    save_root=args.save_root,
                    split_root=args.split_root,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    rerun=args.rerun,
                )
            )
            summarize(results, args.save_root)


if __name__ == "__main__":
    main()
