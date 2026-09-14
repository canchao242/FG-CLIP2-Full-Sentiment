"""Predeclared two-seed replication, using existing train/validation caches only.

Run --preflight first, then --run. This is not an untouched-test experiment.
Existing training implementations and old No-cross results are immutable.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEEDS = (2026, 2027)
SCALES = {"full_v5": .5, "pooled_v5": .25}
OLD_SCALE = ROOT / "checkpoints/fgclip2_full_v5_scale_diagnostic/d9b6a2641cf42689/scale_summary.json"
OLD_V5 = ROOT / "checkpoints/fgclip2_full_v5_development/68dcdbd16a3a9bfe/protocol.json"
SOURCES = (
    "run_full_v5_replication.py", "train_full_model_v5.py", "full_model_v5.py",
    "run_full_model_v2.py", "full_model_v2.py", "full_model_v3.py", "full_model_v4.py",
    "build_contextual_text_cache.py", "full_model_v5_scaled.py",
    "train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus.py",
    "run_fgclip2_submission_cached_repeated.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_plan():
    old, original = read(OLD_SCALE), read(OLD_V5)
    if set(SEEDS) & set(old["seeds"]):
        raise ValueError("Replication seeds must be disjoint from scale-selection seeds")
    if any(old["selected"][variant]["alpha"] != scale for variant, scale in SCALES.items()):
        raise ValueError("Previously selected scale differs; do not retune the replication")
    protected = {str(OLD_SCALE): sha(OLD_SCALE), str(OLD_V5): sha(OLD_V5)}
    for reference in original["no_cross_references"]:
        for name, expected in reference["file_sha256"].items():
            path = Path(reference["directory"]) / name
            if sha(path) != expected:
                raise ValueError("An original No-cross artifact changed")
            protected[str(path)] = expected
    return dict(
        version="fixed_scale_v5_new_seed_replication_v1", seeds=list(SEEDS),
        fixed_scales=SCALES, development_selection_seeds=old["seeds"],
        validation_only=True, untouched_test=False, test_evaluated=False,
        scale_search_on_replication=False, feature_extraction=False,
        parent_training=dict(epochs=20, steps=1200, patience=4, base_lr=5e-5),
        control_refinement=dict(epochs=5, steps=1200, patience=4, base_lr=1e-5,
                                scheduler="ReduceLROnPlateau(factor=0.5, patience=2)"),
        v5_refinement=dict(epochs=5, freeze_epochs=3, steps=1200, base_lr=1e-5,
                           branch_lr=3e-4, joint_branch_lr=1e-4, branch_loss_weight=.1),
        checkpoint_selection="Same original v5 alpha=1 validation selection, including epoch zero; apply locked scale afterward.",
        interpretation="Fresh training RNG only, same development validation set. No independent-test or significance claim.",
        comparison_limit="Same maximum refinement budget; the original No-cross schedule and early stopping are retained, not identical actual updates.",
        source_sha256={name: sha(ROOT / name) for name in SOURCES}, protected_sha256=protected,
    )


def find_protocol(root):
    paths = list(Path(root).glob("*/protocol.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one protocol under {root}, got {len(paths)}")
    return paths[0].parent


def command(stage, root, parents=None, controls=None):
    common = ["--seeds", *map(str, SEEDS), "--steps-per-epoch", "1200", "--eval-batch-size", "64", "--device", "cuda"]
    if stage == "parents":
        return [sys.executable, "-B", "-u", str(ROOT / "run_full_model_v2.py"),
                "--variants", "no_cross_control", "--epochs", "20", "--patience", "4",
                "--cross-dropout", "0.1", "--save-root", str(root / stage), *common]
    if stage == "controls":
        if parents is None:
            raise ValueError("Missing paired parent root")
        return [sys.executable, "-B", "-u", str(ROOT / "run_full_model_v2.py"),
                "--variants", "no_cross_control", "--epochs", "5", "--patience", "4",
                "--cross-dropout", "0", "--base-lr", "0.00001", "--cross-lr", "0.0001",
                "--warm-start-root", str(parents), "--save-root", str(root / stage), *common]
    if stage != "v5" or parents is None or controls is None:
        raise ValueError("Unknown stage or missing paired roots")
    return [sys.executable, "-B", "-u", str(ROOT / "train_full_model_v5.py"),
            "--variants", "pooled_v5", "full_v5", "--epochs", "5", "--freeze-epochs", "3",
            "--base-lr", "0.00001", "--branch-lr", "0.0003", "--joint-branch-lr", "0.0001",
            "--branch-loss-weight", "0.1", "--warm-start-root", str(parents),
            "--reference-root", str(controls), "--save-root", str(root / stage), *common]


def verify_protected(plan):
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Training source changed: {name}")
    for name, expected in plan["protected_sha256"].items():
        if sha(name) != expected:
            raise ValueError(f"Old result changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    identifier = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    root = ROOT / "checkpoints/fgclip2_full_v5_seed_replication" / identifier
    print(f"[REPLICATION ROOT] {root}", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUTF8": "1"}
    if args.preflight:
        cmd = command("parents", root) + ["--preflight"]
        print(subprocess.list2cmdline(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        print("[PREFLIGHT OK] No outputs written; later stages require the new paired parents.", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        plan_path = root / "replication_plan.json"
        if plan_path.exists() and read(plan_path) != plan:
            raise ValueError("Refusing mixed replication plans")
        write(plan_path, plan)
        snapshot = root / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name, expected in plan["source_sha256"].items():
            target = snapshot / name
            if not target.exists():
                shutil.copyfile(ROOT / name, target)
            if sha(target) != expected:
                raise ValueError("Snapshot mismatch")
        parents = controls = None
        for stage in ("parents", "controls", "v5"):
            verify_protected(plan)
            cmd = command(stage, root, parents, controls)
            print(f"[START {stage}] {subprocess.list2cmdline(cmd)}", flush=True)
            with (root / f"{stage}.log").open("a", encoding="utf-8") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = find_protocol(root / stage)
            print(f"[DONE {stage}] {directory}", flush=True)
            write(root / f"{stage}_completed.json", dict(root=str(directory), seeds=list(SEEDS)))
            if stage == "parents":
                parents = directory
            elif stage == "controls":
                controls = directory
        verify_protected(plan)
        print("[TRAINING COMPLETE] Evaluate locked scales only; no new grid search.", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
