"""Fixed-budget v5 development, reusing existing train/validation caches only.

Original No-cross checkpoints and v2/v3/v4 source files are never modified.
The primary No-cross comparator is the already completed v4 continued-training
control, not a weakened or newly chosen checkpoint. --preflight writes nothing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, WeightedRandomSampler

import run_full_model_v2 as shared
from full_model_v5 import StagedResidualFullClassifier
from build_contextual_text_cache import verify_context_cache

core, cached = shared.core, shared.cached
ROOT = Path(__file__).resolve().parent
VARIANTS = ("pooled_v5", "full_v5")


def base_state_digest(state):
    value = hashlib.sha256()
    for name in sorted(state):
        if not name.startswith("cross_attention."):
            tensor = state[name].detach().cpu().contiguous()
            value.update(name.encode())
            value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def build_model(variant, dims):
    if variant not in VARIANTS:
        raise ValueError("Only the explicitly named v5 development arms are supported")
    return StagedResidualFullClassifier(
        use_attention=variant == "full_v5", cross_dropout=0.,
        **dims, proj_dim=core.PROJ_DIM, hidden_dim=core.HIDDEN_DIM,
        num_classes=core.NUM_CLASSES, num_heads=core.NUM_HEADS, dropout=core.DROPOUT)


def make_optimizer(model, args):
    # Include frozen weights so the optimizer state survives the stage transition.
    groups = {"base": [], "cross": []}
    for name, parameter in model.named_parameters():
        groups["cross" if name.startswith("cross_attention.") else "base"].append(parameter)
    return torch.optim.AdamW([
        {"params": groups["base"], "name": "base", "lr": args.base_lr},
        {"params": groups["cross"], "name": "cross", "lr": args.branch_lr},
    ], weight_decay=core.WEIGHT_DECAY)


def configure_stage(model, optimizer, epoch, args):
    frozen = epoch <= args.freeze_epochs
    model.set_base_frozen(frozen)
    for group in optimizer.param_groups:
        group["lr"] = args.base_lr if group["name"] == "base" else (
            args.branch_lr if frozen else args.joint_branch_lr)
    return "branch_only" if frozen else "joint"


def training_loss(logits, aux, labels, criterion, epoch, branch_weight):
    final = criterion(logits.float(), labels)
    content = aux["contextual_token_weights"].sum(-1) > 0
    # No training signal is invented for an empty-content branch.
    if content.any():
        branch = F.cross_entropy(aux["correction_logits"][content].float(), labels[content],
                                 weight=criterion.class_weights, label_smoothing=core.LABEL_SMOOTHING)
    else:
        branch = aux["correction_logits"].sum().float() * 0.
    base_aux = core.compute_aux_loss(aux) if not aux["base_frozen"] else final.new_zeros(())
    aux_weight = core.AUX_LOSS_WEIGHT_MAX * min(1., epoch / max(1, core.AUX_WARMUP_EPOCHS))
    return final + branch_weight * branch + aux_weight * base_aux, final, branch


def train_epoch(model, batches, optimizer, scaler, criterion, epoch, args):
    model.train()
    losses, final_losses, branch_losses, deltas, norms = [], [], [], [], []
    skips, nonfinite = 0, 0
    for batch in batches:
        batch = tuple(t.float() if core.DEVICE == "cpu" and t.is_floating_point() else t for t in batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=core.DEVICE == "cuda", dtype=torch.float16):
            logits, aux, labels, _ = cached.forward_cached(model, batch)
            loss, final, branch = training_loss(logits, aux, labels, criterion, epoch, args.branch_loss_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
        if not scaler.is_enabled() and not torch.isfinite(norm):
            raise FloatingPointError("Non-finite FP32 gradients")
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skips += int(scaler.get_scale() < before)
        if torch.isfinite(norm):
            norms.append(float(norm))
        else:
            nonfinite += 1
        losses.append(float(loss.detach()))
        final_losses.append(float(final.detach()))
        branch_losses.append(float(branch.detach()))
        deltas.append(float(aux["correction_logits"].detach().abs().float().mean()))
    return {"loss": float(np.mean(losses)), "final_loss": float(np.mean(final_losses)),
            "branch_loss": float(np.mean(branch_losses)), "correction_abs_mean": float(np.mean(deltas)),
            "finite_grad_norm_mean": float(np.mean(norms)) if norms else None,
            "amp_skipped_steps": skips, "nonfinite_grad_norm_steps": nonfinite}


def run_one(args, meta, masks, protocol, run_root, variant, seed):
    out = run_root / f"seed_{seed}" / variant
    out.mkdir(parents=True, exist_ok=True)
    config = {"protocol": protocol, "variant": variant, "seed": seed}
    fingerprint = shared.digest(config)
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Refusing mixed configurations")
    elif any(out.iterdir()):
        raise ValueError("Nonempty output without config")
    else:
        shared.write_json(config_path, config)
    if (out / "run_result.json").exists():
        result = json.loads((out / "run_result.json").read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint or not (out / "best_head.ckpt").exists():
            raise ValueError("Invalid completed result")
        print(f"[COMPLETE; SKIP] {seed}/{variant}", flush=True)
        return result
    core.set_seed(seed)
    datasets = {name: shared.ContentDataset(args.cache_root / name, masks[name], args.limit_samples,
                                          contextual_root=args.contextual_cache_root) for name in shared.SPLITS}
    sampler_rng = torch.Generator().manual_seed(seed)
    loader_rng = torch.Generator().manual_seed(seed + 100000)
    weights = core.build_joint_sample_weights(datasets["zh_train"], datasets["en_train"],
        target_en_frac=core.TARGET_EN_FRAC, en_class1_boost=core.EN_CLASS1_BOOST, zh_class2_boost=core.ZH_CLASS2_BOOST)
    sampler = WeightedRandomSampler(weights, args.steps_per_epoch * core.BATCH_SIZE, True, sampler_rng)
    train_loader = shared.loader(ConcatDataset([datasets["zh_train"], datasets["en_train"]]), core.BATCH_SIZE, sampler, loader_rng)
    val_loaders = {name: shared.loader(datasets[name], args.eval_batch_size, generator=loader_rng) for name in ("zh_val", "en_val")}
    model = build_model(variant, meta["dims"]).to(core.DEVICE)
    parent = protocol["warm_start_checkpoints"][str(seed)]
    source = Path(parent["path"])
    if cached.file_sha256(source) != parent["sha256"]:
        raise ValueError("Warm-start checkpoint changed")
    start_protocol = protocol
    if args.limit_samples:
        # Smoke truncation is not a new scientific split. Still validate the
        # complete cache/mask identities, but permit its full-data warm start.
        saved = torch.load(source, map_location="cpu", weights_only=False)
        parent_limit = saved["config"]["protocol"].get("limit_samples")
        if parent_limit not in (0, args.limit_samples):
            raise ValueError("Incompatible smoke warm-start sample limit")
        start_protocol = {**protocol, "limit_samples": parent_limit}
        del saved
    shared.load_shared_start(model, source, seed, start_protocol)
    class_weights, _ = core.get_global_class_weights(datasets["zh_train"], datasets["en_train"])
    criterion = core.SmoothedFocalLoss(class_weights.pow(core.LOSS_CLASS_WEIGHT_POWER).to(core.DEVICE),
                                     core.LABEL_SMOOTHING, core.FOCAL_GAMMA)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=core.DEVICE == "cuda")
    start, history, elapsed, best = 0, [], 0., -1.
    last = out / "last_state.ckpt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, history, elapsed, best = state["epoch"] + 1, state["history"], state["elapsed"], state["best"]
        shared.restore_rng(state["rng"], sampler_rng, loader_rng)
        print(f"[RESUME] {seed}/{variant} epoch={start}", flush=True)
    for epoch in range(start, args.epochs + 1):
        started = time.perf_counter()
        stage = configure_stage(model, optimizer, epoch, args)
        train = None if epoch == 0 else train_epoch(model, train_loader, optimizer, scaler, criterion, epoch, args)
        zh, zh_out = shared.evaluate(model, val_loaders["zh_val"], criterion, "ZH VAL")
        en, en_out = shared.evaluate(model, val_loaders["en_val"], criterion, "EN VAL")
        score = (zh["macro_f1"] + en["macro_f1"]) / 2
        if not all(np.isfinite(x) for x in (score, zh["loss"], en["loss"])):
            raise FloatingPointError("Non-finite validation result")
        row = {"epoch": epoch, "stage": "initial" if epoch == 0 else stage,
               "val_lb_mf1": score, "zh": zh, "en": en, "train": train,
               "lrs": {g["name"]: g["lr"] for g in optimizer.param_groups},
               "correction_output_weight_norm": float(model.cross_attention.classifier[-1].weight.detach().norm()),
               "base_state_sha256": base_state_digest(model.state_dict())}
        if epoch > 0 and stage == "branch_only" and row["base_state_sha256"] != history[0]["base_state_sha256"]:
            raise RuntimeError("Frozen base weights changed; stopping this experiment")
        history.append(row)
        if score > best:
            best = score
            shared.save_torch(out / "best_head.ckpt", {"clf_state_dict": model.state_dict(),
                              "config": config, "fingerprint": fingerprint, "metrics": row})
            core.save_validation_outputs(zh_out, "zh_val", str(out / "best_zh_val_outputs.npz"))
            core.save_validation_outputs(en_out, "en_val", str(out / "best_en_val_outputs.npz"))
        elapsed += time.perf_counter() - started
        shared.save_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": shared.rng_state(sampler_rng, loader_rng),
            "epoch": epoch, "history": history, "elapsed": elapsed, "best": best, "fingerprint": fingerprint})
        shared.write_json(out / "history.json", history)
        print(f"[{seed}/{variant}] epoch={epoch} {stage} LB-MF1={score:.5f} best={best:.5f}", flush=True)
    selected = max(history, key=lambda r: r["val_lb_mf1"])
    result = {"seed": seed, "variant": variant, "fingerprint": fingerprint,
              "validation_only": True, "smoke_only": bool(args.limit_samples),
              "selected_initial_without_refinement": selected["epoch"] == 0,
              "best_val_lb_mf1": best, "selected": selected, "training_seconds": elapsed,
              "head_parameters": sum(p.numel() for p in model.parameters()),
              "branch_parameters": sum(p.numel() for p in model.cross_attention.parameters()), "save_dir": str(out)}
    shared.write_json(out / "run_result.json", result)
    del model, optimizer, scaler, datasets, train_loader, val_loaders
    core.cleanup_memory()
    return result


def summarize(results, references):
    rows = list(results) + [dict(seed=r["seed"], variant="no_cross_control_reused",
                               best_val_lb_mf1=r["score"]) for r in references]
    summary = shared.summarize(rows)
    lookup = {(r["variant"], r["seed"]): r["best_val_lb_mf1"] for r in rows}
    summary["paired"] = {}
    for a, b in (("full_v5", "no_cross_control_reused"), ("pooled_v5", "no_cross_control_reused"), ("full_v5", "pooled_v5")):
        seeds = sorted(s for variant, s in lookup if variant == a and (b, s) in lookup)
        values = [lookup[(a, s)] - lookup[(b, s)] for s in seeds]
        if values:
            summary["paired"][f"{a}_minus_{b}"] = {"seeds": seeds, "values": values,
                "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return summary


def check_reference(root, protocol, seeds):
    original = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    # A different dataset, mask, warm start, update budget or base LR is NOT paired.
    for key in ("core_settings", "cache_metadata_sha256", "content_mask_sha256",
                "epochs", "steps_per_epoch", "base_lr", "limit_samples", "torch_version", "device"):
        if original.get(key) != protocol.get(key):
            raise ValueError(f"No-cross comparator protocol mismatch: {key}")
    for name in set(original["source_sha256"]) & set(protocol["source_sha256"]):
        if original["source_sha256"][name] != protocol["source_sha256"][name]:
            raise ValueError(f"Changed shared source underlying No-cross comparator: {name}")
    references = []
    for seed in seeds:
        if original["warm_start_checkpoints"].get(str(seed)) != protocol["warm_start_checkpoints"][str(seed)]:
            raise ValueError("Different paired parent checkpoint")
        directory = root / f"seed_{seed}/no_cross_control"
        result = json.loads((directory / "run_result.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["protocol"] != original or config["seed"] != seed or config["variant"] != "no_cross_control":
            raise ValueError("Invalid No-cross reference config")
        if result["fingerprint"] != shared.digest(config) or not result["validation_only"] or result["smoke_only"]:
            raise ValueError("Invalid No-cross reference result")
        files = ("config.json", "run_result.json", "best_head.ckpt", "best_zh_val_outputs.npz", "best_en_val_outputs.npz")
        references.append({"seed": seed, "directory": str(directory.resolve()), "score": result["best_val_lb_mf1"],
                           "selected_epoch": result["selected"]["epoch"],
                           "file_sha256": {name: cached.file_sha256(directory / name) for name in files}})
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "checkpoints/fgclip2_submission_v1_feature_cache")
    parser.add_argument("--contextual-cache-root", type=Path, default=ROOT / "checkpoints/fgclip2_contextual_text_v1")
    parser.add_argument("--split-root", type=Path, default=ROOT / "data_submission_v1")
    parser.add_argument("--save-root", type=Path, default=ROOT / "checkpoints/fgclip2_full_v5_development")
    parser.add_argument("--warm-start-root", type=Path, default=ROOT / "checkpoints/fgclip2_full_v2_development/2999018eb95e051a")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "checkpoints/fgclip2_full_v4_development/a04093b760cbe92b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 3407])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--branch-lr", type=float, default=3e-4)
    parser.add_argument("--joint-branch-lr", type=float, default=1e-4)
    parser.add_argument("--branch-loss-weight", type=float, default=.1)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--limit-samples", type=int, default=0, help="Smoke only; no comparison with formal controls")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 0 < args.freeze_epochs < args.epochs or min(args.steps_per_epoch, args.eval_batch_size) < 1 or args.limit_samples < 0:
        parser.error("Require 0 < freeze epochs < total epochs, positive steps/batch size, nonnegative sample limit")
    if min(args.base_lr, args.branch_lr, args.joint_branch_lr, args.grad_clip) <= 0 or args.branch_loss_weight < 0:
        parser.error("Invalid optimization settings")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate seeds or variants")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Use the CUDA-enabled peixun environment")
    for protected in (args.cache_root, args.contextual_cache_root, args.reference_root, args.warm_start_root, args.split_root):
        if args.save_root.resolve().is_relative_to(protected.resolve()):
            parser.error("Output cannot be inside a source, cache or existing result directory")
    core.DEVICE = args.device
    torch.set_num_threads(2)
    meta, frames = shared.verify_cache(args.cache_root, args.split_root)
    context = verify_context_cache(args.contextual_cache_root, meta)
    masks, token_identity = shared.build_content_masks(frames, args.cache_root)
    sources = ("train_full_model_v5.py", "full_model_v5.py", "run_full_model_v2.py", "full_model_v2.py",
               "full_model_v3.py", "full_model_v4.py", "build_contextual_text_cache.py", Path(core.__file__).name, Path(cached.__file__).name)
    protocol = {"version": "staged_contextual_residual_v5_validation_only", "seeds": args.seeds,
                "variants": args.variants, "core_settings": {k: getattr(core, k) for k in shared.SETTING_NAMES},
                "cache_metadata_sha256": shared.digest(meta), **token_identity,
                "context_cache_metadata_sha256": shared.digest(context),
                "source_sha256": {name: cached.file_sha256(ROOT / name) for name in sources},
                "torch_version": torch.__version__, "numpy_version": np.__version__, "device": args.device,
                "epochs": args.epochs, "freeze_epochs": args.freeze_epochs, "steps_per_epoch": args.steps_per_epoch,
                "base_lr": args.base_lr, "branch_lr": args.branch_lr, "joint_branch_lr": args.joint_branch_lr,
                "branch_loss_weight": args.branch_loss_weight, "grad_clip": args.grad_clip,
                "eval_batch_size": args.eval_batch_size, "limit_samples": args.limit_samples,
                "early_stopping": False, "residual_addition_dtype": "float32", "warm_start_checkpoints": {}}
    for seed in args.seeds:
        source = args.warm_start_root / f"seed_{seed}/no_cross_control/best_head.ckpt"
        protocol["warm_start_checkpoints"][str(seed)] = {"path": str(source.resolve()), "sha256": cached.file_sha256(source)}
    references = [] if args.limit_samples else check_reference(args.reference_root, protocol, args.seeds)
    protocol["no_cross_references"] = references
    root = args.save_root / shared.digest(protocol)[:16]
    print(f"[OUTPUT] {root}\n[READ ONLY] Existing caches and No-cross reference files; train/validation only", flush=True)
    if args.preflight:
        print("[PREFLIGHT OK] No training started; no output written", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        shared.write_json(root / "protocol.json", protocol)
        snapshot = root / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name, expected in protocol["source_sha256"].items():
            target = snapshot / name
            if not target.exists():
                shutil.copyfile(ROOT / name, target)
            if cached.file_sha256(target) != expected:
                raise ValueError("Source snapshot mismatch")
        results = []
        for seed in args.seeds:
            for variant in args.variants:
                results.append(run_one(args, meta, masks, protocol, root, variant, seed))
                shared.write_json(root / "validation_summary.json", summarize(results, references))
        print(json.dumps(summarize(results, references), indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
