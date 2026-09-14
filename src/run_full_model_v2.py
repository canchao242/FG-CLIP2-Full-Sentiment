"""Validation-only Full-v2 development on EXISTING frozen feature caches.

No backbone is loaded, no feature cache is written, and test arrays are never
opened. --preflight checks data/configuration without starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

import full_model_v2 as models
import full_model_v3 as models_v3
import full_model_v4 as models_v4
import run_fgclip2_submission_cached_repeated as cached
import train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus as core

ROOT = Path(__file__).resolve().parent
SPLITS = ("zh_train", "en_train", "zh_val", "en_val")
VARIANTS = ("no_cross_control", "full_v2", "full_v3", "pooled_v3", "full_v4", "pooled_v4", "no_cross_legacy", "full_legacy")
REFINEMENT_VARIANTS = ("full_v2", "full_v3", "pooled_v3", "full_v4", "pooled_v4", "no_cross_control")
SETTING_NAMES = (
    "MODEL_ID", "MAX_NUM_PATCHES", "MAX_TEXT_LEN", "TOKEN_FEATURE_SOURCE",
    "BATCH_SIZE", "HEAD_LR", "WEIGHT_DECAY", "PROJ_DIM", "HIDDEN_DIM",
    "NUM_CLASSES", "NUM_HEADS", "DROPOUT", "TARGET_EN_FRAC",
    "EN_CLASS1_BOOST", "ZH_CLASS2_BOOST", "LOSS_CLASS_WEIGHT_POWER",
    "FOCAL_GAMMA", "LABEL_SMOOTHING", "AUX_LOSS_WEIGHT_MAX",
    "AUX_WARMUP_EPOCHS", "GATE_CENTER_LOSS_WEIGHT", "GATE_LANG_BALANCE_WEIGHT",
    "GATE_CENTER_MARGIN", "GATE_DROPOUT", "PATCH_EMO_TEMPERATURE",
    "TOKEN_EMO_TEMPERATURE",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def verify_cache(cache_root, split_root):
    """Fail closed; this runner never calls build_cache or cache_one_split."""
    if core.TOKEN_FEATURE_SOURCE != "embedding":
        raise ValueError("This runner requires embedding features; contextual tokens need a separate versioned cache.")
    marker = cache_root / "cache_ready.json"
    if not marker.is_file():
        raise ValueError("Existing cache_ready.json is required; automatic extraction is disabled.")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    expected = {"model_id": core.MODEL_ID, "token_feature_source": "embedding",
                "max_num_patches": core.MAX_NUM_PATCHES, "max_text_len": core.MAX_TEXT_LEN,
                "storage_dtype": "float16"}
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"Incompatible cache {name}: {meta.get(name)!r} != {value!r}")
    dims = meta["dims"]
    frames = {}
    for split in SPLITS:
        language, partition = split.split("_")
        csv_path = split_root / language / f"{partition}.csv"
        if cached.file_sha256(csv_path) != meta["csv_sha256"][split]:
            raise ValueError(f"Changed CSV for {split}; refusing cache reuse.")
        frame = pd.read_csv(csv_path)
        frames[split] = frame
        rows = len(frame)
        detail = json.loads((cache_root / split / "metadata.json").read_text(encoding="utf-8"))
        if detail["rows"] != rows or detail["dims"] != dims:
            raise ValueError(f"Inconsistent metadata: {split}")
        shapes = {
            "img_global": ((rows, dims["img_dim"]), "float16"),
            "txt_global": ((rows, dims["txt_dim"]), "float16"),
            "patches": ((rows, core.MAX_NUM_PATCHES, dims["patch_dim"]), "float16"),
            "tokens": ((rows, core.MAX_TEXT_LEN, dims["token_dim"]), "float16"),
            "patch_mask": ((rows, core.MAX_NUM_PATCHES), "uint8"),
            "attention_mask": ((rows, core.MAX_TEXT_LEN), "uint8"),
            "labels": ((rows,), "int64"), "language_ids": ((rows,), "uint8"),
        }
        for name, (shape, dtype) in shapes.items():
            array = np.load(cache_root / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.dtype(dtype) or not rows:
                raise ValueError(f"Bad array header: {split}/{name}")
            if name == "labels" and not np.array_equal(array, frame.label.to_numpy()):
                raise ValueError(f"Label/order mismatch: {split}")
            if name == "language_ids" and not np.all(array == (language == "en")):
                raise ValueError(f"Language mismatch: {split}")
            if name.endswith("mask") and (not np.isin(array, [0, 1]).all() or not (array.sum(1) > 0).all()):
                raise ValueError(f"Invalid/all-empty masks: {split}/{name}")
            # Cheap preflight samples; training also checks loss/gradient finiteness.
            if np.issubdtype(array.dtype, np.floating):
                if not np.isfinite(array[[0, rows // 2, rows - 1]]).all():
                    raise ValueError(f"Non-finite features: {split}/{name}")
        print(f"[CACHE READ-ONLY] {split}: {rows} rows checked", flush=True)
    return meta, frames


def build_content_masks(frames, cache_root):
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", local_files_only=True,
                                              trust_remote_code=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A local fast tokenizer with offsets is required.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    masks = {}
    for split, frame in frames.items():
        prefix = "Language: Chinese. Text: " if split.startswith("zh") else "Language: English. Text: "
        original = np.load(cache_root / split / "attention_mask.npy", mmap_mode="r")
        output = np.zeros_like(original)
        texts = frame.text.fillna("").astype(str).tolist()
        for start in range(0, len(texts), 512):
            chunk = texts[start:start + 512]
            encoded = tokenizer([prefix + text for text in chunk], padding="max_length",
                                truncation=True, max_length=core.MAX_TEXT_LEN,
                                return_offsets_mapping=True, return_special_tokens_mask=True)
            if "attention_mask" not in encoded:
                # Mirror ImageTextDataset exactly: this tokenizer can omit the
                # mask and use EOS as PAD, so forcing a new mask changes EOS semantics.
                encoded["attention_mask"] = (
                    np.asarray(encoded["input_ids"]) != tokenizer.pad_token_id
                ).astype(np.uint8).tolist()
            if not np.array_equal(encoded["attention_mask"], original[start:start + len(chunk)]):
                raise ValueError(f"Tokenizer/cache mismatch in {split} at row {start}")
            output[start:start + len(chunk)] = models.make_content_mask(encoded, [len(prefix)] * len(chunk))
        masks[split] = output
        print(f"[CONTENT MASK] {split}: empty={int((output.sum(1) == 0).sum())}; no features extracted", flush=True)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    mask_hashes = {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()}
    return masks, {"tokenizer_sha256": tokenizer_hash, "content_mask_sha256": mask_hashes}


class ContentDataset(cached.CachedFeatureDataset):
    def __init__(self, directory, content_mask=None, limit=0, contextual_root=None):
        super().__init__(directory)
        self.content_mask = content_mask
        self.contextual_tokens = None
        if contextual_root is not None:
            self.contextual_tokens = np.load(contextual_root / directory.name / "tokens.npy", mmap_mode="r", allow_pickle=False)
        if limit:
            self.rows = min(limit, self.rows)
            self.df = self.df.iloc[:self.rows].copy()

    def __getitem__(self, index):
        row = list(super().__getitem__(index))
        if self.content_mask is not None:
            row[5] = self.tensor(self.content_mask[index]).long()
        if self.contextual_tokens is not None:
            row[4] = torch.cat([row[4], self.tensor(self.contextual_tokens[index])], dim=-1)
        return tuple(row)


def loader(dataset, batch_size, sampler=None, generator=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                      generator=generator, pin_memory=core.DEVICE == "cuda", collate_fn=cached.cached_collate)


def build_model(variant, dims, cross_dropout):
    kwargs = dict(img_dim=dims["img_dim"], txt_dim=dims["txt_dim"], patch_dim=dims["patch_dim"],
                  token_dim=dims["token_dim"], proj_dim=core.PROJ_DIM, hidden_dim=core.HIDDEN_DIM,
                  num_classes=core.NUM_CLASSES, num_heads=core.NUM_HEADS, dropout=core.DROPOUT)
    if variant.endswith("legacy"):
        experiment = "full" if variant == "full_legacy" else "no_cross"
        return core.EmotionGatedFusionClassifier(ablation_cfg=core.get_ablation_config(experiment), **kwargs)
    if variant in ("full_v3", "pooled_v3"):
        return models_v3.LogitResidualFullClassifier(
            use_attention=variant == "full_v3", cross_dropout=cross_dropout, **kwargs)
    if variant in ("full_v4", "pooled_v4"):
        return models_v4.ContextualResidualFullClassifier(
            use_attention=variant == "full_v4", cross_dropout=cross_dropout, **kwargs)
    return models.ResidualFullClassifier(use_cross=variant == "full_v2", cross_dropout=cross_dropout, **kwargs)


def load_shared_start(model, checkpoint, seed, protocol):
    """Load the SAME selected no-cross state into both refinement arms."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config["variant"] != "no_cross_control" or config["seed"] != seed:
        raise ValueError("Warm start must be the paired no-cross control for this seed")
    parent = config["protocol"]
    for key in ("core_settings", "cache_metadata_sha256", "content_mask_sha256", "limit_samples"):
        if parent.get(key) != protocol.get(key):
            raise ValueError(f"Warm-start protocol mismatch: {key}")
    incompatible = model.load_state_dict(saved["clf_state_dict"], strict=False)
    if incompatible.unexpected_keys or any(
        not (name == "cross_alpha" or name.startswith("cross_attention."))
        for name in incompatible.missing_keys
    ):
        raise ValueError(f"Incompatible shared warm-start weights: {incompatible}")


def make_optimizer(model, args):
    base_lr = getattr(args, "base_lr", None)
    cross_lr = getattr(args, "cross_lr", None)
    alpha_lr = getattr(args, "alpha_lr", None)
    if base_lr is None and cross_lr is None and alpha_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=core.HEAD_LR, weight_decay=core.WEIGHT_DECAY)
    groups = {"base": [], "cross": [], "alpha": []}
    for name, parameter in model.named_parameters():
        group = "alpha" if name == "cross_alpha" else "cross" if name.startswith("cross_attention.") else "base"
        groups[group].append(parameter)
    rates = {"base": base_lr or core.HEAD_LR, "cross": cross_lr or core.HEAD_LR,
             "alpha": alpha_lr or core.HEAD_LR}
    return torch.optim.AdamW([{"params": values, "lr": rates[name], "name": name}
                             for name, values in groups.items() if values], weight_decay=core.WEIGHT_DECAY)


def rng_state(sampler_generator, loader_generator):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler": sampler_generator.get_state(), "loader": loader_generator.get_state()}


def restore_rng(state, sampler_generator, loader_generator):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    sampler_generator.set_state(state["sampler"])
    loader_generator.set_state(state["loader"])


def train_epoch(model, batches, optimizer, scaler, criterion, epoch, grad_clip):
    model.train()
    aux_weight = core.AUX_LOSS_WEIGHT_MAX * min(1., epoch / max(1, core.AUX_WARMUP_EPOCHS))
    losses, norms = [], []
    nonfinite_norm_steps, amp_skipped_steps = 0, 0
    for batch in batches:
        # CPU Linear cannot consume float16 inputs with float32 parameters.
        batch = tuple(t.float() if core.DEVICE == "cpu" and t.is_floating_point() else t for t in batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=core.DEVICE == "cuda", dtype=torch.float16):
            logits, aux, labels, _ = cached.forward_cached(model, batch)
            loss = criterion(logits, labels) + aux_weight * core.compute_aux_loss(aux)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; stopping without changing the feature cache.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
        # AMP can skip an overflowed step; FP32 non-finite gradients must fail.
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite FP32 gradients")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        amp_skipped_steps += int(scaler.get_scale() < scale_before)
        losses.append(float(loss.detach()))
        norm_value = float(grad_norm)
        if np.isfinite(norm_value):
            norms.append(norm_value)
        else:
            nonfinite_norm_steps += 1
    return {"loss": float(np.mean(losses)),
            "finite_grad_norm_mean": float(np.mean(norms)) if norms else None,
            "nonfinite_grad_norm_steps": nonfinite_norm_steps,
            "amp_skipped_steps": amp_skipped_steps}


def evaluate(model, batches, criterion, name):
    if core.DEVICE == "cpu":
        batches = (tuple(t.float() if t.is_floating_point() else t for t in batch) for batch in batches)
    return cached.evaluate_cached(model, batches, criterion, name, collect_outputs=True)


def run_one(args, meta, masks, protocol, run_root, variant, seed):
    output = run_root / f"seed_{seed}" / variant
    output.mkdir(parents=True, exist_ok=True)
    config = {"protocol": protocol, "variant": variant, "seed": seed}
    fingerprint = digest(config)
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Refusing mixed configurations in {output}")
    elif any(output.iterdir()):
        raise ValueError(f"Nonempty output without a verified configuration: {output}")
    else:
        write_json(config_path, config)
    result_path = output / "run_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint or not (output / "best_head.ckpt").is_file():
            raise ValueError(f"Invalid completed result: {output}")
        print(f"[RESUME COMPLETE] {variant} seed={seed}", flush=True)
        return result
    core.set_seed(seed)
    context_root = getattr(args, "contextual_cache_root", None) if variant in ("full_v4", "pooled_v4") else None
    if variant in ("full_v4", "pooled_v4") and context_root is None:
        raise ValueError("Full-v4 requires a verified contextual text sidecar; no fallback is permitted")
    datasets = {name: ContentDataset(args.cache_root / name,
                                    None if variant.endswith("legacy") else masks[name], args.limit_samples,
                                    contextual_root=context_root)
                for name in SPLITS}
    sampler_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed + 100000)
    weights = core.build_joint_sample_weights(datasets["zh_train"], datasets["en_train"],
                                             target_en_frac=core.TARGET_EN_FRAC,
                                             en_class1_boost=core.EN_CLASS1_BOOST,
                                             zh_class2_boost=core.ZH_CLASS2_BOOST)
    sampler = WeightedRandomSampler(weights, args.steps_per_epoch * core.BATCH_SIZE,
                                    replacement=True, generator=sampler_generator)
    train_loader = loader(ConcatDataset([datasets["zh_train"], datasets["en_train"]]),
                          core.BATCH_SIZE, sampler, loader_generator)
    val_loaders = {name: loader(datasets[name], args.eval_batch_size, generator=loader_generator)
                   for name in ("zh_val", "en_val")}
    model = build_model(variant, meta["dims"], args.cross_dropout).to(core.DEVICE)
    warm_start_root = getattr(args, "warm_start_root", None)
    if warm_start_root is not None:
        if variant not in REFINEMENT_VARIANTS:
            raise ValueError("Refinement requires a residual model or its no-cross control")
        source = warm_start_root / f"seed_{seed}/no_cross_control/best_head.ckpt"
        if cached.file_sha256(source) != protocol["warm_start_checkpoints"][str(seed)]["sha256"]:
            raise ValueError("Warm-start checkpoint changed after protocol creation")
        load_shared_start(model, source, seed, protocol)
    weights, _ = core.get_global_class_weights(datasets["zh_train"], datasets["en_train"])
    criterion = core.SmoothedFocalLoss(weights.pow(core.LOSS_CLASS_WEIGHT_POWER).to(core.DEVICE),
                                     core.LABEL_SMOOTHING, core.FOCAL_GAMMA)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=core.DEVICE == "cuda")
    start_epoch, best, bad, history, elapsed = 1, -1., 0, [], 0.
    last = output / "last_state.ckpt"
    if last.exists():
        # Locally generated trusted resume checkpoint, including NumPy RNG state.
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, bad, history, elapsed = state["epoch"] + 1, state["best"], state["bad"], state["history"], state["elapsed"]
        restore_rng(state["rng"], sampler_generator, loader_generator)
        print(f"[RESUME EPOCH] {variant} seed={seed} next_epoch={start_epoch}", flush=True)
    elif warm_start_root is not None:
        # Epoch 0 is an explicit shared reference, not an invented training score.
        # If it remains selected, the residual is zero: NO demonstrated gain.
        zh, zh_out = evaluate(model, val_loaders["zh_val"], criterion, "ZH VAL INITIAL")
        en, en_out = evaluate(model, val_loaders["en_val"], criterion, "EN VAL INITIAL")
        best = (zh["macro_f1"] + en["macro_f1"]) / 2
        row = {"epoch": 0, "val_lb_mf1": best, "zh": zh, "en": en, "train": None,
               "cross_alpha": 0. if variant == "full_v2" else None,
               "lr": optimizer.param_groups[0]["lr"], **correction_diagnostics(model)}
        history.append(row)
        save_torch(output / "best_head.ckpt", {"clf_state_dict": model.state_dict(),
                   "config": config, "fingerprint": fingerprint, "metrics": row})
        core.save_validation_outputs(zh_out, "zh_val", str(output / "best_zh_val_outputs.npz"))
        core.save_validation_outputs(en_out, "en_val", str(output / "best_en_val_outputs.npz"))
        print(f"[SHARED INITIAL] {variant} seed={seed} LB-MF1={best:.5f}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if bad >= args.patience:
            break
        started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, scaler, criterion, epoch, args.grad_clip)
        zh, zh_out = evaluate(model, val_loaders["zh_val"], criterion, "ZH VAL")
        en, en_out = evaluate(model, val_loaders["en_val"], criterion, "EN VAL")
        score = (zh["macro_f1"] + en["macro_f1"]) / 2
        if not np.isfinite(score) or not np.isfinite(zh["loss"]) or not np.isfinite(en["loss"]):
            raise FloatingPointError("Non-finite validation outputs")
        scheduler.step(score)
        alpha = float(model.cross_alpha.detach()) if getattr(model, "cross_alpha", None) is not None else None
        row = {"epoch": epoch, "val_lb_mf1": score, "zh": zh, "en": en, "train": train,
               "cross_alpha": alpha, "lr": optimizer.param_groups[0]["lr"], **correction_diagnostics(model)}
        history.append(row)
        if score > best:
            best, bad = score, 0
            save_torch(output / "best_head.ckpt", {"clf_state_dict": model.state_dict(),
                       "config": config, "fingerprint": fingerprint, "metrics": row})
            core.save_validation_outputs(zh_out, "zh_val", str(output / "best_zh_val_outputs.npz"))
            core.save_validation_outputs(en_out, "en_val", str(output / "best_en_val_outputs.npz"))
        else:
            bad += 1
        elapsed += time.perf_counter() - started
        save_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "rng": rng_state(sampler_generator, loader_generator), "epoch": epoch,
                   "best": best, "bad": bad, "history": history, "elapsed": elapsed,
                   "fingerprint": fingerprint})
        write_json(output / "history.json", history)
        print(f"[{variant} seed={seed}] epoch={epoch} VAL LB-MF1={score:.5f} best={best:.5f} alpha={alpha}", flush=True)
    selected = max(history, key=lambda row: row["val_lb_mf1"])
    result = {"variant": variant, "seed": seed, "fingerprint": fingerprint,
              "validation_only": True, "smoke_only": bool(args.limit_samples),
              "warm_start": warm_start_root is not None,
              "selected_initial_without_refinement": selected["epoch"] == 0,
              "best_val_lb_mf1": best, "selected": selected, "training_seconds": elapsed,
              "head_parameters": sum(p.numel() for p in model.parameters()), "save_dir": str(output)}
    write_json(result_path, result)
    del model, optimizer, scaler, datasets, train_loader, val_loaders
    core.cleanup_memory()
    return result


def correction_diagnostics(model):
    if isinstance(model, (models_v3.LogitResidualFullClassifier, models_v4.ContextualResidualFullClassifier)):
        output = model.cross_attention.classifier[-1]
        return {"correction_output_weight_norm": float(output.weight.detach().float().norm()),
                "correction_output_bias_norm": float(output.bias.detach().float().norm())}
    return {}


def summarize(results):
    summary = {"validation_only": True, "not_final_test_evidence": True, "variants": {}, "paired_full_v2_minus_control": {}}
    for variant in sorted({row["variant"] for row in results}):
        rows = sorted((r for r in results if r["variant"] == variant), key=lambda r: r["seed"])
        values = [r["best_val_lb_mf1"] for r in rows]
        summary["variants"][variant] = {"seeds": [r["seed"] for r in rows], "values": values,
                  "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    by_key = {(r["variant"], r["seed"]): r["best_val_lb_mf1"] for r in results}
    for seed in sorted({r["seed"] for r in results}):
        if ("full_v2", seed) in by_key and ("no_cross_control", seed) in by_key:
            summary["paired_full_v2_minus_control"][str(seed)] = by_key[("full_v2", seed)] - by_key[("no_cross_control", seed)]
    for candidate, control in (("full_v3", "no_cross_control"), ("pooled_v3", "no_cross_control"),
                               ("full_v3", "pooled_v3"), ("full_v4", "no_cross_control"),
                               ("pooled_v4", "no_cross_control"), ("full_v4", "pooled_v4")):
        paired = {str(seed): by_key[(candidate, seed)] - by_key[(control, seed)]
                  for seed in sorted({r["seed"] for r in results})
                  if (candidate, seed) in by_key and (control, seed) in by_key}
        if paired:
            summary[f"paired_{candidate}_minus_{control}"] = paired
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "checkpoints/fgclip2_submission_v1_feature_cache")
    parser.add_argument("--split-root", type=Path, default=ROOT / "data_submission_v1")
    parser.add_argument("--save-root", type=Path, default=ROOT / "checkpoints/fgclip2_full_v2_development")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["no_cross_control", "full_v2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--cross-dropout", type=float, default=.1)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--limit-samples", type=int, default=0, help="Smoke only: truncate each split; not a quality result")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--warm-start-root", type=Path, default=None,
                        help="Optional protocol directory containing paired no_cross_control checkpoints")
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--cross-lr", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=None)
    parser.add_argument("--contextual-cache-root", type=Path, default=None,
                        help="Verified separate contextual text sidecar for Full-v4/pooled-v4 only")
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.patience, args.eval_batch_size) < 1 or args.limit_samples < 0:
        parser.error("Epochs, steps, patience and batch size must be positive; limit must be nonnegative")
    if not 0 <= args.cross_dropout < 1 or args.grad_clip <= 0:
        parser.error("Invalid cross dropout or gradient clipping")
    if any(value is not None and value <= 0 for value in (args.base_lr, args.cross_lr, args.alpha_lr)):
        parser.error("Learning rates must be positive")
    if args.warm_start_root and any(v not in REFINEMENT_VARIANTS for v in args.variants):
        parser.error("Warm-start refinement requires residual models or no_cross_control")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate seeds or variants")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable: use the peixun environment, or --device cpu for smoke testing")
    if args.save_root.resolve() == args.cache_root.resolve() or args.cache_root.resolve() in args.save_root.resolve().parents:
        parser.error("Output must not be placed inside the read-only feature cache")
    core.DEVICE = args.device
    if args.device == "cpu":
        torch.set_num_threads(2)
    meta, frames = verify_cache(args.cache_root, args.split_root)
    context_identity = None
    if any(v in ("full_v4", "pooled_v4") for v in args.variants):
        if args.contextual_cache_root is None:
            parser.error("Full-v4/pooled-v4 require --contextual-cache-root; extraction is never automatic")
        from build_contextual_text_cache import verify_context_cache
        context_meta = verify_context_cache(args.contextual_cache_root, meta)
        context_identity = digest(context_meta)
    masks, token_identity = build_content_masks(frames, args.cache_root)
    protocol = {"version": "residual_full_v2_validation_only_v1",
                "core_settings": {key: getattr(core, key) for key in SETTING_NAMES},
                "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "patience": args.patience,
                "cross_dropout": args.cross_dropout, "grad_clip": args.grad_clip,
                "eval_batch_size": args.eval_batch_size, "limit_samples": args.limit_samples,
                "cache_metadata_sha256": digest(meta), **token_identity,
                "source_sha256": {name: cached.file_sha256(ROOT / name) for name in
                    (Path(__file__).name, "full_model_v2.py", "full_model_v3.py", "full_model_v4.py",
                     "build_contextual_text_cache.py", Path(core.__file__).name, Path(cached.__file__).name)},
                "torch_version": torch.__version__, "numpy_version": np.__version__, "device": args.device}
    protocol.update({"base_lr": args.base_lr, "cross_lr": args.cross_lr, "alpha_lr": args.alpha_lr})
    if context_identity is not None:
        protocol.update(context_cache_metadata_sha256=context_identity,
                        correction_token_source="text_encoder_last_hidden_state", base_token_source="embedding")
    if args.warm_start_root is not None:
        protocol["warm_start_checkpoints"] = {}
        for seed in args.seeds:
            source = args.warm_start_root / f"seed_{seed}/no_cross_control/best_head.ckpt"
            protocol["warm_start_checkpoints"][str(seed)] = {
                "path": str(source.resolve()), "sha256": cached.file_sha256(source)}
    run_root = args.save_root / digest(protocol)[:16]
    print(f"[OUTPUT] {run_root}\n[POLICY] train/validation only; existing feature cache is read-only", flush=True)
    if args.preflight:
        print("[PREFLIGHT OK] No training started and no experiment outputs written.")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / "RUNNING.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Runner lock exists at {lock}; verify its PID has exited before removing it.") from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        write_json(run_root / "protocol.json", protocol)
        results = []
        for seed in args.seeds:
            for variant in args.variants:
                results.append(run_one(args, meta, masks, protocol, run_root, variant, seed))
                summary = summarize(results)
                write_json(run_root / "validation_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
