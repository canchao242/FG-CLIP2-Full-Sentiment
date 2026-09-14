"""Cache the frozen FG-CLIP2 backbone once, then run paired head experiments.

The cache stores deterministic backbone outputs for the locked submission-v1
splits. All trainable regional, token, gate, matching, and classifier modules
remain inside each seeded run and receive normal gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

import run_fgclip2_submission_core_repeated as repeated
import train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus as core


ROOT = Path(__file__).resolve().parent
SPLIT_ROOT = ROOT / "data_submission_v1"
CACHE_ROOT = ROOT / "checkpoints" / "fgclip2_submission_v1_feature_cache"
SAVE_ROOT = ROOT / "checkpoints" / "fgclip2_submission_v1_core_cached_repeated"
SEEDS = [42, 123, 3407]
EXPERIMENTS = ["global_only", "full", "no_token", "no_region", "no_cross"]
CACHE_BATCH_SIZE = 16
CACHED_EVAL_BATCH_SIZE = 64


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_csvs(split_root: Path) -> dict[str, Path]:
    return {
        f"{language}_{split}": split_root / language / f"{split}.csv"
        for language in ("zh", "en")
        for split in ("train", "val", "test")
    }


def cache_is_current(cache_root: Path, split_root: Path) -> bool:
    marker = cache_root / "cache_ready.json"
    if not marker.exists():
        return False
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("model_id") != core.MODEL_ID:
        return False
    expected = {name: file_sha256(path) for name, path in split_csvs(split_root).items()}
    return metadata.get("csv_sha256") == expected


def create_arrays(directory: Path, rows: int, dims: dict[str, int]) -> dict[str, np.memmap]:
    directory.mkdir(parents=True, exist_ok=True)
    shapes = {
        "img_global": (rows, dims["img_dim"]),
        "txt_global": (rows, dims["txt_dim"]),
        "patches": (rows, core.MAX_NUM_PATCHES, dims["patch_dim"]),
        "patch_mask": (rows, core.MAX_NUM_PATCHES),
        "tokens": (rows, core.MAX_TEXT_LEN, dims["token_dim"]),
        "attention_mask": (rows, core.MAX_TEXT_LEN),
        "labels": (rows,),
        "language_ids": (rows,),
    }
    dtypes = {
        "img_global": np.float16,
        "txt_global": np.float16,
        "patches": np.float16,
        "patch_mask": np.uint8,
        "tokens": np.float16,
        "attention_mask": np.uint8,
        "labels": np.int64,
        "language_ids": np.uint8,
    }
    return {
        name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape)
        for name, shape in shapes.items()
    }


@torch.inference_mode()
def cache_one_split(
    name: str,
    csv_path: Path,
    language: str,
    cache_root: Path,
    image_processor,
    tokenizer,
    model,
    device: torch.device,
) -> dict:
    dataset = core.ImageTextDataset(str(csv_path), image_processor, tokenizer, language)
    loader = DataLoader(
        dataset,
        batch_size=CACHE_BATCH_SIZE,
        shuffle=False,
        num_workers=core.NUM_WORKERS,
        persistent_workers=(core.NUM_WORKERS > 0),
        prefetch_factor=2 if core.NUM_WORKERS > 0 else None,
        pin_memory=(device.type == "cuda"),
        collate_fn=core.collate_fn,
    )
    arrays = None
    dims = None
    offset = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        pv, pm, ss, ids, am, labels, language_ids = [value.to(device, non_blocking=True) for value in batch]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            img, txt, patches, tokens, patch_mask = core.extract_backbone_features(
                model=model,
                pixel_values=pv,
                pixel_attention_mask=pm,
                spatial_shapes=ss,
                input_ids=ids,
                attention_mask=am,
                require_grad=False,
            )
        if arrays is None:
            dims = {
                "img_dim": int(img.shape[-1]),
                "txt_dim": int(txt.shape[-1]),
                "patch_dim": int(patches.shape[-1]),
                "token_dim": int(tokens.shape[-1]),
            }
            arrays = create_arrays(cache_root / name, len(dataset), dims)
        batch_rows = int(labels.shape[0])
        target = slice(offset, offset + batch_rows)
        arrays["img_global"][target] = img.detach().cpu().to(torch.float16).numpy()
        arrays["txt_global"][target] = txt.detach().cpu().to(torch.float16).numpy()
        arrays["patches"][target] = patches.detach().cpu().to(torch.float16).numpy()
        arrays["patch_mask"][target] = patch_mask.detach().cpu().to(torch.uint8).numpy()
        arrays["tokens"][target] = tokens.detach().cpu().to(torch.float16).numpy()
        arrays["attention_mask"][target] = am.detach().cpu().to(torch.uint8).numpy()
        arrays["labels"][target] = labels.detach().cpu().numpy()
        arrays["language_ids"][target] = language_ids.detach().cpu().to(torch.uint8).numpy()
        offset += batch_rows
        if batch_index % 100 == 0 or offset == len(dataset):
            print(f"[CACHE] {name}: {offset}/{len(dataset)}")
    if offset != len(dataset) or arrays is None or dims is None:
        raise RuntimeError(f"Incomplete cache for {name}: {offset}/{len(dataset)}")
    for array in arrays.values():
        array.flush()
    metadata = {
        "name": name,
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "rows": len(dataset),
        "dims": dims,
        "seconds": time.perf_counter() - started,
    }
    (cache_root / name / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def build_cache(cache_root: Path, split_root: Path) -> dict:
    if cache_is_current(cache_root, split_root):
        print(f"[CACHE] reuse: {cache_root}")
        return json.loads((cache_root / "cache_ready.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(core.MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
    core.TOKENIZER = tokenizer
    core.TOKEN_FEATURE_SOURCE = "embedding"
    model = AutoModelForCausalLM.from_pretrained(
        core.MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True, local_files_only=True
    ).to(device)
    core.freeze_backbone(model)
    model.eval()
    model_total = sum(parameter.numel() for parameter in model.parameters())

    sources = split_csvs(split_root)
    split_metadata = {}
    for name, path in sources.items():
        split_metadata[name] = cache_one_split(
            name=name,
            csv_path=path,
            language=name.split("_", 1)[0],
            cache_root=cache_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
    dims = split_metadata["zh_train"]["dims"]
    if any(item["dims"] != dims for item in split_metadata.values()):
        raise RuntimeError("Backbone dimensions differ across cached splits.")
    marker = {
        "model_id": core.MODEL_ID,
        "token_feature_source": "embedding",
        "max_num_patches": core.MAX_NUM_PATCHES,
        "max_text_len": core.MAX_TEXT_LEN,
        "storage_dtype": "float16",
        "backbone_total_params": model_total,
        "dims": dims,
        "csv_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "splits": split_metadata,
    }
    (cache_root / "cache_ready.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    core.cleanup_memory()
    print(f"[CACHE] ready: {cache_root}")
    return marker


class CachedFeatureDataset(Dataset):
    def __init__(self, directory: Path, global_only: bool = False) -> None:
        self.directory = directory
        self.global_only = global_only
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.rows = int(self.metadata["rows"])
        self.img = np.load(directory / "img_global.npy", mmap_mode="r")
        self.txt = np.load(directory / "txt_global.npy", mmap_mode="r")
        self.labels_array = np.load(directory / "labels.npy", mmap_mode="r")
        self.languages = np.load(directory / "language_ids.npy", mmap_mode="r")
        self.labels = self.labels_array.astype(int).tolist()
        # Compatibility with the original weighting/statistics helpers, which
        # intentionally read only dataset.df["label"]. No text or image data is
        # reconstructed here.
        self.df = pd.DataFrame({"label": self.labels})
        if not global_only:
            self.patches = np.load(directory / "patches.npy", mmap_mode="r")
            self.patch_mask = np.load(directory / "patch_mask.npy", mmap_mode="r")
            self.tokens = np.load(directory / "tokens.npy", mmap_mode="r")
            self.attention_mask = np.load(directory / "attention_mask.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.rows

    @staticmethod
    def tensor(row: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(row, copy=True))

    def __getitem__(self, index: int):
        img = self.tensor(self.img[index])
        txt = self.tensor(self.txt[index])
        label = torch.tensor(int(self.labels_array[index]), dtype=torch.long)
        language = torch.tensor(int(self.languages[index]), dtype=torch.long)
        if self.global_only:
            patches = torch.zeros((1, 1), dtype=torch.float16)
            patch_mask = torch.ones((1,), dtype=torch.bool)
            tokens = torch.zeros((1, 1), dtype=torch.float16)
            attention_mask = torch.ones((1,), dtype=torch.long)
        else:
            patches = self.tensor(self.patches[index])
            patch_mask = self.tensor(self.patch_mask[index]).bool()
            tokens = self.tensor(self.tokens[index])
            attention_mask = self.tensor(self.attention_mask[index]).long()
        return img, txt, patches, patch_mask, tokens, attention_mask, label, language


def cached_collate(batch):
    return tuple(torch.stack(items, dim=0) for items in zip(*batch))


def move_cached(batch):
    return tuple(value.to(core.DEVICE, non_blocking=True) for value in batch)


def forward_cached(classifier, batch):
    img, txt, patches, patch_mask, tokens, attention_mask, labels, languages = move_cached(batch)
    logits, aux = classifier(
        img_global=img,
        txt_global=txt,
        flat_patches=patches,
        patch_mask=patch_mask,
        token_embeds=tokens,
        attention_mask=attention_mask,
        language_ids=languages,
        return_aux=True,
    )
    return logits, aux, labels, languages


def train_cached(classifier, loader, optimizer, scaler, criterion, epoch: int) -> dict:
    classifier.train()
    losses = 0.0
    labels_all = []
    preds_all = []
    total = 0
    aux_weight = core.AUX_LOSS_WEIGHT_MAX * min(1.0, epoch / max(1, core.AUX_WARMUP_EPOCHS))
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(core.DEVICE == "cuda"), dtype=torch.float16):
            logits, aux, labels, _languages = forward_cached(classifier, batch)
            classification_loss = criterion(logits, labels)
            auxiliary_loss = core.compute_aux_loss(aux)
            loss = classification_loss + aux_weight * auxiliary_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.shape[0])
        losses += float(loss.detach()) * batch_size
        total += batch_size
        labels_all.append(labels.detach().cpu())
        preds_all.append(predictions.detach().cpu())
        if step % 100 == 0 or step == len(loader):
            print(
                f"[CACHED TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} ce={classification_loss.item():.4f} aux={auxiliary_loss.item():.4f}"
            )
    labels_np = torch.cat(labels_all).numpy()
    preds_np = torch.cat(preds_all).numpy()
    return {
        "loss": losses / max(total, 1),
        "acc": accuracy_score(labels_np, preds_np),
        "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "n": total,
    }


@torch.inference_mode()
def evaluate_cached(classifier, loader, criterion, name: str, collect_outputs: bool = False):
    classifier.eval()
    losses = 0.0
    total = 0
    labels_all = []
    preds_all = []
    logits_all = []
    languages_all = []
    gates_all = []
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=(core.DEVICE == "cuda"), dtype=torch.float16):
            logits, aux, labels, languages = forward_cached(classifier, batch)
            loss = criterion(logits, labels)
        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.shape[0])
        losses += float(loss.detach()) * batch_size
        total += batch_size
        labels_all.append(labels.detach().cpu())
        preds_all.append(predictions.detach().cpu())
        gates_all.append(aux["gate"].mean(dim=-1).detach().float().cpu())
        if collect_outputs:
            logits_all.append(logits.detach().float().cpu())
            languages_all.append(languages.detach().cpu())
    labels_np = torch.cat(labels_all).numpy()
    preds_np = torch.cat(preds_all).numpy()
    gates_np = torch.cat(gates_all).numpy()
    metrics = {
        "loss": losses / max(total, 1),
        "acc": accuracy_score(labels_np, preds_np),
        "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels_np, preds_np, average="weighted", zero_division=0),
        "n": total,
        "gate_mean": float(gates_np.mean()),
    }
    print(
        f"[{name}] n={total} loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    outputs = None
    if collect_outputs:
        logits_np = torch.cat(logits_all).numpy()
        outputs = {
            "logits": logits_np,
            "probs": torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy(),
            "labels": labels_np,
            "preds": preds_np,
            "lang_ids": torch.cat(languages_all).numpy(),
            "gate_mean": gates_np,
        }
    return metrics, outputs


def save_head(path: Path, classifier, epoch: int, score: float, config: dict, metrics: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": score,
            "clf_state_dict": classifier.state_dict(),
            "config": config,
            "extra_metrics": metrics,
        },
        path,
    )


def make_loader(dataset, batch_size: int, sampler=None, shuffle: bool = False, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=(core.DEVICE == "cuda"),
        collate_fn=cached_collate,
    )


def run_cached_experiment(
    experiment: str,
    seed: int,
    cache_root: Path,
    save_root: Path,
    cache_metadata: dict,
    epochs: int,
    steps_per_epoch: int,
) -> dict:
    core.set_seed(seed)
    global_only = experiment == "global_only"
    datasets = {
        name: CachedFeatureDataset(cache_root / name, global_only=global_only)
        for name in ("zh_train", "en_train", "zh_val", "en_val", "zh_test", "en_test")
    }
    train_dataset = ConcatDataset([datasets["zh_train"], datasets["en_train"]])
    weights = core.build_joint_sample_weights(
        datasets["zh_train"], datasets["en_train"],
        target_en_frac=core.TARGET_EN_FRAC,
        num_classes=core.NUM_CLASSES,
        zh_class2_boost=core.ZH_CLASS2_BOOST,
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=steps_per_epoch * core.BATCH_SIZE,
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = make_loader(train_dataset, core.BATCH_SIZE, sampler=sampler, drop_last=True)
    loaders = {
        name: make_loader(datasets[name], CACHED_EVAL_BATCH_SIZE)
        for name in ("zh_val", "en_val", "zh_test", "en_test")
    }

    dims = cache_metadata["dims"]
    config_ablation = core.get_ablation_config(experiment)
    classifier = core.EmotionGatedFusionClassifier(
        img_dim=dims["img_dim"],
        txt_dim=dims["txt_dim"],
        patch_dim=dims["patch_dim"],
        token_dim=dims["token_dim"],
        proj_dim=core.PROJ_DIM,
        hidden_dim=core.HIDDEN_DIM,
        num_classes=core.NUM_CLASSES,
        dropout=core.DROPOUT,
        num_heads=core.NUM_HEADS,
        ablation_cfg=config_ablation,
    ).to(core.DEVICE)
    head_total = sum(parameter.numel() for parameter in classifier.parameters())
    backbone_total = int(cache_metadata["backbone_total_params"])
    class_weights, _counts = core.get_global_class_weights(
        datasets["zh_train"], datasets["en_train"], num_classes=core.NUM_CLASSES
    )
    criterion = core.SmoothedFocalLoss(
        class_weights=torch.pow(class_weights, core.LOSS_CLASS_WEIGHT_POWER).to(core.DEVICE),
        label_smoothing=core.LABEL_SMOOTHING,
        gamma=core.FOCAL_GAMMA,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=core.HEAD_LR, weight_decay=core.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(core.DEVICE == "cuda"))

    output = save_root / f"seed_{seed}" / experiment
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "MODEL_ID": core.MODEL_ID,
        "MAX_NUM_PATCHES": core.MAX_NUM_PATCHES,
        "MAX_TEXT_LEN": core.MAX_TEXT_LEN,
        "BATCH_SIZE": core.BATCH_SIZE,
        "EVAL_BATCH_SIZE": CACHED_EVAL_BATCH_SIZE,
        "NUM_EPOCHS": epochs,
        "HEAD_LR": core.HEAD_LR,
        "WEIGHT_DECAY": core.WEIGHT_DECAY,
        "STEPS_PER_EPOCH": steps_per_epoch,
        "NUM_CLASSES": core.NUM_CLASSES,
        "PROJ_DIM": core.PROJ_DIM,
        "HIDDEN_DIM": core.HIDDEN_DIM,
        "DROPOUT": core.DROPOUT,
        "NUM_HEADS": core.NUM_HEADS,
        "TOKEN_FEATURE_SOURCE": "embedding",
        "SEED": seed,
        "img_dim": dims["img_dim"],
        "txt_dim": dims["txt_dim"],
        "patch_dim": dims["patch_dim"],
        "token_dim": dims["token_dim"],
        "experiment_name": experiment,
        "ablation_cfg": config_ablation,
        "feature_cache": str(cache_root),
        "feature_cache_csv_sha256": cache_metadata["csv_sha256"],
        "SAMPLING_PROTOCOL": "dedicated_weighted_sampler_generator_v1",
        "architecture_variant": "structurally_pruned_no_cross" if not config_ablation["use_cross_attention"] else "cross_attention_instantiated",
    }

    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_values = {}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        print(f"\n[CACHED RUN] experiment={experiment} seed={seed} epoch={epoch}/{epochs}")
        train_cached(classifier, train_loader, optimizer, scaler, criterion, epoch)
        zh_metrics, zh_outputs = evaluate_cached(
            classifier, loaders["zh_val"], criterion, "ZH VAL", collect_outputs=True
        )
        en_metrics, en_outputs = evaluate_cached(
            classifier, loaders["en_val"], criterion, "EN VAL", collect_outputs=True
        )
        balanced_macro, balanced_acc, weighted_macro = core.compute_scores(zh_metrics, en_metrics)
        scheduler.step(balanced_macro)
        combined = {
            "zh_metrics": zh_metrics,
            "en_metrics": en_metrics,
            "balanced_macro_f1": balanced_macro,
            "balanced_acc": balanced_acc,
            "weighted_macro_f1": weighted_macro,
        }
        save_head(output / "last_head.ckpt", classifier, epoch, best_score, config, combined)
        if balanced_macro > best_score:
            best_score = balanced_macro
            best_epoch = epoch
            bad_epochs = 0
            best_values = {
                "best_balanced_acc": balanced_acc,
                "best_balanced_zh_acc": zh_metrics["acc"],
                "best_balanced_en_acc": en_metrics["acc"],
                "best_balanced_zh_macro_f1": zh_metrics["macro_f1"],
                "best_balanced_en_macro_f1": en_metrics["macro_f1"],
                "best_balanced_zh_weighted_f1": zh_metrics["weighted_f1"],
                "best_balanced_en_weighted_f1": en_metrics["weighted_f1"],
            }
            save_head(output / "best_head_balanced.ckpt", classifier, epoch, best_score, config, combined)
            core.save_validation_outputs(zh_outputs, "zh_val", str(output / "best_balanced_zh_val_outputs.npz"))
            core.save_validation_outputs(en_outputs, "en_val", str(output / "best_balanced_en_val_outputs.npz"))
            print(f"[BEST] balanced_macro_f1={best_score:.4f} epoch={epoch}")
        else:
            bad_epochs += 1
            print(f"[NO IMPROVEMENT] {bad_epochs}/{core.PATIENCE}")
        if bad_epochs >= core.PATIENCE:
            print("[STOP] early stopping")
            break

    training_seconds = time.perf_counter() - started
    selected = torch.load(output / "best_head_balanced.ckpt", map_location=core.DEVICE, weights_only=False)
    classifier.load_state_dict(selected["clf_state_dict"])
    zh_test_metrics, zh_test_outputs = evaluate_cached(
        classifier, loaders["zh_test"], criterion, "ZH TEST", collect_outputs=True
    )
    en_test_metrics, en_test_outputs = evaluate_cached(
        classifier, loaders["en_test"], criterion, "EN TEST", collect_outputs=True
    )
    test_balanced_macro, test_balanced_acc, test_weighted_macro = core.compute_scores(
        zh_test_metrics, en_test_metrics
    )
    core.save_validation_outputs(zh_test_outputs, "zh_test", str(output / "test_outputs_zh.npz"))
    core.save_validation_outputs(en_test_outputs, "en_test", str(output / "test_outputs_en.npz"))
    test_metrics = {
        "selected_checkpoint": str(output / "best_head_balanced.ckpt"),
        "zh": zh_test_metrics,
        "en": en_test_metrics,
        "balanced_macro_f1": test_balanced_macro,
        "balanced_acc": test_balanced_acc,
        "weighted_macro_f1": test_weighted_macro,
    }
    (output / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "experiment": experiment,
        "seed": seed,
        "paired_seed": True,
        "split_version": "submission_v1",
        "best_epoch": best_epoch,
        "best_balanced_macro_f1": best_score,
        **best_values,
        "test_balanced_macro_f1": test_balanced_macro,
        "test_balanced_acc": test_balanced_acc,
        "test_weighted_macro_f1": test_weighted_macro,
        "test_zh_acc": zh_test_metrics["acc"],
        "test_en_acc": en_test_metrics["acc"],
        "test_zh_macro_f1": zh_test_metrics["macro_f1"],
        "test_en_macro_f1": en_test_metrics["macro_f1"],
        "backbone_total_params": backbone_total,
        "backbone_trainable_params": 0,
        "classifier_total_params": head_total,
        "classifier_trainable_params": head_total,
        "end_to_end_total_params": backbone_total + head_total,
        "end_to_end_trainable_params": head_total,
        "training_time_seconds": training_seconds,
        "run_wall_time_seconds": training_seconds,
        "save_dir": str(output),
        "execution_mode": "frozen_backbone_feature_cache",
        "feature_cache_csv_sha256": cache_metadata["csv_sha256"],
        "sampling_protocol": "dedicated_weighted_sampler_generator_v1",
    }
    (output / "run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del classifier, optimizer, scheduler, scaler, loaders, datasets, train_dataset, train_loader
    core.cleanup_memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--cache_root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--save_root", type=Path, default=SAVE_ROOT)
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--steps_per_epoch", type=int, default=core.STEPS_PER_EPOCH)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--log_file", type=Path, default=None)
    args = parser.parse_args()
    log_handle = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle

    args.save_root.mkdir(parents=True, exist_ok=True)
    state_path = args.save_root / "runner_state.json"

    def write_state(status: str, **extra) -> None:
        state_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "pid": os.getpid(),
                    "updated_unix": time.time(),
                    **extra,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_state("starting")
    try:
        if args.rebuild_cache and (args.cache_root / "cache_ready.json").exists():
            (args.cache_root / "cache_ready.json").unlink()
        cache_metadata = build_cache(args.cache_root, args.split_root)

        results = []
        total = len(EXPERIMENTS) * len(SEEDS)
        run_number = 0
        for experiment in EXPERIMENTS:
            for seed in SEEDS:
                run_number += 1
                result_path = args.save_root / f"seed_{seed}" / experiment / "run_result.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    if (
                        existing.get("execution_mode") == "frozen_backbone_feature_cache"
                        and existing.get("feature_cache_csv_sha256") == cache_metadata["csv_sha256"]
                        and existing.get("sampling_protocol") == "dedicated_weighted_sampler_generator_v1"
                    ):
                        print(f"[RESUME] completed run {run_number}/{total}: {experiment} seed={seed}")
                        results.append(existing)
                        repeated.summarize(results, args.save_root)
                        continue

                write_state(
                    "running",
                    run_number=run_number,
                    total_runs=total,
                    experiment=experiment,
                    seed=seed,
                    completed_runs=len(results),
                )
                print("\n" + "#" * 100)
                print(f"CACHED SUBMISSION RUN {run_number}/{total} | {experiment} | seed={seed}")
                print("#" * 100)
                result = run_cached_experiment(
                    experiment=experiment,
                    seed=seed,
                    cache_root=args.cache_root,
                    save_root=args.save_root,
                    cache_metadata=cache_metadata,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                )
                results.append(result)
                repeated.summarize(results, args.save_root)
        write_state("complete", total_runs=total, completed_runs=len(results))
    except BaseException as error:
        traceback.print_exc()
        write_state("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if log_handle is not None:
            log_handle.flush()


if __name__ == "__main__":
    main()
