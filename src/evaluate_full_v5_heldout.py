"""Locked, retrospective held-out evaluation; never tune on the test results.

The original test split was previously inspected during model development.
It is weight-held-out, NOT a newly untouched confirmatory test. Existing
feature caches and checkpoints are read-only. Only missing test contextual
tokens are extracted, into this evaluation's separate output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_full_model_v2 as shared
from full_model_v5_scaled import load_scaled_checkpoint

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "checkpoints/fgclip2_submission_v1_feature_cache"
CONTEXT = ROOT / "checkpoints/fgclip2_contextual_text_v1"
OUT = ROOT / "checkpoints/fgclip2_full_v5_heldout_20260914"
SEEDS = (42, 123, 3407, 2026, 2027)
SHA = shared.cached.file_sha256
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
write = shared.write_json
core = shared.core


def metrics(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if logits.shape != (len(labels), 3) or not np.isfinite(logits).all():
        raise ValueError("Invalid predictions")
    pred = logits.argmax(1)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (labels.astype(int), pred), 1)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(3), where=denom != 0)
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros((3, 3)), where=cm.sum(1, keepdims=True) != 0)
    return dict(n=len(labels), accuracy=float(cm.trace() / cm.sum()), macro_f1=float(f1.mean()),
                per_class_f1=f1.tolist(), confusion_counts=cm.tolist(), confusion_row_normalized=norm.tolist())


def stats(values):
    return dict(values=list(values), mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                n=len(values), unit="training RNG seeds on the same held-out examples")


def planned_models():
    first = ROOT / "checkpoints/fgclip2_full_v5_scale_diagnostic/d9b6a2641cf42689"
    repl = ROOT / "checkpoints/fgclip2_full_v5_seed_replication/26e18a7b6b63e575"
    references = {r["seed"]: Path(r["directory"]) for r in read(first / "diagnostic_protocol.json")["no_cross_references"]}
    control_root = Path(read(repl / "controls_completed.json")["root"])
    rows = []
    for seed in SEEDS:
        manifest = ((first if seed in SEEDS[:3] else repl / "locked_scale_evaluation") /
                    "scaled_manifests" / f"full_v5_seed_{seed}.json")
        m = read(manifest)
        if m["residual_scale"] != .5 or m["seed"] != seed:
            raise ValueError("Locked Full manifest changed")
        control = references[seed] if seed in references else control_root / f"seed_{seed}/no_cross_control"
        for variant, checkpoint, mp in (("full_v5", Path(m["checkpoint_path"]), manifest),
                                        ("no_cross_control", control / "best_head.ckpt", None)):
            rows.append(dict(seed=seed, variant=variant, checkpoint=str(checkpoint), checkpoint_sha256=SHA(checkpoint),
                             manifest=str(mp) if mp else None, manifest_sha256=SHA(mp) if mp else None))
    return rows


def make_plan():
    sources = (Path(__file__).name, "full_model_v5_scaled.py", "full_model_v5.py", "full_model_v4.py",
               "full_model_v2.py", "run_full_model_v2.py", "run_fgclip2_submission_cached_repeated.py",
               Path(core.__file__).name, "train_full_model_v5.py")
    protected = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                 for base in (CACHE, CONTEXT) for p in base.rglob("*") if p.is_file()}
    return dict(format="locked_retrospective_full_v5_test_v1", seeds=list(SEEDS),
                selected_deployment_seed=2027, residual_scale=.5, models=planned_models(),
                tuning=False, test_based_selection=False, independent_confirmatory_test=False,
                independence_limitation="Original test outcomes were inspected in earlier ablations. Internal probe was also inspected and is part of the selected v5 model's training partition.",
                evidence=["checkpoints/fgclip2_submission_v1_core_cached_repeated",
                          "checkpoints/fgclip2_teacher_complementarity_pilot/formal/c3a0bfc5222ae069/independent_audit.json"],
                eval_batch_size=64, context_batch_size=64, prediction_rule="argmax over three uncalibrated logits",
                primary_endpoint="LB-MF1 = (ZH macro-F1 + EN macro-F1)/2; mean and sample SD over all five seeds",
                label_order=["Negative", "Neutral", "Positive"],
                source_sha256={n: SHA(ROOT / n) for n in sources}, protected_cache_stamps=protected,
                base_cache_metadata_sha256=shared.digest(read(CACHE / "cache_ready.json")),
                context_training_metadata_sha256=SHA(CONTEXT / "cache_ready.json"),
                csv_sha256={f"{l}_{s}": SHA(ROOT / f"data_submission_v1/{l}/{s}.csv")
                            for l in ("zh", "en") for s in ("train", "val", "test")},
                torch_version=torch.__version__, numpy_version=np.__version__)


def protect(plan):
    for name, expected in plan["source_sha256"].items():
        if SHA(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    for name, expected in plan["protected_cache_stamps"].items():
        p = Path(name)
        if [p.stat().st_size, p.stat().st_mtime_ns] != expected:
            raise ValueError(f"Existing cache changed: {name}")
    for row in plan["models"]:
        if SHA(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("Protected checkpoint changed")


def verify_splits(splits, full_scan=False):
    meta = read(CACHE / "cache_ready.json")
    frames = {}
    for split in splits:
        lang, partition = split.split("_")
        p = ROOT / f"data_submission_v1/{lang}/{partition}.csv"
        if SHA(p) != meta["csv_sha256"][split]:
            raise ValueError("CSV identity mismatch")
        f = pd.read_csv(p)
        frames[split] = f
        n = len(f)
        shape_specs = {"img_global": ((n, 768), "float16"), "txt_global": ((n, 768), "float16"),
                       "patches": ((n, 256, 768), "float16"), "tokens": ((n, 64, 768), "float16"),
                       "patch_mask": ((n, 256), "uint8"), "attention_mask": ((n, 64), "uint8"),
                       "labels": ((n,), "int64"), "language_ids": ((n,), "uint8")}
        for name, (shape, dtype) in shape_specs.items():
            a = np.load(CACHE / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if a.shape != shape or a.dtype != np.dtype(dtype):
                raise ValueError(f"Bad cache shape/type: {split}/{name}")
            if name == "labels":
                np.testing.assert_array_equal(a, f.label.values)
            if name == "language_ids" and not np.all(a == (lang == "en")):
                raise ValueError("Language mismatch")
            if name.endswith("mask") and (not np.isin(a, [0, 1]).all() or not np.all(a.sum(1) > 0)):
                raise ValueError("Mask mismatch")
            if np.issubdtype(a.dtype, np.floating):
                for start in (range(0, n, 256) if full_scan else [0, n // 2, n - 1]):
                    if not np.isfinite(a[start:start + (256 if full_scan else 1)]).all():
                        raise ValueError(f"Non-finite cache: {split}/{name}")
        print(f"[READ-ONLY CACHE VERIFIED] {split} {n} rows", flush=True)
    return meta, frames


def load_backbone(device="cuda"):
    cfg = read(CONTEXT / "config.json")
    snapshot = Path(snapshot_download(core.MODEL_ID, revision=cfg["model_revision"], local_files_only=True))
    for name, expected in cfg["model_files_sha256"].items():
        if SHA(snapshot / name) != expected:
            raise ValueError(f"Backbone source changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL_ID, revision=cfg["model_revision"],
                                            local_files_only=True, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != cfg["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    model = AutoModelForCausalLM.from_pretrained(core.MODEL_ID, revision=cfg["model_revision"],
                                                local_files_only=True, trust_remote_code=True).to(device).eval()
    core.freeze_backbone(model)
    core.TOKENIZER = tokenizer
    return model, tokenizer, cfg


@torch.inference_mode()
def build_test_context(frames, out):
    target = out / "contextual_test"
    target.mkdir(exist_ok=True)
    ready = target / "cache_ready.json"
    if ready.exists():
        marker = read(ready)
        for split, row in marker["splits"].items():
            if SHA(target / split / "tokens.npy") != row["sha256"]:
                raise ValueError("Committed test context cache changed")
        return target
    model, tokenizer, cfg = load_backbone("cpu")
    encoder = core._find_text_encoder(model).cuda().eval()
    embedding = core.get_text_embedding_layer(model, tokenizer)
    rows = {}
    for split, frame in frames.items():
        directory = target / split
        directory.mkdir(exist_ok=True)
        path, progress = directory / "tokens.npy", directory / "progress.json"
        shape = (len(frame), 64, 768)
        if path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != shape or array.dtype != np.float16:
                raise ValueError("Partial context shape mismatch")
            done = read(progress)["completed_rows"] if progress.exists() else 0
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
            done = 0
        original_mask = np.load(CACHE / split / "attention_mask.npy", mmap_mode="r")
        static = np.load(CACHE / split / "tokens.npy", mmap_mode="r")
        prefix = "Language: Chinese. Text: " if split.startswith("zh") else "Language: English. Text: "
        texts = frame.text.fillna("").astype(str).tolist()
        start_time = time.perf_counter()
        for block in range(done, len(frame), 512):
            end = min(block + 512, len(frame))
            for start in range(block, end, 64):
                stop = min(start + 64, end)
                enc = tokenizer([prefix + t for t in texts[start:stop]], padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt")
                ids = enc["input_ids"].cuda()
                mask = enc.get("attention_mask")
                mask = mask.cuda() if mask is not None else (ids != tokenizer.pad_token_id).long()
                np.testing.assert_array_equal(mask.cpu().numpy(), original_mask[start:stop])
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    values = encoder(input_ids=ids, attention_mask=mask, walk_type=core.WALK_TYPE).last_hidden_state
                    if start == block:
                        np.testing.assert_allclose(embedding(ids).half().cpu().numpy(), static[start:stop], atol=1e-5, rtol=1e-3)
                    values = values.half().cpu().numpy()
                if values.shape != (stop - start, 64, 768) or not np.isfinite(values).all():
                    raise ValueError("Invalid contextual features")
                array[start:stop] = values
            array.flush()
            write(progress, dict(completed_rows=end))
            print(f"[NEW TEST TEXT ONLY] {split} {end}/{len(frame)}", flush=True)
        del array
        rows[split] = dict(rows=len(frame), shape=list(shape), sha256=SHA(path), build_or_resume_seconds=time.perf_counter() - start_time)
    write(ready, dict(feature_source="text_encoder_last_hidden_state", training_context_config=cfg, splits=rows))
    del model, encoder, embedding
    core.cleanup_memory()
    return target


def load_control(row, dims):
    saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if cfg["variant"] != "no_cross_control" or cfg["seed"] != row["seed"] or saved["fingerprint"] != shared.digest(cfg):
        raise ValueError("Control checkpoint identity mismatch")
    model = shared.build_model("no_cross_control", dims, 0.)
    model.load_state_dict(saved["clf_state_dict"], strict=True)
    return model.cuda().eval()


@torch.inference_mode()
def evaluate_models(plan, frames, masks, context, out):
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    index_path = out / "prediction_index.json"
    index = read(index_path) if index_path.exists() else {}
    results = []
    dims = read(CACHE / "cache_ready.json")["dims"]
    for row in plan["models"]:
        model = None
        lang_metrics = {}
        for split, frame in frames.items():
            name = f"{row['variant']}_seed_{row['seed']}_{split}.npz"
            path = prediction_dir / name
            if name in index:
                if SHA(path) != index[name]:
                    raise ValueError("Committed predictions changed")
            else:
                if model is None:
                    model = load_scaled_checkpoint(row["manifest"], "cuda")[0] if row["manifest"] else load_control(row, dims)
                dataset = shared.ContentDataset(CACHE / split, masks[split], contextual_root=context if row["manifest"] else None)
                outputs, labels = [], []
                for batch in shared.loader(dataset, plan["eval_batch_size"]):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits, _, label, _ = shared.cached.forward_cached(model, batch)
                    outputs.append(logits.float().cpu().numpy())
                    labels.append(label.cpu().numpy())
                outputs, labels = np.concatenate(outputs), np.concatenate(labels)
                np.testing.assert_array_equal(labels, frame.label.values)
                metrics(labels, outputs)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, logits=outputs, labels=labels, row_indices=np.arange(len(labels)))
                temporary.replace(path)
                index[name] = SHA(path)
                write(index_path, index)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["labels"], frame.label.values)
                lang_metrics[split[:2]] = metrics(saved["labels"], saved["logits"])
        result = dict(seed=row["seed"], variant=row["variant"], **lang_metrics,
                      lb_mf1=(lang_metrics["zh"]["macro_f1"] + lang_metrics["en"]["macro_f1"]) / 2)
        results.append(result)
        write(out / "test_progress.json", results)
        print(f"[LOCKED HELD-OUT] {row['seed']}/{row['variant']} LB-MF1={result['lb_mf1']:.6f}", flush=True)
        del model
        core.cleanup_memory()
    lookup = {(r["seed"], r["variant"]): r for r in results}
    summary = {v: {k: stats([lookup[s, v][k] if k == "lb_mf1" else lookup[s, v][k[:2]][k[3:]] for s in SEEDS])
                   for k in ("lb_mf1", "zh_macro_f1", "en_macro_f1", "zh_accuracy", "en_accuracy")}
               for v in ("full_v5", "no_cross_control")}
    differences = [lookup[s, "full_v5"]["lb_mf1"] - lookup[s, "no_cross_control"]["lb_mf1"] for s in SEEDS]
    summary.update(paired_full_minus_no_cross=stats(differences), rows=results,
                   deployment=lookup[2027, "full_v5"], selected_deployment_seed=2027,
                   independent_confirmatory_test=False, limitation=plan["independence_limitation"],
                   no_test_based_selection=True, prediction_index_sha256=SHA(index_path))
    write(out / "test_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the peixun CUDA environment")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    pp = out / "evaluation_plan.json"
    if pp.exists() and read(pp) != plan:
        raise ValueError("Locked protocol differs; refusing mixed runs")
    write(pp, plan)
    protect(plan)
    meta, frames = verify_splits(("zh_test", "en_test"), full_scan=True)
    masks, identity = shared.build_content_masks(frames, CACHE)
    write(out / "preflight.json", dict(passed=True, rows={s: len(f) for s, f in frames.items()},
                                      masks=identity, independent_confirmatory_test=False))
    if args.preflight:
        print("[PREFLIGHT OK] Checkpoints and caches verified. No forward/test predictions.")
        return
    lock = out / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        context = build_test_context(frames, out)
        evaluate_models(plan, frames, masks, context, out)
        protect(plan)
        write(out / "completion_audit.json", dict(passed=True, source_and_cache_stamps_unchanged=True,
            selected_checkpoints_unchanged=True, test_summary_sha256=SHA(out / "test_summary.json"),
            evaluation_plan_sha256=SHA(pp), independent_confirmatory_test=False))
        print(f"[COMPLETE] {out / 'test_summary.json'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
