"""Matched inference timing of locked Full-v5 and No-cross (no training).

Both use the same validation inputs, GPU, FP32 weights/FP16 autocast and
synchronised per-request wall timing. Image decoding, tokenisation and H2D
copies are excluded. Backbone+head and cached-head-only scopes are separate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import time

import evaluate_full_v5_heldout as ev
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor

core, shared = ev.core, ev.shared


def gpu_status():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()


def audit_partition(out):
    p = ev.ROOT / "data_submission_v1/split_manifest.csv"
    frame = pd.read_csv(p)
    conflicts = int((frame.groupby("duplicate_group")["split"].nunique() > 1).sum())
    if conflicts:
        raise ValueError("Duplicate groups cross partitions")
    for lang in ("zh", "en"):
        for partition in ("train", "val", "test"):
            csv = pd.read_csv(ev.ROOT / f"data_submission_v1/{lang}/{partition}.csv").fillna("")
            subset = frame[(frame.language == lang) & (frame.split == partition)].fillna("")
            for col in ("image_path", "text", "label"):
                if sorted(csv[col].astype(str)) != sorted(subset[col].astype(str)):
                    raise ValueError("Split manifest does not match CSV")
    evidence = sorted((ev.ROOT / "checkpoints/fgclip2_submission_v1_core_cached_repeated").glob("seed_*/**/test_metrics.json"))
    report = dict(manifest_sha256=ev.SHA(p), duplicate_groups_crossing_splits=conflicts,
                  split_counts=frame.groupby(["language", "split"]).size().to_dict(),
                  previously_saved_test_evaluations=[str(p) for p in evidence],
                  independent_confirmatory_test=False,
                  interpretation="No detected grouped overlap, but test results were exposed during previous model development. This is retrospective weight-held-out evaluation.")
    report["split_counts"] = {"/".join(k): int(v) for k, v in report["split_counts"].items()}
    ev.write(out / "data_exposure_audit.json", report)


def extract_once(backbone, static_embedding, batch, need_context):
    pv, pm, ss, ids, am, cm, languages = batch
    vision = core._call_visual_encoder_once(core._find_visual_encoder(backbone), pv, pm, ss)
    img = core._pick_pooled_tensor(vision)
    patches = core._pick_hidden_patch_tensor(vision)
    if patches is None or img is None:
        raise RuntimeError("Visual feature fallback is disabled")
    if patches.shape[1] == pm.shape[1] + 1:
        patches = patches[:, 1:]
    if patches.shape[1] != pm.shape[1]:
        raise ValueError("Patch grid mismatch")
    text = core._call_text_encoder_once(core._find_text_encoder(backbone), ids, am)
    txt = core._apply_text_pooling_head(backbone, core._pick_pooled_tensor(text))
    static = static_embedding(ids).half()
    tokens = torch.cat([static, text.last_hidden_state.half()], -1) if need_context else static
    # Mirror the cached inference contract (FP16 stored backbone features).
    return img.half(), txt.half(), patches.half(), pm.bool(), tokens, cm, languages


def head_forward(model, features):
    return model(*features[:6], language_ids=features[6], return_aux=False)


def gpu_batch(batch):
    return tuple(t.cuda() for t in batch)


@torch.inference_mode()
def validate_inference(plan, out, backbone, tokenizer, processor):
    meta, frames = ev.verify_splits(("zh_val", "en_val"))
    masks, _ = shared.build_content_masks(frames, ev.CACHE)
    selected = {r["variant"]: r for r in plan["models"] if r["seed"] == 2027}
    models = {"full_v5": ev.load_scaled_checkpoint(selected["full_v5"]["manifest"], "cuda")[0],
              "no_cross_control": ev.load_control(selected["no_cross_control"], meta["dims"])}
    raw_by_lang, cached_by_lang, checks = {}, {}, []
    embedding = core.get_text_embedding_layer(backbone, tokenizer)
    for lang in ("zh", "en"):
        split = f"{lang}_val"
        ds = core.ImageTextDataset(str(ev.ROOT / f"data_submission_v1/{lang}/val.csv"), processor, tokenizer, lang)
        raw = [ds[i] for i in range(64)]
        collated = core.collate_fn(raw)
        pv, pm, ss, ids, am, _, languages = collated
        raw_by_lang[lang] = (pv, pm, ss, ids, am, torch.from_numpy(masks[split][:64].copy()).long(), languages)
        cached_ds = shared.ContentDataset(ev.CACHE / split, masks[split], contextual_root=ev.CONTEXT)
        cb = shared.cached.cached_collate([cached_ds[i] for i in range(64)])
        cached_by_lang[lang] = (*cb[:6], cb[7])
        # Check the actual fixed-scale loader against committed validation logits.
        base_dir = Path(selected["full_v5"]["manifest"]).parent.parent
        component = np.load(base_dir / f"full_v5_seed_2027_{lang}_val_components.npz")
        expected = component["base_logits"][:64].astype(np.float32) + .5 * component["correction_logits"][:64].astype(np.float32)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            actual = head_forward(models["full_v5"], gpu_batch(cached_by_lang[lang])).cpu().float().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        # Online one-pass extraction is checked at the profiling batch size.
        for start in range(0, 64, 16):
            inputs = gpu_batch(tuple(t[start:start + 16] for t in raw_by_lang[lang]))
            cached = gpu_batch(tuple(t[start:start + 16] for t in cached_by_lang[lang]))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                online = extract_once(backbone, embedding, inputs, True)
                feature_errors = {}
                for name, x, y in zip(("img_global", "txt_global", "patches", "mask", "tokens"), online[:5], cached[:5]):
                    if x.dtype == torch.bool:
                        assert torch.equal(x, y)
                    else:
                        err = (x.float() - y.float()).abs()
                        feature_errors[name] = dict(max_abs=float(err.max()), mean_abs=float(err.mean()))
                for variant, model in models.items():
                    a, b = list(online), list(cached)
                    if variant == "no_cross_control":
                        a[4], b[4] = a[4][..., :768], b[4][..., :768]
                    x = head_forward(model, a).float().cpu().numpy()
                    y = head_forward(model, b).float().cpu().numpy()
                    np.testing.assert_allclose(x, y, atol=.02, rtol=.02)
                    np.testing.assert_array_equal(x.argmax(1), y.argmax(1))
                    checks.append(dict(language=lang, start=start, variant=variant,
                                       logits_max_abs=float(np.max(np.abs(x - y))), predictions_identical=True,
                                       feature_error=feature_errors))
        print(f"[VALIDATED] {lang}: fixed-scale logits match saved predictions; online/cache predictions agree", flush=True)
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    for model in models.values():
        model.cpu()
    del models, embedding, online, cached, inputs, a, b, cb, cached_ds, x, y
    gc.collect()
    torch.cuda.empty_cache()
    ev.write(out / "inference_validation.json", dict(passed=True, checks=checks, validation_examples=128,
             checkpoint_logit_atol=1e-6, online_cache_atol=.02, online_cache_rtol=.02,
             online_cache_note="Small FP16 batch-size/kernel differences allowed; all validation predictions must agree.",
             head_parameters=counts, backbone_parameters=sum(p.numel() for p in backbone.parameters())))
    return raw_by_lang, cached_by_lang


def batch_bank(by_lang, batch_size, full, raw):
    result = []
    # Balanced language representation; no selection by sentiment/correctness.
    rows = 16 if batch_size == 1 else 64
    for start in range(0, rows, batch_size):
        for lang in ("zh", "en"):
            batch = tuple(t[start:start + batch_size] for t in by_lang[lang])
            if not raw and not full:
                batch = (*batch[:4], batch[4][..., :768].contiguous(), *batch[5:])
            result.append(batch)
    return result


@torch.inference_mode()
def measure(model, backbone, tokenizer, bank, scope, full, round_index):
    embedding = core.get_text_embedding_layer(backbone, tokenizer) if backbone is not None else None
    def forward(batch):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            features = extract_once(backbone, embedding, batch, full) if scope == "backbone_plus_head" else batch
            return head_forward(model, features)
    # Inputs transferred before timing; one active batch, not an artificial GPU
    # bank inflating peak memory. Every request synchronised for wall latency.
    for i in range(20):
        batch = gpu_batch(bank[i % len(bank)])
        y = forward(batch)
        del y, batch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    before = gpu_status()
    for i in range(64):
        batch = gpu_batch(bank[i % len(bank)])
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = forward(batch)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        del y, batch
    b = bank[0][0].shape[0]
    return dict(round=round_index, batch_size=b, scope=scope, variant="full_v5" if full else "no_cross_control",
                latency_ms_per_batch=float(np.mean(timings)), latency_ms_per_sample_amortized=float(np.mean(timings) / b),
                throughput_samples_per_second=float(1000 * b / np.mean(timings)), request_ms=timings,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                device_status_before=before, device_status_after=gpu_status())


def summarize(rows):
    summaries = []
    for scope in ("cached_head_only", "backbone_plus_head"):
        for b in (1, 16):
            for variant in ("full_v5", "no_cross_control"):
                r = [x for x in rows if x["scope"] == scope and x["batch_size"] == b and x["variant"] == variant]
                if len(r) != 5:
                    raise ValueError("Incomplete timing rounds")
                entry = dict(scope=scope, batch_size=b, variant=variant, n_rounds=5, requests_per_round=64)
                for key in ("latency_ms_per_batch", "latency_ms_per_sample_amortized", "throughput_samples_per_second"):
                    values = [x[key] for x in r]
                    entry[key] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
                entry["peak_allocated_mib"] = max(x["peak_allocated_mib"] for x in r)
                summaries.append(entry)
    return summaries


def training_log_times(plan):
    result = []
    for row in plan["models"]:
        p = Path(row["checkpoint"]).parent / "run_result.json"
        saved = ev.read(p)
        result.append(dict(seed=row["seed"], variant=row["variant"], seconds=saved["training_seconds"],
                           source=str(p), source_sha256=ev.SHA(p)))
    return dict(rows=result, scope="Historical refinement wall time including epoch-zero/epoch validation; excludes parent training and feature extraction. Not a fresh controlled training benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ev.OUT / "efficiency")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = ev.read(ev.OUT / "evaluation_plan.json")
    ev.protect(plan)
    cfg = dict(format="full_v5_matched_efficiency_v1", evaluation_plan_sha256=ev.SHA(ev.OUT / "evaluation_plan.json"),
               profiler_sha256=ev.SHA(Path(__file__)), seed=2027, batch_sizes=[1, 16], warmup_requests=20,
               rounds=5, timed_requests_per_round=64, validation_rows_per_language=64,
               precision="FP32 weights, FP16 CUDA autocast, FP32 final Full residual addition",
               timing="per-request perf_counter wall time, CUDA synchronized; CPU model dispatch included",
               excluded=["disk/image decoding", "image preprocessing", "tokenization and content mask construction", "host-to-device transfer", "network/request overhead"],
               environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                                cpu_threads=torch.get_num_threads(), gpu_status=gpu_status()),
               hardware_limitation="Windows WDDM display GPU shared with desktop applications; no clock lock. Timing rounds are technical repeats, not independent model seeds.")
    ev.write(out / "profile_protocol.json", cfg)
    audit_partition(out)
    backbone, tokenizer, bcfg = ev.load_backbone()
    processor = AutoImageProcessor.from_pretrained(core.MODEL_ID, revision=bcfg["model_revision"], trust_remote_code=True, local_files_only=True)
    raw, cached = validate_inference(plan, out, backbone, tokenizer, processor)
    if args.validate_only:
        print("[VALIDATION COMPLETE] No test forward or timing run")
        return
    selected = {r["variant"]: r for r in plan["models"] if r["seed"] == 2027}
    rows = []
    dims = ev.read(ev.CACHE / "cache_ready.json")["dims"]
    for scope in ("cached_head_only", "backbone_plus_head"):
        backbone.to("cpu" if scope == "cached_head_only" else "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        for b in (1, 16):
            for r in range(5):
                # Alternate paired ordering to reduce warm-device/order bias.
                for variant in (("full_v5", "no_cross_control") if r % 2 == 0 else ("no_cross_control", "full_v5")):
                    full = variant == "full_v5"
                    model = ev.load_scaled_checkpoint(selected[variant]["manifest"], "cuda")[0] if full else ev.load_control(selected[variant], dims)
                    model.requires_grad_(False)
                    bank = batch_bank(raw if scope == "backbone_plus_head" else cached, b, full, scope == "backbone_plus_head")
                    value = measure(model, backbone if scope == "backbone_plus_head" else None, tokenizer, bank, scope, full, r)
                    rows.append(value)
                    ev.write(out / "timing_raw.json", rows)
                    print(f"[TIMED] {scope} B{b} round{r+1} {variant}: {value['latency_ms_per_batch']:.3f} ms/batch", flush=True)
                    del model, bank
                    gc.collect()
                    torch.cuda.empty_cache()
    ev.protect(plan)
    summary = dict(protocol=cfg, rows=summarize(rows), parameters=ev.read(out / "inference_validation.json"),
                   historical_refinement_times=training_log_times(plan), timing_raw_sha256=ev.SHA(out / "timing_raw.json"),
                   all_selected_checkpoints_unchanged=True)
    ev.write(out / "efficiency_summary.json", summary)
    print(f"[COMPLETE] {out / 'efficiency_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
