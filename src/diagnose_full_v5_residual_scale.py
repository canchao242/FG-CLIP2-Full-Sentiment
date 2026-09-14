"""Fixed five-value residual-scale diagnostic; no training or test access.

Original v5 checkpoints and independent No-cross controls are read-only.
Alpha is shared across seeds/languages for each arm, never selected per sample.
"""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import train_full_model_v5 as runner

ALPHAS = (0., .1, .25, .5, 1.)
VARIANTS = ("full_v5", "pooled_v5")


def validate_arrays(base, delta, labels):
    if base.ndim != 2 or base.shape[1] != 3 or delta.shape != base.shape:
        raise ValueError("Expected matching [N, 3] base/correction logits")
    if labels.shape != (len(base),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Invalid label shape/type")
    if len(base) == 0 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Empty or invalid labels")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError("Non-finite cached logits")


def scaled_logits(base, delta, alpha):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must lie in [0,1]")
    return base.astype(np.float32) + np.float32(alpha) * delta.astype(np.float32)


def score_language(base, delta, labels, alpha):
    validate_arrays(base, delta, labels)
    logits = scaled_logits(base, delta, alpha)
    predictions = logits.argmax(-1)
    base_predictions = base.argmax(-1)
    return dict(n=len(labels), macro_f1=float(f1_score(labels, predictions, labels=[0,1,2], average="macro", zero_division=0)),
                accuracy=float(accuracy_score(labels,predictions)),
                changed_from_alpha_zero=int((predictions!=base_predictions).sum()),
                fixed_vs_alpha_zero=int(((predictions==labels)&(base_predictions!=labels)).sum()),
                broken_vs_alpha_zero=int(((predictions!=labels)&(base_predictions==labels)).sum()))


def statistics(values):
    values = np.asarray(values,dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite paired seed results required")
    return dict(values=values.tolist(), mean=float(values.mean()), sample_std=float(values.std(ddof=1)))


def summarize(rows, references):
    seeds = sorted(r["seed"] for r in references)
    expected = {(v,s,a) for v in VARIANTS for s in seeds for a in ALPHAS}
    lookup = {(r["variant"],r["seed"],r["alpha"]):r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != expected or len(seeds) != len(set(seeds)):
        raise ValueError("Missing, duplicate or unexpected grid entries")
    control = {r["seed"]:r["score"] for r in references}
    result = dict(validation_only=True, test_evaluated=False, alphas=list(ALPHAS), seeds=seeds,
                  selection_rule="One global alpha per arm maximizing mean validation LB-MF1 across all seeds; ties prefer smaller alpha.",
                  no_cross_control=statistics([control[s] for s in seeds]), grid=[], selected={})
    for alpha in ALPHAS:
        row = dict(alpha=alpha)
        for variant in VARIANTS:
            values = [lookup[(variant,s,alpha)]["lb_mf1"] for s in seeds]
            row[variant] = statistics(values)
            row[f"{variant}_minus_original_no_cross"] = statistics([v-control[s] for s,v in zip(seeds,values)])
            row[f"{variant}_minus_same_checkpoint_alpha_zero"] = statistics([
                v-lookup[(variant,s,0.)]["lb_mf1"] for s,v in zip(seeds,values)])
            row[f"{variant}_seeds_with_changed_predictions"] = sum(
                lookup[(variant,s,alpha)]["prediction_changes"] > 0 for s in seeds)
        row["full_minus_pooled"] = statistics([
            lookup[("full_v5",s,alpha)]["lb_mf1"]-lookup[("pooled_v5",s,alpha)]["lb_mf1"] for s in seeds])
        result["grid"].append(row)
    for variant in VARIANTS:
        best = max(result["grid"],key=lambda row:(row[variant]["mean"],-row["alpha"]))
        result["selected"][variant] = dict(alpha=best["alpha"],**best[variant],
            branch_disabled=best["alpha"]==0, optimistic_validation_selection=True,
            seeds_with_changed_predictions=best[f"{variant}_seeds_with_changed_predictions"])
    return result


@torch.inference_mode()
def collect(model, dataset, batch_size):
    buckets = {"base_logits":[],"correction_logits":[],"labels":[]}
    batches = runner.shared.loader(dataset,batch_size)
    for batch in batches:
        with torch.amp.autocast("cuda",enabled=runner.core.DEVICE=="cuda",dtype=torch.float16):
            full,aux,labels,_ = runner.cached.forward_cached(model,batch)
        np.testing.assert_allclose(full.float().cpu().numpy(),
            scaled_logits(aux["base_logits"].float().cpu().numpy(),aux["correction_logits"].float().cpu().numpy(),1.),
            atol=0,rtol=0)
        buckets["base_logits"].append(aux["base_logits"].float().cpu())
        buckets["correction_logits"].append(aux["correction_logits"].float().cpu())
        buckets["labels"].append(labels.cpu())
    return {name:torch.cat(parts).numpy() for name,parts in buckets.items()}


def load_verified_components(path, reference_path):
    with np.load(path,allow_pickle=False) as archive:
        values = {k:archive[k] for k in ("base_logits","correction_logits","labels")}
    validate_arrays(values["base_logits"],values["correction_logits"],values["labels"])
    with np.load(reference_path,allow_pickle=False) as original:
        np.testing.assert_array_equal(values["labels"],original["labels"])
        full = scaled_logits(values["base_logits"],values["correction_logits"],1.)
        np.testing.assert_array_equal(full.argmax(-1),original["preds"])
        np.testing.assert_allclose(full,original["logits"],atol=1e-6,rtol=1e-6)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=runner.ROOT/"checkpoints/fgclip2_full_v5_development/68dcdbd16a3a9bfe")
    parser.add_argument("--save-root",type=Path,default=runner.ROOT/"checkpoints/fgclip2_full_v5_scale_diagnostic")
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    protocol = json.loads((source/"protocol.json").read_text(encoding="utf-8"))
    if protocol["limit_samples"] or set(protocol["variants"]) != set(VARIANTS) or len(protocol["seeds"]) < 2:
        parser.error("Requires the complete, paired full-data v5 development protocol")
    runner.core.DEVICE = protocol["device"]
    if runner.core.DEVICE != "cuda" or not torch.cuda.is_available() or torch.__version__ != protocol["torch_version"]:
        parser.error("Use the original peixun CUDA runtime for exact prediction verification")
    torch.set_num_threads(2)
    cache = runner.ROOT/"checkpoints/fgclip2_submission_v1_feature_cache"
    context = runner.ROOT/"checkpoints/fgclip2_contextual_text_v1"
    for protected in (source,cache,context,runner.ROOT/"data_submission_v1"):
        if args.save_root.resolve().is_relative_to(protected):
            parser.error("Diagnostic output must be separate from source results, data and feature caches")
    for name,expected in protocol["source_sha256"].items():
        if runner.cached.file_sha256(runner.ROOT/name) != expected:
            raise ValueError(f"Source code differs from training snapshot: {name}")
    references = protocol["no_cross_references"]
    for reference in references:
        for name,expected in reference["file_sha256"].items():
            if runner.cached.file_sha256(Path(reference["directory"])/name) != expected:
                raise ValueError("Original No-cross comparator changed")
    meta,frames = runner.shared.verify_cache(cache,runner.ROOT/"data_submission_v1")
    context_meta = runner.verify_context_cache(context,meta)
    masks,mask_identity = runner.shared.build_content_masks(frames,cache)
    if runner.shared.digest(meta) != protocol["cache_metadata_sha256"] or runner.shared.digest(context_meta) != protocol["context_cache_metadata_sha256"]:
        raise ValueError("Cache identity mismatch")
    if mask_identity["content_mask_sha256"] != protocol["content_mask_sha256"]:
        raise ValueError("Content mask identity mismatch")
    source_files = {str(source/"protocol.json"):runner.cached.file_sha256(source/"protocol.json")}
    for seed in protocol["seeds"]:
        for variant in VARIANTS:
            directory = source/f"seed_{seed}"/variant
            for name in ("config.json","run_result.json","best_head.ckpt","best_zh_val_outputs.npz","best_en_val_outputs.npz"):
                source_files[str(directory/name)] = runner.cached.file_sha256(directory/name)
    plan = dict(version="fixed_five_scales_v1",alphas=list(ALPHAS),source_root=str(source),source_files_sha256=source_files,
                code_sha256=runner.cached.file_sha256(Path(__file__)),torch_version=torch.__version__,
                no_cross_references=references,validation_only=True,test_evaluated=False,
                checkpoint_selection="Existing alpha=1 best checkpoints are fixed; no epoch reselection or retraining",
                alpha_selection="Global per arm across seeds and languages; identical five-value budget for Full and pooled")
    out = args.save_root.resolve()/runner.shared.digest(plan)[:16]
    print(f"[OUTPUT] {out}\n[DIAGNOSTIC] fixed checkpoints; alpha={ALPHAS}; no training",flush=True)
    if args.preflight:
        print("[PREFLIGHT OK] Nothing written; no model forward run",flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    lock = out/"RUNNING.lock"
    with lock.open("x",encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        plan_path = out/"diagnostic_protocol.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Mixed diagnostic protocol")
        runner.shared.write_json(plan_path,plan)
        snapshot = out/Path(__file__).name
        if not snapshot.exists():
            shutil.copyfile(Path(__file__),snapshot)
        if runner.cached.file_sha256(snapshot) != plan["code_sha256"]:
            raise ValueError("Diagnostic snapshot mismatch")
        index_path = out/"components_index.json"
        components_manifest = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        rows = []
        for variant in VARIANTS:
            for seed in protocol["seeds"]:
                directory = source/f"seed_{seed}"/variant
                ckpt = torch.load(directory/"best_head.ckpt",map_location="cpu",weights_only=False)
                config = dict(protocol=protocol,variant=variant,seed=seed)
                if ckpt["config"] != config or ckpt["fingerprint"] != runner.shared.digest(config):
                    raise ValueError("Checkpoint/config mismatch")
                model = None
                per_language = {}
                for lang in ("zh","en"):
                    path = out/f"{variant}_seed_{seed}_{lang}_val_components.npz"
                    if path.name in components_manifest and (not path.exists() or runner.cached.file_sha256(path) != components_manifest[path.name]):
                        raise ValueError("Previously verified component cache changed")
                    if path.name not in components_manifest:
                        if model is None:
                            model = runner.build_model(variant,meta["dims"]).to(runner.core.DEVICE).eval()
                            model.load_state_dict(ckpt["clf_state_dict"])
                        dataset = runner.shared.ContentDataset(cache/f"{lang}_val",masks[f"{lang}_val"],contextual_root=context)
                        values = collect(model,dataset,protocol["eval_batch_size"])
                        temporary = path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle,**values)
                        temporary.replace(path)
                        del dataset
                    values = load_verified_components(path,directory/f"best_{lang}_val_outputs.npz")
                    components_manifest[path.name] = runner.cached.file_sha256(path)
                    runner.shared.write_json(index_path,components_manifest)
                    per_language[lang] = {alpha:score_language(values["base_logits"],values["correction_logits"],values["labels"],alpha) for alpha in ALPHAS}
                for alpha in ALPHAS:
                    zh,en = per_language["zh"][alpha],per_language["en"][alpha]
                    rows.append(dict(variant=variant,seed=seed,alpha=alpha,selected_epoch=ckpt["metrics"]["epoch"],
                                     lb_mf1=(zh["macro_f1"]+en["macro_f1"])/2,zh=zh,en=en,
                                     prediction_changes=zh["changed_from_alpha_zero"]+en["changed_from_alpha_zero"]))
                print(f"[{variant}/{seed}] " + ", ".join(f"a={r['alpha']:g}: {r['lb_mf1']:.6f}" for r in rows[-len(ALPHAS):]),flush=True)
                del model,ckpt
                runner.core.cleanup_memory()
        summary = summarize(rows,references)
        for path,expected in source_files.items():
            if runner.cached.file_sha256(Path(path)) != expected:
                raise RuntimeError("A source experiment file changed during the diagnostic")
        for reference in references:
            for name,expected in reference["file_sha256"].items():
                if runner.cached.file_sha256(Path(reference["directory"])/name) != expected:
                    raise RuntimeError("No-cross changed during the diagnostic")
        summary.update(original_experiment_files_unchanged=True,original_no_cross_files_unchanged=True,
                       components_sha256=components_manifest,rows=rows)
        runner.shared.write_json(out/"scale_summary.json",summary)
        print(json.dumps({"selected":summary["selected"],"no_cross":summary["no_cross_control"]},indent=2),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
