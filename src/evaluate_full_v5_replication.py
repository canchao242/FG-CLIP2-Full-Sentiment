"""Evaluate the prelocked scales on new seeds, without scale/epoch reselection.

All outputs are development-validation results. No test files are opened.
"""

import argparse
from pathlib import Path
import shutil
import os

import numpy as np
import torch

import run_full_v5_replication as replication
import train_full_model_v5 as runner
import diagnose_full_v5_residual_scale as diagnostic
from audit_full_model_v5 import audit
from full_model_v5_scaled import load_scaled_checkpoint


def summarize(rows, seeds):
    variants = ("no_cross_control", "full_v5", "pooled_v5")
    lookup = {(r["variant"], r["seed"]): r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {(v, s) for v in variants for s in seeds}:
        raise ValueError("Incomplete or duplicate paired results")
    result = dict(seeds=list(seeds), validation_only=True, test_evaluated=False,
                  fixed_scales=replication.SCALES, scale_search_on_replication=False,
                  variants={}, paired={})
    for variant in variants:
        result["variants"][variant] = diagnostic.statistics([lookup[variant, seed]["lb_mf1"] for seed in seeds])
    for a, b in (("full_v5", "no_cross_control"), ("pooled_v5", "no_cross_control"), ("full_v5", "pooled_v5")):
        values = [lookup[a, s]["lb_mf1"] - lookup[b, s]["lb_mf1"] for s in seeds]
        result["paired"][f"{a}_minus_{b}"] = dict(**diagnostic.statistics(values),
            wins=sum(x > 1e-12 for x in values), ties=sum(abs(x) <= 1e-12 for x in values),
            losses=sum(x < -1e-12 for x in values))
    for variant in ("full_v5", "pooled_v5"):
        result["variants"][variant]["nonzero_selected_branches"] = sum(lookup[variant, s]["branch_has_nonzero_output"] for s in seeds)
        result["variants"][variant]["same_checkpoint_alpha_zero"] = diagnostic.statistics([lookup[variant, s]["alpha_zero_lb_mf1"] for s in seeds])
        result["variants"][variant]["unscaled_alpha_one"] = diagnostic.statistics([lookup[variant, s]["alpha_one_lb_mf1"] for s in seeds])
    return result


def validate_plan(plan):
    if plan["seeds"] != list(replication.SEEDS) or plan["fixed_scales"] != replication.SCALES:
        raise ValueError("Seeds or scales differ from the locked replication")
    if set(plan["seeds"]) & set(plan["development_selection_seeds"]):
        raise ValueError("A replication seed was used for scale selection")
    if plan["scale_search_on_replication"] or plan["test_evaluated"] or plan["feature_extraction"]:
        raise ValueError("Not the predeclared validation-only experiment")
    replication.verify_protected(plan)


@torch.inference_mode()
def verify_scaled_model(manifest_path, expected, cache, context, masks, batch_size):
    model, manifest = load_scaled_checkpoint(manifest_path, "cuda")
    for lang in ("zh", "en"):
        dataset = runner.shared.ContentDataset(cache / f"{lang}_val", masks[f"{lang}_val"], contextual_root=context)
        logits, labels = [], []
        for batch in runner.shared.loader(dataset, batch_size):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred, _, label, _ = runner.cached.forward_cached(model, batch)
            logits.append(pred.float().cpu())
            labels.append(label.cpu())
        values = expected[lang]
        target = diagnostic.scaled_logits(values["base_logits"], values["correction_logits"], manifest["residual_scale"])
        actual = torch.cat(logits).numpy()
        np.testing.assert_array_equal(torch.cat(labels).numpy(), values["labels"])
        np.testing.assert_array_equal(actual.argmax(-1), target.argmax(-1))
        np.testing.assert_allclose(actual, target, atol=1e-6, rtol=1e-6)
        del dataset
    del model
    runner.core.cleanup_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "RUNNING.lock").exists():
        raise RuntimeError("Training may still be active; wait for its lock to be released before evaluation")
    plan = replication.read(root / "replication_plan.json")
    validate_plan(plan)
    source = replication.find_protocol(root / "v5")
    protocol = replication.read(source / "protocol.json")
    if protocol["seeds"] != plan["seeds"] or protocol["limit_samples"]:
        raise ValueError("Training protocol differs from the predeclared seeds/data")
    training_audit = audit(source)
    if not all(training_audit["current_sources_match_training_fingerprints"].values()):
        raise ValueError("Training implementation changed")
    if not torch.cuda.is_available() or torch.__version__ != protocol["torch_version"]:
        raise ValueError("Use the original peixun CUDA runtime")
    runner.core.DEVICE = "cuda"
    torch.set_num_threads(2)
    cache = runner.ROOT / "checkpoints/fgclip2_submission_v1_feature_cache"
    context = runner.ROOT / "checkpoints/fgclip2_contextual_text_v1"
    meta, frames = runner.shared.verify_cache(cache, runner.ROOT / "data_submission_v1")
    context_meta = runner.verify_context_cache(context, meta)
    masks, mask_identity = runner.shared.build_content_masks(frames, cache)
    if runner.shared.digest(meta) != protocol["cache_metadata_sha256"] or runner.shared.digest(context_meta) != protocol["context_cache_metadata_sha256"]:
        raise ValueError("Feature cache changed")
    if mask_identity["content_mask_sha256"] != protocol["content_mask_sha256"]:
        raise ValueError("Content masks changed")
    out = root / "locked_scale_evaluation"
    out.mkdir(exist_ok=True)
    lock = out / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        evaluation_plan = dict(replication_plan_sha256=replication.sha(root / "replication_plan.json"),
            training_protocol_sha256=replication.sha(source / "protocol.json"), scales=plan["fixed_scales"],
            seeds=plan["seeds"], source_sha256={name: replication.sha(runner.ROOT / name) for name in
                (Path(__file__).name, "diagnose_full_v5_residual_scale.py", "audit_full_model_v5.py", "full_model_v5_scaled.py")},
            scale_search=False, epoch_reselection=False, diagnostic_only_scales=[0., 1.])
        if (out / "evaluation_plan.json").exists() and replication.read(out / "evaluation_plan.json") != evaluation_plan:
            raise ValueError("Refusing mixed evaluation implementations")
        replication.write(out / "evaluation_plan.json", evaluation_plan)
        snapshot = out / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name, expected in evaluation_plan["source_sha256"].items():
            if not (snapshot / name).exists():
                shutil.copyfile(runner.ROOT / name, snapshot / name)
            if replication.sha(snapshot / name) != expected:
                raise ValueError("Evaluation snapshot mismatch")
        replication.write(out / "training_audit.json", training_audit)
        index = replication.read(out / "components_index.json") if (out / "components_index.json").exists() else {}
        rows = [dict(variant="no_cross_control", seed=r["seed"], lb_mf1=r["val_lb_mf1"],
                     selected_epoch=r["selected_epoch"], zh_mf1=r["zh_val_mf1"], en_mf1=r["en_val_mf1"])
                for r in training_audit["runs"] if r["variant"] == "no_cross_control_reused"]
        manifests = out / "scaled_manifests"
        manifests.mkdir(exist_ok=True)
        for seed in plan["seeds"]:
            for variant, alpha in plan["fixed_scales"].items():
                directory = source / f"seed_{seed}" / variant
                checkpoint = directory / "best_head.ckpt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                config = dict(protocol=protocol, variant=variant, seed=seed)
                if saved["config"] != config or saved["fingerprint"] != runner.shared.digest(config):
                    raise ValueError("Checkpoint identity mismatch")
                model, components, by_language = None, {}, {}
                for lang in ("zh", "en"):
                    path = out / f"{variant}_seed_{seed}_{lang}_val_components.npz"
                    if path.name in index and (not path.exists() or replication.sha(path) != index[path.name]):
                        raise ValueError("Previously committed component output changed")
                    if path.name not in index:
                        if model is None:
                            model = runner.build_model(variant, meta["dims"]).cuda().eval()
                            model.load_state_dict(saved["clf_state_dict"], strict=True)
                        dataset = runner.shared.ContentDataset(cache / f"{lang}_val", masks[f"{lang}_val"], contextual_root=context)
                        values = diagnostic.collect(model, dataset, protocol["eval_batch_size"])
                        temporary = path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, **values)
                        temporary.replace(path)
                        del dataset
                    values = diagnostic.load_verified_components(path, directory / f"best_{lang}_val_outputs.npz")
                    components[lang] = values
                    index[path.name] = replication.sha(path)
                    replication.write(out / "components_index.json", index)
                    # Endpoints are reported for diagnosis only; NEVER selected as fallback.
                    by_language[lang] = {key: diagnostic.score_language(values["base_logits"], values["correction_logits"], values["labels"], scale)
                                         for key, scale in (("locked", alpha), ("zero", 0.), ("one", 1.))}
                del model
                runner.core.cleanup_memory()
                manifest = dict(format="fgclip2_v5_fixed_scale_v1", variant=variant, seed=seed,
                    checkpoint_path=str(checkpoint), checkpoint_sha256=replication.sha(checkpoint),
                    residual_scale=alpha, feature_dims=meta["dims"], cache_metadata_sha256=protocol["cache_metadata_sha256"],
                    selected_epoch=saved["metrics"]["epoch"], zero_initial_branch=saved["metrics"]["epoch"] == 0,
                    selected_on_validation=True, scale_selected_on_replication=False, test_evaluated=False,
                    selection_rule=plan["checkpoint_selection"], replication_plan_sha256=evaluation_plan["replication_plan_sha256"])
                path = manifests / f"{variant}_seed_{seed}.json"
                if path.exists() and replication.read(path) != manifest:
                    raise ValueError("Scale manifest changed")
                replication.write(path, manifest)
                verify_scaled_model(path, components, cache, context, masks, protocol["eval_batch_size"])
                score = lambda key: (by_language["zh"][key]["macro_f1"] + by_language["en"][key]["macro_f1"]) / 2
                rows.append(dict(variant=variant, seed=seed, alpha=alpha, lb_mf1=score("locked"),
                    alpha_zero_lb_mf1=score("zero"), alpha_one_lb_mf1=score("one"),
                    zh=by_language["zh"], en=by_language["en"], selected_epoch=manifest["selected_epoch"],
                    zero_initial_branch=manifest["zero_initial_branch"], complete_scaled_forward_verified=True,
                    branch_has_nonzero_output=any(bool(np.any(values["correction_logits"] != 0)) for values in components.values()),
                    checkpoint_sha256=manifest["checkpoint_sha256"]))
                replication.write(out / "progress_rows.json", rows)
                print(f"[LOCKED VERIFIED] {seed}/{variant}: alpha={alpha:g}, LB-MF1={score('locked'):.6f}, epoch={manifest['selected_epoch']}", flush=True)
                del saved, components
        report = summarize(rows, plan["seeds"])
        report.update(rows=rows, complete_validation_forward_verified=True, components_sha256=index,
            limitations=[plan["interpretation"], plan["comparison_limit"],
                        "Only two new RNG seeds; development validation data were already reused extensively.",
                        "Scale remains fixed even if its new-seed result is worse than an endpoint."])
        validate_plan(plan)
        for name, expected in evaluation_plan["source_sha256"].items():
            if replication.sha(runner.ROOT / name) != expected:
                raise ValueError("Evaluation source changed during execution")
        replication.write(out / "replication_summary.json", report)
        print(f"[COMPLETE] {out / 'replication_summary.json'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
