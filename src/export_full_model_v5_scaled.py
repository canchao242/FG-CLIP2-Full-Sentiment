"""Export separate scale manifests and optionally verify every validation output.

No checkpoint is overwritten; no model is retrained. These are calibration
candidates selected on development validation, not new independent results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_full_v5_residual_scale as diagnostic
from full_model_v5_scaled import load_scaled_checkpoint
import train_full_model_v5 as runner


@torch.inference_mode()
def export(root, verify=False):
    summary = json.loads((root/"scale_summary.json").read_text(encoding="utf-8"))
    plan = json.loads((root/"diagnostic_protocol.json").read_text(encoding="utf-8"))
    source = Path(plan["source_root"])
    protocol = json.loads((source/"protocol.json").read_text(encoding="utf-8"))
    for path,expected in plan["source_files_sha256"].items():
        if runner.cached.file_sha256(Path(path)) != expected:
            raise ValueError("Original experiment file changed")
    for ref in protocol["no_cross_references"]:
        for name,expected in ref["file_sha256"].items():
            if runner.cached.file_sha256(Path(ref["directory"])/name) != expected:
                raise ValueError("Original No-cross comparator changed")
    cache = runner.ROOT/"checkpoints/fgclip2_submission_v1_feature_cache"
    context = runner.ROOT/"checkpoints/fgclip2_contextual_text_v1"
    meta = json.loads((cache/"cache_ready.json").read_text(encoding="utf-8"))
    if runner.shared.digest(meta) != protocol["cache_metadata_sha256"]:
        raise ValueError("Different cache metadata")
    rows = []
    for variant in diagnostic.VARIANTS:
        for seed in protocol["seeds"]:
            by_alpha = {}
            for lang in ("zh","en"):
                name = f"{variant}_seed_{seed}_{lang}_val_components.npz"
                if runner.cached.file_sha256(root/name) != summary["components_sha256"][name]:
                    raise ValueError("Diagnostic component cache changed")
                values = diagnostic.load_verified_components(root/name,source/f"seed_{seed}"/variant/f"best_{lang}_val_outputs.npz")
                by_alpha[lang] = {a:diagnostic.score_language(values["base_logits"],values["correction_logits"],values["labels"],a) for a in diagnostic.ALPHAS}
            for alpha in diagnostic.ALPHAS:
                zh,en = by_alpha["zh"][alpha],by_alpha["en"][alpha]
                rows.append(dict(variant=variant,seed=seed,alpha=alpha,lb_mf1=(zh["macro_f1"]+en["macro_f1"])/2,
                                 prediction_changes=zh["changed_from_alpha_zero"]+en["changed_from_alpha_zero"]))
    recomputed = diagnostic.summarize(rows,protocol["no_cross_references"])
    if recomputed["selected"] != summary["selected"] or recomputed["grid"] != summary["grid"]:
        raise ValueError("Scale report does not match recomputed predictions")
    out = root/"scaled_manifests"
    out.mkdir(exist_ok=True)
    manifests = []
    for variant in diagnostic.VARIANTS:
        for seed in protocol["seeds"]:
            checkpoint = source/f"seed_{seed}"/variant/"best_head.ckpt"
            saved_result = json.loads((checkpoint.parent/"run_result.json").read_text(encoding="utf-8"))
            manifest = dict(format="fgclip2_v5_fixed_scale_v1",variant=variant,seed=seed,
                checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=runner.cached.file_sha256(checkpoint),
                residual_scale=summary["selected"][variant]["alpha"],feature_dims=meta["dims"],
                cache_metadata_sha256=protocol["cache_metadata_sha256"],
                selected_epoch=saved_result["selected"]["epoch"],zero_initial_branch=saved_result["selected"]["epoch"]==0,
                selected_on_validation=True,test_evaluated=False,new_training_performed=False,
                selection_rule=summary["selection_rule"],diagnostic_protocol_sha256=runner.cached.file_sha256(root/"diagnostic_protocol.json"))
            path = out/f"{variant}_seed_{seed}.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Refusing to overwrite a different scale manifest")
            runner.shared.write_json(path,manifest)
            manifests.append(path)
    audit = dict(recomputed_full_grid=True,source_files_unchanged=True,original_no_cross_files_unchanged=True,
                 manifest_count=len(manifests),full_validation_inference_verified=False,
                 source_sha256={name:runner.cached.file_sha256(runner.ROOT/name) for name in
                                ("full_model_v5_scaled.py","export_full_model_v5_scaled.py")},checks=[])
    if verify:
        if not torch.cuda.is_available() or torch.__version__ != protocol["torch_version"]:
            raise ValueError("Use original peixun CUDA runtime for verification")
        runner.core.DEVICE="cuda"
        torch.set_num_threads(2)
        checked,frames = runner.shared.verify_cache(cache,runner.ROOT/"data_submission_v1")
        if checked != meta or runner.shared.digest(runner.verify_context_cache(context,meta)) != protocol["context_cache_metadata_sha256"]:
            raise ValueError("Feature cache identity changed")
        masks,identity = runner.shared.build_content_masks(frames,cache)
        if identity["content_mask_sha256"] != protocol["content_mask_sha256"]:
            raise ValueError("Content masks changed")
        for path in manifests:
            model,manifest = load_scaled_checkpoint(path,"cuda")
            variant,seed,alpha = manifest["variant"],manifest["seed"],manifest["residual_scale"]
            for lang in ("zh","en"):
                component = root/f"{variant}_seed_{seed}_{lang}_val_components.npz"
                with np.load(component,allow_pickle=False) as values:
                    expected = diagnostic.scaled_logits(values["base_logits"],values["correction_logits"],alpha)
                    expected_labels = values["labels"].copy()
                dataset = runner.shared.ContentDataset(cache/f"{lang}_val",masks[f"{lang}_val"],contextual_root=context)
                batches=runner.shared.loader(dataset,protocol["eval_batch_size"])
                logits,labels=[],[]
                for batch in batches:
                    with torch.amp.autocast("cuda",dtype=torch.float16):
                        pred,_,label,_ = runner.cached.forward_cached(model,batch)
                    logits.append(pred.float().cpu())
                    labels.append(label.cpu())
                actual=torch.cat(logits).numpy()
                np.testing.assert_array_equal(torch.cat(labels).numpy(),expected_labels)
                np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
                audit["checks"].append(dict(variant=variant,seed=seed,language=lang,n=len(actual),alpha=alpha,predictions_match=True))
                del dataset,batches
            print(f"[VERIFIED] {variant}/{seed} alpha={alpha:g}",flush=True)
            del model
            runner.core.cleanup_memory()
        audit["full_validation_inference_verified"]=True
    runner.shared.write_json(root/"scale_export_audit.json",audit)
    print(out)
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    export(args.root.resolve(),args.verify)
