"""Recompute all v5 validation results and verify unchanged No-cross controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def base_digest(state):
    value = hashlib.sha256()
    for name in sorted(state):
        if not name.startswith("cross_attention."):
            tensor = state[name].detach().cpu().contiguous()
            value.update(name.encode())
            value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def read_scores(directory, labels_reference):
    scores, counts = {}, {}
    for language in ("zh", "en"):
        with np.load(directory / f"best_{language}_val_outputs.npz", allow_pickle=False) as out:
            labels, preds, logits, probs = (out[k] for k in ("labels", "preds", "logits", "probs"))
            if labels.ndim != 1 or preds.shape != labels.shape or logits.shape != (len(labels), 3) or probs.shape != logits.shape:
                raise ValueError("Invalid output shapes")
            if not (np.isfinite(logits).all() and np.isfinite(probs).all()) or set(labels) != {0, 1, 2}:
                raise ValueError("Invalid values or labels")
            np.testing.assert_array_equal(preds, logits.argmax(-1))
            np.testing.assert_allclose(probs, torch.softmax(torch.from_numpy(logits), -1).numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(out["lang_ids"], np.full(len(labels), int(language == "en")))
            if language not in labels_reference:
                labels_reference[language] = labels.copy()
            np.testing.assert_array_equal(labels, labels_reference[language])
            scores[language] = float(f1_score(labels, preds, average="macro", labels=[0,1,2], zero_division=0))
            counts[language] = len(labels)
    return scores, counts


def audit(root):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    if protocol["limit_samples"] != 0 or set(protocol["variants"]) != {"full_v5", "pooled_v5"}:
        raise ValueError("Formal audit requires both arms and full validation splits")
    seeds = sorted(protocol["seeds"])
    if len(seeds) < 2:
        raise ValueError("At least two paired seeds required for sample SD")
    labels_reference, rows = {}, []
    reference_by_seed = {r["seed"]: r for r in protocol["no_cross_references"]}
    if sorted(reference_by_seed) != seeds:
        raise ValueError("Missing fixed No-cross references")
    for seed in seeds:
        reference = reference_by_seed[seed]
        directory = Path(reference["directory"])
        for name, expected in reference["file_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError(f"Original No-cross reference changed: {directory / name}")
        scores, counts = read_scores(directory, labels_reference)
        score = (scores["zh"] + scores["en"]) / 2
        np.testing.assert_allclose(score, reference["score"], atol=1e-12, rtol=0)
        original = torch.load(directory / "best_head.ckpt", map_location="cpu", weights_only=False)
        np.testing.assert_allclose(original["metrics"]["val_lb_mf1"], score, atol=1e-12, rtol=0)
        rows.append(dict(seed=seed, variant="no_cross_control_reused", val_lb_mf1=score,
                         zh_val_mf1=scores["zh"], en_val_mf1=scores["en"], counts=counts,
                         selected_epoch=reference["selected_epoch"], unchanged=True))
        parent_info = protocol["warm_start_checkpoints"][str(seed)]
        parent_path = Path(parent_info["path"])
        if sha(parent_path) != parent_info["sha256"]:
            raise ValueError("Parent weights changed")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        parent_base_hash = base_digest(parent["clf_state_dict"])
        del parent, original
        for variant in ("pooled_v5", "full_v5"):
            directory = root / f"seed_{seed}" / variant
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            result = json.loads((directory / "run_result.json").read_text(encoding="utf-8"))
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            if config != dict(protocol=protocol, seed=seed, variant=variant) or result["fingerprint"] != fingerprint(config):
                raise ValueError("Configuration/fingerprint mismatch")
            if not result["validation_only"] or result["smoke_only"]:
                raise ValueError("Not a full validation development result")
            if [r["epoch"] for r in history] != list(range(protocol["epochs"] + 1)):
                raise ValueError("Incomplete fixed-budget history")
            selected = max(history, key=lambda row: row["val_lb_mf1"])
            if selected != result["selected"]:
                raise ValueError("Selected checkpoint does not maximize validation score")
            for row in history[:protocol["freeze_epochs"] + 1]:
                if row["base_state_sha256"] != parent_base_hash:
                    raise ValueError("Frozen base differs from the unchanged parent")
            scores, counts = read_scores(directory, labels_reference)
            score = (scores["zh"] + scores["en"]) / 2
            np.testing.assert_allclose(score, result["best_val_lb_mf1"], atol=1e-12, rtol=0)
            ckpt = torch.load(directory / "best_head.ckpt", map_location="cpu", weights_only=False)
            if ckpt["metrics"] != selected or ckpt["config"] != config or ckpt["fingerprint"] != fingerprint(config):
                raise ValueError("Checkpoint/config/metrics mismatch")
            state = ckpt["clf_state_dict"]
            if not all(torch.isfinite(p).all() for p in state.values()):
                raise ValueError("Non-finite checkpoint")
            if base_digest(state) != selected["base_state_sha256"]:
                raise ValueError("Selected base state differs from epoch record")
            count = sum(p.numel() for p in state.values())
            branch_count = sum(p.numel() for name,p in state.items() if name.startswith("cross_attention."))
            if count != result["head_parameters"] or branch_count != result["branch_parameters"]:
                raise ValueError("Parameter-count mismatch")
            weight_norm = float(state["cross_attention.classifier.3.weight"].norm())
            if selected["epoch"] == 0 and weight_norm != 0:
                raise ValueError("Initial residual is not zero")
            rows.append(dict(seed=seed, variant=variant, val_lb_mf1=score,
                zh_val_mf1=scores["zh"], en_val_mf1=scores["en"], counts=counts,
                selected_epoch=selected["epoch"], selected_stage=selected["stage"],
                selected_initial=selected["epoch"] == 0, head_parameters=count, branch_parameters=branch_count,
                correction_output_weight_norm=weight_norm, frozen_base_unchanged=True,
                runtime_including_validation_seconds=result["training_seconds"], checkpoint_sha256=sha(directory/"best_head.ckpt")))
            del ckpt, state
    for variant in ("no_cross_control_reused", "pooled_v5", "full_v5"):
        values = [r["val_lb_mf1"] for r in rows if r["variant"] == variant]
        entry = summary["variants"][variant]
        if entry["seeds"] != seeds:
            raise ValueError("Summary seed order changed")
        for actual, expected in ((entry["values"],values),(entry["mean"],np.mean(values)),(entry["sample_std"],np.std(values,ddof=1))):
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    lookup = {(r["variant"],r["seed"]):r["val_lb_mf1"] for r in rows}
    for name, entry in summary["paired"].items():
        a,b = name.split("_minus_")
        values = [lookup[(a,s)]-lookup[(b,s)] for s in seeds]
        if entry["seeds"] != seeds:
            raise ValueError("Unpaired comparison")
        np.testing.assert_allclose(entry["values"],values,atol=1e-12,rtol=0)
        np.testing.assert_allclose(entry["mean"],np.mean(values),atol=1e-12,rtol=0)
        np.testing.assert_allclose(entry["sample_std"],np.std(values,ddof=1),atol=1e-12,rtol=0)
    matches = {}
    for name,expected in protocol["source_sha256"].items():
        if sha(root/"source_snapshot"/name) != expected:
            raise ValueError("Source snapshot mismatch")
        matches[name] = sha(Path(__file__).resolve().parent/name) == expected
    return dict(validation_only=True,test_evaluated=False,new_runs=len(seeds)*2,reused_controls=len(seeds),
                original_no_cross_files_unchanged=True, frozen_base_weights_verified=True,
                current_sources_match_training_fingerprints=matches, runs=rows,
                variants=summary["variants"],paired=summary["paired"],
                limitations="Development validation only; fixed max budget but original controls retain their own early-stopping/scheduler. Pooled is not parameter-matched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root/"audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
