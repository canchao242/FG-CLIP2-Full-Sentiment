"""Build a SEPARATE train/validation contextual-text sidecar; never run vision.

Existing image/static caches remain read-only. No test split is read. Partial
outputs resume from flushed chunk boundaries; metadata/source mismatches fail.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_full_model_v2 as runner


def verify_context_cache(root, base_meta, full_scan=False):
    marker = json.loads((root / "cache_ready.json").read_text(encoding="utf-8"))
    config = marker["config"]
    if config["base_cache_metadata_sha256"] != runner.digest(base_meta):
        raise ValueError("Context cache belongs to a different base cache")
    if config["feature_source"] != "text_encoder_last_hidden_state" or config["splits"] != list(runner.SPLITS):
        raise ValueError("Invalid contextual feature source or development split set")
    dim = base_meta["dims"]["token_dim"]
    for split in runner.SPLITS:
        path = root / split / "tokens.npy"
        row = marker["splits"][split]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        n = base_meta["splits"][split]["rows"]
        if array.shape != (n, runner.core.MAX_TEXT_LEN, dim) or array.dtype != np.float16 or row["rows"] != n:
            raise ValueError(f"Invalid contextual array: {split}")
        if runner.cached.file_sha256(path) != row["sha256"]:
            raise ValueError(f"Context feature file changed: {split}")
        if full_scan:
            for start in range(0, n, 512):
                if not np.isfinite(array[start:start + 512]).all():
                    raise ValueError(f"Non-finite contextual features: {split}")
        elif not np.isfinite(array[[0, n // 2, n - 1]]).all():
            raise ValueError(f"Non-finite contextual features: {split}")
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=runner.ROOT / "checkpoints/fgclip2_submission_v1_feature_cache")
    parser.add_argument("--split-root", type=Path, default=runner.ROOT / "data_submission_v1")
    parser.add_argument("--output", type=Path, default=runner.ROOT / "checkpoints/fgclip2_contextual_text_v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output.resolve() == args.cache_root.resolve() or args.cache_root.resolve() in args.output.resolve().parents:
        parser.error("The context sidecar must not overwrite or be placed inside the existing cache")
    meta, frames = runner.verify_cache(args.cache_root, args.split_root)
    if (args.output / "cache_ready.json").exists():
        verify_context_cache(args.output, meta, full_scan=True)
        print("[REUSE COMPLETE] Verified all contextual text files; no extraction started.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the peixun CUDA environment for text feature extraction")
    snapshot = Path(snapshot_download(runner.core.MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    identity = {path.name: runner.cached.file_sha256(path) for pattern in ("*.safetensors", "*.bin", "*.py", "config.json")
                for path in snapshot.glob(pattern) if path.is_file()}
    config = {"feature_source": "text_encoder_last_hidden_state", "model_id": runner.core.MODEL_ID,
        "model_revision": snapshot.name, "model_files_sha256": identity,
        "base_cache_metadata_sha256": runner.digest(meta), "splits": list(runner.SPLITS),
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "batch_size": args.batch_size, "max_text_len": runner.core.MAX_TEXT_LEN,
        "walk_type": runner.core.WALK_TYPE, "torch_version": torch.__version__,
        "precision": "float32 weights with float16 CUDA autocast and storage",
        "builder_sha256": runner.cached.file_sha256(Path(__file__)),
        "core_sha256": runner.cached.file_sha256(Path(runner.core.__file__))}
    print(f"[TEXT ONLY] {sum(len(f) for f in frames.values())} rows; output={args.output}", flush=True)
    if args.preflight:
        print("[PREFLIGHT OK] No model forward or cache writes")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        config_path = args.output / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Context cache build configuration changed; use a new output directory")
        elif any(p.name not in ("RUNNING.lock",) for p in args.output.iterdir()):
            raise ValueError("Nonempty sidecar directory without verified configuration")
        else:
            runner.write_json(config_path, config)
        # Keep the Hub model/revision identity so Transformers reuses the already
        # installed dynamic-code namespace rather than creating a new namespace
        # from the local snapshot directory's hexadecimal basename.
        model = AutoModelForCausalLM.from_pretrained(
            runner.core.MODEL_ID, revision=snapshot.name,
            trust_remote_code=True, local_files_only=True,
        )
        encoder = runner.core._find_text_encoder(model)
        if encoder is None:
            raise RuntimeError("Text encoder not found; embedding fallback is DISABLED")
        encoder.to("cuda").eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        runner.core.TOKENIZER = tokenizer
        embedding = runner.core.get_text_embedding_layer(model, tokenizer=tokenizer)
        completed = {}
        for split, frame in frames.items():
            directory = args.output / split
            directory.mkdir(exist_ok=True)
            path, progress_path = directory / "tokens.npy", directory / "progress.json"
            n, dim = len(frame), meta["dims"]["token_dim"]
            shape = (n, runner.core.MAX_TEXT_LEN, dim)
            if path.exists():
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
                if array.shape != shape or array.dtype != np.float16:
                    raise ValueError(f"Invalid partial array: {split}")
                done = json.loads(progress_path.read_text())["completed_rows"] if progress_path.exists() else 0
                if not 0 <= done <= n:
                    raise ValueError("Invalid cache resume boundary")
            else:
                array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
                done = 0
            prefix = "Language: Chinese. Text: " if split.startswith("zh") else "Language: English. Text: "
            texts = frame.text.fillna("").astype(str).tolist()
            original_mask = np.load(args.cache_root / split / "attention_mask.npy", mmap_mode="r")
            original_tokens = np.load(args.cache_root / split / "tokens.npy", mmap_mode="r")
            for block in range(done, n, 512):
                end = min(n, block + 512)
                for start in range(block, end, args.batch_size):
                    stop = min(end, start + args.batch_size)
                    encoded = tokenizer([prefix + t for t in texts[start:stop]], padding="max_length",
                        truncation=True, max_length=runner.core.MAX_TEXT_LEN, return_tensors="pt")
                    ids = encoded["input_ids"].to("cuda")
                    mask = encoded.get("attention_mask")
                    mask = mask.to("cuda") if mask is not None else (ids != tokenizer.pad_token_id).long()
                    if not np.array_equal(mask.cpu().numpy(), original_mask[start:stop]):
                        raise ValueError(f"Tokenizer mask differs from base cache: {split}/{start}")
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = encoder(input_ids=ids, attention_mask=mask, walk_type=runner.core.WALK_TYPE)
                        # Require the actual encoder output: never call the legacy fallback helper.
                        hidden = getattr(outputs, "last_hidden_state", None)
                        if hidden is None or hidden.shape != (stop - start, shape[1], shape[2]):
                            raise RuntimeError("Missing/wrong contextual last_hidden_state; refusing static fallback")
                        if start == block:
                            static = embedding(ids).half().cpu().numpy()
                            if not np.allclose(static, original_tokens[start:stop], atol=1e-5, rtol=1e-3):
                                raise ValueError("Token IDs/embedding weights differ from the existing static cache")
                            if np.allclose(hidden.half().cpu().numpy(), static, atol=1e-5, rtol=1e-3):
                                raise RuntimeError("Contextual output equals static embeddings; refusing cache mislabelling")
                        values = hidden.half().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Non-finite contextual features")
                    array[start:stop] = values
                array.flush()
                runner.write_json(progress_path, {"completed_rows": end})
                print(f"[CONTEXT TEXT] {split} {end}/{n}", flush=True)
            del array
            completed[split] = {"rows": n, "shape": list(shape), "sha256": runner.cached.file_sha256(path)}
        runner.write_json(args.output / "cache_ready.json", {"config": config, "splits": completed})
        verify_context_cache(args.output, meta, full_scan=True)
        print("[COMPLETE] Context text sidecar verified; image/static/test caches were not written.", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
