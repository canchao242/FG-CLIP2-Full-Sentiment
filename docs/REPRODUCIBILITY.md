# Reproduction guide

Run commands from the repository root. Use a separate Python 3.12 environment. Install PyTorch 2.9.1 appropriate to the hardware, followed by `python -m pip install -r requirements.txt`. The recorded research build was PyTorch `2.9.1+cu128`; CPU tests do not need CUDA.

## 1. Verify public evidence first

```bash
python tools/verify_results.py
python -m unittest discover -s tests -p "test_release_tools.py" -v
python tools/test_models.py
```

The first two commands require NumPy and pandas only. The last runs the retained three synthetic model-test files, with CUDA hidden and CPU thread counts limited to two. No private feature caches or upstream downloads are accessed.

## 2. Prepare private input data

Read [the dataset and licensing notes](DATA_AND_LICENSES.md). Supply the same usable subset as local `image_path,text,label` CSVs. Example paths below are placeholders for your own legally obtained files:

```bash
python tools/build_private_splits.py --zh-csv /path/to/cisd_usable.csv --en-csv /path/to/mvsa_usable.csv
```

Repeat `--zh-csv`/`--en-csv` if the usable subset is stored across several CSVs. Output is six CSVs beneath `src/data_submission_v1/{zh,en}/{train,val,test}.csv`. Existing outputs are not overwritten. Sample content, multiplicity and order must match; missing examples are not silently excluded. This reconstructs fixed membership, not a new random split or a new duplicate search.

## 3. Obtain the fixed backbone and build caches explicitly

Download the pinned upstream revision **after reviewing its license and custom code**:

```bash
hf download qihoo360/fg-clip2-base --revision 430fbc8a912c86fd4de601381b6245a0edab22f0
python tools/build_cache.py
python tools/build_cache.py --run
python src/build_contextual_text_cache.py --preflight
python src/build_contextual_text_cache.py
```

`build_cache.py` first prints a plan unless `--run` is passed. It builds/reuses static frozen features for all six splits. This includes test features but does not optimize a head on test labels. The contextual sidecar reads **train/validation only**, runs the frozen text encoder, and does not rerun vision. Full-v5 training reads this contextual sidecar in addition to the static cache. Contextual extraction requires CUDA in the retained implementation.

Static cache: `src/checkpoints/fgclip2_submission_v1_feature_cache/`.
Context sidecar: `src/checkpoints/fgclip2_contextual_text_v1/`.

These arrays can be large. Check disk space first. Training scripts require complete metadata, exact local CSV hashes and matching masks; they never invoke extraction implicitly. Do not point a training output into a cache directory. If a process is interrupted, verify the owning process has exited before handling its `RUNNING.lock`; do not delete a live lock or regenerate verified vision features reflexively.

## 4. Shared parents and matched refinements

```bash
python tools/reproduce.py --stage parents
python tools/reproduce.py --stage parents --preflight
python tools/reproduce.py --stage parents --run
python tools/reproduce.py --stage controls --preflight
python tools/reproduce.py --stage controls --run
python tools/reproduce.py --stage v5 --preflight
python tools/reproduce.py --stage v5 --run
```

The default seed list is `42 123 3407 2026 2027`. Stage outputs go to `src/checkpoints/public_reproduction/<stage>/<protocol fingerprint>/`. Subsequent stages require exactly one prior protocol or an explicit `--parents /path/to/protocol-directory` / `--controls /path/to/protocol-directory`.

Parents: up to 20 epochs, No-cross base, initial learning rate 5e−5. Continued controls: up to five epochs, learning rate 1e−5, original plateau scheduler and patience four. Full-v5 and pooled-v5: five epochs, first three branch-only, then two joint epochs; rates and branch loss are in `configs/full_v5.json`. All use 1,200 sampled batches of 16 per epoch. The staged and continued schedules differ: do not call them equal-update-budget arms.

Only train and validation inputs determine these runs. Do not introduce test-based checkpoint/scale selection. A new reproduction can produce different values; record every seed and retain epoch-zero selections.

## 5. Preserve the fixed inference scale

```bash
python tools/export_fixed_scale.py --run-root /path/to/v5/protocol-directory
python tools/export_fixed_scale.py --run-root /path/to/v5/protocol-directory --run
```

The helper emits manifests for locally generated checkpoints with the published scales (Full 0.5, pooled 0.25). It performs **no new scale search**. Manifests refer to local weights; they are not a distribution of weights. Use `full_model_v5_scaled.load_scaled_checkpoint()` for hash-checked inference. Do not replace a manifest with a bare checkpoint and silently assume alpha=1.

## Historical drivers and provenance

`evaluate_full_v5_heldout.py`, `profile_full_v5_locked.py` and the replication drivers retain archived experiment-directory contracts for audit. Their default historical checkpoint directories are **not present** in this public package. They are not the starting commands for a fresh run. The released numerical predictions can be independently verified without those unavailable files.

`docs/source_provenance.json` records the original and release hashes. Portability edits replace author-machine I/O defaults, correct an obsolete “untouched” docstring and pin previously unpinned upstream loading calls to the recorded revision. AST checks establish that no tensor calculations, optimizer logic or No-cross model definitions changed. Original checkpoint source hashes will differ where these release edits apply. Keep the old source/runtime with old weights; never disable its identity checks. Fresh training creates its own release-local source snapshots and cache/protocol identities.

## Validation performed for this release

Synthetic CPU tests and public prediction recalculation are executed during packaging. The entire GPU pipeline is **not rerun** merely to publish the code. No claim of clean-machine end-to-end reproduction, accepted publication or new independent-test superiority is made.
