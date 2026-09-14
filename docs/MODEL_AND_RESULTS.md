# Model identity and interpretation

## Three names that must not be conflated

**Historical Full** is the older direct cross-attention ablation head. Its single-seed 90/10 validation result is not the current five-seed Full-v5 result.

**Continued No-cross** is the structurally pruned shared base, with regional visual weighting, text-token pooling, a sample-centred gate and difference/product features. Its matched refinement arm remains intact. It contains 3,102,213 head parameters.

**Full-v5** adds a contextual-text residual correction to that base. The final scaled prediction is base logits plus **0.5 × correction logits**; a checkpoint alone does not encode this external scale. The head contains 3,765,513 parameters. The frozen backbone contains 383,803,394 parameters. The pooled-v5 control uses a 0.25 scale; it is not No-cross.

Training uses three branch-only epochs and two joint epochs, selecting checkpoints on validation with scale 1.0. Scales were selected on validation seeds 42, 123 and 3407, then held fixed for seeds 2026 and 2027. This historical distinction remains explicit even if a reader launches all five seeds in one new reproduction command.

## Numerical evidence

`results/predictions/` includes labels, logits and zero-based row indices for every model/seed/language test combination. `prediction_index.json` hashes the **released** NPZ files. `original_prediction_index.json` and internal hash fields inside historical reports refer to the original archive, not the repackaged JSON/NPZ byte streams. `docs/source_provenance.json` maps original and release hashes. Repacking removes any non-allowlisted fields; the published observation arrays are unchanged.

The five-seed Full-v5 and continued No-cross held-out means are 0.7233314292 and 0.7291995905. The paired mean difference is −0.0058681613. There is one Full win, two ties and two losses. These are training-seed replicates on the same examples, not five independently sampled datasets. Report sample SD (`ddof=1`).

The validation-selected deployment seed 2027 scores 0.7406203677 for Full-v5 and 0.7368478261 for No-cross. That single checkpoint comparison does not overturn the five-seed mean. Selected epoch-zero/zero-correction runs and their ties must not be removed.

The test split excludes gradient updates for these checkpoints, but its earlier outcomes informed development. It is a **retrospective held-out benchmark**, not fresh independent confirmation. No further test-directed tuning was performed to prepare this package.

## Efficiency scope

`results/efficiency/` contains the protocol, raw timings and summaries. Profiling uses the locked seed-2027 checkpoint, batch sizes 1 and 16, twenty warmups, and five technical rounds of 64 timed requests. The recorded machine is a Windows WDDM RTX 5080 shared with desktop applications, with no clock lock.

Single-sample backbone-plus-head forward latency is approximately 18.49 ± 0.57 ms for Full-v5 and 16.84 ± 0.54 ms for No-cross. This excludes disk/image decoding, preprocessing, tokenization, host-to-device transfer and service overhead. Cached-head timing is a separate scope. Timing rounds must not be labelled training seeds or end-to-end user latency.

## Scope of the release

This package supports code inspection, synthetic testing, exact numerical reanalysis, and new training with locally obtained inputs. It does not distribute the historical weights or guarantee one-command regeneration of every historical table. Evaluation/profiling drivers under `src/` retain archived manifests and fail-closed checks; porting them to newly trained checkpoints requires a new explicit evaluation plan. Never replace their checks with unconditional success.
