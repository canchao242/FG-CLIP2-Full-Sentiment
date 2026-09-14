# FG-CLIP2 Full Sentiment

Research code and numerical evidence for **Full-v5**, an emotion-aware bilingual image–text sentiment model with a contextual residual correction on a frozen FG-CLIP2 backbone. Author: **Can Chao**, College of Electronics and Information, Southwest Minzu University.

This is the Full-v5 publication package, **not** the earlier No-cross release. Full-v5, the historical Full attention head, and continued No-cross are distinct models. See [the model and evidence notes](docs/MODEL_AND_RESULTS.md).

## What is public

- Nineteen research modules covering the shared base, Full-v5, its matched controls, training, inference and profiling.
- Fixed settings, exact ordered split membership for 41,362 usable examples, and private-data reconstruction checks.
- All twenty numerical held-out prediction files: two models × five seeds × two languages; timing records and source provenance.
- CPU synthetic tests and an independent result verifier that require no photos, weights or GPU.

Raw posts, dataset images, model weights, feature caches and manuscript photographs are **not included**. Obtain original data and FG-CLIP2 through their respective providers. MIT applies to this repository's original code, not to third-party data or weights. See [data access](docs/DATA_AND_LICENSES.md).

## Quick start: check the published numbers

Python 3.12 was used. From the repository root:

```bash
python -m pip install numpy==2.0.1 pandas==2.3.1
python tools/verify_results.py
python -m unittest discover -s tests -p "test_release_tools.py" -v
```

The verifier recomputes three-class macro-F1 from logits, checks exact test-row/label alignment, and checks each released prediction file's SHA-256. No network, model loading or training is involved.

| Retrospective held-out LB-MF1 | Mean ± sample SD (5 training seeds) |
|---|---:|
| Full-v5 | 0.7233 ± 0.0145 |
| Continued No-cross | 0.7292 ± 0.0102 |
| Paired Full-v5 minus No-cross | −0.0059 ± 0.0114 |

LB-MF1 is the arithmetic mean of Chinese and English macro-F1, not macro-F1 on their pooled examples. The higher Full-v5 **validation** mean did not transfer to the five-seed held-out mean. These partitions had been inspected during development and are **not a newly untouched confirmatory test**. All seeds, ties and adverse comparisons are retained.

## Reproduce training with your legally obtained local data

Install a suitable PyTorch 2.9.1 build for your platform, then `python -m pip install -r requirements.txt`. The recorded environment used Python 3.12.11, PyTorch 2.9.1+cu128 and an RTX 5080; the dependency list is a recorded environment, not a guarantee of cross-platform bitwise identity.

Follow [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) in order: reconstruct the exact private splits, obtain the pinned upstream model, build static and contextual features, train shared parents, continue No-cross, and run staged Full-v5/pooled controls. Cache construction is explicit; training never silently rebuilds features. For a safe preview:

```bash
python tools/build_cache.py
python tools/reproduce.py --stage parents
```

These commands print plans only. Add `--run` deliberately after completing the prerequisites. Historical evaluation drivers retain archived path/hash contracts; they are supplied for inspection, not presented as one-command fresh evaluation of unavailable author checkpoints.

## Repository layout

```text
src/         original research modules, with documented portability edits
tools/       safe entry points, exact split reconstruction, numerical verification
tests/       synthetic model and release-tool tests
configs/     fixed Full-v5 protocol
data/        derived split identifiers and counts; no posts or photos
results/     numerical predictions, metrics and timing records
docs/        reproduction, model limitations, third-party terms and provenance
```

Generated private data live in `src/data_submission_v1/`; generated weights and caches live in `src/checkpoints/`. Both are ignored by Git. Do not force-add them.

## Citation and contact

The accompanying manuscript is a submission, not a claimed accepted publication. Cite the repository URL and the specific Git commit you used; see [CITATION.md](CITATION.md). Contact: canchao242@gmail.com. ORCID: [0009-0009-4050-9964](https://orcid.org/0009-0009-4050-9964).

### Verification boundary

Packaging validates source provenance, synthetic CPU tests, and the released numerical evidence. It does **not** rerun the full GPU training pipeline or assert that a new run must reproduce each recorded score exactly. No original experiment or comparator was changed to prepare this release.
