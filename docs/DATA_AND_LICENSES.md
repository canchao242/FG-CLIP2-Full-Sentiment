# Data, third-party software and release boundaries

## Original resources

- **CISD:** Xiao et al., “Collaborative Fine-Grained Interaction Learning for Image–Text Sentiment Analysis,” Knowledge-Based Systems, DOI [10.1016/j.knosys.2023.110951](https://doi.org/10.1016/j.knosys.2023.110951). Obtain the study data through the original authors' resources and their applicable conditions. This repository is not a mirror or a grant of image/post reproduction rights.
- **MVSA-Single:** obtain data through the [official MCRLab MVSA page](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) and cite the dataset's original publications. The availability of a download does not establish unrestricted photograph redistribution rights.
- **FG-CLIP2:** [qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base), recorded revision `430fbc8a912c86fd4de601381b6245a0edab22f0`. Its model card identifies Apache-2.0. The upstream model code/weights retain their own terms and attribution requirements; they are not bundled or relicensed here. Loading uses `trust_remote_code=True`: review the pinned upstream code before execution.

The repository's MIT license covers the original research software. Dependencies retain their own licenses. Numerical predictions and derived split metadata are provided for research verification subject to any applicable original-data terms; no broader third-party rights are asserted.

## Public split metadata

`data/split_manifest_public.csv` contains **41,362** usable examples:

| Dataset/language | Train | Validation | Test |
|---|---:|---:|---:|
| CISD / zh | 31,010 | 3,877 | 3,883 |
| MVSA-Single / en | 2,069 | 260 | 263 |

| Column | Meaning |
|---|---|
| sample_id | Language-prefixed study row identifier, unique in this manifest |
| image_file | Original basename, not an image URL or local absolute path |
| language | `zh` or `en` |
| split | `train`, `val`, `test` |
| split_row | Zero-based row order within this language/split; prediction alignment key |
| label | 0 negative, 1 neutral, 2 positive |
| duplicate_group | Existing duplicate-connected group identifier; groups do not cross splits |
| text_sha256 | SHA-256 of the exact study text encoded as UTF-8 |

These are derived identifiers, **not anonymous data**. Checksums do not prevent dictionary matching, and filenames may already encode public source identifiers. No raw text, token strings, token IDs, photo bytes or photo URLs are released. The grouped manifest preserves the study split; group disjointness does not prove that every possible near-duplicate has been detected.

## Reconstruction contract

Prepare local CSVs containing `image_path,text,label`, after obtaining permission/access and the same usable study subset. Relative image paths resolve against the CSV's own directory. Preserve exact text and labels; do not normalize text merely to make a hash match. Multiple input CSVs can be supplied per language.

The reconstruction helper joins on language, image basename, label and exact-text checksum, retains multiplicity, and restores the published `split_row` order. It rejects missing images, missing/extra samples, changed text, duplicated IDs, cross-split groups and existing output files. If your data version differs, contact the corresponding author; do not silently drop examples. Local paths/CSV byte hashes will differ from the author machine even when sample content agrees, so create new local cache/protocol records.

## Excluded assets

No raw dataset, third-party photograph, pretrained or trained checkpoint, feature array, manuscript figure, credential, browser profile or authentication helper is part of this repository. Permission to access a dataset is separate from permission to reproduce its photos in a paper. Contact: canchao242@gmail.com.
