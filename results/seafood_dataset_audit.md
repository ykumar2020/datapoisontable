# Seafood Dataset Duplicate Audit

Local root: `Datasets/`

The seafood folder contains 522 JPG files on disk, but only 400 unique SHA-256 image hashes. The 122 extra files are duplicate split copies under `Anchovy/working/`, not additional independent training examples.

| Quantity | Count |
|---|---:|
| JPG files on disk | 522 |
| Unique SHA-256 image hashes | 400 |
| Duplicate hash groups | 122 |
| Files under `Anchovy/working/` | 122 |
| Excluded duplicate copies | 122 |

Unique image counts used in the seafood experiment:

| Class | Unique images |
|---|---:|
| Fresh | 200 |
| NonFresh | 200 |
| Total | 400 |

Example duplicate pair:

- `Anchovy/Fresh/HamsiTaze (1).jpg`
- `Anchovy/working/train/Fresh/HamsiTaze (1).jpg`

The runner excludes `working/` to avoid duplicate train/validation leakage and uses all 400 unique images in a deterministic stratified 75/25 split.
