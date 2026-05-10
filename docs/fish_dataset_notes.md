# Fish Dataset Notes From SC25 AdversaGuard Poster

The added `__fish2/` folder matches the fish freshness setup referenced by the SC25 poster paper:

Yulia Kumar, Solomon Thomas, Dejaun Gayle, J. Jenny Li, and Dov Kruger. 2025. "AdversaGuard: A Distributed Data-Poisoning Benchmark for Parallel AI." SC25 Posters. https://sc25.supercomputing.org/proceedings/posters/poster_files/post110s2-file3.pdf

Relevant extracted details:

- Dataset context: food-domain freshness classification used in the AdversaGuard distributed data-poisoning benchmark.
- Source described in the SC25 poster references: Kaggle seafood / fish freshness data from Turkey.
- Task used here: binary classification for horse mackerel freshness.
- Local classes: `fresh2` and `non-fresh2`.
- Local size: 40 images total, 20 per class.
- Layout in this repo: flat class folders, not explicit `train/val` folders.
- Split used by the runner: deterministic stratified split with `val_fraction=0.25`, producing 30 train images and 10 validation images.
- Safety-critical direction: `non-fresh2 -> fresh2`, because accepting non-fresh fish as fresh is the risky failure.
- Model/attack setup: same MobileNetV3-Small transfer model and same 20-method attack table used for the fungi comparison.

Primary command:

```powershell
python scripts\fungi_attack_comparison.py --dataset-root __fish2 --dataset-name fish --source-class non-fresh2 --target-class fresh2 --eval-samples 12 --zoo-samples 4 --epochs 12
```

The comparison summary is generated in `results/food_dataset_comparison.md`.
