# Data Poisoning Risk Toolkit

This folder contains a cleaned-up technical expansion for the data poisoning project.

The original poster framed several test-time adversarial examples as data poisoning. The stronger framing is:

- **Data poisoning** changes training, fine-tuning, feedback, retrieval, or supply-chain data.
- **Backdoors** are a high-risk poisoning subfamily because they preserve normal accuracy while adding trigger-activated behavior.
- **Evasion attacks** such as FGSM, PGD, CW, DeepFool, Boundary, One-Pixel, and UAP are related adversarial ML techniques, but they are not data poisoning unless used to craft poisoned training samples.
- **The final taxonomy is a Mendeleev-like risk-by-mechanism table, not a contradictory chemical clone:** the paper uses readable native LaTeX tables with element-style symbols, while the generated PNG/SVG keeps colored element tiles for posters and slides.

## Contents

- `catalog/data_poisoning_techniques.csv` - expanded technique catalog with risk scores and primary sources.
- `docs/poster_upgrade_notes.md` - recommended changes to the poster/table.
- `docs/visual_taxonomy_audit.md` - rationale for replacing the literal chemical-table shape with an auditable risk-by-mechanism matrix.
- `docs/fish_dataset_notes.md` - extracted notes from the SC25 AdversaGuard poster for the `__fish2/` fish freshness dataset.
- `docs/data_poisoning_theory.tex` - IEEE two-column conference-style LaTeX paper with abstract, introduction, related work, methodology, results, discussion, and conclusion.
- `docs/data_poisoning_access.tex` - IEEE Access version of the paper.
- `scripts/fungi_attack_comparison.py` - fungi dataset comparison using lazy PyTorch datasets, a pretrained MobileNetV3-Small transfer model, conditional evasion metrics, every original poster method (FGSM, PGD, DeepFool, Carlini-Wagner, Boundary, One-Pixel, UAP), and the additional shared methods including Poison Frogs, Sleeper-Agent-style backdoor, Witches' Brew-style gradient matching, Adversarial Patch, JSMA, ZOO, SparseFool, EAD, subpopulation poisoning, and BadNets.
- `scripts/generate_reporting_artifacts.py` - derived report tables and figures: fungi dataset statistics, sample image grid, dataset split chart, grouped attack statistics, implementation coverage audit, table layout audit, reporting dashboard, paper risk matrix, poster risk matrix, and cross-dataset stage comparison.
- `pptx_media/periodic_table_slide.jpg` - extracted image from the supplied PPTX for reference.

## Run The Food-Image Experiments

Place the fungi image dataset under `fungi/` with this folder layout:

```text
fungi/
  train/
    edible/
    poisonous/
  val/
    edible/
    poisonous/
```

Then run:

```powershell
python scripts\fungi_attack_comparison.py
```

For the fish freshness dataset referenced by the SC25 AdversaGuard poster, use the flat `fish/` class-folder layout. The safety-critical direction is non-fresh fish being accepted as fresh:

```powershell
python scripts\fungi_attack_comparison.py --dataset-root fish --dataset-name fish --source-class non-fresh2 --target-class fresh2 --eval-samples 12 --zoo-samples 4 --epochs 12
```

For the larger seafood freshness benchmark, use the local `Datasets/` hierarchy. The runner excludes duplicate `working/` split copies and uses all 400 unique image hashes:

```powershell
python scripts\fungi_attack_comparison.py --dataset-root Datasets --dataset-name seafood --source-class NonFresh --target-class Fresh --image-size 96 --epochs 2 --batch-size 32 --eval-samples 8 --zoo-samples 2 --trigger-size 12 --adversarial-patch-size 20 --attack-scale 0.12 --seed 2027 --cache-frozen-features
```

Generate the additional reporting artifacts:

```powershell
python scripts\generate_reporting_artifacts.py
```

The script writes:

- `results/fungi_attack_comparison.csv`
- `results/fungi_attack_comparison.json`
- `results/fungi_attack_comparison.md`
- `results/fish_attack_comparison.csv`
- `results/fish_attack_comparison.json`
- `results/fish_attack_comparison.md`
- `results/seafood_attack_comparison.csv`
- `results/seafood_attack_comparison.json`
- `results/seafood_attack_comparison.md`
- `results/seafood_dataset_audit.md`
- `results/food_dataset_comparison.csv`
- `results/food_dataset_comparison.json`
- `results/food_dataset_comparison.md`
- `results/fungi_dataset_statistics.csv`
- `results/fungi_dataset_statistics.json`
- `results/fungi_dataset_statistics.md`
- `results/attack_summary_statistics.csv`
- `results/attack_summary_statistics.json`
- `results/attack_summary_statistics.md`
- `results/periodic_table_entries.csv`
- `results/periodic_table_entries.json`
- `results/periodic_table_entries.md`
- `results/table_layout_audit.csv`
- `results/table_layout_audit.json`
- `results/table_layout_audit.md`
- `results/implementation_coverage.csv`
- `results/implementation_coverage.json`
- `results/implementation_coverage.md`
- `results/citation_verification.csv`
- `results/citation_verification.md`
- `figures/fungi_attack_comparison_bars.png`
- `figures/fungi_attack_comparison_split.png`
- `figures/fungi_attack_examples.png`
- `figures/fish_attack_comparison_bars.png`
- `figures/fish_attack_comparison_split.png`
- `figures/fish_attack_examples.png`
- `figures/seafood_attack_comparison_bars.png`
- `figures/seafood_attack_comparison_split.png`
- `figures/seafood_attack_examples.png`
- `figures/fungi_dataset_samples.png`
- `figures/fungi_dataset_split.png`
- `figures/fungi_reporting_dashboard.png`
- `figures/attack_success_by_stage.png`
- `figures/periodic_table_data_poisoning.png`
- `figures/periodic_table_data_poisoning.svg`
- `figures/paper_taxonomy_panels.png`

The fungi, fish, seafood dataset folders and pretrained-weight cache are intentionally ignored by git.
