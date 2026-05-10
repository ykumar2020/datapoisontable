# Data Poisoning Risk Toolkit

This folder contains a cleaned-up technical expansion for the data poisoning project.

The original poster framed several test-time adversarial examples as data poisoning. The stronger framing is:

- **Data poisoning** changes training, fine-tuning, feedback, retrieval, or supply-chain data.
- **Backdoors** are a high-risk poisoning subfamily because they preserve normal accuracy while adding trigger-activated behavior.
- **Evasion attacks** such as FGSM, PGD, CW, DeepFool, Boundary, One-Pixel, and UAP are related adversarial ML techniques, but they are not data poisoning unless used to craft poisoned training samples.
- **The final visual is a Mendeleev-like risk-by-mechanism matrix, not a contradictory chemical clone:** it keeps colored element tiles, symbols, and a table key, while rows encode this study's assigned `R1`-`R4` levels, columns are the mechanism families, and evasion analogs are separated from the poisoning grid.

## Contents

- `catalog/data_poisoning_techniques.csv` - expanded technique catalog with risk scores and primary sources.
- `docs/poster_upgrade_notes.md` - recommended changes to the poster/table.
- `docs/visual_taxonomy_audit.md` - rationale for replacing the literal chemical-table shape with an auditable risk-by-mechanism matrix.
- `docs/fish_dataset_notes.md` - extracted notes from the attached FoodDDP PDF for the `__fish2/` fish freshness dataset.
- `docs/data_poisoning_theory.tex` - IEEE two-column conference-style LaTeX paper with abstract, introduction, related work, methodology, results, discussion, and conclusion.
- `scripts/mnist_poisoning_experiment.py` - actual training-time poisoning implementations on real MNIST: random label flipping, targeted source-to-target label flipping, and a BadNets-style visible trigger backdoor.
- `scripts/mnist_attack_comparison.py` - 14-method MNIST comparison suite covering poisoning, backdoors, and related evasion attacks.
- `scripts/fungi_attack_comparison.py` - fungi dataset comparison using lazy PyTorch datasets, a pretrained MobileNetV3-Small transfer model, conditional evasion metrics, every original poster method (FGSM, PGD, DeepFool, Carlini-Wagner, Boundary, One-Pixel, UAP), and the additional shared methods including Poison Frogs, Sleeper-Agent-style backdoor, Witches' Brew-style gradient matching, Adversarial Patch, JSMA, ZOO, SparseFool, EAD, subpopulation poisoning, and BadNets.
- `scripts/generate_reporting_artifacts.py` - derived report tables and figures: fungi dataset statistics, sample image grid, dataset split chart, grouped attack statistics, implementation coverage audit, table layout audit, reporting dashboard, paper risk matrix, poster risk matrix, and cross-dataset stage comparison.
- `pptx_media/periodic_table_slide.jpg` - extracted image from the supplied PPTX for reference.

## Run The MNIST Experiment

```powershell
python scripts\mnist_poisoning_experiment.py --download
```

Run the expanded comparison table:

```powershell
python scripts\mnist_attack_comparison.py --download
```

The script uses real MNIST data through `torchvision`, trains real scikit-learn classifiers, and writes:

- `results/mnist_poisoning_results.csv`
- `results/mnist_poisoning_results.json`
- `results/mnist_poisoning_results.md`
- `figures/mnist_clean_samples.png`
- `figures/mnist_backdoor_examples.png`
- `figures/mnist_metric_bars.png`
- `figures/mnist_confusion_*.png`
- `figures/mnist_confusion_visible_patch_backdoor_triggered.png`

The expanded comparison additionally writes:

- `results/mnist_attack_comparison.csv`
- `results/mnist_attack_comparison.json`
- `results/mnist_attack_comparison.md`
- `figures/mnist_attack_comparison_bars.png`
- `figures/mnist_attack_comparison_split.png`
- `figures/mnist_attack_examples.png`

The experiment is local to this project: it uses a public research dataset, trains real classifiers, applies real poisoning transformations to the training split, and reports the exact poison rates and metrics used.

## Run The Fungi Experiment

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

For the fish freshness dataset from the attached FoodDDP PDF, use the flat `__fish2/` class-folder layout. The safety-critical direction is non-fresh fish being accepted as fresh:

```powershell
python scripts\fungi_attack_comparison.py --dataset-root __fish2 --dataset-name fish --source-class non-fresh2 --target-class fresh2 --eval-samples 12 --zoo-samples 4 --epochs 12
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
- `figures/fungi_dataset_samples.png`
- `figures/fungi_dataset_split.png`
- `figures/fungi_reporting_dashboard.png`
- `figures/attack_success_by_stage.png`
- `figures/periodic_table_data_poisoning.png`
- `figures/periodic_table_data_poisoning.svg`
- `figures/paper_taxonomy_panels.png`

The fungi dataset and pretrained-weight cache are intentionally ignored by git.
