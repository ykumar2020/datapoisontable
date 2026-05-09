# Data Poisoning Risk Toolkit

This folder contains a cleaned-up technical expansion for the data poisoning project.

The original poster framed several test-time adversarial examples as data poisoning. The stronger framing is:

- **Data poisoning** changes training, fine-tuning, feedback, retrieval, or supply-chain data.
- **Backdoors** are a high-risk poisoning subfamily because they preserve normal accuracy while adding trigger-activated behavior.
- **Evasion attacks** such as FGSM, PGD, CW, DeepFool, Boundary, One-Pixel, and UAP are related adversarial ML techniques, but they are not data poisoning unless used to craft poisoned training samples.

## Contents

- `catalog/data_poisoning_techniques.csv` - expanded technique catalog with risk scores and primary sources.
- `docs/poster_upgrade_notes.md` - recommended changes to the poster/table.
- `docs/data_poisoning_theory.tex` - IEEE two-column conference-style LaTeX paper with abstract, introduction, related work, methodology, results, discussion, and conclusion.
- `scripts/mnist_poisoning_experiment.py` - actual training-time poisoning implementations on real MNIST: random label flipping, targeted source-to-target label flipping, and a BadNets-style visible trigger backdoor.
- `scripts/mnist_attack_comparison.py` - 14-method MNIST comparison suite covering poisoning, backdoors, and related evasion attacks.
- `scripts/fungi_attack_comparison.py` - fungi dataset comparison using lazy PyTorch datasets, a pretrained MobileNetV3-Small transfer model, conditional evasion metrics, true latent-space clean-label Poison Frogs, gradient-matching poisoning, Sleeper-Agent-style backdoor, EOT adversarial patch, vectorized JSMA, batched ZOO, and other baseline attacks.
- `scripts/generate_reporting_artifacts.py` - derived report tables and figures: fungi dataset statistics, sample image grid, dataset split chart, grouped attack statistics, reporting dashboard, and cross-dataset stage comparison.
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

The expanded comparison additionally writes:

- `results/mnist_attack_comparison.csv`
- `results/mnist_attack_comparison.json`
- `results/mnist_attack_comparison.md`
- `figures/mnist_attack_comparison_bars.png`
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

Generate the additional reporting artifacts:

```powershell
python scripts\generate_reporting_artifacts.py
```

The script writes:

- `results/fungi_attack_comparison.csv`
- `results/fungi_attack_comparison.json`
- `results/fungi_attack_comparison.md`
- `results/fungi_dataset_statistics.csv`
- `results/fungi_dataset_statistics.json`
- `results/fungi_dataset_statistics.md`
- `results/attack_summary_statistics.csv`
- `results/attack_summary_statistics.json`
- `results/attack_summary_statistics.md`
- `figures/fungi_attack_comparison_bars.png`
- `figures/fungi_attack_examples.png`
- `figures/fungi_dataset_samples.png`
- `figures/fungi_dataset_split.png`
- `figures/fungi_reporting_dashboard.png`
- `figures/attack_success_by_stage.png`

The fungi dataset and pretrained-weight cache are intentionally ignored by git.
