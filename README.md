# Data Poisoning Risk Toolkit

This folder contains a cleaned-up technical expansion for the data poisoning project.

The original poster framed several test-time adversarial examples as data poisoning. The stronger framing is:

- **Data poisoning** changes training, fine-tuning, feedback, retrieval, or supply-chain data.
- **Backdoors** are a high-risk poisoning subfamily because they preserve normal accuracy while adding trigger-activated behavior.
- **Evasion attacks** such as FGSM, PGD, CW, DeepFool, Boundary, One-Pixel, and UAP are related adversarial ML techniques, but they are not data poisoning unless used to craft poisoned training samples.

## Contents

- `catalog/data_poisoning_techniques.csv` - expanded technique catalog with risk scores and primary sources.
- `docs/poster_upgrade_notes.md` - recommended changes to the poster/table.
- `docs/data_poisoning_theory.tex` - LaTeX write-up of the theory, scoring model, MNIST experiment, and metrics.
- `scripts/mnist_poisoning_experiment.py` - actual training-time poisoning implementations on real MNIST: random label flipping, targeted source-to-target label flipping, and a BadNets-style visible trigger backdoor.
- `pptx_media/periodic_table_slide.jpg` - extracted image from the supplied PPTX for reference.

## Run The MNIST Experiment

```powershell
python scripts\mnist_poisoning_experiment.py --download
```

The script uses real MNIST data through `torchvision`, trains real scikit-learn classifiers, and writes:

- `results/mnist_poisoning_results.csv`
- `results/mnist_poisoning_results.json`
- `results/mnist_poisoning_results.md`
- `figures/mnist_clean_samples.png`
- `figures/mnist_backdoor_examples.png`
- `figures/mnist_metric_bars.png`
- `figures/mnist_confusion_*.png`

The experiment is local to this project: it uses a public research dataset, trains real classifiers, applies real poisoning transformations to the training split, and reports the exact poison rates and metrics used.
