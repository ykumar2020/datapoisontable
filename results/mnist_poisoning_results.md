# MNIST Poisoning Experiment Results

Dataset: real MNIST handwritten digits loaded through `torchvision.datasets.MNIST`.

| Scenario | Poison rate | Clean accuracy | Attack metric | Value | Train size | Notes |
|---|---:|---:|---|---:|---:|---|
| dataset | 0 |  | train/test images |  | 60000 | MNIST train=60000, test=10000, image_shape=28x28 |
| clean_baseline | 0 | 0.9153 | 7_to_1_confusion | 0.0117 | 60000 | Unpoisoned reference model |
| random_label_flip | 10.00% | 0.8935 | clean_accuracy_drop | 0.0218 | 60000 | 6000 real MNIST labels changed |
| targeted_label_flip | 35.00% of digit 7 | 0.8859 | 7_to_1_confusion | 0.1761 | 60000 | 2193 digit-7 labels changed to 1 |
| visible_patch_backdoor | 5.00% appended triggered samples | 0.9164 | trigger_ASR_to_1 | 0.9985 | 63000 | 3000 non-target MNIST images copied with 5x5 trigger |
