# MNIST Attack Epsilon Sweep

| Attack | Eps | Step | Iters | Success | Meets threshold | Backend |
|---|---:|---:|---:|---:|---|---|
| PGD | 0.04 | 0.008 | 5 | 0.065 | False | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
| PGD | 0.08 | 0.016 | 5 | 0.19 | False | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
| PGD | 0.12 | 0.024 | 5 | 0.385 | False | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
| PGD | 0.16 | 0.032 | 5 | 0.65 | False | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
| PGD | 0.2 | 0.04 | 5 | 0.825 | True | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
| PGD | 0.25 | 0.05 | 5 | 0.96 | True | local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD |
