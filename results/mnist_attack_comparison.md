# MNIST Attack Comparison

| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| Clean baseline | reference | none | none | 0 | 0.94 | 7_to_1_confusion | 0.01 |  |  | PyTorch 2-layer CNN; defense=none; undefended training. |
| Random label flip | poisoning | M | R3 | 10.00% | 0.925 | clean_accuracy_drop | 0.015 |  |  | 600 labels changed; undefended training. |
| Targeted label flip | poisoning | M | R2 | 35% of digit 7 | 0.945 | 7_to_1_confusion | 0.08 |  |  | 210 source labels changed; undefended training. |
| Subpopulation label poison | poisoning | M | R2 | 80% of selected subgroup | 0.929 | subpopulation_to_target_confusion | 0.5667 |  |  | 144 source-subgroup labels changed; undefended training. |
| BadNets patch backdoor | poisoning | B | R1 | 5.00% appended | 0.944 | trigger_ASR_to_1 | 1.0 |  |  | 300 triggered non-target copies; undefended training. |
| Clean-label patch backdoor | poisoning | B | R1 | 5.00% target-class copies | 0.953 | trigger_ASR_to_1 | 0.37 |  |  | 300 correctly labeled target images patched; undefended training. |
| True feature collision | poisoning | O | R1 | 600 appended | 0.952 | 1_to_7_confusion | 0.0 |  |  | Clean-label source poisons optimized in CNN latent space with L-infinity projection; undefended training. |
| FGSM | evasion | G | R3 | 0 | 0.94 | conditional_untargeted_misclassification | 0.76 | 481.69 | 0.22 | eps=0.22; local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD. |
| PGD | evasion | G | R2 | 0 | 0.94 | conditional_untargeted_misclassification | 0.96 | 536.49 | 0.25 | eps=0.25, step=0.05, iters=5; local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD. |
| PGD eps sweep | evasion | G | R2 | 0 | 0.94 | lowest_eps_for_threshold | 0.2 |  | 0.2 | Lowest eps reaching >= 0.80 conditional success; success=0.825; local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD. |
| Elastic Net EAD | evasion | O | R2 | 0 | 0.94 | conditional_targeted_ASR_to_1 | 0.0 | 10.97 | 0.0016 | Formal ISTA proximal solver for targeted elastic-net objective. |
| JSMA saliency | evasion | M | R3 | 0 | 0.94 | conditional_targeted_ASR_to_1 | 0.575 | 36.4 | 0.9953 | Vectorized sparse-pixel saliency using CNN input Jacobian. |
| SparseFool-style boundary | evasion | M | R4 | 0 | 0.94 | conditional_untargeted_misclassification | 0.99 | 14.48 | 0.9901 | Sparse gradient boundary approximation for the CNN baseline. |
| Adversarial patch | evasion | U | R2 | 0 | 0.94 | conditional_untargeted_misclassification | 0.76 | 56.59 | 1.0 | Universal fixed-position patch optimized by CNN loss gradients. |
| ZOO finite difference | evasion | O | R3 | 0 | 0.94 | conditional_targeted_ASR_to_1 | 0.125 | 224.5 | 0.3013 | Black-box coordinate finite differences; 8255 model queries. |
| HopSkipJump-style boundary | evasion | O | R3 | 0 | 0.94 | conditional_targeted_ASR_to_1 | 0.99 | 192.04 | 0.5506 | Decision-only binary search between clean input and target-class guide image. |
