# Fungi Attack Comparison

| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| Clean baseline | reference | none | none | 0 | 0.6667 | poisonous_to_edible_confusion | 0.2708 |  |  | Model=mobilenet_v3_small, pretrained=True, freeze_features=True; classes=['edible', 'poisonous']. |
| Random label flip | poisoning | M | R3 | 10.00% | 0.5778 | clean_accuracy_drop | 0.0889 |  |  | 36 labels changed. |
| Targeted label flip | poisoning | M | R2 | 35% of poisonous | 0.7111 | poisonous_to_edible_confusion | 0.3125 |  |  | 71 source labels changed. |
| Subpopulation label poison | poisoning | M | R2 | 80% of selected subgroup | 0.5667 | subpopulation_to_target_confusion | 0.6842 |  |  | 65 source-subgroup labels changed. |
| BadNets patch backdoor | poisoning | B | R1 | 25.00% appended | 0.5778 | trigger_ASR_to_edible | 0.2941 |  |  | 90 triggered source copies. |
| Clean-label patch backdoor | poisoning | B | R1 | 25.00% target-class copies | 0.6111 | trigger_ASR_to_edible | 0.1714 |  |  | 90 correctly labeled target images patched. |
| Sleeper Agent-style backdoor | poisoning | B | R1 | 40 appended | 0.5667 | stealth_trigger_ASR_to_edible | 0.6296 |  |  | Clean-label target poisons optimized by gradient alignment to a low-amplitude trigger objective. |
| Clean-label Poison Frogs | poisoning | O | R1 | 60 appended | 0.6333 | poisonous_to_edible_confusion | 0.3542 |  |  | Optimized edible clean-label poisons toward poisonous latent features with L-infinity projection. |
| Witches' Brew gradient match | poisoning | G | R2 | 40 appended | 0.6889 | poisonous_to_edible_confusion | 0.25 |  |  | Clean-label poisons optimized so classifier-parameter gradients align with a poisonous-to-edible target objective. |
| FGSM | evasion | G | R3 | 0 | 0.6667 | conditional_untargeted_misclassification | 0.8571 | 76727.0 | 0.08 | Single-step gradient sign. |
| PGD | evasion | G | R2 | 0 | 0.6667 | conditional_untargeted_misclassification | 1.0 | 76755.86 | 0.1 | Projected gradient ascent. |
| DeepFool L2 | evasion | M | R3 | 0 | 0.6667 | conditional_untargeted_misclassification | 1.0 | 75912.62 | 0.0133 | Poster method: iterative local linearization of the nearest logit boundary. |
| Carlini-Wagner L2 | evasion | O | R2 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.7429 | 76675.54 | 0.0769 | Poster method: targeted CW margin objective with L2 regularization. |
| Elastic Net EAD | evasion | O | R2 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.0286 | 1303.06 | 0.0039 | Formal ISTA proximal solver for targeted elastic-net objective. |
| JSMA saliency | evasion | M | R3 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.9375 | 134.25 | 0.9539 | Vectorized sparse-feature saliency using input Jacobian. |
| One-Pixel DE | evasion | M | R4 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.0625 | 3.0 | 0.757 | Poster method: differential evolution over one pixel coordinate and RGB value. |
| SparseFool-style boundary | evasion | M | R4 | 0 | 0.6667 | conditional_untargeted_misclassification | 0.0625 | 41.81 | 0.889 | Vectorized sparse local-boundary crossing approximation. |
| Universal adversarial perturbation | evasion | U | R2 | 0 | 0.6667 | conditional_untargeted_misclassification | 1.0 | 66187.95 | 0.08 | Poster method: one learned L-infinity perturbation shared by all source-class images. |
| Adversarial patch | evasion | U | R2 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.2857 | 3071.4 | 0.9715 | EOT-trained patch with random training locations; evaluated lower-right. |
| ZOO finite difference | evasion | O | R3 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.125 | 565.0 | 0.07 | Batched black-box finite differences; 9159 model queries. |
| Boundary / HopSkipJump search | evasion | O | R3 | 0 | 0.6667 | conditional_targeted_ASR_to_edible | 0.6875 | 76407.56 | 0.6702 | Poster Boundary method represented by decision-only binary search to target-class guide images. |
