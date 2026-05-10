# Fish Attack Comparison

Few-shot micro-batch proof of concept. The validation split has only 10 images, with five non-fresh validation samples in the safety-critical source class, so rates such as 0.8000 and 0.4000 correspond to 4/5 and 2/5 images. Perfect clean accuracy in poisoned rows should be read as tiny-split variance/overfitting, not as evidence that poisoned data improves the classifier.

| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| Clean baseline | reference | none | none | 0 | 0.8 | non-fresh2_to_fresh2_confusion | 0.0 |  |  | Dataset=fish; split=deterministic stratified split from flat class folders; val_fraction=0.25; model=mobilenet_v3_small, pretrained=True, freeze_features=True; classes=['fresh2', 'non-fresh2']. |
| Random label flip | poisoning | M | R3 | 10.00% | 0.9 | clean_accuracy_drop | 0.0 |  |  | 3 labels changed. |
| Targeted label flip | poisoning | M | R2 | 35% of non-fresh2 | 0.8 | non-fresh2_to_fresh2_confusion | 0.0 |  |  | 5 source labels changed. |
| Subpopulation label poison | poisoning | M | R2 | 80% of selected subgroup | 0.8 | subpopulation_to_target_confusion | 0.0 |  |  | 5 source-subgroup labels changed. |
| BadNets patch backdoor | poisoning | B | R1 | 25.00% appended | 1.0 | trigger_ASR_to_fresh2 | 0.0 |  |  | 8 triggered source copies. |
| Clean-label patch backdoor | poisoning | B | R1 | 25.00% target-class copies | 1.0 | trigger_ASR_to_fresh2 | 0.0 |  |  | 8 correctly labeled target images patched. |
| Sleeper Agent-style backdoor | poisoning | B | R1 | 15 appended | 1.0 | stealth_trigger_ASR_to_fresh2 | 0.8 |  |  | Clean-label target poisons optimized by gradient alignment to a low-amplitude trigger objective. |
| Clean-label Poison Frogs | poisoning | O | R1 | 15 appended | 1.0 | non-fresh2_to_fresh2_confusion | 0.0 |  |  | Optimized fresh2 clean-label poisons toward non-fresh2 latent features with L-infinity projection. |
| Witches' Brew gradient match | poisoning | G | R2 | 15 appended | 1.0 | non-fresh2_to_fresh2_confusion | 0.0 |  |  | Clean-label fresh2 poisons optimized so classifier-parameter gradients align with a non-fresh2-to-fresh2 target objective. |
| FGSM | evasion | G | R3 | 0 | 0.8 | conditional_untargeted_misclassification | 1.0 | 73686.4 | 0.08 | Single-step gradient sign. |
| PGD | evasion | G | R2 | 0 | 0.8 | conditional_untargeted_misclassification | 1.0 | 75083.6 | 0.1 | Projected gradient ascent. |
| DeepFool L2 | evasion | M | R3 | 0 | 0.8 | conditional_untargeted_misclassification | 1.0 | 74704.6 | 0.0115 | Poster method: iterative local linearization of the nearest logit boundary. |
| Carlini-Wagner L2 | evasion | O | R2 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 1.0 | 72680.4 | 0.0618 | Poster method: targeted CW margin objective with L2 regularization. |
| Elastic Net EAD | evasion | O | R2 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.8 | 20093.0 | 0.0152 | Formal ISTA proximal solver for targeted elastic-net objective. |
| JSMA saliency | evasion | M | R3 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.4 | 224.4 | 1.0 | Vectorized sparse-feature saliency using input Jacobian. |
| One-Pixel DE | evasion | M | R4 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.0 | 3.0 | 0.8031 | Poster method: differential evolution over one pixel coordinate and RGB value. |
| SparseFool-style boundary | evasion | M | R4 | 0 | 0.8 | conditional_untargeted_misclassification | 0.0 | 16.2 | 0.96 | Vectorized sparse local-boundary crossing approximation. |
| Universal adversarial perturbation | evasion | U | R2 | 0 | 0.8 | conditional_untargeted_misclassification | 1.0 | 73786.4 | 0.075 | Poster method: one learned L-infinity perturbation shared by all source-class images. |
| Adversarial patch | evasion | U | R2 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.0 | 3034.4 | 0.9937 | EOT-trained patch with random training locations; evaluated lower-right. |
| ZOO finite difference | evasion | O | R3 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.0 | 602.75 | 0.08 | Batched black-box finite differences; 5160 model queries. |
| Boundary / HopSkipJump search | evasion | O | R3 | 0 | 0.8 | conditional_targeted_ASR_to_fresh2 | 0.8 | 76383.6 | 0.6807 | Poster Boundary method represented by decision-only binary search to target-class guide images. |
