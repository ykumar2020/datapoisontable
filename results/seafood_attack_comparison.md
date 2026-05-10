# Seafood Attack Comparison

| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| Clean baseline | reference | none | none | 0 | 1.0 | NonFresh_to_Fresh_confusion | 0.0 |  |  | Dataset=seafood; split=deterministic stratified split from class folders; label_mode=leaf; val_fraction=0.25; excluded=['__pycache__', 'working']; model=mobilenet_v3_small, pretrained=True, freeze_features=True; cache_frozen_features=True; attack_scale=0.12; classes=['Fresh', 'NonFresh']. |
| Random label flip | poisoning | M | R3 | 10.00% | 1.0 | clean_accuracy_drop | 0.0 |  |  | 30 labels changed. |
| Targeted label flip | poisoning | M | R2 | 35% of NonFresh | 0.98 | NonFresh_to_Fresh_confusion | 0.04 |  |  | 52 source labels changed. |
| Subpopulation label poison | poisoning | M | R2 | 80% of selected subgroup | 0.99 | subpopulation_to_target_confusion | 0.05 |  |  | 48 source-subgroup labels changed. |
| BadNets patch backdoor | poisoning | B | R1 | 25.00% appended | 0.99 | trigger_ASR_to_Fresh | 0.5306 |  |  | 75 triggered source copies. |
| Clean-label patch backdoor | poisoning | B | R1 | 25.00% target-class copies | 1.0 | trigger_ASR_to_Fresh | 0.0 |  |  | 75 correctly labeled target images patched. |
| Sleeper Agent-style backdoor | poisoning | B | R1 | 5 appended | 1.0 | stealth_trigger_ASR_to_Fresh | 0.94 |  |  | Clean-label target poisons optimized by gradient alignment to a low-amplitude trigger objective. |
| Clean-label Poison Frogs | poisoning | O | R1 | 7 appended | 1.0 | NonFresh_to_Fresh_confusion | 0.0 |  |  | Optimized Fresh clean-label poisons toward NonFresh latent features with L-infinity projection. |
| Witches' Brew gradient match | poisoning | G | R2 | 5 appended | 1.0 | NonFresh_to_Fresh_confusion | 0.0 |  |  | Clean-label Fresh poisons optimized so classifier-parameter gradients align with a NonFresh-to-Fresh target objective. |
| FGSM | evasion | G | R3 | 0 | 1.0 | conditional_untargeted_misclassification | 1.0 | 27593.5 | 0.08 | Single-step gradient sign. |
| PGD | evasion | G | R2 | 0 | 1.0 | conditional_untargeted_misclassification | 1.0 | 27612.88 | 0.1 | Projected gradient ascent. |
| DeepFool L2 | evasion | M | R3 | 0 | 1.0 | conditional_untargeted_misclassification | 1.0 | 27523.25 | 0.005 | Poster method: iterative local linearization of the nearest logit boundary. |
| Carlini-Wagner L2 | evasion | O | R2 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 1.0 | 27566.25 | 0.0885 | Poster method: targeted CW margin objective with L2 regularization. |
| Elastic Net EAD | evasion | O | R2 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 1.0 | 20711.38 | 0.0076 | Formal ISTA proximal solver for targeted elastic-net objective. |
| JSMA saliency | evasion | M | R3 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 0.25 | 29.25 | 0.8559 | Vectorized sparse-feature saliency using input Jacobian. |
| One-Pixel DE | evasion | M | R4 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 0.0 | 3.0 | 0.5423 | Poster method: differential evolution over one pixel coordinate and RGB value. |
| SparseFool-style boundary | evasion | M | R4 | 0 | 1.0 | conditional_untargeted_misclassification | 0.0 | 29.88 | 0.8289 | Vectorized sparse local-boundary crossing approximation. |
| Universal adversarial perturbation | evasion | U | R2 | 0 | 1.0 | conditional_untargeted_misclassification | 1.0 | 17417.88 | 0.06 | Poster method: one learned L-infinity perturbation shared by all source-class images. |
| Adversarial patch | evasion | U | R2 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 1.0 | 1200.0 | 0.8206 | EOT-trained patch with random training locations; evaluated lower-right. |
| ZOO finite difference | evasion | O | R3 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 0.0 | 24.0 | 0.04 | Batched black-box finite differences; 102 model queries. |
| Boundary / HopSkipJump search | evasion | O | R3 | 0 | 1.0 | conditional_targeted_ASR_to_Fresh | 1.0 | 27642.25 | 0.4219 | Poster Boundary method represented by decision-only binary search to target-class guide images. |
