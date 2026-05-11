# Implementation Coverage

| requirement_group | method | stage | implemented_in | result_row | status |
|---|---|---|---|---|---|
| Poster-required | FGSM | evasion | scripts/fungi_attack_comparison.py::fgsm; optional torchattacks.FGSM backend | FGSM | implemented |
| Poster-required | PGD | evasion | scripts/fungi_attack_comparison.py::pgd; optional torchattacks.PGD backend | PGD | implemented |
| Poster-required | DeepFool | evasion | scripts/fungi_attack_comparison.py::deepfool_l2 | DeepFool L2 | implemented |
| Poster-required | Carlini-Wagner | evasion | scripts/fungi_attack_comparison.py::carlini_wagner_l2_target | Carlini-Wagner L2 | implemented |
| Poster-required | Boundary Attack | evasion | scripts/fungi_attack_comparison.py::boundary_search | Boundary / HopSkipJump search | implemented |
| Poster-required | One-Pixel Attack | evasion | scripts/fungi_attack_comparison.py::one_pixel_de_target | One-Pixel DE | implemented |
| Poster-required | Universal Adversarial Perturbation | evasion | scripts/fungi_attack_comparison.py::universal_adversarial_perturbation | Universal adversarial perturbation | implemented |
| Additional-10 | Clean-Label Poisoning / Poison Frogs | poisoning | scripts/fungi_attack_comparison.py::true_feature_collision | Clean-label Poison Frogs | implemented |
| Additional-10 | Sleeper Agent | poisoning | scripts/fungi_attack_comparison.py::sleeper_agent_poison | Sleeper Agent-style backdoor | implemented |
| Additional-10 | Adversarial Patch | evasion | scripts/fungi_attack_comparison.py::adversarial_patch_eot | Adversarial patch | implemented |
| Additional-10 | Witches' Brew / Gradient Matching | poisoning | scripts/fungi_attack_comparison.py::gradient_matching_poison | Witches' Brew gradient match | implemented |
| Additional-10 | JSMA | evasion | scripts/fungi_attack_comparison.py::vectorized_jsma_saliency | JSMA saliency | implemented |
| Additional-10 | ZOO | evasion | scripts/fungi_attack_comparison.py::zoo_target | ZOO finite difference | implemented |
| Additional-10 | SparseFool | evasion | scripts/fungi_attack_comparison.py::sparse_boundary | SparseFool-style boundary | implemented |
| Additional-10 | Elastic Net EAD | evasion | scripts/fungi_attack_comparison.py::ead_target | Elastic Net EAD | implemented |
| Additional-10 | Subpopulation Poisoning | poisoning | scripts/fungi_attack_comparison.py::subpopulation_label_flip | Subpopulation label poison | implemented |
| Additional-10 | BadNets Patch Backdoor | poisoning | scripts/fungi_attack_comparison.py::backdoor_dataset | BadNets patch backdoor | implemented |
| Extra implemented | Random Label Flipping | poisoning | scripts/fungi_attack_comparison.py::random_label_flip | Random label flip | implemented |
| Extra implemented | Targeted Label Flipping | poisoning | scripts/fungi_attack_comparison.py::targeted_label_flip | Targeted label flip | implemented |
| Extra implemented | Clean-Label Patch Backdoor | poisoning | scripts/fungi_attack_comparison.py::clean_label_backdoor_dataset | Clean-label patch backdoor | implemented |
| Implementation-upgrade | Food transfer-learning baseline | reference | scripts/fungi_attack_comparison.py::TransferFungiModel | Clean baseline | implemented |
| Implementation-upgrade | Frozen-feature linear probing cache | training | scripts/fungi_attack_comparison.py::train_model(cache_frozen_features=True) | Seafood cached frozen features | implemented |
| Implementation-upgrade | PGD epsilon sweep | evaluation | scripts/fungi_attack_comparison.py::run_pgd_epsilon_sweep | *_attack_sweep.* | implemented |
