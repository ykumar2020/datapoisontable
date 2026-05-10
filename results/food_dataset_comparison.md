# Food Dataset Comparison

| dataset | methods | train_size | clean_accuracy | critical_confusion_metric | critical_confusion | top_poisoning_method | top_poisoning_success | top_evasion_method | top_evasion_success |
|---|---|---|---|---|---|---|---|---|---|
| fungi | 20 | 360 | 0.6667 | poisonous_to_edible_confusion | 0.2708 | Subpopulation label poison | 0.6842 | PGD | 1.0 |
| fish | 20 | 30 | 0.8 | non-fresh2_to_fresh2_confusion | 0.0 | Sleeper Agent-style backdoor | 0.8 | FGSM | 1.0 |
