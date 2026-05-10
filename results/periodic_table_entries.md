# Risk Matrix Entries

| number | symbol | technique | risk | mechanism | mechanism_name | stage | section | risk_row | mechanism_column | slot |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Ss | Split-view web | R1 | S | Supply Chain | Poison | poisoning | 0 | 5 | 0 |
| 2 | Sd | Single-doc RAG | R1 | R | Retrieval / RAG | Poison | poisoning | 0 | 6 | 0 |
| 3 | Fr | Frontrun web | R1 | S | Supply Chain | Poison | poisoning | 0 | 5 | 1 |
| 4 | Wa | WaNet | R1 | B | Backdoor / Trigger | Poison | poisoning | 0 | 3 | 0 |
| 5 | Rg | RAG poison | R1 | R | Retrieval / RAG | Poison | poisoning | 0 | 6 | 1 |
| 6 | Ns | Nightshade | R1 | N | Generative | Poison | poisoning | 0 | 7 | 0 |
| 7 | Fl | Fed replace | R1 | D | Distributed / Federated | Poison | poisoning | 0 | 4 | 0 |
| 8 | Db | DBA | R1 | D | Distributed / Federated | Poison | poisoning | 0 | 4 | 1 |
| 9 | Sp | Subpopulation | R2 | M | Minimal / Structural | Poison | poisoning | 1 | 0 | 0 |
| 10 | Wb | Witches Brew | R2 | G | Gradient | Poison | poisoning | 1 | 2 | 0 |
| 11 | Bn | BadNets | R2 | B | Backdoor / Trigger | Poison | poisoning | 1 | 3 | 0 |
| 12 | Hb | Hidden backdoor | R2 | B | Backdoor / Trigger | Poison | poisoning | 1 | 3 | 1 |
| 13 | Sa | Sleeper agent | R1 | B | Backdoor / Trigger | Poison | poisoning | 0 | 3 | 1 |
| 14 | Bv | Best-of-Venom | R2 | P | Preference / RLHF | Poison | poisoning | 1 | 8 | 0 |
| 15 | Rl | RLHF backdoor | R2 | P | Preference / RLHF | Poison | poisoning | 1 | 8 | 1 |
| 16 | Td | TrojDiff | R2 | N | Generative | Poison | poisoning | 1 | 7 | 0 |
| 17 | Ip | Input-aware BD | R2 | B | Backdoor / Trigger | Poison | poisoning | 1 | 3 | 2 |
| 18 | Gg | Gradient RAG | R2 | R | Retrieval / RAG | Poison | poisoning | 1 | 6 | 0 |
| 19 | Tf | Target label flip | R3 | M | Minimal / Structural | Poison | poisoning | 2 | 0 | 0 |
| 20 | Fc | Feature collision | R1 | O | Optimization | Poison | poisoning | 0 | 1 | 0 |
| 21 | Bp | Bullseye poly | R2 | O | Optimization | Poison | poisoning | 1 | 1 | 0 |
| 22 | If | Influence | R2 | O | Optimization | Poison | poisoning | 1 | 1 | 1 |
| 23 | Hp | Hessian | R2 | O | Optimization | Poison | poisoning | 1 | 1 | 2 |
| 24 | Cp | Curriculum | R2 | O | Optimization | Poison | poisoning | 1 | 1 | 3 |
| 25 | Cl | Clean-label BD | R1 | B | Backdoor / Trigger | Poison | poisoning | 0 | 3 | 2 |
| 26 | Nl | NLP trigger | R2 | B | Backdoor / Trigger | Poison | poisoning | 1 | 3 | 3 |
| 27 | Al | ALIE | R2 | D | Distributed / Federated | Poison | poisoning | 1 | 4 | 0 |
| 28 | Gd | Glaze | R3 | N | Generative | Defense | poisoning | 2 | 7 | 0 |
| 29 | Dp | DPO poison | R3 | P | Preference / RLHF | Poison | poisoning | 2 | 8 | 0 |
| 30 | Rp | Reward poison | R3 | P | Preference / RLHF | Poison | poisoning | 2 | 8 | 1 |
| 31 | Lf | Random label flip | R4 | M | Minimal / Structural | Poison | poisoning | 3 | 0 | 0 |
| 32 | Svm | SVM poison | R4 | O | Optimization | Poison | poisoning | 3 | 1 | 0 |
| 33 | Ko | KNN poison | R3 | M | Minimal / Structural | Poison | poisoning | 2 | 0 | 1 |
| 34 | Oo | Outlier inject | R4 | M | Minimal / Structural | Poison | poisoning | 3 | 0 | 1 |
| 35 | RgN | Regression poison | R4 | O | Optimization | Poison | poisoning | 3 | 1 | 1 |
| 36 | Ts | Time-series | R3 | M | Minimal / Structural | Poison | poisoning | 2 | 0 | 2 |
| 37 | Cf | Collab filter | R2 | M | Minimal / Structural | Poison | poisoning | 1 | 0 | 1 |
| 38 | Js | JSMA | R3 | M | Minimal / Structural | Evasion | evasion_analog | 2 | 0 | 0 |
| 39 | Sf | SparseFool | R4 | M | Minimal / Structural | Evasion | evasion_analog | 3 | 0 | 0 |
| 40 | Fg | FGSM | R3 | G | Gradient | Evasion | evasion_analog | 2 | 2 | 0 |
| 41 | Pg | PGD | R2 | G | Gradient | Evasion | evasion_analog | 1 | 2 | 0 |
| 42 | Df | DeepFool | R3 | M | Minimal / Structural | Evasion | evasion_analog | 2 | 0 | 1 |
| 43 | Cw | Carlini-Wagner | R2 | O | Optimization | Evasion | evasion_analog | 1 | 1 | 0 |
| 44 | Ead | Elastic Net | R2 | O | Optimization | Evasion | evasion_analog | 1 | 1 | 1 |
| 45 | Zo | ZOO | R3 | O | Optimization | Evasion | evasion_analog | 2 | 1 | 0 |
| 46 | Bd | Boundary | R3 | O | Optimization | Evasion | evasion_analog | 2 | 1 | 1 |
| 47 | Op | One-Pixel | R4 | M | Minimal / Structural | Evasion | evasion_analog | 3 | 0 | 1 |
| 48 | Ap | Adv. patch | R2 | U | Universal / Transfer | Evasion | evasion_analog | 1 | 3 | 0 |
| 49 | Uap | Universal pert. | R2 | U | Universal / Transfer | Evasion | evasion_analog | 1 | 3 | 1 |
| 50 | Eot | EOT | R2 | U | Universal / Transfer | Evasion | evasion_analog | 1 | 3 | 2 |
| 51 | St | Surrogate transfer | R2 | U | Universal / Transfer | Evasion | evasion_analog | 1 | 3 | 3 |
