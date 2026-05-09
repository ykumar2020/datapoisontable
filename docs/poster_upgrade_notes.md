# Poster Upgrade Notes

## Main Correction

The current poster includes FGSM, PGD, DeepFool, Carlini-Wagner, Boundary, One-Pixel, and UAP in the periodic table. Those are mostly **test-time evasion/adversarial-example attacks**, not data poisoning. They can remain in a small "related adversarial ML" box, but the main periodic table should focus on attacks that contaminate:

- training datasets
- labels or rewards
- pretraining/fine-tuning corpora
- preference/RLHF datasets
- retrieval/RAG knowledge bases
- federated updates or client data
- model supply-chain artifacts

## Better Column Blocks

Use mechanism-based blocks that fit real data poisoning literature:

| Block | Meaning | Examples |
|---|---|---|
| A | Availability poisoning | random label flips, outlier injection, regression poisoning |
| T | Targeted integrity poisoning | targeted label flips, feature collision, gradient matching, subpopulation poisoning |
| B | Backdoor/trigger poisoning | BadNets, clean-label backdoors, hidden trigger, WaNet, NLP triggers |
| S | Supply-chain and web-scale poisoning | split-view poisoning, frontrunning poisoning, pretrained weight poisoning |
| F | Federated/distributed poisoning | model replacement, distributed backdoors, Byzantine update attacks |
| R | Retrieval/RAG poisoning | PoisonedRAG, single-document corpus poisoning, metadata poisoning |
| G | Generative/multimodal poisoning | Nightshade, diffusion model Trojans, diffusion backdoors |
| P | Preference/reward poisoning | RLHF preference poisoning, reward-model poisoning, RL reward poisoning |

## Risk Formula

Keep the poster's idea, but make the score explicit:

```text
danger_score = impact x stealth x scalability x cost_efficiency
```

Each factor is scored from 1 to 5, where 5 means worse for defenders. Cost-efficiency means the attack is cheaper/easier for an adversary, not that it is costly.

Suggested levels:

| Level | Score | Meaning |
|---|---:|---|
| R1 Extreme | 400-625 | severe impact, stealthy, scalable, or practical in modern pipelines |
| R2 High | 200-399 | strong real-world concern, but with more constraints |
| R3 Moderate | 100-199 | meaningful but narrower, noisier, or more detectable |
| R4 Low | 1-99 | mainly educational, costly, or limited to constrained settings |

## Stronger Results Section

Replace "most mathematically powerful" with a clearer takeaway:

> The highest operational risk comes from poisoning that is stealthy, durable, and aligned with modern ML pipelines: web-scale dataset poisoning, supply-chain/pretrained-model backdoors, RAG knowledge-base poisoning, federated model replacement, and trigger-based backdoors. Simple label flipping remains useful for demonstrations, but it is easier to detect and usually causes visible accuracy loss.

## MNIST Technical Experiments To Report

The runnable script in `scripts/mnist_poisoning_experiment.py` uses the real MNIST handwritten-digit dataset and reports:

- clean baseline accuracy on the official MNIST test split
- random label-flip availability poisoning
- targeted source-to-target label poisoning, defaulting to digit `7` toward digit `1`
- visible patch-trigger backdoor poisoning with clean accuracy and attack success rate
- figures showing clean digits, poisoned trigger examples, metric bars, and confusion matrices

This gives a visually clear conference presentation: the audience can see the handwritten digits, see the trigger, and compare clean accuracy against targeted failure behavior.

The expanded script `scripts/mnist_attack_comparison.py` implements a larger comparison suite: random label flipping, targeted label flipping, subpopulation poisoning, BadNets patch backdoor, clean-label patch backdoor, prototype feature collision, FGSM, PGD, EAD, JSMA, SparseFool-style sparse boundary attack, adversarial patch, ZOO finite-difference attack, and a HopSkipJump-style boundary search. Its summary table is written to `results/mnist_attack_comparison.md`.

## Suggested Future Work

- Add a "data poisoning vs evasion" comparison panel.
- Validate the risk scores with consistent MNIST experiments and literature-backed case studies.
- Add defenses beside each block: provenance, deduplication, anomaly detection, trusted data snapshots, robust aggregation, backdoor scanning, and retrieval-source auditing.
- Extend the visual periodic table to LLM/RAG/federated settings, which are now more relevant than only image classifiers.
