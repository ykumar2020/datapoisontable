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

## Layout Integrity Update

The conference version should not force the attacks into the literal shape of the chemical periodic table. The final poster figure now uses the periodic-table visual language only at the cell level: each method has a compact element-style tile with a number, symbol, name, and risk/mechanism label. The coordinates are functional:

- rows are strictly R1, R2, R3, and R4
- columns are strictly mechanism families: M, O, G, B, D, S, R, N, and P
- risk levels printed inside each cell must match the row position
- inference-time evasion attacks are below a dashed divider in a separate companion panel, not inside the data-poisoning grid
- `results/table_layout_audit.*` verifies panel assignment, expected risk row, mechanism column, and poisoning-versus-evasion separation

This removes the reviewer-facing contradiction where a cell could be labeled R1 while physically sitting in an R2/R3 row.

## Stronger Results Section

Replace "most mathematically powerful" with a clearer takeaway:

> The highest operational risk comes from poisoning that is stealthy, durable, and aligned with modern ML pipelines: web-scale dataset poisoning, supply-chain/pretrained-model backdoors, RAG knowledge-base poisoning, federated model replacement, and trigger-based backdoors. Simple label flipping remains useful for demonstrations, but it is easier to detect and usually causes visible accuracy loss.

## Food-Image Technical Experiments To Report

The runnable experiment in `scripts/fungi_attack_comparison.py` is now the main empirical benchmark. It evaluates fungi, fish, and the larger seafood freshness dataset using the same transfer-learning/linear-probing threat model and produces comparable CSV, Markdown, JSON, and figure artifacts for the report.

The script implements all original poster evasion methods (FGSM, PGD, DeepFool, Carlini-Wagner, Boundary, One-Pixel, and UAP) plus the additional shared methods: Poison Frogs, Sleeper-Agent-style backdoor, Adversarial Patch, Witches' Brew-style gradient matching, JSMA, ZOO, SparseFool, EAD, subpopulation poisoning, and BadNets. The implementation audit is generated in `results/implementation_coverage.md`.

Important implementation audit: the code reproduces the core experimental mechanisms in a local fungi setting. It should not claim industrial-scale transferability, physical-world patch certification, or long-horizon LLM sleeper-agent persistence.

## Suggested Future Work

- Keep the "data poisoning vs evasion" separation explicit in the visual.
- Validate the risk scores with larger food-image datasets and literature-backed case studies.
- Add defenses beside each block: provenance, deduplication, anomaly detection, trusted data snapshots, robust aggregation, backdoor scanning, and retrieval-source auditing.
- Extend the risk matrix to LLM/RAG/federated settings, which are now more relevant than only image classifiers.
