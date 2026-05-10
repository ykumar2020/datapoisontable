# Visual Taxonomy Audit

This note records the correction from a literal chemical-table layout to a functional risk matrix.

## Reviewer Risk

The earlier chemical-table shape was visually memorable but structurally fragile. Chemical-table gaps imply physical properties; in an attack taxonomy they create arbitrary empty space. Worse, some cells could be labeled with one R-level while physically appearing in a different visual row.

## Adopted Rule

The final figure uses periodic-table-style cells, but the grid coordinates are no longer chemical:

| Coordinate | Meaning |
|---|---|
| Row | Exact operational risk level: R1, R2, R3, or R4 |
| Column | Mechanism family |
| Panel | Threat stage: data poisoning grid or related evasion analogs |

Inference-time attacks such as FGSM, PGD, JSMA, ZOO, SparseFool, CW, and UAP remain available for comparison, but they are displayed in a separate evasion-analog panel and are not counted as data poisoning.

## Generated Audit

`scripts/generate_reporting_artifacts.py` writes `results/table_layout_audit.csv`, `.json`, and `.md`. Each row reports the technique, printed risk label, expected visual row, mechanism column, and a status value. The expected status for every row is `consistent`.

