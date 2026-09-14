# Baselines / Ablations

Two representational baselines are used to evaluate what the reified TQAVE-KG
schema (`etd:Relation`, `etd:Evidence`, `etd:DiscourseUnit`, `etd:FieldAssertion`,
provenance/confidence properties, etc.) contributes over simpler graph
representations. Both are built from the same source data
(`data/ai_ethics_datasets.json`) or the full pipeline output, but with most of
the ontology's structure stripped out.

## `flat_triple_baseline` (this folder)

`scripts/build_flat_triple_baseline.py` takes the frozen manuscript graph
(`output/etd_hub_enhanced_kg47.ttl`) and re-flattens it into plain
subject–predicate–object triples, dropping reified relation/evidence/
discourse/vote/answer/question/theme/expert nodes. It keeps the full ontology
prefix set (aipo, airo, dpv, fmo, etc.).

```bash
python baselines/scripts/build_flat_triple_baseline.py \
    --input output/etd_hub_enhanced_kg47.ttl \
    --output baselines/output/flat_triple_baseline.ttl \
    --audit-output baselines/results/flat_triple_baseline_audit.json \
    --query-results-output baselines/results/flat_triple_query_results.json \
    --queries-output baselines/queries/flat_triple_queries.sparql \
    --log-output baselines/results/flat_triple_generation_log.txt
```

(These are also the script's defaults, so `python baselines/scripts/build_flat_triple_baseline.py` with no arguments reproduces the same files.)

Outputs: [`output/flat_triple_baseline.ttl`](output/flat_triple_baseline.ttl),
[`results/flat_triple_baseline_audit.json`](results/flat_triple_baseline_audit.json),
[`results/flat_triple_query_results.json`](results/flat_triple_query_results.json),
SHACL shapes at [`queries/flat_triple_baseline_shapes.ttl`](queries/flat_triple_baseline_shapes.ttl),
query set at [`queries/flat_triple_queries.sparql`](queries/flat_triple_queries.sparql).

## `representational_baseline/` (QA-only, no semantics)

`representational_baseline/scripts/build_qa_only_baseline.py` builds directly
from `data/ai_ethics_datasets.json` (not from pipeline output) and allows
**only** the bare discourse skeleton — `Theme`, `Question`, `Answer`, `Expert`,
`Vote` — explicitly excluding all relation/evidence/provenance predicates
(`subject`, `predicate`, `object`, `hasEvidence`, `evidenceText`,
`confidence`, etc.). It's the more minimal of the two baselines: no semantic
extraction at all, just the raw Q&A/voting structure.

```bash
python baselines/representational_baseline/scripts/build_qa_only_baseline.py \
    --input data/ai_ethics_datasets.json \
    --output baselines/representational_baseline/output/qa_only_baseline.ttl \
    --audit-output baselines/representational_baseline/results/qa_only_baseline_audit.json \
    --query-results-output baselines/representational_baseline/results/qa_only_query_results.json \
    --queries-output baselines/representational_baseline/queries/qa_only_queries.sparql \
    --log-output baselines/representational_baseline/results/qa_only_generation_log.txt
```

**Note:** the script's own `--output`/`--audit-output`/etc. *defaults* point
at `baselines/output/...` and `baselines/results/...` (i.e. the top-level
`flat_triple_baseline` paths) — running it with no arguments will **not**
reproduce the files currently checked in under
`representational_baseline/`. Use the explicit flags above (matching the
paths the checked-in `representational_baseline/` artifacts actually live
at) to regenerate them correctly.

Outputs: [`representational_baseline/output/qa_only_baseline.ttl`](representational_baseline/output/qa_only_baseline.ttl),
[`representational_baseline/results/`](representational_baseline/results/),
[`representational_baseline/queries/`](representational_baseline/queries/).

## Comparing against the full KG

For an apples-to-apples comparison, run the EvQ1–EvQ8 queries (see
[`../queries/README.md`](../queries/README.md)) or the baseline-specific
`.sparql` query sets above against each of the three graphs:
`output/etd_hub_enhanced_kg47.ttl` (full reified KG), `flat_triple_baseline.ttl`,
and `qa_only_baseline.ttl`. See [`../RESULTS_MAP.md`](../RESULTS_MAP.md) for
which paper table/figure (if any) each comparison feeds.
