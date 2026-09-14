# KG / Data Card

Statistics below are measured directly from the frozen manuscript
artifact, `output/etd_hub_enhanced_kg47.ttl` (run ID
`run-b0eafdda0235eb48`, generated 2026-09-11), not copied from any
cached/older figure. `output/etd_hub_enhanced_kg.ttl` (the default-named
pipeline output also checked in) is byte-for-byte identical to this
file — same SHA-256, same run ID; they are the same graph, not two
different experimental runs. See [`RESULTS_MAP.md`](RESULTS_MAP.md) for
the full verification detail behind every number below.

## Graph statistics (published RDF graph — `output/etd_hub_enhanced_kg47.ttl`)

These are measured directly against the RDF graph itself (`rdflib.Graph().parse(...)`
and SPARQL/type queries) and describe what a reviewer querying the released
file will actually see.

| Metric | Value | Source |
| --- | --- | --- |
| Triples | 38,806 | `output/etd_hub_enhanced_kg47.ttl` |
| SHA-256 | `88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f` | `output/etd_hub_enhanced_kg47.ttl` |
| Run ID | `run-b0eafdda0235eb48` | `etd:runId` on the graph's `etd:Export` node |
| Distinct subjects / predicates / objects | 1,867 / 231 / 5,541 | RDF graph |
| `etd:Entity` nodes | 301 | RDF graph, `rdf:type etd:Entity` |
| `etd:Relation` nodes (total) | 333 | RDF graph — 211 semantic (`relation_kind="extracted_semantic"`) + 122 structural |
| Relations (inferred, this run) | 0 | RDF graph, `etd:isInferred` — all 211 semantic relations are `false` (no code path currently sets this flag `true`) |
| `etd:Evidence` nodes (total) | 337 | RDF graph — 215 attached to semantic relations, 122 to structural, zero overlap |
| `etd:DiscourseUnit` nodes | 174 | RDF graph, `rdf:type etd:DiscourseUnit` |
| `etd:Proposition` nodes | 1 | RDF graph, `rdf:type etd:Proposition` |
| `etd:FieldAssertion` nodes | 317 | RDF graph, `rdf:type etd:FieldAssertion` |
| Entities with a Wikidata identity link | 20/301 (6.64%), all via `skos:closeMatch` (heuristic, non-authoritative) | RDF graph — `owl:sameAs` is reserved for an authoritative identity; 0 entities currently qualify |
| Entities with ontology alignment (`etd:ontology_uri`) | 300/301 (99.67%) | RDF graph |
| Confidence — semantic relations (n=211) | min 0.760, max 0.920, mean 0.9125, median 0.920 | RDF graph, `etd:confidence` |
| Confidence — structural relations (n=122) | min 0.960, max 0.990, mean 0.9775, median 0.980 | RDF graph, `etd:confidence` |
| OWL classes / ObjectProperty / DatatypeProperty / plain RDF Property | 85 / 72 / 223 / 4 | RDF graph |
| SHACL conformance | Conforms: True, 0 violations, 0 warnings, 0 Info-severity results | verified 2026-09-11, `scripts/kg_quality_gate.py`'s SHACL sub-check run directly against this exact file |
| Quality gate (`scripts/kg_quality_gate.py`) | PASS, zero failure reasons | verified 2026-09-11, gate run directly against this exact file |

**Source-level TQAVE coverage**: 12 Theme, 8 Question, 21 Answer, 26 Vote,
57 Expert instances — structurally invariant, driven by
`data/ai_ethics_datasets.json` and unaffected by extraction changes.

**Structural relation preservation**: `createdBy` 12, `belongsToTheme` 8,
`askedBy` 8, `answersQuestion` 21, `authoredBy` 21, `receivesVote` 26,
`castBy` 26 — all exactly match source counts.

**Relation/evidence completeness (semantic relations only)**: **211/211
(100%)** semantic relations have complete subject/predicate/object, ≥1
evidence link, confidence, inference-status, and source provenance
(`scripts/kg_quality_gate.py`'s `relation_missing_*` metrics are all zero
for this file). Of those 211, **207 carry exactly one `etd:hasEvidence`
link and 4 carry two** (a relation citing two corroborating textual
quotes), for **215 distinct semantic `etd:Evidence` nodes** in total —
report "211/211 relations, 215 evidence nodes", not "215/215". All 215
semantic Evidence resources have `etd:evidenceText` — 100%. Field
breakdown (by relation, not evidence node): `theme.description` 126,
`answer.description` 67, `question.body` 18 (sum 211).

**Vote metadata**: 26/26 votes typed and linked; 10 of 21 answers have
≥1 vote (11 without).

**Ontology reuse** (of the 300 aligned entities, by `etd:ontology_uri`
target namespace): FOAF 60, SKOS 58, PROV-O 49, Schema.org 29, DCAT 25,
AIRO 22, AIPO 15, Fairness Metrics Ontology 3, BIBO 2, W3C ORG 1, VAIR 1
(sum 265); the remaining 35 aligned entities map to a project-local
`etd:` term rather than an external ontology. `skos:Concept` 70,
`skos:ConceptScheme` 4, `skos:prefLabel` 12, `skos:altLabel` 14.

Entity-type and relation-type distributions in full are in
`output/etd_hub_enhanced_kg_quality_metrics.json` (see caveat below —
that file's own entity/relation *totals* are pre-publication pipeline
metrics, not graph counts; the type-distribution breakdowns within it
are still useful and are not affected by the same discrepancy).

## Pre-publication pipeline metrics (NOT the published graph)

`output/etd_hub_enhanced_kg_quality_metrics.json` snapshots pipeline
state (`pipeline.global_registry` / `annotations_dict`) at a point
**before** two RDF-build-time steps run: isolated-entity pruning and
duplicate-relation consolidation. Its entity/relation totals are
therefore higher than the published graph's, and describe raw
extraction yield rather than KG size.

| Metric | Value | Source |
| --- | --- | --- |
| Entities (pre-pruning pipeline registry) | 600 | `output/etd_hub_enhanced_kg_quality_metrics.json` |
| Relations (pre-consolidation, doc-summed) | 340 | `output/etd_hub_enhanced_kg_quality_metrics.json` |
| Average entity extraction confidence | 0.889 | `output/etd_hub_enhanced_kg_quality_metrics.json` |
| Wikidata-linked entities (registry-based) | 57 (9.5% linking rate) | `output/etd_hub_enhanced_kg_quality_metrics.json` |
| Documents with entities / relations | see file | `output/etd_hub_enhanced_kg_quality_metrics.json` (`coverage_metrics`) |
| Overall quality score (pipeline's own 0-10 composite) | 6.88 | `output/etd_hub_enhanced_kg_quality_metrics.json` |

The published graph's 301 entities / 333 relations differ from the
600 / 340 pre-publication figures above because of isolated-entity
pruning and duplicate-relation consolidation at RDF-build time (see
`serializers/rdf_serializer.py`'s `prune_isolated_entities()` and
`consolidate_duplicate_reified_relations()`). The exact per-mechanism
breakdown (how many of the 299/7-count difference comes from pruning vs.
consolidation) was not independently re-derived for this specific run
and is intentionally not stated here rather than presenting an
unverified number.

**Do not combine figures from the two tables above into one number** —
e.g. do not compute a "linking rate" using the 301 published-graph
entity count against the 57 figure from the pre-publication population.
Each table is internally consistent; the tables are not cross-compatible.

## Source data provenance

`data/ai_ethics_datasets.json` is a curated set of 12 "Theme" case studies
about AI-ethics incidents (e.g. COMPAS recidivism prediction, healthcare
risk-algorithm bias, mortgage-lending discrimination, AI recruitment
screening) — the same 12 Themes reported in "Source-level TQAVE coverage"
above — each with an expert-authored description, a Q&A discourse
thread (questions/answers per theme), and votes on answers. Each theme
description references and links to the public third-party dataset it
discusses (e.g. ProPublica's COMPAS dataset, MIMIC-CXR, HMDA, NIH Chest
X-ray, UCI Adult/Census Income) — those third-party datasets are **not**
included in this repository and are subject to their own separate licenses
and access terms at the URLs cited in `data/ai_ethics_datasets.json`.

The curated theme/Q&A/vote content itself (`data/ai_ethics_datasets.json`)
does not carry an explicit license marking in the file. **Flagged, not
assumed**: confirm with the paper authors what license applies to this
curated input dataset before redistribution; it may differ from the CC BY
4.0 license applied to the derived KG below.

## KG output license

The generated knowledge graph (`output/*.ttl`, `.rdf`, `.jsonld`, `.owl`,
and baseline outputs under `baselines/`) is licensed
**CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/), per author
direction. Attribution: cite this work per
[`CITATION.cff`](CITATION.cff).

## Known reproducibility caveat

LLM-based extraction (`extractors/*.py`) runs at `temperature=0.0` but
without an OpenAI `seed` parameter — see the "Reproducibility note" in
[`README.md`](README.md). Regenerating `output/` from `data/ai_ethics_datasets.json`
is expected to closely match, but is not guaranteed to bit-for-bit match,
the statistics above. Full regeneration also requires an `OPENAI_API_KEY`
(see `README.md`); it is not required to verify any figure in this card —
every number above is measured directly from the already-checked-in
`output/etd_hub_enhanced_kg47.ttl` and `output/etd_hub_enhanced_kg_quality_metrics.json`.
