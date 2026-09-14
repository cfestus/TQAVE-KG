# Results Map

Maps each reported evaluation result to the exact script/query/config in
this repository that produces or verifies it. The "Published RDF graph —
golden numbers" section condenses a full manuscript-figure verification
pass — every metric computed independently via `rdflib` SPARQL against the
canonical graph. A separate "Pre-publication pipeline metrics" section
explains numbers that are legitimate but measure something different
(pipeline yield before RDF-build-time pruning/consolidation) and must not
be substituted for the published-graph figures.

| Result | Producing script/query | Config | Output location |
|---|---|---|---|
| Frozen manuscript graph | frozen artifact (see "Run the pipeline" in `README.md` for how it was generated) | — | `output/etd_hub_enhanced_kg47.ttl` — byte-identical to `output/etd_hub_enhanced_kg.ttl` (same SHA-256, same run ID; not two different experimental graphs) |
| Full KG build (entities, relations, triples) | `python main.py --input data/ai_ethics_datasets.json --output output` — optional, requires `OPENAI_API_KEY`; **not** required to verify any reported result (see `README.md`) | `config.py` defaults; `key/.env` for `OPENAI_API_KEY` | `output/etd_hub_enhanced_kg.{ttl,rdf,jsonld,owl}`, `output/etd_hub_enhanced_kg_quality_metrics.json` |
| KG quality/coverage metrics | `metrics/quality_metrics.py` (invoked by `main.py`/`pipeline.py`) | same as above | `output/etd_hub_enhanced_kg_quality_metrics.json` |
| SHACL conformance / KG quality gate | `python scripts/kg_quality_gate.py --ttl output/etd_hub_enhanced_kg47.ttl --shapes ontology/shacl_shapes.ttl --output-dir output` — no API key required | `ontology/shacl_shapes.ttl` — **must** validate with `allow_infos=True, allow_warnings=True, meta_shacl=True, advanced=True` (the project's own `validation/shacl_validator.py` defaults); a bare `pyshacl.validate(ttl, shapes)` call without those flags reports `conforms=False` on the *same* Info-only results — not a graph regression, just pySHACL's stricter defaults | `output/shacl_report.{ttl,txt}` |
| Graph audit (structural/provenance cross-checks) | `scripts/audit_harness.py` — no API key required | run against `output/etd_hub_enhanced_kg47.ttl` | ad hoc (audit output, not persisted by default) |
| Flat-triple baseline comparison | `baselines/scripts/build_flat_triple_baseline.py --input output/etd_hub_enhanced_kg47.ttl ...` (see `baselines/README.md` for the full invocation) — no API key required | script defaults except `--input` | `baselines/output/`, `baselines/results/` |
| QA-only (representational) baseline comparison | `baselines/representational_baseline/scripts/build_qa_only_baseline.py` (see `baselines/README.md` — **must** use explicit `--output`/etc. flags, script defaults point at the flat-triple baseline's paths) — no API key required | script defaults except paths | `baselines/representational_baseline/output/`, `baselines/representational_baseline/results/` |
| Theme/discourse rollups, provenance/evidence tracing (EvQ1–EvQ8) | `queries/EvQ1.rq` … `queries/EvQ8.rq` (see `queries/README.md`) — all 8 verified to execute cleanly under the pinned rdflib version; no API key required | run against `output/etd_hub_enhanced_kg47.ttl` | ad hoc (query results, not persisted) |
| Manual semantic evaluation | two independent evaluator workbooks — deterministic stratified sample, no script re-run needed to inspect | tied to `output/etd_hub_enhanced_kg47.ttl` by SHA-256 (embedded in each workbook's manifest); see "Manual semantic evaluation" section below | `manual_semantic_evaluation/Manual_semantic_evaluation_Evaluator{1,2}.xlsx` |

## Published RDF graph — golden numbers (verified against `output/etd_hub_enhanced_kg47.ttl`, the frozen manuscript artifact)

**These are the authoritative numbers for describing the released KG file itself** — every one is a direct `rdflib==6.3.2` SPARQL/graph query against `output/etd_hub_enhanced_kg47.ttl` (SHA-256 `88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f`, run ID `run-b0eafdda0235eb48`). `output/etd_hub_enhanced_kg.ttl` (the default-named pipeline output also checked in) is byte-for-byte identical to this file — same SHA-256, same run ID; they are the same graph, not two different experimental runs.

A separate section below ("Pre-publication pipeline metrics") records numbers that are legitimate but measure something **different** — the pipeline's internal state before RDF-serialization-time pruning/consolidation — and must not be substituted for the numbers here. See that section for exactly why they differ and which to cite when.

- **Graph identity**: `etd:runId "run-b0eafdda0235eb48"`.
- **Graph-scale statistics**: 38,806 total triples; 1,867 distinct subjects; 231 distinct predicates; 5,541 distinct objects; 85 OWL classes; 72 OWL object properties; 223 OWL datatype properties; 4 RDF properties; 211 semantic reified relations (`relation_kind="extracted_semantic"`) + 122 structural = **333 total `etd:Relation` nodes**; **337 total `etd:Evidence` nodes** (215 attached to semantic relations + 122 attached to structural relations, zero overlap); **301 `etd:Entity` nodes**; 174 discourse-unit nodes; 1 Proposition node; 317 field-assertion nodes.
- **Source-level TQAVE coverage**: 12 Theme, 8 Question, 21 Answer, 26 Vote, 57 Expert instances (all four counts structurally invariant across every regeneration to date).
- **Structural relation preservation**: `createdBy` 12, `belongsToTheme` 8, `askedBy` 8, `answersQuestion` 21, `authoredBy` 21, `receivesVote` 26, `castBy` 26 — all exactly match source counts, unchanged from every prior run.
- **Relation/evidence completeness (semantic relations only)**: **211/211 (100%)** semantic relations have complete subject/predicate/object, ≥1 evidence link, confidence, inference-status, and source provenance. Of those 211, **207 carry exactly one `etd:hasEvidence` link and 4 carry two** (a relation citing two corroborating textual quotes), for **215 distinct semantic `etd:Evidence` nodes** in total — this 215 is a genuine node count, not the relation count (do not report "215/215 relations"; report "211/211 relations, 215 evidence nodes"). All 215 semantic Evidence resources have text, provenance metadata, and grounding info — 100%. Field breakdown (by relation, not evidence node): `theme.description` 126, `answer.description` 67, `question.body` 18 (sum 211).
- **Vote metadata**: 26/26 votes typed and linked; 10 of 21 answers have ≥1 vote (11 without) — unchanged from every prior run.
- **Confidence — reported separately per `relation_kind`, since the two populations are not comparable** (structural relations are deterministic template extractions from source-data fields and score consistently high; semantic relations are open-vocabulary LLM extractions and score lower/wider):
  - Semantic (n=211): min 0.760, max 0.920, mean 0.9125, median 0.920.
  - Structural (n=122): min 0.960, max 0.990, mean 0.9775, median 0.980.
- **Ontology reuse and entity alignment**: 300/301 (99.67%) entities have `etd:ontology_uri`; 20/301 (6.64%) have a Wikidata identity link — all via `skos:closeMatch` (heuristic, non-authoritative); `owl:sameAs` is reserved for an authoritative identity and 0 entities currently qualify. `skos:Concept` 70, `skos:ConceptScheme` 4, `skos:prefLabel` 12, `skos:altLabel` 14. Per-namespace (`etd:ontology_uri` target, of the 300 aligned): FOAF 60, SKOS 58, PROV-O 49, Schema.org 29, DCAT 25, AIRO 22, AIPO 15, Fairness Metrics Ontology 3, BIBO 2, W3C ORG 1, VAIR 1 (sum 265); the remaining 35 aligned entities map to a project-local `etd:` term rather than an external ontology.
- **EvQ1/EvQ7 spot checks**: EvQ1's top theme = Theme 10 with 3 questions / 11 answers / **53 relations**. EvQ7: all 211 semantic relations have `isInferred=false` (0 inferred relations this run).
- **SHACL**: Conforms = True, 0 violations, 0 warnings, **0 Info-severity results**.
- **Quality gate** (`scripts/kg_quality_gate.py`): **PASS, zero failure reasons**, verified directly against this exact file.

## Pre-publication pipeline metrics (NOT the published graph — do not cite as KG size)

`output/etd_hub_enhanced_kg_quality_metrics.json` is computed by `metrics/quality_metrics.py`, called from `main.py` **after** `RDFSerializer` builds the graph but reading `pipeline.global_registry`/`annotations_dict` — data structures that predate two RDF-build-time steps: `RDFSerializer.prune_isolated_entities()` (drops entities with no relation connecting them to anything) and `consolidate_duplicate_reified_relations()` (merges exact subject/predicate/object duplicates). Its entity/relation counts are therefore **pre-pruning/pre-consolidation pipeline yield**, not published-graph size:

| | Pre-publication (quality_metrics.json) | Published graph (this file, above) | Reconciliation |
| --- | --- | --- | --- |
| Entities | 600 (`entity_metrics.total_entities` = `len(global_registry.entities)`) | **301** | Differ due to isolated-entity pruning (`RDFSerializer.prune_isolated_entities()`). |
| Relations | 340 (`relation_metrics.total_relations`, summed from `annotations_dict`) | **333** | Differ due to `consolidate_duplicate_reified_relations()`; `duplicate_relation_spo_groups_count: 0` in the gate output is correctly measuring the graph **after** this consolidation, not before. |

Use 600/340 only if explicitly describing raw pre-publication extraction yield; use 301/333 for anything describing the released `output/etd_hub_enhanced_kg47.ttl`. Other `quality_metrics.json` figures (avg_confidence 0.889, linking_rate 0.095, overall_quality_score 6.88) are registry/document-level composites with no direct graph-level equivalent and can be cited as pipeline-level quality indicators, distinct from the graph-population counts above.

**Do not combine figures from the two tables above into one number** — e.g. do not compute a "linking rate" using the 301 published-graph entity count against a pre-publication-population figure. Each table is internally consistent; the tables are not cross-compatible.

## Representational baseline comparison

| | TQAVE-KG | Q&A-only baseline | Flat triple baseline |
| --- | --- | --- | --- |
| Discourse/theme tracing | ✅ (EvQ1–4, non-empty) | ✅ (1.0, Q1–Q4) | ❌ (0.0, Q3–Q6) |
| Evidence-backed claim tracing | ✅ (EvQ5/6: 215 evidence rows) | ❌ (0.0, Q5) | ❌ (0.0, Q7) |
| Confidence-aware filtering | ✅ (EvQ8: 170 rows) | ❌ (0.0, Q6) | ❌ (0.0, Q8) |
| Inference-aware querying | ✅ (EvQ7: 1 row) | ❌ (0.0, Q7) | ❌ (0.0, Q9) |
| Reified relation provenance | ✅ | ❌ (0.0, Q9) | ❌ (0.0, Q11) |
| Direct entity/relation retrieval | ✅ (structural) | n/a | ✅ (1.0, Q1+Q2, combined into one capability) |
| **Mean representational score (no SHACL)** | **1.000** (passes all 6 capability checks above — TQAVE-KG's reified schema satisfies every capability the two ablations were scored against, by construction; no separate 0/1 probe artifact was built for TQAVE-KG itself since EvQ1–8 already demonstrate each capability directly) | **0.4** (mean of 10 per-query scores: Q1–Q4=1.0, Q5–Q10=0.0) | **0.1** (mean of 10 capability dimensions, where Q1+Q2 are combined into a single "Entity/direct relation retrieval" score and Q12 is excluded as a structural negative control, not a capability) |

Both ablation means verified by direct arithmetic against `baselines/results/flat_triple_baseline_audit.json` and `baselines/representational_baseline/results/qa_only_baseline_audit.json` — not recomputed by re-running either baseline script. Note: 10 of the flat-triple baseline's 12 probe queries (Q3–Q12) correctly return 0 rows by design (`passed: true` there means "correctly confirmed this baseline cannot represent the concept," not "capability present") — only Q1/Q2, which are not part of the EvQ-equivalent capability set, score positively.

## Manual semantic evaluation

`manual_semantic_evaluation/` contains two independent evaluator
workbooks (`Manual_semantic_evaluation_Evaluator1.xlsx`,
`Manual_semantic_evaluation_Evaluator2.xlsx`), no separate script — the
sampling parameters and the sampled rows themselves are embedded in each
workbook's own "Audit & Manifest" / "Sampling Summary" / "Manual
Validation" sheets. Verified package facts:

- **Population**: 211 (all `relation_kind="extracted_semantic"` relations in `output/etd_hub_enhanced_kg47.ttl`).
- **Sample**: 140.
- **Seed**: 42.
- **Allocation**: proportional stratification by source field, Hamilton/largest-remainder allocation — `theme.description` 84, `answer.description` 44, `question.body` 12 (sum 140).
- **Two independent evaluator workbooks**, both scoring the identical 140 sampled `Relation_URI`s in the same order.
- **Manifest tied to the frozen graph**: each workbook records `Frozen TTL SHA-256 = 88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f` (= `output/etd_hub_enhanced_kg47.ttl`) with a self-reported "Checksum verified: YES".

No sampling script is included in this repository — the selection rule
itself ("SHA-256(seed|Relation_URI) rank within source-field stratum") is
documented in prose inside the workbooks.
