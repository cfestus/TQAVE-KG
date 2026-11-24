# KG / Data Card

This card documents the frozen TQAVE-KG evaluation artifact:

`output/etd_hub_enhanced_kg47.ttl`

The artifact corresponds to run:

`run-b0eafdda0235eb48`

and has SHA-256:

`88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f`

`output/etd_hub_enhanced_kg.ttl` is byte-for-byte identical to the frozen `kg47` artifact and has the same run ID and SHA-256.

See [`RESULTS_MAP.md`](RESULTS_MAP.md) for the mapping between reported evaluation results, scripts, and reproducibility artefacts.

## Frozen evaluation graph statistics

The following values are measured directly from `output/etd_hub_enhanced_kg47.ttl`.

| Metric | Value |
|---|---|
| RDF triples | 38,806 |
| Distinct subjects / predicates / objects | 1,867 / 231 / 5,541 |
| `etd:Entity` nodes | 301 |
| `etd:Relation` nodes | 333 |
| Semantic relations | 211 |
| Structural relations | 122 |
| `etd:Evidence` nodes | 337 |
| Semantic Evidence nodes | 215 |
| Structural Evidence nodes | 122 |
| `etd:DiscourseUnit` nodes | 174 |
| `etd:Proposition` nodes | 1 |
| `etd:FieldAssertion` nodes | 317 |
| Entities with `etd:ontology_uri` | 300 / 301 (99.7%) |
| Wikidata-aligned entities | 20 / 301 (6.6%) |
| Semantic relation confidence | min 0.760, max 0.920, mean 0.913, median 0.920 |
| OWL classes | 85 |
| OWL object properties | 72 |
| OWL datatype properties | 223 |
| RDF properties | 4 |
| SHACL conformance | Conforms: True |
| SHACL violations / warnings | 0 / 0 |
| Project-specific quality gate | PASS |

All 211 semantic relations in the frozen evaluation artifact have:

```text
etd:isInferred = false
```

No inferred semantic relation was retained in this evaluation run.

## Source-level TQAVE coverage

The frozen graph preserves all source TQAVE components:

| Component | Source records | Graph records | Coverage |
|---|---:|---:|---:|
| Theme | 12 | 12 | 100% |
| Question | 8 | 8 | 100% |
| Answer | 21 | 21 | 100% |
| Vote | 26 | 26 | 100% |
| Expert | 57 | 57 | 100% |

## Structural relation preservation

The following discourse-backbone relations are fully preserved:

| Relation | Count |
|---|---:|
| `etd:createdBy` | 12 |
| `etd:belongsToTheme` | 8 |
| `etd:askedBy` | 8 |
| `etd:answersQuestion` | 21 |
| `etd:authoredBy` | 21 |
| `etd:receivesVote` | 26 |
| `etd:castBy` | 26 |

Each count matches the corresponding source structure.

## Semantic relation and evidence completeness

All 211 semantic relations contain:

- subject
- predicate
- object
- at least one Evidence link
- confidence metadata
- inference-status metadata
- source provenance
- recoverable discourse context

A total of 215 distinct semantic Evidence nodes support the 211 semantic relations.

- 207 relations have one Evidence link.
- 4 relations have two Evidence links.

All 215 semantic Evidence resources contain evidence text, provenance metadata, and grounding information.

Semantic relations are distributed across source fields as follows:

| Source field | Relations |
|---|---:|
| `theme.description` | 126 |
| `answer.description` | 67 |
| `question.body` | 18 |
| **Total** | **211** |

## Vote preservation

All 26 source Votes are represented and linked to both:

- the corresponding Answer; and
- the Expert who cast the Vote.

Ten of the 21 Answers received at least one Vote.

## Ontology reuse and entity alignment

Of the 301 typed `etd:Entity` resources:

- 300 have an `etd:ontology_uri` mapping;
- 20 are linked to Wikidata;
- Wikidata links use `skos:closeMatch`;
- no `owl:sameAs` Wikidata assertions are present.

The 300 ontology mappings span 23 distinct ontology URIs.

External ontology namespace usage includes:

| Namespace | Mapped entities |
|---|---:|
| FOAF | 60 |
| SKOS | 58 |
| PROV-O | 49 |
| Schema.org | 29 |
| DCAT | 25 |
| AIRO | 22 |
| AIPO | 15 |
| Fairness Metrics Ontology | 3 |
| BIBO | 2 |
| VAIR | 1 |
| W3C ORG | 1 |

A further 35 aligned entities map to project-local `etd:` terms.

Additional SKOS usage:

| Construct | Count |
|---|---:|
| `skos:Concept` | 70 |
| `skos:ConceptScheme` | 4 |
| `skos:prefLabel` | 12 |
| `skos:altLabel` | 14 |

## Pre-materialisation pipeline metrics

`output/etd_hub_enhanced_kg_quality_metrics.json` contains pipeline-state metrics captured before final RDF-build steps such as isolated-entity pruning and duplicate-relation consolidation.

These figures describe intermediate pipeline yield and should not be treated as final graph counts.

| Metric | Value |
|---|---:|
| Entities in pre-materialisation registry | 600 |
| Relations before consolidation | 340 |
| Average entity extraction confidence | 0.889 |
| Registry-based Wikidata-linked entities | 57 |
| Pipeline composite quality score | 6.88 |

The final materialised graph contains 301 entities and 333 relations.

The intermediate and final metric sets use different populations and should not be combined when calculating derived rates.

## Source data provenance

`data/ai_ethics_datasets.json` contains 12 AI-ethics case-study Themes with associated Questions, Answers, Votes, and Experts.

The Themes cover examples including:

- COMPAS recidivism prediction;
- healthcare risk-algorithm bias;
- mortgage-lending discrimination;
- AI recruitment screening.

Theme descriptions reference public third-party datasets where relevant. Those third-party datasets are not distributed in this repository and remain subject to their own licences and access conditions.

The curated source file itself does not contain an explicit licence statement. Users should therefore verify the applicable licence before redistributing the source dataset.

## Licensing

Code is licensed under the [MIT License](LICENSE).

Generated knowledge-graph artefacts in `output/` and baseline outputs are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

See [`CITATION.cff`](CITATION.cff) for citation information.

## Reproducibility note

LLM extraction uses `temperature=0.0`, but no OpenAI `seed` parameter is supplied. Consequently, regeneration is not guaranteed to reproduce the frozen graph bit-for-bit across API runs.

Exact verification of the reported evaluation should therefore use the checked-in frozen artifact:

```text
output/etd_hub_enhanced_kg47.ttl
```

No OpenAI API key is required to inspect, query, or validate this frozen graph.
