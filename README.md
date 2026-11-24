# TQAVE-KG

TQAVE-KG is a provenance-aware semantic knowledge-graph construction pipeline for expert-contributed AI-ethics Q&A discourse. It transforms the curated source dataset (`data/ai_ethics_datasets.json`) into a reified RDF knowledge graph using LLM-assisted entity and relation extraction, ontology alignment, Wikidata linking, canonicalisation, reasoning, provenance modelling, and SHACL validation.

Supporting documentation:

- [`RESULTS_MAP.md`](RESULTS_MAP.md): mapping from reported evaluation results to scripts and artefacts.
- [`KG_CARD.md`](KG_CARD.md): graph statistics, provenance, and dataset information.

## Frozen evaluation artifact

The evaluation reported for TQAVE-KG is based on the following frozen graph:

| Item | Value |
|---|---|
| File | [`output/etd_hub_enhanced_kg47.ttl`](output/etd_hub_enhanced_kg47.ttl) |
| Run ID | `run-b0eafdda0235eb48` |
| SHA-256 | `88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f` |
| RDF triples | 38,806 |
| Reified relations | 333 total: 211 semantic + 122 structural |
| Evidence resources | 337 |

`output/etd_hub_enhanced_kg.ttl` is byte-for-byte identical to the frozen `etd_hub_enhanced_kg47.ttl` artifact and has the same SHA-256 and run ID. The `kg47` filename is retained as the immutable version-stamped artifact used for evaluation.

## Reproduce the reported evaluation

No OpenAI API key is required to verify the reported results. The checked-in graph and evaluation artefacts can be inspected directly.

### SHACL validation and quality gate

```bash
python scripts/kg_quality_gate.py \
  --ttl output/etd_hub_enhanced_kg47.ttl \
  --shapes ontology/shacl_shapes.ttl \
  --output-dir output
```

### Provenance-oriented SPARQL queries

The eight evaluation queries, EvQ1–EvQ8, are documented in:

[`queries/README.md`](queries/README.md)

Run them against:

```text
output/etd_hub_enhanced_kg47.ttl
```

### Representational ablations

The Q&A-only and flat-triple ablations are documented in:

[`baselines/README.md`](baselines/README.md)

### Manual semantic evaluation

Independent evaluator workbooks and related artefacts are available in:

```text
manual_semantic_evaluation/
```

See [`RESULTS_MAP.md`](RESULTS_MAP.md) for the mapping between evaluation results and artefacts.

### Tests

```bash
python -m unittest discover -v
```

## Repository structure

```text
config.py, models.py, pipeline.py, main.py   Pipeline core and CLI entry point
extractors/                                  LLM entity/relation extraction
ontology/                                    Schema, mappings, and SHACL shapes
reasoning/                                   Reasoning and consistency checks
serializers/                                 RDF serialization
validation/                                  SHACL validation
metrics/                                     Quality metrics
scripts/                                     Evaluation and audit utilities
data/                                        Source dataset
output/                                      Frozen and generated graph artefacts
baselines/                                   Representational ablations
queries/                                     SPARQL evaluation queries
manual_semantic_evaluation/                  Manual validation artefacts
tests/                                       Unit and integration tests
key/                                         Local environment configuration
```

## Installation

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

The repository uses the pinned dependencies defined in `requirements.txt`.

## Configuration

To regenerate the graph, copy:

```text
key/.env.example
```

to:

```text
key/.env
```

and set:

```text
OPENAI_API_KEY
```

`key/.env` is git-ignored and must not be committed or shared.

An API key is required only for LLM-based graph regeneration. It is not required to verify the frozen graph or reported evaluation results.

## Run the pipeline

```bash
python main.py \
  --input data/ai_ethics_datasets.json \
  --output output
```

The pipeline performs:

```text
LLM extraction
→ canonicalisation and referential resolution
→ ontology alignment
→ Wikidata linking
→ reasoning and consistency checks
→ provenance-aware RDF materialisation
→ SHACL validation
→ output generation
```

Use:

```bash
python main.py --help
```

for available command-line options.

## Reproducibility note

LLM extraction uses `temperature=0.0`, but no OpenAI `seed` parameter is supplied. Consequently, graph regeneration is deterministic where implemented locally, but LLM-generated extraction output is not guaranteed to be bit-for-bit identical across API runs.

For exact verification of the reported evaluation, use the frozen checked-in artifact:

```text
output/etd_hub_enhanced_kg47.ttl
```

## Evaluation artefacts

The principal reproducibility resources are:

- Frozen graph: `output/etd_hub_enhanced_kg47.ttl`
- Graph statistics and provenance: [`KG_CARD.md`](KG_CARD.md)
- Results-to-artefact mapping: [`RESULTS_MAP.md`](RESULTS_MAP.md)
- SHACL/quality-gate script: `scripts/kg_quality_gate.py`
- SPARQL evaluation queries: `queries/`
- Representational ablations: `baselines/`
- Manual semantic validation: `manual_semantic_evaluation/`

## License

Code is licensed under the [MIT License](LICENSE).

Generated knowledge-graph artefacts in `output/` are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [`KG_CARD.md`](KG_CARD.md) for dataset and licensing notes.

See [`CITATION.cff`](CITATION.cff) for citation information.
