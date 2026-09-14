# TQAVE-KG

Semantic knowledge-graph extraction pipeline for the TQAVE-KG paper: turns
curated AI-ethics case-study Q&A discourse (`data/ai_ethics_datasets.json`)
into a reified RDF knowledge graph (`etd:` namespace) via LLM-based entity
and relation extraction, ontology grounding, Wikidata linking, reasoning,
and SHACL validation.

See also: [`RESULTS_MAP.md`](RESULTS_MAP.md) (paper table/figure → script
mapping) and [`KG_CARD.md`](KG_CARD.md) (dataset/KG statistics and
provenance).

## Frozen manuscript artifact

The evaluation reported in the manuscript is computed against one frozen,
checked-in graph:

| | |
|---|---|
| File | [`output/etd_hub_enhanced_kg47.ttl`](output/etd_hub_enhanced_kg47.ttl) |
| Run ID | `run-b0eafdda0235eb48` |
| SHA-256 | `88c321e97b3e8c58fb3fb4aa49863700e6f61428513de130a05e1a5f4d54fd8f` |
| RDF triples | 38,806 |
| Reified relations | 333 total — 211 semantic (`relation_kind="extracted_semantic"`) + 122 structural |
| Evidence resources | 337 |

`output/etd_hub_enhanced_kg.ttl` (the default-named pipeline output also
checked in) is byte-for-byte identical to `etd_hub_enhanced_kg47.ttl` —
same SHA-256, same run ID. They are the same graph, not two different
experimental runs; `etd_hub_enhanced_kg47.ttl` is the immutable,
version-stamped filename to cite for the frozen manuscript artifact. See
[`KG_CARD.md`](KG_CARD.md) for full graph statistics and
[`RESULTS_MAP.md`](RESULTS_MAP.md) for the paper-result → artefact
mapping.

### A. Reviewer verification — no OpenAI API key required

Every one of the following reads or queries the already-generated,
checked-in files; none makes an LLM or network call:

- **KG quality gate / SHACL validation**: `python scripts/kg_quality_gate.py --ttl output/etd_hub_enhanced_kg47.ttl --shapes ontology/shacl_shapes.ttl --output-dir output`
- **Provenance/evidence-tracing queries (EvQ1–EvQ8)**: see [`queries/README.md`](queries/README.md), run against `output/etd_hub_enhanced_kg47.ttl`
- **Representational baselines** (flat-triple, Q&A-only): see [`baselines/README.md`](baselines/README.md)
- **Graph-analysis metrics**: read the checked-in `output/etd_hub_enhanced_kg_quality_metrics.json`
- **Manual semantic evaluation artefacts**: `manual_semantic_evaluation/` (two independent evaluator workbooks; see [`RESULTS_MAP.md`](RESULTS_MAP.md))
- **Unit/integration tests**: `python -m unittest discover -v`

### B. Optional — full LLM graph regeneration (requires an API key)

Only regenerating the graph from raw source text —
`python main.py --input data/ai_ethics_datasets.json --output output`
(below) — requires `OPENAI_API_KEY` (see Configuration below). This is
**not** required to verify any reported result; it is provided only for
reviewers who want to re-run the extraction pipeline itself, and, per the
reproducibility note below, is not guaranteed to reproduce KG47
bit-for-bit.

## Repository layout

```
config.py, models.py, pipeline.py, main.py   Pipeline core + CLI entry point
ontology_mappings.py                         Ontology term registry used by config.py
extractors/                                  Entity/relation/typing LLM extractors
ontology/                                    Class/property schema (add_schema.py), SHACL shapes, mapper
utils/                                       Canonicalization, linking, chunking, policy helpers
reasoning/                                   Multi-hop reasoning + conflict detection
serializers/                                 RDF/Turtle/JSON-LD/OWL serialization
validation/                                  SHACL validation wrapper
metrics/                                     Quality-metrics computation
scripts/                                     CLI tools: quality gate, audits (see RESULTS_MAP.md)
data/                                        Input dataset
output/                                      Canonical pipeline output (generated; see KG_CARD.md)
baselines/                                   Ablation baselines (see baselines/README.md)
queries/                                     Evaluation SPARQL queries EvQ1-EvQ8 (see queries/README.md)
manual_semantic_evaluation/                  Manual semantic evaluation workbooks (see RESULTS_MAP.md)
tests/                                       Unit/integration tests (unittest)
key/                                         Local .env (git-ignored) — see key/.env.example
```

## Install

Python 3.9+ (developed against 3.9.13). Pinned runtime dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
openai==1.14.0
rdflib==6.3.2
pyshacl==0.23.0
requests==2.31.0
jsonschema==4.17.0
python-dotenv==1.0.0
networkx==3.0
tqdm==4.65.0
tenacity==8.2.0
httpx==0.27.2
httpcore==1.0.9
tiktoken==0.12.0
```

No additional system-level (non-Python) dependencies are required — every
pinned package is pure-Python/HTTP-client, so no Dockerfile is provided.
Tests use only the Python standard library's `unittest` (no `pytest`-only
features are used anywhere in `tests/`), so `requirements.txt` alone is
sufficient to install, run, and test the repository.

## Configuration

Copy [`key/.env.example`](key/.env.example) to `key/.env` and set
`OPENAI_API_KEY` (required — the pipeline is LLM-only extraction and refuses
to run without it, see `Config.validate()` in `config.py`). `config.py`
auto-discovers any `**/.env` file under the working directory, so `key/.env`
is picked up without extra flags. All other settings have working defaults;
see `key/.env.example` for the full list of variables `config.py` reads.

**`key/.env` must never be committed or shared** — it is git-ignored. If you
have an existing `key/.env` with a live API key, treat it as a secret you
manage locally, not part of this repository's deliverable.

## Run the pipeline

```bash
python main.py --input data/ai_ethics_datasets.json --output output
```

This reproduces the canonical output already checked in under `output/`
(entity/relation extraction → ontology grounding → Wikidata linking →
canonical/coref linking → reasoning → SHACL validation → RDF serialization
to `.ttl`/`.rdf`/`.jsonld`/`.owl` → quality metrics). See `main.py --help`
for flags (`--no-reasoning`, `--no-wikidata`, `--batch-size`, `--debug`,
`--config <json>`).

Useful flags for a faster/no-network smoke run: `--no-wikidata` skips
external Wikidata SPARQL calls; there is no offline mode for the extraction
step itself since it requires the OpenAI API.

### Reproducibility note (stochasticity)

All LLM extraction calls use `temperature=0.0`, hardcoded at each call site
in `extractors/*.py` (not read from `config.py` or an environment variable).
No OpenAI `seed` parameter is passed at any
call site, so even at `temperature=0.0` the API does not guarantee
bit-identical output run-to-run — exact reproduction of extraction output is
therefore best-effort, not guaranteed. No other part of the pipeline (no
`random`/`np.random` used for data-affecting logic; retry/backoff jitter is
the only other use of `random.uniform`) is stochastic, and there is no
seed/random_state config to set.

### Quality gate

```bash
python scripts/kg_quality_gate.py --ttl output/etd_hub_enhanced_kg47.ttl --shapes ontology/shacl_shapes.ttl --output-dir output
```

## Run baselines / ablations

See [`baselines/README.md`](baselines/README.md) for the two ablation
baselines (`flat_triple_baseline` and `representational_baseline`
(QA-only)) and their exact build commands.

## Run the evaluation queries (EvQ1–EvQ8)

See [`queries/README.md`](queries/README.md) for what each query computes
and how to run it (rdflib one-liner or any SPARQL 1.1 store) against
`output/etd_hub_enhanced_kg47.ttl`.

## Reproduce reported evaluation results

See [`RESULTS_MAP.md`](RESULTS_MAP.md) for the full table/figure → script
mapping, including flagged reproducibility gaps. In summary:

- **KG statistics** (triple/entity/relation counts, the published-graph figures): [`KG_CARD.md`](KG_CARD.md), measured directly against `output/etd_hub_enhanced_kg47.ttl`. `output/etd_hub_enhanced_kg_quality_metrics.json` reports a *different*, pre-pruning pipeline-yield figure — see KG_CARD.md's explanation before citing it as KG size.
- **SHACL conformance / quality gate**: `python scripts/kg_quality_gate.py ...` (above).
- **Baseline comparisons**: `baselines/README.md`.
- **Provenance/evidence tracing queries**: `queries/README.md` (EvQ1-EvQ8).
- **Manual semantic evaluation**: `manual_semantic_evaluation/` (see `RESULTS_MAP.md`).

## Tests

```bash
python -m unittest discover -v
```

## License

Code is licensed under [MIT](LICENSE). The generated knowledge graph
(`output/*.ttl`/`.rdf`/`.jsonld`/`.owl`) is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[`KG_CARD.md`](KG_CARD.md) for details and a caveat on the underlying
source dataset's license. See [`CITATION.cff`](CITATION.cff) to cite this
work.
