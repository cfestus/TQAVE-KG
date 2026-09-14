# Evaluation Queries (EvQ1–EvQ8)

SPARQL queries used to trace and audit the TQAVE-KG output graph. Each query
targets `output/etd_hub_enhanced_kg47.ttl` (the frozen manuscript artifact —
see [`../RESULTS_MAP.md`](../RESULTS_MAP.md)) via the `etd:` namespace
(`https://alfie-project.eu/etd-hub#`).

| Query | Purpose |
|---|---|
| [`EvQ1.rq`](EvQ1.rq) | Theme-level rollup: question / answer / semantic-relation counts per `etd:Theme`. |
| [`EvQ2.rq`](EvQ2.rq) | Question → answer → expert tracing, with semantic-relation counts per answer (`field = "answer.description"`). |
| [`EvQ3.rq`](EvQ3.rq) | Expert contribution breakdown: every `answer`/`question`/`theme`/`vote` item an expert authored, asked, created, or cast. |
| [`EvQ4.rq`](EvQ4.rq) | Vote-supported answer analysis: vote value/polarity cast on each answer. |
| [`EvQ5.rq`](EvQ5.rq) | Claim provenance tracing: relation → evidence → source-entity chain, with evidence text and originating discourse units. |
| [`EvQ6.rq`](EvQ6.rq) | Evidence and confidence audit: extracted semantic relations with evidence text, ranked by `etd:confidence` descending. |
| [`EvQ7.rq`](EvQ7.rq) | Inference-aware querying: extracted semantic relation counts grouped by `etd:isInferred`. |
| [`EvQ8.rq`](EvQ8.rq) | Multi-hop provenance: theme → question → answer → author → vote → relation → evidence, joined in one query. |

## Running the queries

Any SPARQL 1.1 engine that can load a Turtle file works. Two options:

**A. rdflib (no server required, matches the dependency pinned in `requirements.txt`)**

```bash
python -c "
import rdflib, sys
g = rdflib.Graph()
g.parse('output/etd_hub_enhanced_kg47.ttl', format='turtle')
q = open(sys.argv[1]).read()
for row in g.query(q):
    print(row)
" queries/EvQ1.rq
```

Repeat for `EvQ2.rq` … `EvQ8.rq`. For large result sets, redirect to a file:
`... > results_EvQ1.txt`.

**B. A triple store (GraphDB, Apache Jena Fuseki, etc.)**

Load `output/etd_hub_enhanced_kg47.ttl` into a repository/dataset, then submit
each `.rq` file to the store's SPARQL endpoint (e.g. via `curl` or the
store's web UI). See `utils/graphdb_client.py` if you want to reuse this
repo's GraphDB upload path (`USE_GRAPHDB_UPLOAD=true`, see
`key/.env.example`).

## Note on `relation_kind` filtering

Most queries filter `etd:relation_kind = "extracted_semantic"` to scope
results to LLM-extracted semantic relations (excluding backbone/structural
relations such as `answersQuestion`, `castBy`, `belongsToTheme`). Structural
relations are represented as direct graph properties between the entities
*and* are also materialised as reified `etd:Relation` nodes
(`relation_kind = "structural"`) carrying their own confidence/evidence
metadata — the `relation_kind` filter is what scopes queries to the
semantic subset, not an absence of reification on the structural side.
