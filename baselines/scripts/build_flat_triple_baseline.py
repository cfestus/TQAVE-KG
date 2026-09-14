#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS
from rdflib.term import Identifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ETD_NS = "https://alfie-project.eu/etd-hub#"
ETD = Namespace(ETD_NS)
PROV = Namespace("http://www.w3.org/ns/prov#")

ALLOWED_LABEL_PREDICATES: Tuple[URIRef, ...] = (
    RDFS.label,
    ETD.name,
    ETD.canonicalLabel,
    SKOS.prefLabel,
)

EXCLUDED_CLASS_URIS: Tuple[URIRef, ...] = (
    ETD.Relation,
    ETD.Evidence,
    ETD.DiscourseUnit,
    ETD.FieldAssertion,
    ETD.Vote,
    ETD.Answer,
    ETD.Question,
    ETD.Theme,
    ETD.Expert,
)

EXCLUDED_PREDICATES: Tuple[URIRef, ...] = (
    ETD.subject,
    ETD.predicate,
    ETD.object,
    ETD.hasEvidence,
    ETD.evidenceText,
    ETD.confidence,
    ETD.isInferred,
    ETD.inferenceRule,
    ETD.hasSourceEntity,
    ETD.source,
    ETD.field,
    ETD.source_doc_id,
    ETD.context_source,
    ETD.groundingStatus,
    ETD.groundingMethod,
    ETD.answersQuestion,
    ETD.belongsToTheme,
    ETD.authoredBy,
    ETD.hasVote,
    ETD.receivesVote,
    ETD.castBy,
    ETD.castVote,
    ETD.parentAnswer,
    OWL.sameAs,
    ETD.ontology_uri,
)

EXCLUDED_CLASS_COUNT_KEYS = {
    ETD.Relation: "relation_nodes",
    ETD.Evidence: "evidence_nodes",
    ETD.DiscourseUnit: "discourse_unit_nodes",
    ETD.FieldAssertion: "field_assertion_nodes",
    ETD.Vote: "vote_nodes",
    ETD.Answer: "answer_nodes",
    ETD.Question: "question_nodes",
    ETD.Theme: "theme_nodes",
    ETD.Expert: "expert_nodes",
}

EXCLUDED_PREDICATE_COUNT_KEYS = {
    ETD.subject: "subject_triples",
    ETD.predicate: "predicate_triples",
    ETD.object: "object_triples",
    ETD.hasEvidence: "hasEvidence_triples",
    ETD.evidenceText: "evidenceText_triples",
    ETD.confidence: "confidence_triples",
    ETD.isInferred: "isInferred_triples",
    ETD.inferenceRule: "inferenceRule_triples",
    ETD.hasSourceEntity: "hasSourceEntity_triples",
    ETD.source: "source_triples",
    ETD.field: "field_triples",
    ETD.source_doc_id: "source_doc_id_triples",
    ETD.context_source: "context_source_triples",
    ETD.groundingStatus: "groundingStatus_triples",
    ETD.groundingMethod: "groundingMethod_triples",
    ETD.answersQuestion: "answersQuestion_triples",
    ETD.belongsToTheme: "belongsToTheme_triples",
    ETD.authoredBy: "authoredBy_triples",
    ETD.hasVote: "hasVote_triples",
    ETD.receivesVote: "receivesVote_triples",
    ETD.castBy: "castBy_triples",
    ETD.castVote: "castVote_triples",
    ETD.parentAnswer: "parentAnswer_triples",
    OWL.sameAs: "owl_sameAs_triples",
    ETD.ontology_uri: "ontology_uri_triples",
}

QUERY_FILE_TEXT = """# Query 1: Direct triple retrieval
# Expected: more than 0 rows
# Capability: entity/direct relation retrieval
# Score: 1.0 if rows > 0
PREFIX etd: <https://alfie-project.eu/etd-hub#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object .
  FILTER(?predicate NOT IN (rdf:type, rdfs:label))
}
LIMIT 25

# Query 2: Predicate distribution
# Expected: more than 0 predicate groups
# Capability: entity/direct relation retrieval
# Score: 1.0 if groups > 0
PREFIX etd: <https://alfie-project.eu/etd-hub#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?predicate (COUNT(*) AS ?count)
WHERE {
  ?subject ?predicate ?object .
  FILTER(?predicate NOT IN (rdf:type, rdfs:label))
}
GROUP BY ?predicate
ORDER BY DESC(?count)

# Query 3: Question-answer tracing negative control
# Expected: 0 rows
# Capability: question-answer tracing
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer ?question
WHERE {
  ?answer a etd:Answer ;
          etd:answersQuestion ?question .
}

# Query 4: Theme-level aggregation negative control
# Expected: 0 rows
# Capability: theme-level aggregation
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?theme (COUNT(DISTINCT ?answer) AS ?answerCount)
WHERE {
  ?answer a etd:Answer ;
          etd:answersQuestion ?question .
  ?question etd:belongsToTheme ?theme .
}
GROUP BY ?theme

# Query 5: Expert contribution tracing negative control
# Expected: 0 rows
# Capability: expert contribution tracing
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer ?expert
WHERE {
  ?answer a etd:Answer ;
          etd:authoredBy ?expert .
}

# Query 6: Vote-supported answer analysis negative control
# Expected: 0 rows
# Capability: vote-supported answer analysis
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer (COUNT(DISTINCT ?vote) AS ?voteCount)
WHERE {
  ?answer a etd:Answer ;
          etd:hasVote ?vote .
}
GROUP BY ?answer

# Query 7: Evidence-backed claim tracing negative control
# Expected: 0 rows
# Capability: evidence-backed claim tracing
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?evidence
WHERE {
  ?relation a etd:Relation ;
            etd:hasEvidence ?evidence .
}

# Query 8: Confidence-aware filtering negative control
# Expected: 0 rows
# Capability: confidence-aware filtering
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?confidence
WHERE {
  ?relation a etd:Relation ;
            etd:confidence ?confidence .
}

# Query 9: Inference-aware querying negative control
# Expected: 0 rows
# Capability: inference-aware querying
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?isInferred
WHERE {
  ?relation a etd:Relation ;
            etd:isInferred ?isInferred .
}

# Query 10: Claim-to-discourse traceability negative control
# Expected: 0 rows
# Capability: claim-to-discourse traceability
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?answer ?question ?theme ?expert
WHERE {
  ?relation a etd:Relation ;
            etd:hasSourceEntity ?answer .
  ?answer a etd:Answer ;
          etd:answersQuestion ?question ;
          etd:authoredBy ?expert .
  ?question etd:belongsToTheme ?theme .
}

# Query 11: Reified relation provenance negative control
# Expected: 0 rows
# Capability: reified relation provenance
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?subject ?predicate ?object
WHERE {
  ?relation a etd:Relation ;
            etd:subject ?subject ;
            etd:predicate ?predicate ;
            etd:object ?object .
}

# Query 12: Evidence node absence
# Expected: 0 rows
# Capability: evidence-backed claim tracing
# Score: 0.0
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?evidence
WHERE {
  ?evidence a etd:Evidence .
}
"""

SHACL_SHAPES = '''@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix etd: <https://alfie-project.eu/etd-hub#> .

etd:NoRelationInstancesShape
    a sh:NodeShape ;
    sh:targetSubjectsOf rdf:type ;
    sh:sparql [
        sh:message "Flat baseline must not contain etd:Relation instances." ;
        sh:select """
            SELECT $this
            WHERE {
              $this a etd:Relation .
            }
        """ ;
    ] .

etd:NoEvidenceInstancesShape
    a sh:NodeShape ;
    sh:targetSubjectsOf rdf:type ;
    sh:sparql [
        sh:message "Flat baseline must not contain etd:Evidence instances." ;
        sh:select """
            SELECT $this
            WHERE {
              $this a etd:Evidence .
            }
        """ ;
    ] .

etd:NoConfidencePredicateShape
    a sh:NodeShape ;
    sh:targetSubjectsOf etd:confidence ;
    sh:property [
        sh:path etd:confidence ;
        sh:maxCount 0 ;
    ] .

etd:NoSourceEntityPredicateShape
    a sh:NodeShape ;
    sh:targetSubjectsOf etd:hasSourceEntity ;
    sh:property [
        sh:path etd:hasSourceEntity ;
        sh:maxCount 0 ;
    ] .

etd:NoAnswersQuestionPredicateShape
    a sh:NodeShape ;
    sh:targetSubjectsOf etd:answersQuestion ;
    sh:property [
        sh:path etd:answersQuestion ;
        sh:maxCount 0 ;
    ] .

etd:AtLeastOneDirectRelationShape
    a sh:NodeShape ;
    sh:targetNode etd:FlatTripleBaselineCheck ;
    sh:sparql [
        sh:message "Flat baseline must contain at least one direct non-metadata relation triple." ;
        sh:select """
            SELECT $this
            WHERE {
              FILTER NOT EXISTS {
                ?subject ?predicate ?object .
                FILTER(?predicate NOT IN (rdf:type, <http://www.w3.org/2000/01/rdf-schema#label>))
              }
            }
        """ ;
    ] .
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic flat text-to-triple baseline from the validated TQAVE-KG TTL."
    )
    parser.add_argument("--input", default="output/etd_hub_enhanced_kg.ttl")
    parser.add_argument("--output", default="baselines/output/flat_triple_baseline.ttl")
    parser.add_argument("--audit-output", default="baselines/results/flat_triple_baseline_audit.json")
    parser.add_argument(
        "--query-results-output",
        default="baselines/results/flat_triple_query_results.json",
    )
    parser.add_argument("--queries-output", default="baselines/queries/flat_triple_queries.sparql")
    parser.add_argument("--log-output", default="baselines/results/flat_triple_generation_log.txt")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directories(paths: Sequence[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(path: Path) -> Graph:
    graph = Graph()
    try:
        graph.parse(path, format="turtle")
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load input TTL from {display_path(path)}: {exc}") from exc
    return graph


def copy_namespace_bindings(source_graph: Graph, target_graph: Graph) -> None:
    for prefix, namespace in source_graph.namespaces():
        if prefix is None:
            continue
        target_graph.bind(prefix, namespace, replace=False, override=False)
    target_graph.bind("etd", ETD, replace=True, override=True)
    target_graph.bind("rdf", RDF, replace=True, override=True)
    target_graph.bind("rdfs", RDFS, replace=True, override=True)
    target_graph.bind("owl", OWL, replace=True, override=True)
    target_graph.bind("skos", SKOS, replace=True, override=True)
    target_graph.bind("prov", PROV, replace=True, override=True)


def term_sort_key(term: Identifier) -> Tuple[int, str, str]:
    if isinstance(term, URIRef):
        return (0, str(term), "")
    if isinstance(term, Literal):
        datatype = str(term.datatype) if term.datatype else ""
        language = term.language or ""
        return (1, f"{language}|{datatype}|{str(term)}", term.n3())
    return (2, term.n3(), "")


def triple_sort_key(triple: Tuple[Identifier, Identifier, Identifier]) -> Tuple[Tuple[int, str, str], ...]:
    subject, predicate, obj = triple
    return (term_sort_key(subject), term_sort_key(predicate), term_sort_key(obj))


def relation_uri_sort_key(value: Identifier) -> Tuple[int, str, str]:
    return term_sort_key(value)


def is_excluded_endpoint_node(source_graph: Graph, node: Identifier) -> bool:
    if not isinstance(node, URIRef):
        return False
    return any((node, RDF.type, class_uri) in source_graph for class_uri in EXCLUDED_CLASS_URIS)


def useful_type(type_uri: Identifier) -> bool:
    if not isinstance(type_uri, URIRef):
        return False
    if type_uri in EXCLUDED_CLASS_URIS:
        return False
    if type_uri in {PROV.Entity, RDFS.Resource, OWL.Thing, OWL.Class, RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty}:
        return False
    return True


def predicate_distribution(triples: Iterable[Tuple[Identifier, Identifier, Identifier]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for _, predicate, _ in triples:
        counts[str(predicate)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def strict_exclusion_counts(graph: Graph) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for class_uri, key in EXCLUDED_CLASS_COUNT_KEYS.items():
        counts[key] = len({subject for subject in graph.subjects(RDF.type, class_uri)})
    for predicate_uri, key in EXCLUDED_PREDICATE_COUNT_KEYS.items():
        counts[key] = sum(1 for _ in graph.triples((None, predicate_uri, None)))
    return counts


def strict_exclusion_errors(counts: Dict[str, int]) -> List[str]:
    errors: List[str] = []
    for key, count in counts.items():
        if count > 0:
            errors.append(f"Strict exclusion violation: {key}={count}")
    return errors


def graph_count_summary(graph: Graph) -> Dict[str, int]:
    subjects = {subject for subject, _, _ in graph}
    predicates = {predicate for _, predicate, _ in graph}
    objects = {obj for _, _, obj in graph}
    return {
        "total_triples": len(graph),
        "unique_subjects": len(subjects),
        "unique_predicates": len(predicates),
        "unique_objects": len(objects),
    }


def serialize_deterministic_turtle(graph: Graph, output_path: Path) -> None:
    ensure_parent(output_path)

    prefixes: List[Tuple[str, str]] = []
    for prefix, namespace in graph.namespaces():
        if prefix is None:
            continue
        prefixes.append((prefix, str(namespace)))
    prefixes = sorted(set(prefixes), key=lambda item: item[0])

    grouped: Dict[Identifier, Dict[Identifier, List[Identifier]]] = defaultdict(lambda: defaultdict(list))
    for subject, predicate, obj in sorted(graph, key=triple_sort_key):
        grouped[subject][predicate].append(obj)

    lines: List[str] = [f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in prefixes]
    if lines:
        lines.append("")

    manager = graph.namespace_manager
    sorted_subjects = sorted(grouped.keys(), key=term_sort_key)
    for subject in sorted_subjects:
        predicate_map = grouped[subject]
        predicate_items = sorted(predicate_map.items(), key=lambda item: term_sort_key(item[0]))
        subject_text = subject.n3(namespace_manager=manager)
        lines.append(f"{subject_text}")

        for index, (predicate, objects) in enumerate(predicate_items):
            predicate_text = predicate.n3(namespace_manager=manager)
            object_texts = [obj.n3(namespace_manager=manager) for obj in sorted(objects, key=term_sort_key)]
            separator = " ;" if index < len(predicate_items) - 1 else " ."
            if len(object_texts) == 1:
                lines.append(f"    {predicate_text} {object_texts[0]}{separator}")
                continue

            first_object, remaining_objects = object_texts[0], object_texts[1:]
            lines.append(f"    {predicate_text} {first_object}")
            for object_index, object_text in enumerate(remaining_objects):
                object_separator = "," if object_index < len(remaining_objects) - 1 else separator
                lines.append(f"        {object_text}{object_separator}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_flat_graph(source_graph: Graph) -> Dict[str, Any]:
    relation_nodes = sorted({node for node in source_graph.subjects(RDF.type, ETD.Relation)}, key=relation_uri_sort_key)

    incomplete_relations: List[Dict[str, Any]] = []
    policy_exclusion_counts: Counter[str] = Counter()
    policy_exclusion_samples: List[Dict[str, Any]] = []
    flattened_candidates: List[Tuple[Identifier, Identifier, Identifier]] = []

    complete_relation_count = 0

    for relation in relation_nodes:
        subjects = sorted(set(source_graph.objects(relation, ETD.subject)), key=term_sort_key)
        predicates = sorted(set(source_graph.objects(relation, ETD.predicate)), key=term_sort_key)
        objects = sorted(set(source_graph.objects(relation, ETD.object)), key=term_sort_key)

        issues: List[str] = []
        if len(subjects) != 1:
            issues.append("missing_subject" if len(subjects) == 0 else "multiple_subject_values")
        if len(predicates) != 1:
            issues.append("missing_predicate" if len(predicates) == 0 else "multiple_predicate_values")
        if len(objects) != 1:
            issues.append("missing_object" if len(objects) == 0 else "multiple_object_values")

        if issues:
            incomplete_relations.append(
                {
                    "relation": str(relation),
                    "issue": issues,
                    "subject_count": len(subjects),
                    "predicate_count": len(predicates),
                    "object_count": len(objects),
                }
            )
            continue

        subject = subjects[0]
        predicate = predicates[0]
        obj = objects[0]
        complete_relation_count += 1

        if not isinstance(subject, URIRef):
            policy_exclusion_counts["non_uri_subject"] += 1
            if len(policy_exclusion_samples) < 25:
                policy_exclusion_samples.append(
                    {"relation": str(relation), "reason": "non_uri_subject", "subject": subject.n3()}
                )
            continue

        if not isinstance(predicate, URIRef):
            policy_exclusion_counts["non_uri_predicate"] += 1
            if len(policy_exclusion_samples) < 25:
                policy_exclusion_samples.append(
                    {"relation": str(relation), "reason": "non_uri_predicate", "predicate": predicate.n3()}
                )
            continue

        if predicate in EXCLUDED_PREDICATES:
            policy_exclusion_counts[f"forbidden_predicate::{predicate}"] += 1
            if len(policy_exclusion_samples) < 25:
                policy_exclusion_samples.append(
                    {"relation": str(relation), "reason": "forbidden_predicate", "predicate": str(predicate)}
                )
            continue

        if is_excluded_endpoint_node(source_graph, subject):
            policy_exclusion_counts["excluded_subject_class"] += 1
            if len(policy_exclusion_samples) < 25:
                policy_exclusion_samples.append(
                    {"relation": str(relation), "reason": "excluded_subject_class", "subject": str(subject)}
                )
            continue

        if is_excluded_endpoint_node(source_graph, obj):
            policy_exclusion_counts["excluded_object_class"] += 1
            if len(policy_exclusion_samples) < 25:
                policy_exclusion_samples.append(
                    {"relation": str(relation), "reason": "excluded_object_class", "object": str(obj)}
                )
            continue

        flattened_candidates.append((subject, predicate, obj))

    unique_flattened_triples = sorted(set(flattened_candidates), key=triple_sort_key)

    output_graph = Graph()
    copy_namespace_bindings(source_graph, output_graph)

    for triple in unique_flattened_triples:
        output_graph.add(triple)

    referenced_nodes = set()
    for subject, _, obj in unique_flattened_triples:
        if isinstance(subject, URIRef):
            referenced_nodes.add(subject)
        if isinstance(obj, URIRef):
            referenced_nodes.add(obj)

    for node in sorted(referenced_nodes, key=term_sort_key):
        for label_predicate in ALLOWED_LABEL_PREDICATES:
            for label_value in sorted(set(source_graph.objects(node, label_predicate)), key=term_sort_key):
                output_graph.add((node, label_predicate, label_value))

        for type_uri in sorted(set(source_graph.objects(node, RDF.type)), key=term_sort_key):
            if useful_type(type_uri):
                output_graph.add((node, RDF.type, type_uri))

    return {
        "relation_nodes": relation_nodes,
        "complete_relation_count": complete_relation_count,
        "incomplete_relations": incomplete_relations,
        "policy_exclusion_counts": dict(sorted(policy_exclusion_counts.items(), key=lambda item: item[0])),
        "policy_exclusion_samples": policy_exclusion_samples,
        "flattened_candidates": flattened_candidates,
        "unique_flattened_triples": unique_flattened_triples,
        "output_graph": output_graph,
    }


def execute_queries(graph: Graph) -> Dict[str, Any]:
    query_specs = [
        {
            "query_id": "Q1",
            "capability": "Entity/direct relation retrieval",
            "expected": "> 0 rows",
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?subject ?predicate ?object
                WHERE {
                  ?subject ?predicate ?object .
                  FILTER(?predicate NOT IN (rdf:type, rdfs:label))
                }
                LIMIT 25
            """,
            "pass_rule": lambda count: count > 0,
        },
        {
            "query_id": "Q2",
            "capability": "Predicate distribution",
            "expected": "> 0 predicate groups",
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?predicate (COUNT(*) AS ?count)
                WHERE {
                  ?subject ?predicate ?object .
                  FILTER(?predicate NOT IN (rdf:type, rdfs:label))
                }
                GROUP BY ?predicate
                ORDER BY DESC(?count)
            """,
            "pass_rule": lambda count: count > 0,
        },
        {
            "query_id": "Q3",
            "capability": "Question-answer tracing",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer ?question
                WHERE {
                  ?answer a etd:Answer ;
                          etd:answersQuestion ?question .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q4",
            "capability": "Theme-level aggregation",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?theme (COUNT(DISTINCT ?answer) AS ?answerCount)
                WHERE {
                  ?answer a etd:Answer ;
                          etd:answersQuestion ?question .
                  ?question etd:belongsToTheme ?theme .
                }
                GROUP BY ?theme
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q5",
            "capability": "Expert contribution tracing",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer ?expert
                WHERE {
                  ?answer a etd:Answer ;
                          etd:authoredBy ?expert .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q6",
            "capability": "Vote-supported answer analysis",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer (COUNT(DISTINCT ?vote) AS ?voteCount)
                WHERE {
                  ?answer a etd:Answer ;
                          etd:hasVote ?vote .
                }
                GROUP BY ?answer
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q7",
            "capability": "Evidence-backed claim tracing",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation ?evidence
                WHERE {
                  ?relation a etd:Relation ;
                            etd:hasEvidence ?evidence .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q8",
            "capability": "Confidence-aware filtering",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation ?confidence
                WHERE {
                  ?relation a etd:Relation ;
                            etd:confidence ?confidence .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q9",
            "capability": "Inference-aware querying",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation ?isInferred
                WHERE {
                  ?relation a etd:Relation ;
                            etd:isInferred ?isInferred .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q10",
            "capability": "Claim-to-discourse traceability",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation ?answer ?question ?theme ?expert
                WHERE {
                  ?relation a etd:Relation ;
                            etd:hasSourceEntity ?answer .
                  ?answer a etd:Answer ;
                          etd:answersQuestion ?question ;
                          etd:authoredBy ?expert .
                  ?question etd:belongsToTheme ?theme .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q11",
            "capability": "Reified relation provenance",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation ?subject ?predicate ?object
                WHERE {
                  ?relation a etd:Relation ;
                            etd:subject ?subject ;
                            etd:predicate ?predicate ;
                            etd:object ?object .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q12",
            "capability": "Evidence node absence",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?evidence
                WHERE {
                  ?evidence a etd:Evidence .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
    ]

    query_results: List[Dict[str, Any]] = []
    pass_lookup: Dict[str, bool] = {}
    for spec in query_specs:
        rows = list(graph.query(spec["sparql"]))
        actual_count = len(rows)
        passed = bool(spec["pass_rule"](actual_count))
        score = spec["score_if_supported"] if passed else 0.0
        query_results.append(
            {
                "query_id": spec["query_id"],
                "capability": spec["capability"],
                "expected": spec["expected"],
                "actual_count": actual_count,
                "score": score,
                "passed": passed,
            }
        )
        pass_lookup[spec["query_id"]] = passed

    entity_direct_score = 1.0 if pass_lookup.get("Q1") and pass_lookup.get("Q2") else 0.0
    capability_scores = {
        "Question-answer tracing": 0.0,
        "Theme-level aggregation": 0.0,
        "Expert contribution tracing": 0.0,
        "Vote-supported answer analysis": 0.0,
        "Evidence-backed claim tracing": 0.0,
        "Claim-to-discourse traceability": 0.0,
        "Reified relation provenance": 0.0,
        "Confidence-aware filtering": 0.0,
        "Inference-aware querying": 0.0,
        "Entity/direct relation retrieval": entity_direct_score,
    }
    mean_score = round(sum(capability_scores.values()) / len(capability_scores), 4)

    return {
        "query_results": query_results,
        "capability_scores": capability_scores,
        "mean_representational_score_without_shacl": mean_score,
        "notes": (
            "Flat text-to-triple baseline supports direct entity-relation retrieval but deliberately lacks "
            "discourse, evidence, confidence, inference and reified relation provenance."
        ),
    }


def console_summary(
    input_path: Path,
    output_path: Path,
    source_graph_counts: Dict[str, int],
    flattening_counts: Dict[str, int],
    output_graph_counts: Dict[str, int],
    strict_counts: Dict[str, int],
    query_payload: Dict[str, Any],
    passed: bool,
) -> str:
    query_map = {item["query_id"]: item for item in query_payload["query_results"]}
    return "\n".join(
        [
            "Flat text-to-triple baseline generated.",
            f"Input: {display_path(input_path)}",
            f"Output: {display_path(output_path)}",
            "",
            "Source graph:",
            f"Total triples: {source_graph_counts['total_triples']}",
            f"Reified relations: {source_graph_counts['reified_relation_nodes']}",
            "",
            "Flattening:",
            f"Complete relations: {flattening_counts['complete_reified_relation_count']}",
            f"Incomplete relations: {flattening_counts['incomplete_reified_relation_count']}",
            f"Flattened triple attempts: {flattening_counts['flattened_triple_attempt_count']}",
            f"Unique flattened triples: {flattening_counts['unique_flattened_triple_count']}",
            f"Duplicate flattened triples removed: {flattening_counts['duplicate_flattened_triple_count']}",
            "",
            "Output graph:",
            f"Total triples: {output_graph_counts['total_triples']}",
            f"Unique subjects: {output_graph_counts['unique_subjects']}",
            f"Unique predicates: {output_graph_counts['unique_predicates']}",
            f"Unique objects: {output_graph_counts['unique_objects']}",
            "",
            "Strict exclusions:",
            f"Relation nodes: {strict_counts['relation_nodes']}",
            f"Evidence nodes: {strict_counts['evidence_nodes']}",
            f"DiscourseUnit nodes: {strict_counts['discourse_unit_nodes']}",
            f"FieldAssertion nodes: {strict_counts['field_assertion_nodes']}",
            f"Answer nodes: {strict_counts['answer_nodes']}",
            f"Question nodes: {strict_counts['question_nodes']}",
            f"Theme nodes: {strict_counts['theme_nodes']}",
            f"Expert nodes: {strict_counts['expert_nodes']}",
            f"Vote nodes: {strict_counts['vote_nodes']}",
            f"confidence triples: {strict_counts['confidence_triples']}",
            f"isInferred triples: {strict_counts['isInferred_triples']}",
            f"hasSourceEntity triples: {strict_counts['hasSourceEntity_triples']}",
            f"answersQuestion triples: {strict_counts['answersQuestion_triples']}",
            f"belongsToTheme triples: {strict_counts['belongsToTheme_triples']}",
            f"authoredBy triples: {strict_counts['authoredBy_triples']}",
            "",
            "Query results:",
            f"Q1 Direct triple retrieval: actual_count={query_map['Q1']['actual_count']}, score={query_map['Q1']['score']}",
            f"Q2 Predicate distribution: actual_count={query_map['Q2']['actual_count']}, supporting evidence",
            f"Q3 Question-answer tracing: actual_count={query_map['Q3']['actual_count']}, score={query_map['Q3']['score']}",
            f"Q4 Theme-level aggregation: actual_count={query_map['Q4']['actual_count']}, score={query_map['Q4']['score']}",
            f"Q5 Expert contribution tracing: actual_count={query_map['Q5']['actual_count']}, score={query_map['Q5']['score']}",
            f"Q6 Vote-supported answer analysis: actual_count={query_map['Q6']['actual_count']}, score={query_map['Q6']['score']}",
            f"Q7 Evidence-backed claim tracing: actual_count={query_map['Q7']['actual_count']}, score={query_map['Q7']['score']}",
            f"Q8 Confidence-aware filtering: actual_count={query_map['Q8']['actual_count']}, score={query_map['Q8']['score']}",
            f"Q9 Inference-aware querying: actual_count={query_map['Q9']['actual_count']}, score={query_map['Q9']['score']}",
            f"Q10 Claim-to-discourse traceability: actual_count={query_map['Q10']['actual_count']}, score={query_map['Q10']['score']}",
            f"Q11 Reified relation provenance: actual_count={query_map['Q11']['actual_count']}, score={query_map['Q11']['score']}",
            f"Q12 Evidence node absence: actual_count={query_map['Q12']['actual_count']}, score={query_map['Q12']['score']}",
            "",
            f"Mean score without SHACL: {query_payload['mean_representational_score_without_shacl']}",
            "",
            "Final status:",
            "PASS" if passed else "FAIL",
        ]
    )


def generation_log_text(
    timestamp: str,
    script_path: Path,
    input_path: Path,
    output_path: Path,
    input_sha256: str,
    output_sha256: str,
    source_graph_counts: Dict[str, int],
    flattening_counts: Dict[str, int],
    predicate_counts: Dict[str, int],
    policy_exclusion_counts: Dict[str, int],
    strict_counts: Dict[str, int],
    query_payload: Dict[str, Any],
    passed: bool,
) -> str:
    query_lines = [
        f"{item['query_id']} {item['capability']}: actual_count={item['actual_count']}, "
        f"score={item['score']}, passed={'true' if item['passed'] else 'false'}"
        for item in query_payload["query_results"]
    ]
    predicate_lines = [f"{predicate}: {count}" for predicate, count in predicate_counts.items()]
    policy_lines = (
        [f"{key}: {value}" for key, value in policy_exclusion_counts.items()]
        if policy_exclusion_counts
        else ["none"]
    )

    return "\n".join(
        [
            "Flat text-to-triple baseline generation log",
            "",
            f"Generated at: {timestamp}",
            f"Input file: {display_path(input_path)}",
            f"Input SHA-256: {input_sha256}",
            f"Output file: {display_path(output_path)}",
            f"Output SHA-256: {output_sha256}",
            f"Script path: {display_path(script_path)}",
            "",
            "Method description:",
            (
                "Deterministic flattening of validated TQAVE-KG reified relation nodes. Each complete "
                "etd:Relation node was transformed from relation-level subject/predicate/object metadata "
                "into a direct RDF triple. Relation-level evidence, confidence, inference, source-entity, "
                "discourse, vote and provenance metadata were deliberately excluded."
            ),
            "",
            "LLM used: No",
            "Manual sampling used: No",
            "Manual filtering used: No",
            "",
            f"Relation count read from TQAVE-KG: {flattening_counts['reified_relation_count']}",
            f"Complete relation count: {flattening_counts['complete_reified_relation_count']}",
            f"Incomplete relation count: {flattening_counts['incomplete_reified_relation_count']}",
            f"Flattened triple attempt count: {flattening_counts['flattened_triple_attempt_count']}",
            f"Unique flattened triple count: {flattening_counts['unique_flattened_triple_count']}",
            f"Duplicate flattened triple count: {flattening_counts['duplicate_flattened_triple_count']}",
            "",
            "Predicate distribution summary:",
            *predicate_lines,
            "",
            "Included content:",
            "Direct subject-predicate-object triples derived from eligible reified relations.",
            "rdfs:label, etd:name, etd:canonicalLabel and skos:prefLabel values for referenced subject/object nodes when present.",
            "Useful rdf:type assertions for referenced subject/object entity nodes, excluding TQAVE discourse artefacts.",
            "",
            "Excluded content:",
            "etd:Relation, etd:Evidence, etd:DiscourseUnit, etd:FieldAssertion, etd:Vote, etd:Answer, etd:Question, etd:Theme, etd:Expert.",
            "Relation-level provenance, evidence, confidence, inference, source-entity, vote and discourse predicates.",
            "owl:sameAs and etd:ontology_uri.",
            "",
            "Policy exclusions applied while flattening:",
            *policy_lines,
            "",
            "Source graph counts:",
            f"Total triples: {source_graph_counts['total_triples']}",
            f"Reified relation nodes: {source_graph_counts['reified_relation_nodes']}",
            "",
            "Strict exclusion counts:",
            *[f"{key}: {value}" for key, value in sorted(strict_counts.items())],
            "",
            "Query summary:",
            *query_lines,
            f"Mean score without SHACL: {query_payload['mean_representational_score_without_shacl']}",
            "",
            "Final status:",
            "PASS" if passed else "FAIL",
        ]
    )


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    audit_output_path = resolve_path(args.audit_output)
    query_results_output_path = resolve_path(args.query_results_output)
    queries_output_path = resolve_path(args.queries_output)
    log_output_path = resolve_path(args.log_output)
    shapes_output_path = queries_output_path.parent / "flat_triple_baseline_shapes.ttl"

    ensure_directories(
        [
            output_path.parent,
            audit_output_path.parent,
            query_results_output_path.parent,
            queries_output_path.parent,
            log_output_path.parent,
        ]
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    errors: List[str] = []

    write_text(queries_output_path, QUERY_FILE_TEXT)
    write_text(shapes_output_path, SHACL_SHAPES)

    if not input_path.exists():
        raise RuntimeError(
            f"Input TTL not found at {display_path(input_path)}. Pass an alternative with --input."
        )

    source_graph = load_graph(input_path)
    source_graph_counts = {
        "total_triples": len(source_graph),
        "reified_relation_nodes": len({node for node in source_graph.subjects(RDF.type, ETD.Relation)}),
    }

    build_payload = build_flat_graph(source_graph)
    output_graph: Graph = build_payload["output_graph"]
    unique_flattened_triples = build_payload["unique_flattened_triples"]
    flattened_candidates = build_payload["flattened_candidates"]
    incomplete_relations = build_payload["incomplete_relations"]

    if build_payload["complete_relation_count"] == 0:
        errors.append("No complete reified relations were found in the input graph.")

    flattened_triple_attempt_count = len(flattened_candidates)
    unique_flattened_triple_count = len(unique_flattened_triples)
    duplicate_flattened_triple_count = flattened_triple_attempt_count - unique_flattened_triple_count

    if unique_flattened_triple_count == 0:
        errors.append("No eligible flattened direct triples remained after exclusions were applied.")

    serialize_deterministic_turtle(output_graph, output_path)

    strict_counts = strict_exclusion_counts(output_graph)
    errors.extend(strict_exclusion_errors(strict_counts))

    if strict_counts["relation_nodes"] > 0:
        errors.append("Output graph contains etd:Relation nodes.")
    if strict_counts["evidence_nodes"] > 0:
        errors.append("Output graph contains etd:Evidence nodes.")
    if strict_counts["answer_nodes"] > 0:
        errors.append("Output graph contains etd:Answer nodes.")
    if strict_counts["question_nodes"] > 0:
        errors.append("Output graph contains etd:Question nodes.")
    if strict_counts["theme_nodes"] > 0:
        errors.append("Output graph contains etd:Theme nodes.")
    if strict_counts["expert_nodes"] > 0:
        errors.append("Output graph contains etd:Expert nodes.")
    if strict_counts["vote_nodes"] > 0:
        errors.append("Output graph contains etd:Vote nodes.")
    if strict_counts["confidence_triples"] > 0:
        errors.append("Output graph contains etd:confidence triples.")
    if strict_counts["isInferred_triples"] > 0:
        errors.append("Output graph contains etd:isInferred triples.")
    if strict_counts["hasSourceEntity_triples"] > 0:
        errors.append("Output graph contains etd:hasSourceEntity triples.")
    if strict_counts["answersQuestion_triples"] > 0:
        errors.append("Output graph contains etd:answersQuestion triples.")
    if strict_counts["belongsToTheme_triples"] > 0:
        errors.append("Output graph contains etd:belongsToTheme triples.")
    if strict_counts["authoredBy_triples"] > 0:
        errors.append("Output graph contains etd:authoredBy triples.")

    input_sha256 = sha256_file(input_path)
    output_sha256 = sha256_file(output_path)

    output_graph_counts = graph_count_summary(output_graph)
    predicate_counts = predicate_distribution(unique_flattened_triples)

    flattening_counts = {
        "reified_relation_count": source_graph_counts["reified_relation_nodes"],
        "complete_reified_relation_count": build_payload["complete_relation_count"],
        "incomplete_reified_relation_count": len(incomplete_relations),
        "flattened_triple_attempt_count": flattened_triple_attempt_count,
        "unique_flattened_triple_count": unique_flattened_triple_count,
        "duplicate_flattened_triple_count": duplicate_flattened_triple_count,
        "policy_excluded_relation_count": build_payload["complete_relation_count"] - flattened_triple_attempt_count,
    }

    query_payload = execute_queries(output_graph)

    passed = len(errors) == 0 and all(
        item["passed"] for item in query_payload["query_results"] if item["query_id"] in {"Q1", "Q2"}
    )
    if not query_payload["capability_scores"]["Entity/direct relation retrieval"]:
        errors.append("Entity/direct relation retrieval queries did not both pass.")
        passed = False

    audit_payload: Dict[str, Any] = {
        "input_file": display_path(input_path),
        "output_file": display_path(output_path),
        "generation_method": "deterministic_flattening_of_tqave_reified_relations",
        "llm_used": False,
        "manual_filtering": False,
        "manual_sampling": False,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "source_graph_counts": source_graph_counts,
        "flattening_counts": flattening_counts,
        "output_graph_counts": output_graph_counts,
        "predicate_distribution": predicate_counts,
        "strict_exclusion_counts": strict_counts,
        "policy_exclusion_counts": build_payload["policy_exclusion_counts"],
        "policy_exclusion_samples": build_payload["policy_exclusion_samples"],
        "incomplete_relations": incomplete_relations,
        "flat_triple_shacl_shapes_created": True,
        "errors": errors,
        "passed": passed,
    }

    query_results_payload = {
        "graph_file": display_path(output_path),
        "query_results": query_payload["query_results"],
        "capability_scores": query_payload["capability_scores"],
        "mean_representational_score_without_shacl": query_payload["mean_representational_score_without_shacl"],
        "notes": query_payload["notes"],
    }

    log_text = generation_log_text(
        timestamp=timestamp,
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        source_graph_counts=source_graph_counts,
        flattening_counts=flattening_counts,
        predicate_counts=predicate_counts,
        policy_exclusion_counts=build_payload["policy_exclusion_counts"],
        strict_counts=strict_counts,
        query_payload=query_payload,
        passed=passed,
    )

    write_json(audit_output_path, audit_payload)
    write_json(query_results_output_path, query_results_payload)
    write_text(log_output_path, log_text)

    print(
        console_summary(
            input_path=input_path,
            output_path=output_path,
            source_graph_counts=source_graph_counts,
            flattening_counts=flattening_counts,
            output_graph_counts=output_graph_counts,
            strict_counts=strict_counts,
            query_payload=query_payload,
            passed=passed,
        )
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
