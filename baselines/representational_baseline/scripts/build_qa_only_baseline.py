#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ETD_NS = "https://alfie-project.eu/etd-hub#"
ETD = Namespace(ETD_NS)

ALLOWED_CLASS_LOCAL_NAMES = ("Theme", "Question", "Answer", "Expert", "Vote")
FORBIDDEN_CLASS_LOCAL_NAMES = ("Relation", "Evidence", "Entity", "DiscourseUnit", "FieldAssertion")
FORBIDDEN_PREDICATE_LOCAL_NAMES = (
    "subject",
    "predicate",
    "object",
    "hasEvidence",
    "evidenceText",
    "confidence",
    "isInferred",
    "inferenceRule",
    "hasSourceEntity",
    "groundingStatus",
    "groundingMethod",
    "ontology_uri",
)
FORBIDDEN_CLASS_COUNT_KEYS = {
    "Relation": "relation_nodes",
    "Evidence": "evidence_nodes",
    "Entity": "entity_nodes",
    "DiscourseUnit": "discourse_unit_nodes",
    "FieldAssertion": "field_assertion_nodes",
}

SPARQL_QUERIES = """# Q1: Answer to Question and Theme tracing
# Expected result: 21 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer ?question ?theme
WHERE {
  ?answer a etd:Answer ;
          etd:answersQuestion ?question .
  ?question a etd:Question ;
            etd:belongsToTheme ?theme .
}
ORDER BY ?answer

# Q2: Theme-level answer aggregation
# Expected result: themes with one or more answers
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?theme (COUNT(DISTINCT ?answer) AS ?answerCount)
WHERE {
  ?answer a etd:Answer ;
          etd:answersQuestion ?question .
  ?question etd:belongsToTheme ?theme .
}
GROUP BY ?theme
ORDER BY DESC(?answerCount)

# Q3: Expert contribution tracing through answers
# Expected result: 21 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer ?expert
WHERE {
  ?answer a etd:Answer ;
          etd:authoredBy ?expert .
}
ORDER BY ?answer

# Q4: Vote-supported answer analysis
# Expected result: answers with one or more votes
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?answer (COUNT(DISTINCT ?vote) AS ?voteCount)
WHERE {
  ?answer a etd:Answer ;
          etd:hasVote ?vote .
}
GROUP BY ?answer
ORDER BY DESC(?voteCount)

# Q5: Evidence-backed claim tracing negative control
# Expected result: 0 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?evidence
WHERE {
  ?relation a etd:Relation ;
            etd:hasEvidence ?evidence .
}

# Q6: Confidence-aware filtering negative control
# Expected result: 0 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?confidence
WHERE {
  ?relation a etd:Relation ;
            etd:confidence ?confidence .
}

# Q7: Inference-aware querying negative control
# Expected result: 0 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation ?isInferred
WHERE {
  ?relation a etd:Relation ;
            etd:isInferred ?isInferred .
}

# Q8: Claim-to-discourse traceability negative control
# Expected result: 0 rows
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

# Q9: Reified relation node absence
# Expected result: 0 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?relation
WHERE {
  ?relation a etd:Relation .
}

# Q10: Evidence node absence
# Expected result: 0 rows
PREFIX etd: <https://alfie-project.eu/etd-hub#>

SELECT ?evidence
WHERE {
  ?evidence a etd:Evidence .
}
"""

SHACL_SHAPES = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix etd: <https://alfie-project.eu/etd-hub#> .

etd:QuestionThemeShape
    a sh:NodeShape ;
    sh:targetClass etd:Question ;
    sh:property [
        sh:path etd:belongsToTheme ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class etd:Theme ;
    ] .

etd:AnswerQuestionShape
    a sh:NodeShape ;
    sh:targetClass etd:Answer ;
    sh:property [
        sh:path etd:answersQuestion ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class etd:Question ;
    ] .

etd:AnswerAuthorShape
    a sh:NodeShape ;
    sh:targetClass etd:Answer ;
    sh:property [
        sh:path etd:authoredBy ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class etd:Expert ;
    ] .

etd:VoteAnswerShape
    a sh:NodeShape ;
    sh:targetClass etd:Vote ;
    sh:property [
        sh:path etd:receivesVote ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class etd:Answer ;
    ] .

etd:VoteExpertShape
    a sh:NodeShape ;
    sh:targetClass etd:Vote ;
    sh:property [
        sh:path etd:castBy ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class etd:Expert ;
    ] .
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Q&A-only RDF baseline from the consultation dataset."
    )
    parser.add_argument("--input", default="data/ai_ethics_datasets.json")
    parser.add_argument("--output", default="baselines/output/qa_only_baseline.ttl")
    parser.add_argument("--audit-output", default="baselines/results/qa_only_baseline_audit.json")
    parser.add_argument("--query-results-output", default="baselines/results/qa_only_query_results.json")
    parser.add_argument("--queries-output", default="baselines/queries/qa_only_queries.sparql")
    parser.add_argument("--log-output", default="baselines/results/qa_only_generation_log.txt")
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


def load_dataset(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(f"Failed to load input JSON from {display_path(path)}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected the input JSON root to be an object, got {type(payload).__name__}.")

    return payload


def get_records(payload: Dict[str, Any], key: str, missing_top_level_arrays: List[str]) -> List[Dict[str, Any]]:
    value = payload.get(key)
    if value is None:
        missing_top_level_arrays.append(key)
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"Top-level key {key!r} must contain a list, got {type(value).__name__}.")
    return [item for item in value if isinstance(item, dict)]


def sorted_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed = list(enumerate(records))
    indexed.sort(key=lambda pair: (sort_key(pair[1].get("id")), pair[0]))
    return [item for _, item in indexed]


def sort_key(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def require_id(record: Dict[str, Any], collection_name: str) -> Any:
    if "id" not in record or record.get("id") is None:
        raise RuntimeError(f"Record in {collection_name} is missing a required 'id' field: {record!r}")
    return record["id"]


def uri_for(kind: str, identifier: Any) -> URIRef:
    return URIRef(f"{ETD_NS}{kind}-{identifier}")


def meaningful_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_datetime_literal(value: Any) -> Optional[Literal]:
    text = meaningful_text(value)
    if text is None:
        return None

    candidate = text
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return Literal(text)
    return Literal(text, datatype=XSD.dateTime)


def int_literal(value: Any) -> Optional[Literal]:
    if value is None or value == "":
        return None
    try:
        return Literal(int(value), datatype=XSD.integer)
    except (TypeError, ValueError):
        return None


def bool_literal(value: Any) -> Optional[Literal]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return Literal(True, datatype=XSD.boolean)
        if lowered in {"false", "0", "no"}:
            return Literal(False, datatype=XSD.boolean)
    return None


def add_literal(graph: Graph, subject: URIRef, predicate: URIRef, literal: Optional[Literal]) -> None:
    if literal is not None:
        graph.add((subject, predicate, literal))


def add_text_literal(graph: Graph, subject: URIRef, predicate: URIRef, value: Any) -> None:
    text = meaningful_text(value)
    if text is not None:
        graph.add((subject, predicate, Literal(text)))


def answer_label(description: Any) -> Optional[str]:
    text = meaningful_text(description)
    if text is None:
        return None
    return collapse_whitespace(text)[:80]


def class_count(graph: Graph, class_name: str) -> int:
    return len({subject for subject in graph.subjects(RDF.type, ETD[class_name])})


def triple_count(graph: Graph, predicate: URIRef) -> int:
    return sum(1 for _ in graph.triples((None, predicate, None)))


def node_type_count(graph: Graph, class_uri: URIRef) -> int:
    return sum(1 for _ in graph.triples((None, RDF.type, class_uri)))


def forbidden_counts(graph: Graph) -> Dict[str, int]:
    counts = {
        "relation_nodes": node_type_count(graph, ETD.Relation),
        "evidence_nodes": node_type_count(graph, ETD.Evidence),
        "entity_nodes": node_type_count(graph, ETD.Entity),
        "discourse_unit_nodes": node_type_count(graph, ETD.DiscourseUnit),
        "field_assertion_nodes": node_type_count(graph, ETD.FieldAssertion),
        "subject_triples": triple_count(graph, ETD.subject),
        "predicate_triples": triple_count(graph, ETD.predicate),
        "object_triples": triple_count(graph, ETD.object),
        "hasEvidence_triples": triple_count(graph, ETD.hasEvidence),
        "evidenceText_triples": triple_count(graph, ETD.evidenceText),
        "confidence_triples": triple_count(graph, ETD.confidence),
        "isInferred_triples": triple_count(graph, ETD.isInferred),
        "inferenceRule_triples": triple_count(graph, ETD.inferenceRule),
        "hasSourceEntity_triples": triple_count(graph, ETD.hasSourceEntity),
        "groundingStatus_triples": triple_count(graph, ETD.groundingStatus),
        "groundingMethod_triples": triple_count(graph, ETD.groundingMethod),
        "ontology_uri_triples": triple_count(graph, ETD.ontology_uri),
        "owl_sameAs_triples": triple_count(graph, OWL.sameAs),
    }
    return counts


def expected_theme_aggregation_count(
    questions: Sequence[Dict[str, Any]],
    answers: Sequence[Dict[str, Any]],
    valid_theme_ids: Sequence[Any],
) -> int:
    theme_ids = set(valid_theme_ids)
    question_to_theme: Dict[Any, Any] = {}
    for question in questions:
        qid = question.get("id")
        theme_id = question.get("theme_id")
        if qid is not None and theme_id in theme_ids:
            question_to_theme[qid] = theme_id

    reachable_theme_ids = {
        question_to_theme.get(answer.get("question_id"))
        for answer in answers
        if answer.get("question_id") in question_to_theme
    }
    return len({theme_id for theme_id in reachable_theme_ids if theme_id is not None})


def expected_vote_supported_answer_count(
    votes: Sequence[Dict[str, Any]],
    valid_answer_ids: Sequence[Any],
) -> int:
    answer_ids = set(valid_answer_ids)
    reachable_answers = {
        vote.get("answer_id")
        for vote in votes
        if vote.get("answer_id") in answer_ids
    }
    return len({answer_id for answer_id in reachable_answers if answer_id is not None})


def execute_queries(
    graph: Graph,
    source_counts: Dict[str, int],
    expected_theme_rows: int,
    expected_vote_rows: int,
) -> Dict[str, Any]:
    query_specs = [
        {
            "query_id": "Q1",
            "capability": "Question-answer tracing",
            "expected": f"{source_counts['answers']} rows",
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer ?question ?theme
                WHERE {
                  ?answer a etd:Answer ;
                          etd:answersQuestion ?question .
                  ?question a etd:Question ;
                            etd:belongsToTheme ?theme .
                }
                ORDER BY ?answer
            """,
            "pass_rule": lambda count: count == source_counts["answers"],
        },
        {
            "query_id": "Q2",
            "capability": "Theme-level aggregation",
            "expected": "themes with one or more answers",
            "expected_count": expected_theme_rows,
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?theme (COUNT(DISTINCT ?answer) AS ?answerCount)
                WHERE {
                  ?answer a etd:Answer ;
                          etd:answersQuestion ?question .
                  ?question etd:belongsToTheme ?theme .
                }
                GROUP BY ?theme
                ORDER BY DESC(?answerCount)
            """,
            "pass_rule": lambda count: count == expected_theme_rows,
        },
        {
            "query_id": "Q3",
            "capability": "Expert contribution tracing",
            "expected": f"{source_counts['answers']} rows",
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer ?expert
                WHERE {
                  ?answer a etd:Answer ;
                          etd:authoredBy ?expert .
                }
                ORDER BY ?answer
            """,
            "pass_rule": lambda count: count == source_counts["answers"],
        },
        {
            "query_id": "Q4",
            "capability": "Vote-supported answer analysis",
            "expected": "answers with one or more votes",
            "expected_count": expected_vote_rows,
            "score_if_supported": 1.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?answer (COUNT(DISTINCT ?vote) AS ?voteCount)
                WHERE {
                  ?answer a etd:Answer ;
                          etd:hasVote ?vote .
                }
                GROUP BY ?answer
                ORDER BY DESC(?voteCount)
            """,
            "pass_rule": lambda count: count == expected_vote_rows,
        },
        {
            "query_id": "Q5",
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
            "query_id": "Q6",
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
            "query_id": "Q7",
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
            "query_id": "Q8",
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
            "query_id": "Q9",
            "capability": "Reified relation node absence",
            "expected": "0 rows",
            "score_if_supported": 0.0,
            "sparql": """
                PREFIX etd: <https://alfie-project.eu/etd-hub#>
                SELECT ?relation
                WHERE {
                  ?relation a etd:Relation .
                }
            """,
            "pass_rule": lambda count: count == 0,
        },
        {
            "query_id": "Q10",
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
    for spec in query_specs:
        rows = list(graph.query(spec["sparql"]))
        actual_count = len(rows)
        passed = bool(spec["pass_rule"](actual_count))
        score = spec["score_if_supported"] if passed else 0.0

        result = {
            "query_id": spec["query_id"],
            "capability": spec["capability"],
            "expected": spec["expected"],
            "actual_count": actual_count,
            "score": score,
            "passed": passed,
        }
        if "expected_count" in spec:
            result["expected_count"] = spec["expected_count"]
        query_results.append(result)

    mean_score = round(sum(item["score"] for item in query_results) / len(query_results), 4)
    return {
        "query_results": query_results,
        "mean_representational_score_without_shacl": mean_score,
        "notes": (
            "Q&A-only baseline supports discourse-level navigation but deliberately lacks "
            "claim-level evidence, confidence, inference and reified relation provenance."
        ),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def console_summary(
    input_path: Path,
    output_path: Path,
    rdf_counts: Dict[str, int],
    link_counts: Dict[str, int],
    strict_counts: Dict[str, int],
    query_payload: Dict[str, Any],
    passed: bool,
) -> str:
    query_lines = []
    label_overrides = {
        "Q1": "Question-answer tracing",
        "Q2": "Theme aggregation",
        "Q3": "Expert contribution tracing",
        "Q4": "Vote-supported answer analysis",
        "Q5": "Evidence-backed claim tracing",
        "Q6": "Confidence-aware filtering",
        "Q7": "Inference-aware querying",
        "Q8": "Claim-to-discourse traceability",
        "Q9": "Reified relation node absence",
        "Q10": "Evidence node absence",
    }
    for result in query_payload["query_results"]:
        query_lines.append(
            f"{result['query_id']} {label_overrides[result['query_id']]}: "
            f"actual_count={result['actual_count']}, score={result['score']}"
        )

    return "\n".join(
        [
            "Q&A-only baseline generated.",
            f"Input: {display_path(input_path)}",
            f"Output: {display_path(output_path)}",
            "",
            "Counts:",
            f"Themes: {rdf_counts['themes']}",
            f"Questions: {rdf_counts['questions']}",
            f"Answers: {rdf_counts['answers']}",
            f"Experts: {rdf_counts['experts']}",
            f"Votes: {rdf_counts['votes']}",
            f"Triples: {rdf_counts['total_triples']}",
            "",
            "Links:",
            f"Question-theme links: {link_counts['question_theme_links']}",
            f"Answer-question links: {link_counts['answer_question_links']}",
            f"Answer-expert links: {link_counts['answer_expert_links']}",
            f"Vote-answer links: {link_counts['vote_answer_links']}",
            f"Vote-expert links: {link_counts['vote_expert_links']}",
            "",
            "Strict exclusions:",
            f"Relation nodes: {strict_counts['relation_nodes']}",
            f"Evidence nodes: {strict_counts['evidence_nodes']}",
            f"Entity nodes: {strict_counts['entity_nodes']}",
            f"Confidence triples: {strict_counts['confidence_triples']}",
            f"Inference triples: {strict_counts['isInferred_triples']}",
            f"hasSourceEntity triples: {strict_counts['hasSourceEntity_triples']}",
            "",
            "Query results:",
            *query_lines,
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
    source_counts: Dict[str, int],
    rdf_counts: Dict[str, int],
    link_counts: Dict[str, int],
    strict_counts: Dict[str, int],
    query_payload: Dict[str, Any],
    passed: bool,
) -> str:
    query_lines = [
        f"{item['query_id']} {item['capability']}: actual_count={item['actual_count']}, "
        f"score={item['score']}, passed={'true' if item['passed'] else 'false'}"
        for item in query_payload["query_results"]
    ]

    return "\n".join(
        [
            "Q&A-only baseline generation log",
            "",
            f"Generated at: {timestamp}",
            "",
            "Input:",
            display_path(input_path),
            "",
            "Output:",
            display_path(output_path),
            "",
            "Script:",
            display_path(script_path),
            "",
            "Method:",
            (
                "Deterministic RDF export from the original consultation dataset. No LLM extraction, "
                "no relation extraction, no evidence extraction, no inference, no manual filtering, "
                "and no manual sampling were used."
            ),
            "",
            "LLM used: No",
            "Manual sampling used: No",
            "Manual filtering used: No",
            "",
            "Included RDF classes:",
            "etd:Theme",
            "etd:Question",
            "etd:Answer",
            "etd:Expert",
            "etd:Vote",
            "",
            "Excluded RDF classes:",
            "etd:Relation",
            "etd:Evidence",
            "etd:Entity",
            "etd:DiscourseUnit",
            "etd:FieldAssertion",
            "",
            "Source counts:",
            f"Themes: {source_counts['themes']}",
            f"Questions: {source_counts['questions']}",
            f"Answers: {source_counts['answers']}",
            f"Experts: {source_counts['experts']}",
            f"Votes: {source_counts['votes']}",
            "",
            "RDF node counts:",
            f"Themes: {rdf_counts['themes']}",
            f"Questions: {rdf_counts['questions']}",
            f"Answers: {rdf_counts['answers']}",
            f"Experts: {rdf_counts['experts']}",
            f"Votes: {rdf_counts['votes']}",
            f"Triples: {rdf_counts['total_triples']}",
            "",
            "Link counts:",
            f"Question-theme links: {link_counts['question_theme_links']}",
            f"Answer-question links: {link_counts['answer_question_links']}",
            f"Answer-expert links: {link_counts['answer_expert_links']}",
            f"Answer-parent links: {link_counts['answer_parent_links']}",
            f"Answer-vote links: {link_counts['answer_vote_links']}",
            f"Vote-answer links: {link_counts['vote_answer_links']}",
            f"Vote-expert links: {link_counts['vote_expert_links']}",
            f"Expert-vote links: {link_counts['expert_vote_links']}",
            "",
            "Strict exclusion counts:",
            f"Relation nodes: {strict_counts['relation_nodes']}",
            f"Evidence nodes: {strict_counts['evidence_nodes']}",
            f"Entity nodes: {strict_counts['entity_nodes']}",
            f"DiscourseUnit nodes: {strict_counts['discourse_unit_nodes']}",
            f"FieldAssertion nodes: {strict_counts['field_assertion_nodes']}",
            f"hasEvidence triples: {strict_counts['hasEvidence_triples']}",
            f"confidence triples: {strict_counts['confidence_triples']}",
            f"isInferred triples: {strict_counts['isInferred_triples']}",
            f"hasSourceEntity triples: {strict_counts['hasSourceEntity_triples']}",
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
    shapes_output_path = queries_output_path.parent / "qa_only_baseline_shapes.ttl"

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
    graph = Graph()
    graph.bind("etd", ETD)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)

    missing_top_level_arrays: List[str] = []
    payload = load_dataset(input_path)

    themes = sorted_records(get_records(payload, "Theme", missing_top_level_arrays))
    questions = sorted_records(get_records(payload, "Question", missing_top_level_arrays))
    answers = sorted_records(get_records(payload, "Answer", missing_top_level_arrays))
    experts = sorted_records(get_records(payload, "Expert", missing_top_level_arrays))
    votes = sorted_records(get_records(payload, "Vote", missing_top_level_arrays))

    theme_uri_by_id = {require_id(item, "Theme"): uri_for("theme", require_id(item, "Theme")) for item in themes}
    question_uri_by_id = {
        require_id(item, "Question"): uri_for("question", require_id(item, "Question")) for item in questions
    }
    answer_uri_by_id = {require_id(item, "Answer"): uri_for("answer", require_id(item, "Answer")) for item in answers}
    expert_uri_by_id = {require_id(item, "Expert"): uri_for("expert", require_id(item, "Expert")) for item in experts}
    vote_uri_by_id = {require_id(item, "Vote"): uri_for("vote", require_id(item, "Vote")) for item in votes}

    source_counts = {
        "themes": len(themes),
        "questions": len(questions),
        "answers": len(answers),
        "experts": len(experts),
        "votes": len(votes),
    }

    link_counts = {
        "question_theme_links": 0,
        "answer_question_links": 0,
        "answer_expert_links": 0,
        "answer_parent_links": 0,
        "answer_vote_links": 0,
        "vote_answer_links": 0,
        "vote_expert_links": 0,
        "expert_vote_links": 0,
    }
    optional_link_counts = {
        "theme_authored_by_links": 0,
        "question_authored_by_links": 0,
    }
    missing_references = {
        "missing_question_theme_links": [],
        "missing_answer_question_links": [],
        "missing_answer_expert_links": [],
        "missing_answer_parent_links": [],
        "missing_vote_answer_links": [],
        "missing_vote_expert_links": [],
    }

    for theme in themes:
        identifier = require_id(theme, "Theme")
        subject = theme_uri_by_id[identifier]
        graph.add((subject, RDF.type, ETD.Theme))
        add_literal(graph, subject, ETD.themeId, int_literal(identifier))
        add_text_literal(graph, subject, ETD.name, theme.get("name"))
        add_text_literal(graph, subject, RDFS.label, theme.get("name"))
        add_text_literal(graph, subject, ETD.description, theme.get("description"))
        add_literal(graph, subject, ETD.views, int_literal(theme.get("views")))
        add_literal(graph, subject, ETD.expertId, int_literal(theme.get("expert_id")))
        add_text_literal(graph, subject, ETD.problem_category, theme.get("problem_category"))
        add_text_literal(graph, subject, ETD.model_category, theme.get("model_category"))
        add_text_literal(graph, subject, ETD.domain_category, theme.get("domain_category"))
        add_literal(graph, subject, ETD.created_at, parse_datetime_literal(theme.get("created_at")))

        expert_id = theme.get("expert_id")
        if expert_id in expert_uri_by_id:
            graph.add((subject, ETD.authoredBy, expert_uri_by_id[expert_id]))
            optional_link_counts["theme_authored_by_links"] += 1

    for question in questions:
        identifier = require_id(question, "Question")
        subject = question_uri_by_id[identifier]
        graph.add((subject, RDF.type, ETD.Question))
        add_literal(graph, subject, ETD.questionId, int_literal(identifier))
        add_text_literal(graph, subject, ETD.title, question.get("title"))
        add_text_literal(graph, subject, RDFS.label, question.get("title"))
        add_text_literal(graph, subject, ETD.body, question.get("body"))
        add_literal(graph, subject, ETD.created_at, parse_datetime_literal(question.get("created_at")))
        add_literal(graph, subject, ETD.views, int_literal(question.get("views")))
        add_literal(graph, subject, ETD.expertId, int_literal(question.get("expert_id")))
        add_literal(graph, subject, ETD.themeId, int_literal(question.get("theme_id")))

        theme_id = question.get("theme_id")
        if theme_id in theme_uri_by_id:
            graph.add((subject, ETD.belongsToTheme, theme_uri_by_id[theme_id]))
            link_counts["question_theme_links"] += 1
        else:
            missing_references["missing_question_theme_links"].append(
                {"question_id": identifier, "theme_id": theme_id}
            )

        expert_id = question.get("expert_id")
        if expert_id in expert_uri_by_id:
            graph.add((subject, ETD.authoredBy, expert_uri_by_id[expert_id]))
            optional_link_counts["question_authored_by_links"] += 1

    for answer in answers:
        identifier = require_id(answer, "Answer")
        subject = answer_uri_by_id[identifier]
        graph.add((subject, RDF.type, ETD.Answer))
        add_literal(graph, subject, ETD.answerId, int_literal(identifier))
        add_text_literal(graph, subject, ETD.description, answer.get("description"))
        add_text_literal(graph, subject, RDFS.label, answer_label(answer.get("description")))
        add_literal(graph, subject, ETD.created_at, parse_datetime_literal(answer.get("created_at")))
        add_literal(graph, subject, ETD.questionId, int_literal(answer.get("question_id")))
        add_literal(graph, subject, ETD.expertId, int_literal(answer.get("expert_id")))
        if answer.get("parent_id") is not None:
            add_literal(graph, subject, ETD.parentId, int_literal(answer.get("parent_id")))

        question_id = answer.get("question_id")
        if question_id in question_uri_by_id:
            graph.add((subject, ETD.answersQuestion, question_uri_by_id[question_id]))
            link_counts["answer_question_links"] += 1
        else:
            missing_references["missing_answer_question_links"].append(
                {"answer_id": identifier, "question_id": question_id}
            )

        expert_id = answer.get("expert_id")
        if expert_id in expert_uri_by_id:
            graph.add((subject, ETD.authoredBy, expert_uri_by_id[expert_id]))
            link_counts["answer_expert_links"] += 1
        else:
            missing_references["missing_answer_expert_links"].append(
                {"answer_id": identifier, "expert_id": expert_id}
            )

        parent_id = answer.get("parent_id")
        if parent_id is not None:
            if parent_id in answer_uri_by_id:
                graph.add((subject, ETD.parentAnswer, answer_uri_by_id[parent_id]))
                link_counts["answer_parent_links"] += 1
            else:
                missing_references["missing_answer_parent_links"].append(
                    {"answer_id": identifier, "parent_id": parent_id}
                )

    for expert in experts:
        identifier = require_id(expert, "Expert")
        subject = expert_uri_by_id[identifier]
        graph.add((subject, RDF.type, ETD.Expert))
        add_literal(graph, subject, ETD.expertId, int_literal(identifier))
        add_literal(graph, subject, ETD.userId, int_literal(expert.get("user_id")))
        add_literal(graph, subject, ETD.isDeleted, bool_literal(expert.get("is_deleted")))
        add_text_literal(graph, subject, ETD.area_of_expertise, expert.get("area_of_expertise"))
        add_text_literal(graph, subject, RDFS.label, expert.get("area_of_expertise"))
        add_text_literal(graph, subject, ETD.bio, expert.get("bio"))
        add_literal(graph, subject, ETD.date_joined, parse_datetime_literal(expert.get("date_joined")))
        add_text_literal(graph, subject, ETD.profile_picture, expert.get("profile_picture"))

    for vote in votes:
        identifier = require_id(vote, "Vote")
        subject = vote_uri_by_id[identifier]
        graph.add((subject, RDF.type, ETD.Vote))
        add_literal(graph, subject, ETD.voteId, int_literal(identifier))
        add_literal(graph, subject, ETD.expertId, int_literal(vote.get("expert_id")))
        add_literal(graph, subject, ETD.answerId, int_literal(vote.get("answer_id")))
        add_literal(graph, subject, ETD.vote_value, int_literal(vote.get("vote_value")))
        add_literal(graph, subject, ETD.created_at, parse_datetime_literal(vote.get("created_at")))
        add_literal(graph, subject, ETD.updated_at, parse_datetime_literal(vote.get("updated_at")))

        answer_id = vote.get("answer_id")
        if answer_id in answer_uri_by_id:
            answer_uri = answer_uri_by_id[answer_id]
            graph.add((subject, ETD.receivesVote, answer_uri))
            graph.add((answer_uri, ETD.hasVote, subject))
            link_counts["vote_answer_links"] += 1
            link_counts["answer_vote_links"] += 1
        else:
            missing_references["missing_vote_answer_links"].append(
                {"vote_id": identifier, "answer_id": answer_id}
            )

        expert_id = vote.get("expert_id")
        if expert_id in expert_uri_by_id:
            expert_uri = expert_uri_by_id[expert_id]
            graph.add((subject, ETD.castBy, expert_uri))
            graph.add((expert_uri, ETD.castVote, subject))
            link_counts["vote_expert_links"] += 1
            link_counts["expert_vote_links"] += 1
        else:
            missing_references["missing_vote_expert_links"].append(
                {"vote_id": identifier, "expert_id": expert_id}
            )

    strict_counts = forbidden_counts(graph)

    rdf_counts = {
        "themes": class_count(graph, "Theme"),
        "questions": class_count(graph, "Question"),
        "answers": class_count(graph, "Answer"),
        "experts": class_count(graph, "Expert"),
        "votes": class_count(graph, "Vote"),
        "total_triples": len(graph),
    }

    expected_theme_rows = expected_theme_aggregation_count(
        questions=questions,
        answers=answers,
        valid_theme_ids=theme_uri_by_id.keys(),
    )
    expected_vote_rows = expected_vote_supported_answer_count(
        votes=votes,
        valid_answer_ids=answer_uri_by_id.keys(),
    )
    query_payload = execute_queries(graph, source_counts, expected_theme_rows, expected_vote_rows)

    fatal_errors: List[str] = []
    required_missing_keys = [
        "missing_question_theme_links",
        "missing_answer_question_links",
        "missing_answer_expert_links",
        "missing_vote_answer_links",
        "missing_vote_expert_links",
    ]
    for key in required_missing_keys:
        if missing_references[key]:
            fatal_errors.append(f"{key}={len(missing_references[key])}")

    for class_name in FORBIDDEN_CLASS_LOCAL_NAMES:
        count_key = FORBIDDEN_CLASS_COUNT_KEYS[class_name]
        if count_key in strict_counts and strict_counts[count_key] > 0:
            fatal_errors.append(f"{count_key}={strict_counts[count_key]}")

    for predicate_name in FORBIDDEN_PREDICATE_LOCAL_NAMES:
        count_key = f"{predicate_name}_triples"
        if count_key in strict_counts and strict_counts[count_key] > 0:
            fatal_errors.append(f"{count_key}={strict_counts[count_key]}")

    if strict_counts["owl_sameAs_triples"] > 0:
        fatal_errors.append(f"owl_sameAs_triples={strict_counts['owl_sameAs_triples']}")

    passed = not fatal_errors

    ensure_parent(output_path)
    try:
        graph.serialize(destination=str(output_path), format="turtle")
    except Exception as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(f"Failed to write Turtle output to {display_path(output_path)}: {exc}") from exc

    write_text(queries_output_path, SPARQL_QUERIES)
    write_text(shapes_output_path, SHACL_SHAPES)

    audit_payload = {
        "input_file": display_path(input_path),
        "output_file": display_path(output_path),
        "generation_method": "deterministic_q_and_a_only_rdf_export",
        "llm_used": False,
        "manual_filtering": False,
        "manual_sampling": False,
        "qa_only_shacl_shapes_created": True,
        "included_classes": [f"etd:{name}" for name in ALLOWED_CLASS_LOCAL_NAMES],
        "excluded_classes": [f"etd:{name}" for name in FORBIDDEN_CLASS_LOCAL_NAMES],
        "missing_top_level_arrays": missing_top_level_arrays,
        "source_counts": source_counts,
        "rdf_counts": rdf_counts,
        "link_counts": link_counts,
        "optional_link_counts": optional_link_counts,
        "strict_exclusion_counts": strict_counts,
        "missing_references": missing_references,
        "query_summary": {
            "query_results_file": display_path(query_results_output_path),
            "mean_representational_score_without_shacl": query_payload[
                "mean_representational_score_without_shacl"
            ],
        },
        "errors": fatal_errors,
        "passed": passed,
    }

    query_results_payload = {
        "graph_file": display_path(output_path),
        "query_results": query_payload["query_results"],
        "mean_representational_score_without_shacl": query_payload[
            "mean_representational_score_without_shacl"
        ],
        "notes": query_payload["notes"],
    }

    write_json(audit_output_path, audit_payload)
    write_json(query_results_output_path, query_results_payload)
    write_text(
        log_output_path,
        generation_log_text(
            timestamp=timestamp,
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            source_counts=source_counts,
            rdf_counts=rdf_counts,
            link_counts=link_counts,
            strict_counts=strict_counts,
            query_payload=query_payload,
            passed=passed,
        ),
    )

    print(
        console_summary(
            input_path=input_path,
            output_path=output_path,
            rdf_counts=rdf_counts,
            link_counts=link_counts,
            strict_counts=strict_counts,
            query_payload=query_payload,
            passed=passed,
        )
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
