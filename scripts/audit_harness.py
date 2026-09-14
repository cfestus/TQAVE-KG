#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from validation.shacl_validator import SHACLValidator


def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _resolve_etd_namespace(cli_value: Optional[str]) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()

    try:
        from config import Config  # local import keeps CLI standalone

        etd = (Config.NAMESPACES or {}).get("etd")
        if etd and str(etd).strip():
            return str(etd).strip()
    except Exception:
        pass

    raise ValueError("Missing ETD namespace. Provide --etd-namespace or ensure Config.NAMESPACES['etd'] is set.")


def _entity_nodes(graph: Graph, etd_ns: str) -> Set[URIRef]:
    ignored = {
        URIRef(f"{etd_ns}Relation"),
        URIRef(f"{etd_ns}Evidence"),
        URIRef(f"{etd_ns}FieldAssertion"),
        URIRef(f"{etd_ns}ontology"),
        URIRef(f"{etd_ns}Ontology"),
        URIRef(f"{etd_ns}Export"),
        URIRef(f"{etd_ns}DiscourseUnit"),
    }
    nodes: Set[URIRef] = set()
    for s in set(graph.subjects(RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        types = {o for o in graph.objects(s, RDF.type) if isinstance(o, URIRef)}
        if any(t in ignored for t in types):
            continue
        if any(str(t).startswith(etd_ns) for t in types):
            nodes.add(s)
    return nodes


def _normalized_label(label: str) -> str:
    return " ".join(label.lower().split())


def _duplicate_entity_groups_by_label(graph: Graph, entity_nodes: Set[URIRef]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for entity in sorted(entity_nodes, key=str):
        labels = [str(v) for v in graph.objects(entity, RDFS.label)]
        if not labels:
            continue
        key = _normalized_label(labels[0])
        groups.setdefault(key, []).append(str(entity))
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def _evidence_chunk_unknown_stats(graph: Graph, etd_ns: str) -> Dict[str, float]:
    etd_evidence = URIRef(f"{etd_ns}Evidence")
    etd_chunk_index = URIRef(f"{etd_ns}chunk_index")
    total = 0
    unknown = 0
    for ev in graph.subjects(RDF.type, etd_evidence):
        if not isinstance(ev, URIRef):
            continue
        total += 1
        vals = [str(v).strip() for v in graph.objects(ev, etd_chunk_index)]
        if any(v in {"-1", "-1.0"} for v in vals):
            unknown += 1
    pct = (float(unknown) / float(total) * 100.0) if total > 0 else 0.0
    return {
        "evidence_total_count": total,
        "evidence_chunk_index_unknown_count": unknown,
        "evidence_chunk_index_unknown_pct": round(pct, 2),
    }


def _isolated_entities(graph: Graph, entity_nodes: Set[URIRef], etd_ns: str) -> List[str]:
    etd_subject = URIRef(f"{etd_ns}subject")
    etd_object = URIRef(f"{etd_ns}object")
    etd_has_source_entity = URIRef(f"{etd_ns}hasSourceEntity")

    connected: Set[URIRef] = set()
    for node in graph.objects(None, etd_subject):
        if isinstance(node, URIRef):
            connected.add(node)
    for node in graph.objects(None, etd_object):
        if isinstance(node, URIRef):
            connected.add(node)
    for node in graph.objects(None, etd_has_source_entity):
        if isinstance(node, URIRef):
            connected.add(node)

    return [str(n) for n in sorted(entity_nodes, key=str) if n not in connected]


def _iter_reified_spo(graph: Graph, etd_ns: str) -> Iterable[Tuple[URIRef, URIRef, URIRef, URIRef]]:
    etd_relation = URIRef(f"{etd_ns}Relation")
    etd_subject = URIRef(f"{etd_ns}subject")
    etd_predicate = URIRef(f"{etd_ns}predicate")
    etd_object = URIRef(f"{etd_ns}object")

    for rel in graph.subjects(RDF.type, etd_relation):
        if not isinstance(rel, URIRef):
            continue
        subjects = [x for x in graph.objects(rel, etd_subject) if isinstance(x, URIRef)]
        predicates = [x for x in graph.objects(rel, etd_predicate) if isinstance(x, URIRef)]
        objects = [x for x in graph.objects(rel, etd_object) if isinstance(x, URIRef)]
        for s in subjects:
            for p in predicates:
                for o in objects:
                    yield rel, s, p, o


def _duplicate_reified_relation_spo(
    graph: Graph,
    etd_ns: str,
    output_dir: Path,
) -> Dict[str, object]:
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    for rel, s, p, o in _iter_reified_spo(graph, etd_ns):
        key = (str(s), str(p), str(o))
        groups.setdefault(key, []).append(str(rel))

    duplicate_map: Dict[str, List[str]] = {}
    total_rel_nodes = 0
    for key, rels in groups.items():
        unique_sorted = sorted(set(rels))
        if len(unique_sorted) > 1:
            key_str = f"{key[0]}|{key[1]}|{key[2]}"
            duplicate_map[key_str] = unique_sorted
            total_rel_nodes += len(unique_sorted)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "duplicate_relation_spo.json"
    out_path.write_text(json.dumps(duplicate_map, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "duplicate_relation_spo_groups_count": len(duplicate_map),
        "duplicate_relation_spo_total_relations_involved": total_rel_nodes,
        "duplicate_relation_spo_report_path": str(out_path),
    }


def _vote_typing_issues(graph: Graph, etd_ns: str) -> Dict[str, int]:
    etd_vote = URIRef(f"{etd_ns}Vote")
    etd_expert = URIRef(f"{etd_ns}Expert")
    etd_person = URIRef(f"{etd_ns}Person")
    etd_answer = URIRef(f"{etd_ns}Answer")
    etd_cast_by = URIRef(f"{etd_ns}castBy")
    etd_receives_vote = URIRef(f"{etd_ns}receivesVote")

    # index reified vote edges:
    # - castBy: Vote -> Expert
    # - receivesVote: Answer -> Vote
    reified_cast_by: Dict[URIRef, Set[URIRef]] = {}
    reified_receives_by_vote: Dict[URIRef, Set[URIRef]] = {}
    for _, s, p, o in _iter_reified_spo(graph, etd_ns):
        if p == etd_cast_by:
            reified_cast_by.setdefault(s, set()).add(o)
        elif p == etd_receives_vote:
            reified_receives_by_vote.setdefault(o, set()).add(s)

    issues = Counter(
        {
            "missing_castBy": 0,
            "multiple_castBy": 0,
            "castBy_wrong_type": 0,
            "missing_receivesVote": 0,
            "multiple_receivesVote": 0,
            "receivesVote_wrong_type": 0,
        }
    )

    for vote in graph.subjects(RDF.type, etd_vote):
        if not isinstance(vote, URIRef):
            continue

        direct_cast = {o for o in graph.objects(vote, etd_cast_by) if isinstance(o, URIRef)}
        direct_recv = {s for s in graph.subjects(etd_receives_vote, vote) if isinstance(s, URIRef)}

        cast_candidates = direct_cast.union(reified_cast_by.get(vote, set()))
        recv_candidates = direct_recv.union(reified_receives_by_vote.get(vote, set()))

        if len(cast_candidates) == 0:
            issues["missing_castBy"] += 1
        if len(cast_candidates) > 1:
            issues["multiple_castBy"] += 1
        for obj in sorted(cast_candidates, key=str):
            if (obj, RDF.type, etd_expert) not in graph and (obj, RDF.type, etd_person) not in graph:
                issues["castBy_wrong_type"] += 1

        if len(recv_candidates) == 0:
            issues["missing_receivesVote"] += 1
        if len(recv_candidates) > 1:
            issues["multiple_receivesVote"] += 1
        for obj in sorted(recv_candidates, key=str):
            if (obj, RDF.type, etd_answer) not in graph:
                issues["receivesVote_wrong_type"] += 1

    return dict(issues)


def _same_chunk_duplicate_labels_count(graph: Graph, etd_ns: str) -> int:
    etd_relation = URIRef(f"{etd_ns}Relation")
    etd_entity = URIRef(f"{etd_ns}Entity")
    etd_subject = URIRef(f"{etd_ns}subject")
    etd_object = URIRef(f"{etd_ns}object")
    etd_has_evidence = URIRef(f"{etd_ns}hasEvidence")
    etd_source = URIRef(f"{etd_ns}source")
    etd_discourse_id = URIRef(f"{etd_ns}discourse_id")
    etd_chunk_index = URIRef(f"{etd_ns}chunk_index")

    groups: Dict[Tuple[str, str, str], Dict[str, Set[str]]] = {}
    for rel in graph.subjects(RDF.type, etd_relation):
        if not isinstance(rel, URIRef):
            continue
        subj = next((o for o in graph.objects(rel, etd_subject) if isinstance(o, URIRef)), None)
        obj = next((o for o in graph.objects(rel, etd_object) if isinstance(o, URIRef)), None)
        if subj is None and obj is None:
            continue
        labels: List[Tuple[str, str]] = []
        for node in (subj, obj):
            if node is None:
                continue
            if (node, RDF.type, etd_entity) not in graph:
                continue
            lbl = next((str(v).strip() for v in graph.objects(node, RDFS.label) if str(v).strip()), None)
            if lbl:
                labels.append((_normalized_label(lbl), str(node)))

        for ev in graph.objects(rel, etd_has_evidence):
            if not isinstance(ev, URIRef):
                continue
            src = next((str(v).strip() for v in graph.objects(ev, etd_source) if str(v).strip()), "unknown")
            did = next((str(v).strip() for v in graph.objects(ev, etd_discourse_id) if str(v).strip()), "unknown")
            chunk = next((str(v).strip() for v in graph.objects(ev, etd_chunk_index) if str(v).strip()), "unknown")
            key = (src, did, chunk)
            bucket = groups.setdefault(key, {})
            for norm_lbl, uri in labels:
                bucket.setdefault(norm_lbl, set()).add(uri)

    count = 0
    for _, label_to_uris in groups.items():
        for uris in label_to_uris.values():
            if len(uris) > 1:
                count += 1
    return count


def _entity_chunk_contexts(graph: Graph, etd_ns: str) -> Dict[str, Set[str]]:
    etd_relation = URIRef(f"{etd_ns}Relation")
    etd_subject = URIRef(f"{etd_ns}subject")
    etd_object = URIRef(f"{etd_ns}object")
    etd_has_evidence = URIRef(f"{etd_ns}hasEvidence")
    etd_discourse_id = URIRef(f"{etd_ns}discourse_id")
    etd_chunk_index = URIRef(f"{etd_ns}chunk_index")

    contexts: Dict[str, Set[str]] = {}
    for rel in graph.subjects(RDF.type, etd_relation):
        if not isinstance(rel, URIRef):
            continue
        subj = next((o for o in graph.objects(rel, etd_subject) if isinstance(o, URIRef)), None)
        obj = next((o for o in graph.objects(rel, etd_object) if isinstance(o, URIRef)), None)
        if subj is None and obj is None:
            continue

        relation_contexts: Set[str] = set()
        for ev in graph.objects(rel, etd_has_evidence):
            if not isinstance(ev, URIRef):
                continue
            did = next((str(v).strip() for v in graph.objects(ev, etd_discourse_id) if str(v).strip()), "")
            chunk_num: Optional[int] = None
            for v in graph.objects(ev, etd_chunk_index):
                try:
                    n = int(str(v))
                except Exception:
                    continue
                if n >= 0:
                    chunk_num = n
                    break
            if did and chunk_num is not None:
                relation_contexts.add(f"{did}|{chunk_num}")

        for node in (subj, obj):
            if node is None:
                continue
            contexts.setdefault(str(node), set()).update(relation_contexts)
    return contexts


def _cross_chunk_duplicate_labels(graph: Graph, etd_ns: str, output_dir: Path) -> Dict[str, object]:
    etd_entity = URIRef(f"{etd_ns}Entity")
    chunk_contexts = _entity_chunk_contexts(graph, etd_ns)
    duplicate_map: Dict[str, Dict[str, List[str]]] = {}
    labels_to_uris: Dict[str, Set[str]] = {}
    labels_uri_chunks: Dict[str, Dict[str, Set[str]]] = {}

    for ent in graph.subjects(RDF.type, etd_entity):
        if not isinstance(ent, URIRef):
            continue
        labels = [str(v).strip() for v in graph.objects(ent, RDFS.label) if str(v).strip()]
        if not labels:
            continue
        norm = _normalized_label(labels[0])
        uri = str(ent)
        labels_to_uris.setdefault(norm, set()).add(uri)
        labels_uri_chunks.setdefault(norm, {}).setdefault(uri, set()).update(chunk_contexts.get(uri, set()))

    cross_chunk_count = 0
    for lbl in sorted(labels_uri_chunks.keys()):
        uri_map = labels_uri_chunks[lbl]
        if len(uri_map) <= 1:
            continue
        all_chunks: Set[str] = set()
        for ch in uri_map.values():
            all_chunks.update(ch)
        if len(all_chunks) <= 1:
            continue
        cross_chunk_count += 1
        duplicate_map[lbl] = {
            uri: sorted(list(uri_map[uri])) for uri in sorted(uri_map.keys())
        }

    # Merge metric: same external ID across chunks.
    ext_to_uris: Dict[str, Set[str]] = {}
    ext_to_chunks: Dict[str, Set[str]] = {}
    for ent in graph.subjects(RDF.type, etd_entity):
        if not isinstance(ent, URIRef):
            continue
        ent_uri = str(ent)
        for ext in graph.objects(ent, OWL.sameAs):
            if not isinstance(ext, URIRef):
                continue
            ext_s = str(ext)
            if ext_s.startswith(etd_ns):
                continue
            ext_to_uris.setdefault(ext_s, set()).add(ent_uri)
            ext_to_chunks.setdefault(ext_s, set()).update(chunk_contexts.get(ent_uri, set()))

    cross_chunk_merges_details: Dict[str, Dict[str, List[str]]] = {}
    for ext in sorted(ext_to_uris.keys()):
        uris = ext_to_uris[ext]
        chunks = ext_to_chunks.get(ext, set())
        if len(uris) > 1 and len(chunks) > 1:
            cross_chunk_merges_details[ext] = {
                "entity_uris": sorted(list(uris)),
                "chunks": sorted(list(chunks)),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cross_chunk_linking_report.json"
    payload = {
        "cross_chunk_collisions_count": cross_chunk_count,
        "cross_chunk_collisions_details": duplicate_map,
        "cross_chunk_merges_count": len(cross_chunk_merges_details),
        "cross_chunk_merges_details": cross_chunk_merges_details,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "cross_chunk_collisions_count": cross_chunk_count,
        "cross_chunk_collisions_details": duplicate_map,
        "cross_chunk_merges_count": len(cross_chunk_merges_details),
        "cross_chunk_merges_details": cross_chunk_merges_details,
        "cross_chunk_linking_report_path": str(out_path),
        # Backward-compatible aliases:
        "cross_chunk_duplicate_labels_count": cross_chunk_count,
        "cross_chunk_duplicate_labels_report_path": str(out_path),
    }


def run_audit(ttl_path: Path, shapes_path: Path, output_dir: Path, etd_ns: str) -> Dict[str, object]:
    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")

    entity_nodes = _entity_nodes(graph, etd_ns)
    duplicate_entities = _duplicate_entity_groups_by_label(graph, entity_nodes)
    isolated = _isolated_entities(graph, entity_nodes, etd_ns)
    vote_issues = _vote_typing_issues(graph, etd_ns)
    chunk_stats = _evidence_chunk_unknown_stats(graph, etd_ns)
    same_chunk_dups = _same_chunk_duplicate_labels_count(graph, etd_ns)
    cross_chunk_dups = _cross_chunk_duplicate_labels(graph, etd_ns, output_dir)

    dup_spo = _duplicate_reified_relation_spo(graph, etd_ns, output_dir)

    validator = SHACLValidator(shapes_path=shapes_path, output_dir=output_dir)
    shacl = validator.validate_graph(graph)
    counts = shacl.get("counts") or {}

    return {
        "shacl_conforms": bool(shacl.get("conforms", False)),
        "shacl_violations": int(counts.get("violations", 0)),
        "shacl_warnings": int(counts.get("warnings", 0)),
        "shacl_infos": int(counts.get("infos", 0)),
        "duplicate_entities_by_label_count": len(duplicate_entities),
        "isolated_entities_count": len(isolated),
        "vote_typing_issues": vote_issues,
        "same_chunk_duplicate_labels_count": same_chunk_dups,
        **chunk_stats,
        **cross_chunk_dups,
        **dup_spo,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast deterministic KG audit harness")
    parser.add_argument("--ttl", default="output/etd_hub_enhanced_kg.ttl", help="Path to generated KG TTL")
    parser.add_argument("--shapes", default="ontology/shacl_shapes.ttl", help="Path to SHACL shapes TTL")
    parser.add_argument("--output-dir", default="output", help="Directory for SHACL + audit artifacts")
    parser.add_argument("--etd-namespace", default=None, help="ETD namespace base URI")

    _bool_flag(parser, "fail-on-shacl", True, "Fail when SHACL does not conform")
    _bool_flag(parser, "fail-on-duplicate-spo", True, "Fail when duplicate reified relation SPO groups exist")
    _bool_flag(parser, "fail-on-isolated-entities", False, "Fail on isolated entities")
    _bool_flag(parser, "fail-on-vote-issues", True, "Fail on vote typing issues")
    _bool_flag(parser, "fail-on-unknown-chunk-index", False, "Fail if any evidence has chunk_index=-1")
    _bool_flag(parser, "fail-on-dup-labels", False, "Fail on same/cross chunk duplicate labels")

    args = parser.parse_args()

    ttl_path = Path(args.ttl)
    shapes_path = Path(args.shapes)
    output_dir = Path(args.output_dir)

    if not ttl_path.exists():
        print(f"[ERROR] TTL not found: {ttl_path}")
        return 2
    if not shapes_path.exists():
        print(f"[ERROR] SHACL shapes not found: {shapes_path}")
        return 2

    try:
        etd_ns = _resolve_etd_namespace(args.etd_namespace)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    report = run_audit(ttl_path=ttl_path, shapes_path=shapes_path, output_dir=output_dir, etd_ns=etd_ns)

    print("=== Audit Harness ===")
    print(f"TTL: {ttl_path}")
    print(f"SHACL shapes: {shapes_path}")
    print(f"ETD namespace: {etd_ns}")
    print(f"SHACL conforms: {report['shacl_conforms']}")
    print(f"SHACL violations: {report['shacl_violations']}")
    print(f"duplicate_relation_spo_groups_count: {report['duplicate_relation_spo_groups_count']}")
    print(f"duplicate_relation_spo_total_relations_involved: {report['duplicate_relation_spo_total_relations_involved']}")
    print(f"isolated_entities_count: {report['isolated_entities_count']}")
    print(f"vote_typing_issues: {report['vote_typing_issues']}")
    print(f"evidence_chunk_index_unknown_count: {report['evidence_chunk_index_unknown_count']}")
    print(f"evidence_chunk_index_unknown_pct: {report['evidence_chunk_index_unknown_pct']}")
    print(f"same_chunk_duplicate_labels_count: {report['same_chunk_duplicate_labels_count']}")
    print(f"cross_chunk_duplicate_labels_count: {report['cross_chunk_duplicate_labels_count']}")
    print(f"cross_chunk_collisions_count: {report['cross_chunk_collisions_count']}")
    print(f"cross_chunk_merges_count: {report['cross_chunk_merges_count']}")
    print(f"cross_chunk_linking_report_path: {report['cross_chunk_linking_report_path']}")
    print(f"duplicate_entities_by_label_count (info): {report['duplicate_entities_by_label_count']}")

    fail_reasons: List[str] = []
    if args.fail_on_shacl and (not report["shacl_conforms"] or report["shacl_violations"] > 0):
        fail_reasons.append("shacl")
    if args.fail_on_duplicate_spo and report["duplicate_relation_spo_groups_count"] > 0:
        fail_reasons.append("duplicate_spo")
    if args.fail_on_isolated_entities and report["isolated_entities_count"] > 0:
        fail_reasons.append("isolated_entities")
    if args.fail_on_vote_issues and sum(int(v) for v in report["vote_typing_issues"].values()) > 0:
        fail_reasons.append("vote_issues")
    if args.fail_on_unknown_chunk_index and int(report["evidence_chunk_index_unknown_count"]) > 0:
        fail_reasons.append("unknown_chunk_index")
    if args.fail_on_dup_labels and (
        int(report["same_chunk_duplicate_labels_count"]) > 0 or int(report["cross_chunk_duplicate_labels_count"]) > 0
    ):
        fail_reasons.append("duplicate_labels")

    if fail_reasons:
        print(f"[FAIL] Audit checks failed: {', '.join(fail_reasons)}")
        return 1

    print("[PASS] Audit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
