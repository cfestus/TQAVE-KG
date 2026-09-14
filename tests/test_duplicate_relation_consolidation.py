import importlib.util
import unittest

from config import Config
from models import Entity

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self.entities = {e.uri: e for e in entities}

    def get_all_entities(self):
        return list(self.entities.values())


def _relation_dict(
    uri, source, relation, target, *, target_is_literal=True, target_literal=None,
    target_datatype=None, evidence_quote="", field="theme.description", discourse_id="d1",
    confidence=0.9, is_inferred=False, extra_meta=None,
):
    meta = {
        "evidence_quote": evidence_quote,
        "evidence": evidence_quote,
        "field": field,
        "source_field": field,
        "discourse_id": discourse_id,
        "source_doc_id": discourse_id,
        "origin_key": f"{field}|{discourse_id}",
        "chunk_index": 0,
        "source": "llm_relation_extractor",
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "uri": uri, "source": source, "relation": relation, "target": target,
        "confidence": confidence, "is_inferred": is_inferred,
        "target_literal": target_literal if target_is_literal else None,
        "target_datatype": target_datatype, "target_is_literal": target_is_literal,
        "object_type": "string" if target_is_literal else "entity",
        "qualifiers": {}, "predicate_text": relation, "predicate_mapping_status": "mapped",
        "negated": False, "temporal_information": {}, "quantitative_information": {},
        "metadata": meta,
    }


def _make_graph(entities, relations_by_record):
    from serializers.rdf_serializer import RDFSerializer

    reg = _Registry(entities)
    themes = {}
    for rec_id, rels in relations_by_record.items():
        themes[rec_id] = {"entities": [], "relations": rels}
    ann = {
        "themes": themes, "questions": {}, "answers": {},
        "votes": {}, "experts": {}, "documents": {},
    }
    s = RDFSerializer(
        annotations_dict=ann, registry=reg, namespaces=Config.NAMESPACES,
        prune_isolated_entities=False,
    )
    return s.graph, s


def _subject(local="subject_x", label="Some Subject"):
    return Entity(
        uri=f"https://alfie-project.eu/etd-hub#{local}", label=label,
        entity_type="Concept", confidence=0.9, metadata={},
    )


def _object_entity(local="object_y", label="Some Object"):
    return Entity(
        uri=f"https://alfie-project.eu/etd-hub#{local}", label=label,
        entity_type="Concept", confidence=0.9, metadata={},
    )


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class SameDiscourseDifferentEvidenceTests(unittest.TestCase):
    def test_same_discourse_same_spo_different_evidence_yields_one_relation_two_evidence(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/occ_a", subj.uri, "hasAttribute", "a shared literal value",
            evidence_quote="first supporting span",
        )
        rel_b = _relation_dict(
            "https://kg/r/occ_b", subj.uri, "hasAttribute", "a shared literal value",
            evidence_quote="second supporting span",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        self.assertEqual(len(evidence_nodes), 2)
        texts = {str(t) for ev in evidence_nodes for t in g.objects(ev, etd.evidenceText)}
        self.assertEqual(texts, {"first supporting span", "second supporting span"})


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class SameDiscourseIdenticalEvidenceTests(unittest.TestCase):
    def test_same_discourse_same_spo_identical_evidence_yields_one_relation_one_evidence(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/occ_c", subj.uri, "hasAttribute", "identical literal value",
            evidence_quote="identical supporting span",
        )
        rel_b = _relation_dict(
            "https://kg/r/occ_d", subj.uri, "hasAttribute", "identical literal value",
            evidence_quote="identical supporting span",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        self.assertEqual(len(evidence_nodes), 1)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class DiscourseScopeProtectionTests(unittest.TestCase):
    def test_same_spo_different_discourse_yields_two_relations(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/disc_a", subj.uri, "hasAttribute", "same literal value",
            evidence_quote="span in discourse one", discourse_id="d1",
        )
        rel_b = _relation_dict(
            "https://kg/r/disc_b", subj.uri, "hasAttribute", "same literal value",
            evidence_quote="span in discourse two", discourse_id="d2",
        )
        g, s = _make_graph([subj], {"t1": [rel_a], "t2": [rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 2)

    def test_same_spo_different_field_yields_two_relations(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/field_a", subj.uri, "hasAttribute", "same literal value",
            evidence_quote="span in field one", field="theme.description", discourse_id="d1",
        )
        rel_b = _relation_dict(
            "https://kg/r/field_b", subj.uri, "hasAttribute", "same literal value",
            evidence_quote="span in field two", field="answer.text", discourse_id="d1",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class PredicateAndDatatypeDistinctionTests(unittest.TestCase):
    def test_same_entities_different_predicate_stays_separate(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/pred_a", subj.uri, "hasAttribute", "shared value",
            evidence_quote="span one",
        )
        rel_b = _relation_dict(
            "https://kg/r/pred_b", subj.uri, "hasDescription", "shared value",
            evidence_quote="span two",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 2)

    def test_same_lexical_value_different_datatype_stays_separate(self):
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/dt_a", subj.uri, "hasAttribute", "10",
            target_literal="10", target_datatype="xsd:integer",
            evidence_quote="an integer value of 10",
        )
        rel_b = _relation_dict(
            "https://kg/r/dt_b", subj.uri, "hasAttribute", "10",
            target_literal="10%", target_datatype="xsd:string",
            evidence_quote="a percentage value of 10%",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class PostRefinementConsolidationTests(unittest.TestCase):
    def test_two_raw_relations_identical_only_after_refinement_still_consolidate(self):
        # Simulates two raw extraction occurrences that arrived at the SAME
        # final predicate only after upstream predicate refinement already
        # ran (refinement happens earlier in the pipeline, before relations
        # reach RDF serialization) -- consolidation must key off the FINAL
        # resolved predicate, which is all this stage ever sees.
        from rdflib import RDF, Namespace
        subj = _subject()
        rel_a = _relation_dict(
            "https://kg/r/refined_a", subj.uri, "hasDescription", "a refined descriptive value",
            evidence_quote="span from an originally generic 'is' predicate",
            extra_meta={"predicate_text": "is"},
        )
        rel_b = _relation_dict(
            "https://kg/r/refined_b", subj.uri, "hasDescription", "a refined descriptive value",
            evidence_quote="span from an originally generic 'has' predicate",
            extra_meta={"predicate_text": "has"},
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        self.assertEqual(len(evidence_nodes), 2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class CanonicalEntityRedirectTests(unittest.TestCase):
    def test_canonical_entity_redirect_does_not_leave_duplicate_relations(self):
        from rdflib import RDF, Namespace
        subj = Entity(
            uri="https://alfie-project.eu/etd-hub#canonical_subject", label="Canonical Subject",
            entity_type="Concept", confidence=0.9,
            metadata={"wikidata_uri": "http://www.wikidata.org/entity/Q999999"},
        )
        rel_internal = _relation_dict(
            "https://kg/r/redir_internal", subj.uri, "hasAttribute", "redirect-tested value",
            evidence_quote="span referencing the internal URI directly",
        )
        rel_external = _relation_dict(
            "https://kg/r/redir_external", "http://www.wikidata.org/entity/Q999999",
            "hasAttribute", "redirect-tested value",
            evidence_quote="span referencing the external wikidata URI",
        )
        g, s = _make_graph([subj], {"t1": [rel_internal, rel_external]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        self.assertEqual(len(evidence_nodes), 2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class RelationUriDeterminismTests(unittest.TestCase):
    def test_relation_uri_stable_across_extraction_order(self):
        from rdflib import RDF, Namespace
        subj = _subject(local="order_subject")
        rel_a = _relation_dict(
            "https://kg/r/order_a", subj.uri, "hasAttribute", "order-independent value",
            evidence_quote="first occurrence span",
        )
        rel_b = _relation_dict(
            "https://kg/r/order_b", subj.uri, "hasAttribute", "order-independent value",
            evidence_quote="second occurrence span",
        )
        g1, _ = _make_graph([subj], {"t1": [rel_a, rel_b]})
        g2, _ = _make_graph([subj], {"t1": [rel_b, rel_a]})
        etd = Namespace(Config.NAMESPACES["etd"])
        uris1 = sorted(str(u) for u in g1.subjects(RDF.type, etd.Relation))
        uris2 = sorted(str(u) for u in g2.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(uris1), 1)
        self.assertEqual(uris1, uris2)

    def test_relation_uri_stable_regardless_of_duplicate_count(self):
        # Within an actual duplicate-merge group, the winning URI must not
        # depend on exactly how many raw occurrences happened to be
        # extracted this run (2 vs 3) -- both scenarios ARE duplicate groups
        # for the same final assertion, so both must land on the same
        # canonical URI. (A true singleton -- never duplicated at all --
        # intentionally keeps its extraction-time URI; that is a separate,
        # pre-existing invariant this task does not change, so it is not
        # exercised by this test.)
        from rdflib import RDF, Namespace
        subj = _subject(local="count_subject")
        two_occurrences = [
            _relation_dict(
                "https://kg/r/count_a", subj.uri, "hasAttribute", "count-independent value",
                evidence_quote="occurrence span a",
            ),
            _relation_dict(
                "https://kg/r/count_b", subj.uri, "hasAttribute", "count-independent value",
                evidence_quote="occurrence span b",
            ),
        ]
        three_occurrences = two_occurrences + [
            _relation_dict(
                "https://kg/r/count_c", subj.uri, "hasAttribute", "count-independent value",
                evidence_quote="occurrence span c",
            ),
        ]
        g1, _ = _make_graph([subj], {"t1": two_occurrences})
        g2, _ = _make_graph([subj], {"t1": three_occurrences})
        etd = Namespace(Config.NAMESPACES["etd"])
        uris1 = sorted(str(u) for u in g1.subjects(RDF.type, etd.Relation))
        uris2 = sorted(str(u) for u in g2.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(uris1), 1)
        self.assertEqual(uris1, uris2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class EvidenceIdentityAndProvenanceIntegrityTests(unittest.TestCase):
    def test_evidence_nodes_retain_exactly_one_evidence_text_each(self):
        from rdflib import RDF, Namespace
        subj = _subject(local="ev_identity_subject")
        rel_a = _relation_dict(
            "https://kg/r/evid_a", subj.uri, "hasAttribute", "shared literal target",
            evidence_quote="distinct span alpha",
        )
        rel_b = _relation_dict(
            "https://kg/r/evid_b", subj.uri, "hasAttribute", "shared literal target",
            evidence_quote="distinct span beta",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_nodes = list(g.subjects(RDF.type, etd.Evidence))
        self.assertGreaterEqual(len(evidence_nodes), 2)
        for ev in evidence_nodes:
            texts = list(g.objects(ev, etd.evidenceText))
            self.assertEqual(len(texts), 1)

    def test_provenance_context_links_remain_intact_after_consolidation(self):
        from rdflib import RDF, Namespace
        subj = _subject(local="prov_subject")
        shared_context = "Some Subject has a shared literal target described in this passage."
        rel_a = _relation_dict(
            "https://kg/r/prov_a", subj.uri, "hasAttribute", "shared literal target",
            evidence_quote="a shared literal target",
            extra_meta={"source_context": shared_context},
        )
        rel_b = _relation_dict(
            "https://kg/r/prov_b", subj.uri, "hasAttribute", "shared literal target",
            evidence_quote="literal target",
            extra_meta={"source_context": shared_context},
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        for ev in evidence_nodes:
            discourse_units = list(g.objects(ev, etd.extractedFrom))
            self.assertEqual(len(discourse_units), 1)
            context_texts = list(g.objects(discourse_units[0], etd.contextText))
            self.assertIn(shared_context, [str(c) for c in context_texts])


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for duplicate-relation-consolidation tests")
class ShaclConformanceTests(unittest.TestCase):
    def test_shacl_still_conforms_after_consolidation(self):
        if importlib.util.find_spec("pyshacl") is None:
            self.skipTest("pyshacl not installed")
        from validation.shacl_validator import SHACLValidator
        import tempfile
        from pathlib import Path
        from rdflib import Namespace

        subj = _subject(local="shacl_subject")
        rel_a = _relation_dict(
            "https://kg/r/shacl_a", subj.uri, "hasAttribute", "some descriptive value",
            evidence_quote="some descriptive value",
        )
        rel_b = _relation_dict(
            "https://kg/r/shacl_b", subj.uri, "hasAttribute", "some descriptive value",
            evidence_quote="some other descriptive value",
        )
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        for pfx, uri in Config.NAMESPACES.items():
            g.bind(pfx, Namespace(uri))
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            validator = SHACLValidator(output_dir=out_dir)
            result = validator.validate_graph(g)
            self.assertTrue(result.get("conforms", False), result.get("results_text", "SHACL failed"))


if __name__ == "__main__":
    unittest.main()
