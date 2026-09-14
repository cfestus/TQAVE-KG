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
    evidence_quote="", source_context="", field="theme.description", discourse_id="d1",
    origin_key=None, chunk_index=0, extra_meta=None,
):
    meta = {
        "evidence_quote": evidence_quote,
        "evidence": evidence_quote,
        "field": field,
        "source_field": field,
        "discourse_id": discourse_id,
        "source_doc_id": discourse_id,
        "origin_key": origin_key or f"{field}|{discourse_id}",
        "chunk_index": chunk_index,
        "source": "llm_relation_extractor",
    }
    if source_context:
        meta["source_context"] = source_context
    if extra_meta:
        meta.update(extra_meta)
    return {
        "uri": uri, "source": source, "relation": relation, "target": target,
        "confidence": 0.9, "is_inferred": False,
        "target_literal": target_literal if target_is_literal else None,
        "target_datatype": None, "target_is_literal": target_is_literal,
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


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for provenance/context tests")
class MinimalEvidenceVsSourceContextTests(unittest.TestCase):
    def _subject(self):
        return Entity(
            uri="https://alfie-project.eu/etd-hub#subject_x", label="Some Subject",
            entity_type="Concept", confidence=0.9, metadata={},
        )

    def test_minimal_evidence_shorter_than_source_context(self):
        from rdflib import Namespace
        subj = self._subject()
        full_sentence = "Some Subject was described in detail: a short descriptive value, among other things."
        rel = _relation_dict(
            "https://kg/r/r1", subj.uri, "hasDescription", "a short descriptive value",
            evidence_quote="a short descriptive value", source_context=full_sentence,
        )
        g, s = _make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_texts = [str(o) for o in g.objects(None, etd.evidenceText)]
        context_texts = [str(o) for o in g.objects(None, etd.contextText)]
        self.assertIn("a short descriptive value", evidence_texts)
        self.assertIn(full_sentence, context_texts)
        self.assertTrue(len(evidence_texts[0]) < len(context_texts[0]) or "a short descriptive value" != full_sentence)

    def test_evidence_equals_full_sentence_where_appropriate(self):
        from rdflib import Namespace
        subj = self._subject()
        sentence = "Some Subject is a fairly short standalone assertion."
        rel = _relation_dict(
            "https://kg/r/r2", subj.uri, "hasAttribute", sentence,
            evidence_quote=sentence, source_context=sentence,
        )
        g, s = _make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_texts = {str(o) for o in g.objects(None, etd.evidenceText)}
        context_texts = {str(o) for o in g.objects(None, etd.contextText)}
        self.assertIn(sentence, evidence_texts)
        self.assertIn(sentence, context_texts)

    def test_subject_omitted_from_minimal_evidence_but_present_in_context(self):
        from rdflib import Namespace
        subj = self._subject()
        full_context = "Some Subject, when evaluated under the described conditions, shows a specific measurable outcome."
        rel = _relation_dict(
            "https://kg/r/r3", subj.uri, "hasAttribute", "a specific measurable outcome",
            evidence_quote="shows a specific measurable outcome", source_context=full_context,
        )
        g, s = _make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        context_texts = [str(o) for o in g.objects(None, etd.contextText)]
        self.assertTrue(any("Some Subject" in c for c in context_texts))
        evidence_texts = [str(o) for o in g.objects(None, etd.evidenceText)]
        self.assertFalse(any("Some Subject" in e for e in evidence_texts))


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for provenance/context tests")
class SharedContextDeduplicationTests(unittest.TestCase):
    def test_multiple_relations_share_one_discourseunit_context(self):
        from rdflib import RDF, Namespace
        subj = Entity(uri="https://alfie-project.eu/etd-hub#s1", label="Some Subject",
                      entity_type="Concept", confidence=0.9, metadata={})
        shared_context = "Some Subject supports several distinct facts described in this same passage."
        rel_a = _relation_dict("https://kg/r/a", subj.uri, "hasAttribute", "fact one",
                                evidence_quote="fact one", source_context=shared_context, field="theme.description", discourse_id="d1")
        rel_b = _relation_dict("https://kg/r/b", subj.uri, "hasAttribute", "fact two",
                                evidence_quote="fact two", source_context=shared_context, field="theme.description", discourse_id="d1")
        g, s = _make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        discourse_nodes = list(g.subjects(RDF.type, etd.DiscourseUnit))
        self.assertEqual(len(discourse_nodes), 1)
        context_texts = list(g.objects(discourse_nodes[0], etd.contextText))
        self.assertEqual(len(context_texts), 1)
        self.assertEqual(str(context_texts[0]), shared_context)

    def test_context_serialization_does_not_duplicate_full_text_per_relation(self):
        from rdflib import Namespace
        subj = Entity(uri="https://alfie-project.eu/etd-hub#s2", label="Some Subject",
                      entity_type="Concept", confidence=0.9, metadata={})
        shared_context = "A single shared passage referenced by three separate relations here."
        rels = [
            _relation_dict(f"https://kg/r/{i}", subj.uri, "hasAttribute", f"value {i}",
                            evidence_quote=f"value {i}", source_context=shared_context,
                            field="theme.description", discourse_id="d1")
            for i in range(3)
        ]
        g, s = _make_graph([subj], {"t1": rels})
        etd = Namespace(Config.NAMESPACES["etd"])
        all_context_triples = list(g.triples((None, etd.contextText, None)))
        self.assertEqual(len(all_context_triples), 1)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for provenance/context tests")
class MultipleEvidenceSpansTests(unittest.TestCase):
    def test_multiple_evidence_spans_remain_distinct(self):
        from rdflib import Namespace
        subj = Entity(uri="https://alfie-project.eu/etd-hub#s3", label="Some Subject",
                      entity_type="Concept", confidence=0.9, metadata={})
        rel = _relation_dict(
            "https://kg/r/multi", subj.uri, "hasAttribute", "shared target value",
            evidence_quote="first supporting span", source_context="Full passage context text.",
            extra_meta={
                "evidence_spans": [
                    {"chunk_index": 0, "char_start": 0, "char_end": 10, "evidence": "first supporting span"},
                    {"chunk_index": 0, "char_start": 20, "char_end": 40, "evidence": "second supporting span"},
                ]
            },
        )
        g, s = _make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_texts = {str(o) for o in g.objects(None, etd.evidenceText)}
        self.assertIn("first supporting span", evidence_texts)
        self.assertIn("second supporting span", evidence_texts)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for provenance/context tests")
class SubstringAndOffsetIntegrityTests(unittest.TestCase):
    def test_evidence_text_is_exact_context_substring_when_grounded(self):
        from rdflib import Namespace
        subj = Entity(uri="https://alfie-project.eu/etd-hub#s4", label="Some Subject",
                      entity_type="Concept", confidence=0.9, metadata={})
        context = "Some Subject clearly exhibits a specific verifiable property in this sentence."
        span = "exhibits a specific verifiable property"
        self.assertIn(span, context)
        rel = _relation_dict(
            "https://kg/r/exact", subj.uri, "hasAttribute", span,
            evidence_quote=span, source_context=context,
        )
        g, s = _make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_texts = [str(o) for o in g.objects(None, etd.evidenceText)]
        context_texts = [str(o) for o in g.objects(None, etd.contextText)]
        self.assertTrue(any(e in c for e in evidence_texts for c in context_texts))


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for provenance/context tests")
class ShaclConformanceTests(unittest.TestCase):
    def test_shacl_still_conforms_with_context_text_present(self):
        importlib.util.find_spec("pyshacl")
        if importlib.util.find_spec("pyshacl") is None:
            self.skipTest("pyshacl not installed")
        from validation.shacl_validator import SHACLValidator
        import tempfile
        from pathlib import Path
        from rdflib import Namespace

        subj = Entity(uri="https://alfie-project.eu/etd-hub#s5", label="Some Subject",
                      entity_type="Concept", confidence=0.9, metadata={})
        rel = _relation_dict(
            "https://kg/r/shacl", subj.uri, "hasAttribute", "some descriptive value",
            evidence_quote="some descriptive value", source_context="Some Subject has some descriptive value in context.",
        )
        g, s = _make_graph([subj], {"t1": [rel]})
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
