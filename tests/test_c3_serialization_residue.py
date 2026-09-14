"""C.3 serialization residue.

RDFSerializer._add_entity_metadata previously mixed two source occurrences
on one entity: field / discourse_id / span / chunk_index came from the
chosen primary_mention (occurrence Y), while source_doc_id and the primary
extractedFrom DiscourseUnit still came from the top-level scalar tuple
(occurrence X, kept by first-writer-wins). The per-mention extractedFrom
loop also inherited a conflicting field / origin_key from the scalars.

Fix (serialization only): when a primary_mention exists, every
single-valued primary source scalar (field, discourse_id, source_doc_id)
and the primary extractedFrom DiscourseUnit describe THAT occurrence; each
additional mention gets its own DiscourseUnit keyed by that mention's own
(field, discourse_id). The top-level scalar tuple is untouched.

Synthetic ids / fields only.
"""
import importlib.util
import unittest

from config import Config
from models import Entity

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]
BASE = ETD.rstrip("#")


class _Registry:
    def __init__(self, ents):
        self._e = list(ents)

    def get_all_entities(self):
        return self._e


def _annotations():
    return {"themes": {}, "questions": {}, "answers": {}, "votes": {},
            "experts": {}, "documents": {},
            "metadata": {"datasetId": "ds", "runId": "run"}}


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class C3SerializationResidueTests(unittest.TestCase):
    def _serialize(self, entity):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer(annotations_dict=_annotations(), registry=_Registry([entity]),
                          namespaces=Config.NAMESPACES, prune_isolated_entities=False)
        return s, Namespace(ETD)

    def _prov(self, s, ns, uri):
        from rdflib import URIRef
        u = URIRef(str(uri))
        one = lambda p: sorted(str(o) for o in s.graph.objects(u, p))
        ef = sorted(str(o).split("discourse_")[-1] for o in s.graph.objects(u, ns.extractedFrom))
        return {
            "field": one(ns.field),
            "discourse_id": one(ns.discourse_id),
            "source_doc_id": one(ns.source_doc_id),
            "start_char": one(ns.start_char),
            "end_char": one(ns.end_char),
            "chunk_index": one(ns.chunk_index),
            "extractedFrom": ef,
        }

    # scalar tuple = occurrence X (theme.description / 6);
    # mentions hold X and Y (answer.description / 11); "answer.*" sorts
    # first so primary_mention == Y.
    def _two_field_entity(self, slug="x", *, x_field="theme.description", x_did="6",
                          y_field="answer.description", y_did="11"):
        return Entity(
            uri=f"{BASE}/entity/concept/{slug}", label="Systemic Bias",
            entity_type="Concept", confidence=0.9,
            metadata={
                "surface_form": "systemic bias",
                "field": x_field, "source_field": x_field,
                "discourse_id": x_did, "source_doc_id": x_did,
                "origin_key": f"{x_field}|{x_did}",
                "start_char": 40, "end_char": 53, "chunk_index": 0,
                "context_source": "dataset_theme", "source": "llm",
                "mentions": [
                    {"field": y_field, "discourse_id": y_did,
                     "start_char": 12, "end_char": 25, "chunk_index": 0},
                    {"field": x_field, "discourse_id": x_did,
                     "start_char": 40, "end_char": 53, "chunk_index": 0},
                ],
            },
        )

    def test_primary_provenance_all_resolves_to_primary_mention(self):
        e = self._two_field_entity("two-field")
        s, ns = self._serialize(e)
        p = self._prov(s, ns, e.uri)
        # every single-valued primary scalar == occurrence Y
        self.assertEqual(p["field"], ["answer.description"])
        self.assertEqual(p["discourse_id"], ["11"])
        self.assertEqual(p["source_doc_id"], ["11"])
        self.assertEqual(p["start_char"], ["12"])
        self.assertEqual(p["end_char"], ["25"])
        self.assertEqual(p["chunk_index"], ["0"])
        # extractedFrom contains Y's DU (the primary) ...
        self.assertIn("answer.description%7C11", p["extractedFrom"])
        # ... and still records X as an additional occurrence
        self.assertIn("theme.description%7C6", p["extractedFrom"])

    def test_no_duplicate_extractedfrom_for_primary_mention(self):
        e = self._two_field_entity("nodup")
        s, ns = self._serialize(e)
        p = self._prov(s, ns, e.uri)
        self.assertEqual(sorted(p["extractedFrom"]),
                         ["answer.description%7C11", "theme.description%7C6"])
        self.assertEqual(len(p["extractedFrom"]), len(set(p["extractedFrom"])))

    def test_same_numeric_discourse_id_in_two_different_fields(self):
        # X = theme.description/7 (scalars), Y = answer.description/7 (mention).
        # Same numeric id, different field -> primary must be the answer.* one
        # and its DU keyed by field, not by the bare number.
        e = self._two_field_entity("samenum", x_field="theme.description", x_did="7",
                                   y_field="answer.description", y_did="7")
        s, ns = self._serialize(e)
        p = self._prov(s, ns, e.uri)
        self.assertEqual(p["field"], ["answer.description"])
        self.assertEqual(p["discourse_id"], ["7"])
        self.assertEqual(p["source_doc_id"], ["7"])
        self.assertIn("answer.description%7C7", p["extractedFrom"])
        self.assertIn("theme.description%7C7", p["extractedFrom"])
        self.assertEqual(len(p["extractedFrom"]), 2)

    def test_single_occurrence_entity_unchanged(self):
        e = Entity(
            uri=f"{BASE}/entity/concept/single", label="Systemic Bias",
            entity_type="Concept", confidence=0.9,
            metadata={
                "surface_form": "systemic bias",
                "field": "answer.description", "source_field": "answer.description",
                "discourse_id": "11", "source_doc_id": "11",
                "origin_key": "answer.description|11",
                "start_char": 12, "end_char": 25, "chunk_index": 0,
                "context_source": "dataset_answer", "source": "llm",
            },
        )
        s, ns = self._serialize(e)
        p = self._prov(s, ns, e.uri)
        self.assertEqual(p["field"], ["answer.description"])
        self.assertEqual(p["discourse_id"], ["11"])
        self.assertEqual(p["source_doc_id"], ["11"])
        self.assertEqual(p["extractedFrom"], ["answer.description%7C11"])

    def test_scalar_tuple_on_entity_object_is_not_mutated(self):
        e = self._two_field_entity("nomutate")
        before = dict(e.metadata)
        self._serialize(e)
        for k in ("field", "source_field", "discourse_id", "source_doc_id",
                  "origin_key", "start_char", "end_char", "chunk_index"):
            self.assertEqual(e.metadata.get(k), before.get(k), f"scalar {k} mutated")

    def test_fieldless_mention_falls_back_to_top_level_field_not_origin_key(self):
        # older-style mention with no field of its own -> DU keyed
        # top_field|did, never the scalar origin_key of a different pair.
        e = Entity(
            uri=f"{BASE}/entity/concept/fieldless", label="Systemic Bias",
            entity_type="Concept", confidence=0.9,
            metadata={
                "surface_form": "systemic bias",
                "field": "answer.description", "source_field": "answer.description",
                "discourse_id": "3", "source_doc_id": "3",
                "origin_key": "answer.description|3",
                "start_char": 1, "end_char": 14, "chunk_index": 0,
                "context_source": "dataset_answer", "source": "llm",
                "mentions": [
                    {"discourse_id": "3", "start_char": 1, "end_char": 14, "chunk_index": 0},
                    {"discourse_id": "9", "start_char": 5, "end_char": 18, "chunk_index": 0},
                ],
            },
        )
        s, ns = self._serialize(e)
        p = self._prov(s, ns, e.uri)
        self.assertEqual(set(p["extractedFrom"]),
                         {"answer.description%7C3", "answer.description%7C9"})


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class C3NoRegressionForStructuralOccurrenceTests(unittest.TestCase):
    """A span-less structural entity (no mentions) -> primary_mention is
    None -> the fix's primary_mention branches are skipped -> behaviour is
    exactly the scalar-tuple behaviour, and any source_occurrences /
    Expert-profile DiscourseUnit is still materialised."""

    def _serialize(self, entities):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer(annotations_dict=_annotations(), registry=_Registry(entities),
                          namespaces=Config.NAMESPACES, prune_isolated_entities=False)
        return s, Namespace(ETD)

    def test_structural_entity_provenance_from_scalars(self):
        from rdflib import URIRef
        e = Entity(
            uri=f"{BASE}/entity/person/expert-5-abcdef01", label="Expert 5",
            entity_type="Expert", confidence=0.95,
            metadata={
                "surface_form": "Expert 5",
                "field": "vote.vote_value", "source_field": "vote.vote_value",
                "discourse_id": "900", "source_doc_id": "900",
                "origin_key": "vote.vote_value|900",
                "chunk_index": -1, "chunk_index_unknown": True,
                "context_source": "dataset_vote", "source": "structural",
                "source_occurrences": [{
                    "field": "expert.profile", "discourse_id": "5", "source_doc_id": "5",
                    "origin_key": "expert.profile|5", "source": "structural",
                    "context_source": "dataset_expert", "chunk_index": -1,
                    "chunk_index_unknown": True,
                }],
            },
        )
        s, ns = self._serialize([e])
        u = URIRef(str(e.uri))
        self.assertEqual([str(o) for o in s.graph.objects(u, ns.field)], ["vote.vote_value"])
        self.assertEqual([str(o) for o in s.graph.objects(u, ns.source_doc_id)], ["900"])
        ef = {str(o).split("discourse_")[-1] for o in s.graph.objects(u, ns.extractedFrom)}
        self.assertIn("vote.vote_value%7C900", ef)
        # source_occurrence DU still materialised + linked
        self.assertIn("expert.profile%7C5", ef)
        prof = URIRef(f"{ETD}discourse_expert.profile%7C5")
        self.assertIn(u, list(s.graph.objects(prof, ns.hasSourceEntity)))


if __name__ == "__main__":
    unittest.main()
