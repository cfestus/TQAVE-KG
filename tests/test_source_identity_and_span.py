"""
C.2 (source-span over-capture / bad offset arithmetic) and
C.3 (wrong field/discourse provenance assignment).

C.2 invariant: for a genuine source-grounded primary mention,
    source_text[start_char:end_char]
must equal the entity's surface form OR a recorded justified alias /
normalization form. A span that cannot be verified deterministically is
WITHHELD (numeric offsets not emitted) -- never silently truncated, and
the entity keeps its (field, discourse_id, surface_form) identity.

C.3 invariant: (field, discourse_id) is ONE atomic source-identity pair.
A cross-record merge never updates one half from a different occurrence;
the emitted (field, discourse_id) is the pair whose source text the
emitted span actually indexes into.

All source text here is tiny synthetic strings -- no dataset dependency,
no network. Named phrases are regression fixtures only.
"""
import importlib.util
import unittest

from config import Config
from models import Entity
from utils.helpers import (
    GlobalEntityRegistry,
    dedup_mentions,
    mention_from_source_identity,
)

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]


class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


def _lookup(table):
    def resolve(field, did):
        return table.get((str(field or ""), str(did or "")))
    return resolve


def _annotations(entity_uri, target_uri, discourse_id="d1"):
    # one relation so the entity under test survives isolated-entity pruning
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": [{
            "source": entity_uri, "relation": "relatedTo", "target": target_uri,
            "confidence": 0.9,
            "metadata": {"discourse_id": discourse_id, "field": "answer.description",
                         "chunk_index": 0, "source": "llm_relation_extractor"},
        }]}},
        "votes": {}, "experts": {}, "documents": {}, "metadata": {},
    }


def _serialize(entity_metadata, *, table, surface_form=None, label="X", aliases=None):
    from rdflib import Namespace
    from serializers.rdf_serializer import RDFSerializer

    uri = f"{ETD}entity_under_test"
    target_uri = f"{ETD}entity_target"
    md = dict(entity_metadata)
    if surface_form is not None:
        md.setdefault("surface_form", surface_form)
    if aliases is not None:
        md.setdefault("aliases", list(aliases))
    entities = [
        Entity(uri=uri, label=label, entity_type="Concept", confidence=0.8, metadata=md),
        Entity(uri=target_uri, label="Target", entity_type="Concept", confidence=0.8,
               metadata={"surface_form": "Target"}),
    ]
    s = RDFSerializer(
        annotations_dict=_annotations(uri, target_uri),
        registry=_Registry(entities),
        namespaces=Config.NAMESPACES,
        source_text_lookup=_lookup(table),
    )
    return s, Namespace(ETD), uri


def _one(graph, subj, pred):
    vals = list(graph.objects(subj, pred))
    return vals[0] if vals else None


# =====================================================================
# C.2 -- span over-capture / verification
# =====================================================================

@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class SpanVerificationTests(unittest.TestCase):
    def test_exact_mention_span_is_emitted(self):
        text = "intro words, Signal Term appears here."
        start = text.index("Signal Term")
        end = start + len("Signal Term")
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "7",
             "mentions": [{"field": "answer.description", "discourse_id": "7",
                           "chunk_index": 0, "start_char": start, "end_char": end}]},
            table={("answer.description", "7"): text},
            surface_form="Signal Term",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        self.assertEqual(int(_one(s.graph, subj, ns.start_char)), start)
        self.assertEqual(int(_one(s.graph, subj, ns.end_char)), end)
        self.assertEqual(str(_one(s.graph, subj, ns.field)), "answer.description")
        self.assertEqual(str(_one(s.graph, subj, ns.discourse_id)), "7")

    def test_end_over_capture_mention_is_rejected_for_a_verifying_one(self):
        # Two recorded occurrences: one over-captures (bounds "Signal Term X"
        # while surface_form is "Signal Term"); the other bounds it exactly.
        text = "The Signal Term X thing, and later just Signal Term alone."
        over_s = text.index("Signal Term X")
        over_e = over_s + len("Signal Term X")
        good_s = text.index("Signal Term alone")
        good_e = good_s + len("Signal Term")
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "7",
             "mentions": [
                 {"field": "answer.description", "discourse_id": "7", "chunk_index": 0,
                  "start_char": over_s, "end_char": over_e},
                 {"field": "answer.description", "discourse_id": "7", "chunk_index": 1,
                  "start_char": good_s, "end_char": good_e},
             ]},
            table={("answer.description", "7"): text},
            surface_form="Signal Term",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        s_out = int(_one(s.graph, subj, ns.start_char))
        e_out = int(_one(s.graph, subj, ns.end_char))
        self.assertEqual(text[s_out:e_out], "Signal Term")
        self.assertNotEqual(s_out, over_s)

    def test_normalized_alias_preserved_mention_verifies(self):
        # Raw text has the pre-normalization form; surface_form is the
        # normalized form; the raw form is a recorded alias.
        text = "context ... Data's Source in the middle ... end"
        raw = "Data's Source"
        start = text.index(raw)
        end = start + len(raw)
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "7",
             "mentions": [{"field": "answer.description", "discourse_id": "7",
                           "chunk_index": 0, "start_char": start, "end_char": end}]},
            table={("answer.description", "7"): text},
            surface_form="Data Source",     # normalized
            aliases=["Data's Source"],       # preserved original
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        self.assertEqual(int(_one(s.graph, subj, ns.start_char)), start)
        self.assertEqual(int(_one(s.graph, subj, ns.end_char)), end)

    def test_repeated_mention_occurrence_all_verify_deterministic_pick(self):
        text = "bias here, bias again, and bias once more."
        occ = []
        i = text.find("bias")
        while i != -1:
            occ.append((i, i + 4))
            i = text.find("bias", i + 1)
        self.assertEqual(len(occ), 3)
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "7",
             "mentions": [
                 {"field": "answer.description", "discourse_id": "7", "chunk_index": 0,
                  "start_char": a, "end_char": b} for a, b in occ
             ]},
            table={("answer.description", "7"): text},
            surface_form="bias",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        s_out = int(_one(s.graph, subj, ns.start_char))
        e_out = int(_one(s.graph, subj, ns.end_char))
        self.assertEqual(text[s_out:e_out], "bias")
        self.assertEqual((s_out, e_out), occ[0])  # deterministic: earliest

    def test_unverifiable_span_is_withheld_not_truncated(self):
        # The only recorded occurrence's excerpt matches neither the
        # surface form nor any alias, and the pair still resolves.
        text = "completely unrelated words fill this record entirely."
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "7",
             "mentions": [{"field": "answer.description", "discourse_id": "7",
                           "chunk_index": 0, "start_char": 5, "end_char": 20}]},
            table={("answer.description", "7"): text},
            surface_form="Signal Term",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        # numeric offsets withheld ...
        self.assertIsNone(_one(s.graph, subj, ns.start_char))
        self.assertIsNone(_one(s.graph, subj, ns.end_char))
        # ... identity kept ...
        self.assertEqual(str(_one(s.graph, subj, ns.field)), "answer.description")
        self.assertEqual(str(_one(s.graph, subj, ns.discourse_id)), "7")
        self.assertEqual(str(_one(s.graph, subj, ns.surface_form)), "Signal Term")
        # ... and it is flagged, not silently dropped.
        self.assertTrue(any(
            e.get("category") == "entity_span_withheld_unverifiable" for e in s.quality_events
        ))


# =====================================================================
# C.3 -- field / discourse provenance is one atomic pair
# =====================================================================

class RegistryAtomicPairTests(unittest.TestCase):
    def test_same_numeric_discourse_id_in_two_different_fields(self):
        reg = GlobalEntityRegistry()
        uri = f"{ETD}dup_num_did"
        a = Entity(uri=uri, label="Term", entity_type="Concept", confidence=0.8,
                   metadata={"field": "answer.description", "discourse_id": "10",
                             "source_doc_id": "10", "start_char": 3, "end_char": 7,
                             "mentions": [{"field": "answer.description", "discourse_id": "10",
                                           "chunk_index": 0, "start_char": 3, "end_char": 7}]})
        b = Entity(uri=uri, label="Term", entity_type="Concept", confidence=0.8,
                   metadata={"field": "question.body", "discourse_id": "10",
                             "source_doc_id": "10", "start_char": 40, "end_char": 44,
                             "mentions": [{"field": "question.body", "discourse_id": "10",
                                           "chunk_index": 0, "start_char": 40, "end_char": 44}]})
        reg.get_or_create(a)
        merged = reg.get_or_create(b)
        # top-level tuple stays the first coherent one, undivided
        self.assertEqual(merged.metadata["field"], "answer.description")
        self.assertEqual(merged.metadata["discourse_id"], "10")
        self.assertEqual(merged.metadata["start_char"], 3)
        # both field-distinct occurrences of discourse "10" survive as mentions
        pairs = {(m["field"], m["discourse_id"], m["start_char"]) for m in merged.metadata["mentions"]}
        self.assertIn(("answer.description", "10", 3), pairs)
        self.assertIn(("question.body", "10", 40), pairs)

    def test_cross_discourse_merged_entity_keeps_pairs_intact(self):
        reg = GlobalEntityRegistry()
        uri = f"{ETD}xdisc"
        a = Entity(uri=uri, label="Term", entity_type="Concept", confidence=0.8,
                   metadata={"field": "theme.description", "discourse_id": "6",
                             "start_char": 12, "end_char": 16,
                             "mentions": [{"field": "theme.description", "discourse_id": "6",
                                           "chunk_index": 0, "start_char": 12, "end_char": 16}]})
        b = Entity(uri=uri, label="Term", entity_type="Concept", confidence=0.9,
                   metadata={"field": "theme.description", "discourse_id": "11",
                             "start_char": 90, "end_char": 94,
                             "mentions": [{"field": "theme.description", "discourse_id": "11",
                                           "chunk_index": 2, "start_char": 90, "end_char": 94}]})
        reg.get_or_create(a)
        merged = reg.get_or_create(b)
        # discourse_id did NOT drift to the last-merged occurrence
        self.assertEqual(merged.metadata["discourse_id"], "6")
        self.assertEqual(merged.metadata["start_char"], 12)
        by_did = {m["discourse_id"]: m for m in merged.metadata["mentions"]}
        self.assertEqual(by_did["6"]["start_char"], 12)
        self.assertEqual(by_did["11"]["start_char"], 90)
        self.assertEqual(by_did["11"]["field"], "theme.description")

    def test_mention_from_source_identity_carries_field(self):
        m = mention_from_source_identity(
            {"field": "answer.description", "discourse_id": "9", "start_char": 1, "end_char": 5}
        )
        self.assertEqual(m, {"field": "answer.description", "discourse_id": "9",
                             "start_char": 1, "end_char": 5})
        self.assertIsNone(mention_from_source_identity(
            {"field": "answer.description", "discourse_id": "9"}  # no span
        ))

    def test_dedup_folds_scalar_and_explicit_occurrence(self):
        folded = dedup_mentions([
            {"discourse_id": "9", "start_char": 10, "end_char": 20},  # scalar-derived, no field/chunk
            {"field": "answer.description", "discourse_id": "9", "chunk_index": 3,
             "start_char": 10, "end_char": 20},
        ])
        self.assertEqual(len(folded), 1)
        self.assertEqual(folded[0]["field"], "answer.description")
        self.assertEqual(folded[0]["chunk_index"], 3)


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class SerializerAtomicPairTests(unittest.TestCase):
    def test_primary_mention_selection_preserves_field_and_discourse_together(self):
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer.__new__(RDFSerializer)
        s._source_text_lookup = None
        meta = {
            "field": "answer.description", "discourse_id": "10",
            "mentions": [
                {"field": "question.body", "discourse_id": "10", "chunk_index": 0,
                 "start_char": 5, "end_char": 9},
                {"field": "answer.description", "discourse_id": "10", "chunk_index": 1,
                 "start_char": 50, "end_char": 54},
            ],
        }
        pm = s._primary_mention(meta, surface_form="Term", aliases=[])
        # the returned field + discourse_id + span are ONE record -- the
        # field is never taken from a different mention than the offsets
        self.assertIn(pm["field"], {"question.body", "answer.description"})
        if pm["field"] == "question.body":
            self.assertEqual((pm["start_char"], pm["end_char"]), (5, 9))
        else:
            self.assertEqual((pm["start_char"], pm["end_char"]), (50, 54))

    def test_serializer_emits_the_same_pair_used_for_the_span(self):
        ans_text = "aaaa Concept Alpha bbbb"
        que_text = "no such phrase in this question body at all"
        a_s = ans_text.index("Concept Alpha")
        a_e = a_s + len("Concept Alpha")
        s, ns, uri = _serialize(
            {"field": "question.body", "discourse_id": "10",   # drifted top-level field
             "mentions": [
                 {"field": "answer.description", "discourse_id": "10", "chunk_index": 0,
                  "start_char": a_s, "end_char": a_e},
             ]},
            table={("answer.description", "10"): ans_text, ("question.body", "10"): que_text},
            surface_form="Concept Alpha",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        f_out = str(_one(s.graph, subj, ns.field))
        d_out = str(_one(s.graph, subj, ns.discourse_id))
        s_out = int(_one(s.graph, subj, ns.start_char))
        e_out = int(_one(s.graph, subj, ns.end_char))
        # emitted pair actually indexes the emitted span
        self.assertEqual(f_out, "answer.description")
        self.assertEqual(d_out, "10")
        self.assertEqual(_lookup(
            {("answer.description", "10"): ans_text}
        )(f_out, d_out)[s_out:e_out], "Concept Alpha")

    def test_nonexistent_source_pair_is_flagged_not_invented(self):
        s, ns, uri = _serialize(
            {"field": "answer.description", "discourse_id": "999",
             "mentions": [{"field": "answer.description", "discourse_id": "999",
                           "chunk_index": 0, "start_char": 1, "end_char": 5}]},
            table={("answer.description", "10"): "real record text only"},  # 999 does not exist
            surface_form="Term",
        )
        from rdflib import URIRef
        subj = URIRef(uri)
        # discourse id is NOT renumbered / invented ...
        self.assertEqual(str(_one(s.graph, subj, ns.discourse_id)), "999")
        # ... but the span it cannot resolve is withheld + flagged
        self.assertIsNone(_one(s.graph, subj, ns.start_char))
        self.assertTrue(any(
            e.get("category") == "entity_span_withheld_unverifiable" for e in s.quality_events
        ))


if __name__ == "__main__":
    unittest.main()
