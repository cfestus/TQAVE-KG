"""
Regression tests for the source-field/span consistency defect found during
the 2026-09 linker-validation preflight: an entity consolidated across two
or more discourse mentions could serialize a `discourse_id` that no longer
matched the `start_char`/`end_char` also emitted for it, because
GlobalEntityRegistry.get_or_create correctly accumulates every occurrence
into metadata["mentions"] but the TOP-LEVEL scalar
discourse_id/start_char/end_char/chunk_index were separately overwritten
by whichever mention was merged last. Confirmed empirically against 6 real
graph entities (Cultural Bias, communities, model, users, Algorithms, one
of two OpenAI mentions) whose recorded span matched a DIFFERENT
discourse_id than the first one listed.

Fix: serializers/rdf_serializer.py::_add_entity_metadata now derives
discourse_id, start_char, end_char, and chunk_index from the SAME
deterministic primary_mention record, so the emitted quadruple is always
internally consistent (verifiable against one real source text), never a
discourse_id from one mention paired with an offset from another.
"""
import importlib.util
import unittest

from config import Config
from models import Entity
from utils.helpers import GlobalEntityRegistry

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


def _base_annotations(relations=None, discourse_id="d1"):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations or []}},
        "votes": {}, "experts": {}, "documents": {},
        "metadata": {},
    }


class RegistryMergeScalarDriftTests(unittest.TestCase):
    """C.3: get_or_create must treat the (field, discourse_id, start_char,
    end_char, chunk_index) tuple as ONE atomic source-identity record. A
    cross-record merge keeps the first coherent tuple intact and records
    the other occurrence in metadata["mentions"] -- it never overwrites
    one scalar of the tuple from a different occurrence."""

    def test_source_identity_tuple_does_not_drift_across_merge(self):
        reg = GlobalEntityRegistry()
        uri = "https://alfie-project.eu/etd-hub#entity_drift"
        e1 = Entity(
            uri=uri, label="Cultural Bias", entity_type="Bias", confidence=0.8,
            metadata={
                "discourse_id": "11", "field": "theme.description",
                "start_char": 999, "end_char": 1012,
                "mentions": [{"discourse_id": "11", "field": "theme.description",
                              "chunk_index": 0, "start_char": 999, "end_char": 1012}],
            },
        )
        e2 = Entity(
            uri=uri, label="Cultural Bias", entity_type="Bias", confidence=0.8,
            metadata={
                "discourse_id": "14", "field": "theme.description",
                "start_char": 404, "end_char": 417,
                "mentions": [{"discourse_id": "14", "field": "theme.description",
                              "chunk_index": 0, "start_char": 404, "end_char": 417}],
            },
        )
        reg.get_or_create(e1)
        merged = reg.get_or_create(e2)

        # The first-registered coherent tuple survives intact...
        self.assertEqual(merged.metadata["discourse_id"], "11")
        self.assertEqual(merged.metadata["start_char"], 999)
        self.assertEqual(merged.metadata["end_char"], 1012)
        self.assertEqual(merged.metadata["field"], "theme.description")
        # ...and the second occurrence is preserved as a mention, with its
        # own field paired to its own discourse_id.
        self.assertEqual(len(merged.metadata["mentions"]), 2)
        by_did = {m["discourse_id"]: m for m in merged.metadata["mentions"]}
        self.assertEqual(by_did["11"]["start_char"], 999)
        self.assertEqual(by_did["14"]["start_char"], 404)
        self.assertEqual(by_did["14"]["field"], "theme.description")


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class SerializedSpanConsistencyTests(unittest.TestCase):
    """End-to-end: for an entity whose scalar metadata reflects a DIFFERENT
    mention than mentions[0]/primary, the SERIALIZED graph's
    discourse_id/start_char/end_char/chunk_index must all come from one
    single, same mention -- never a mix."""

    etd_base = Config.NAMESPACES["etd"]

    def _serialize(self, metadata, uri_suffix=""):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        uri = f"{self.etd_base}entity_span_consistency_00{uri_suffix}"
        target_uri = f"{self.etd_base}entity_span_consistency_target{uri_suffix}"
        entities = [
            Entity(uri=uri, label="X", entity_type="Concept", confidence=0.8, metadata=metadata),
            Entity(uri=target_uri, label="Target", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Target"}),
        ]
        # An isolated (relation-free) entity gets pruned before final
        # output (see test_isolated_entity_pruning.py) -- give it exactly
        # one relation so it survives to be inspected, same as every other
        # RDFSerializer-level test in this suite.
        relations = [{
            "source": uri, "relation": "relatedTo", "target": target_uri, "confidence": 0.9,
            "metadata": {"discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor"},
        }]
        serializer = RDFSerializer(annotations_dict=_base_annotations(relations), registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer, etd_ns, uri

    def test_discourse_id_and_span_come_from_the_same_mention(self):
        # Scalar fields simulate the drifted state produced by
        # get_or_create's last-merge-wins overwrite: they belong to
        # discourse "9" (a mention NOT present in `mentions` at all here,
        # standing in for "whatever the stale top-level scalar happened to
        # end up as"), while `mentions` holds the two REAL, individually
        # correct occurrences.
        metadata = {
            "discourse_id": "9", "field": "theme.description",
            "start_char": 9999, "end_char": 10010,
            "chunk_index": 7,
            "mentions": [
                {"discourse_id": "10", "chunk_index": 0, "start_char": 612, "end_char": 635},
                {"discourse_id": "9", "chunk_index": 1, "start_char": 153, "end_char": 176},
            ],
        }
        serializer, etd_ns, uri = self._serialize(metadata)
        from rdflib import URIRef
        subj = URIRef(uri)

        dids = [str(v) for v in serializer.graph.objects(subj, etd_ns.discourse_id)]
        starts = [int(v) for v in serializer.graph.objects(subj, etd_ns.start_char)]
        ends = [int(v) for v in serializer.graph.objects(subj, etd_ns.end_char)]
        chunk_idx = [int(v) for v in serializer.graph.objects(subj, etd_ns.chunk_index)]

        # Exactly ONE discourse_id triple, ONE start_char, ONE end_char --
        # no ambiguity about which pairs with which.
        self.assertEqual(len(dids), 1)
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)

        # And they must be a matched set from ONE mention record -- the
        # stale scalar (9999/10010/discourse "9" mismatched with chunk 7)
        # must NOT appear at all.
        matched = (
            (dids[0] == "10" and starts[0] == 612 and ends[0] == 635) or
            (dids[0] == "9" and starts[0] == 153 and ends[0] == 176)
        )
        self.assertTrue(matched, f"discourse_id={dids[0]} start={starts[0]} end={ends[0]} is not a real mention pair")
        self.assertNotEqual(starts[0], 9999, "the stale drifted scalar start_char must never be emitted when mentions exist")
        # chunk_index must belong to the SAME mention as the emitted span.
        if dids[0] == "10":
            self.assertEqual(chunk_idx[0], 0)
        else:
            self.assertEqual(chunk_idx[0], 1)

    def test_falls_back_to_scalar_when_no_mentions_recorded(self):
        """Entities that were never consolidated (no `mentions` list at
        all) must keep working exactly as before -- the fix only changes
        behavior when there is a real ambiguity to resolve."""
        metadata = {
            "discourse_id": "5", "field": "answer.description",
            "start_char": 20, "end_char": 30, "chunk_index": 2,
        }
        serializer, etd_ns, uri = self._serialize(metadata)
        from rdflib import URIRef
        subj = URIRef(uri)
        self.assertEqual([str(v) for v in serializer.graph.objects(subj, etd_ns.discourse_id)], ["5"])
        self.assertEqual([int(v) for v in serializer.graph.objects(subj, etd_ns.start_char)], [20])
        self.assertEqual([int(v) for v in serializer.graph.objects(subj, etd_ns.end_char)], [30])

    def test_single_mention_matching_scalar_is_unaffected(self):
        """The common case (one mention, scalar already agrees) must be
        completely unaffected."""
        metadata = {
            "discourse_id": "5", "field": "answer.description",
            "start_char": 20, "end_char": 30, "chunk_index": 2,
            "mentions": [{"discourse_id": "5", "chunk_index": 2, "start_char": 20, "end_char": 30}],
        }
        serializer, etd_ns, uri = self._serialize(metadata)
        from rdflib import URIRef
        subj = URIRef(uri)
        self.assertEqual([str(v) for v in serializer.graph.objects(subj, etd_ns.discourse_id)], ["5"])
        self.assertEqual([int(v) for v in serializer.graph.objects(subj, etd_ns.start_char)], [20])
        self.assertEqual([int(v) for v in serializer.graph.objects(subj, etd_ns.end_char)], [30])


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class SpanAgainstSourceFieldTests(unittest.TestCase):
    """Deterministic tests that verify the recorded span against the
    EXACT source field text used for extraction (per the release-preflight
    requirement), using small synthetic field texts rather than the live
    dataset so the test has no external dependency."""

    etd_base = Config.NAMESPACES["etd"]

    # Both mentions are individually correct within their OWN discourse's
    # text (as they would be in real production data -- each was found via
    # a real literal/fuzzy search in its own chunk at extraction time). The
    # defect this test guards against is a SERIALIZED discourse_id from
    # one mention getting paired with a start/end from a DIFFERENT one --
    # not either mention's own recorded pair being individually wrong.
    SOURCE_TEXTS = {
        ("theme.description", "11"): "xxxx Cultural Bias is discussed here.",
        ("theme.description", "14"): "Some intro. Cultural Bias is a real concern in this domain.",
    }

    def test_recorded_span_matches_exact_source_field_text(self):
        from rdflib import Namespace, URIRef
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        uri = f"{self.etd_base}entity_span_source_check_00"
        metadata = {
            # Deliberately drifted top-level scalar (simulating
            # get_or_create's last-merge-wins overwrite): if the
            # serializer trusted this instead of `mentions`, it would
            # under- or over-shoot the real text.
            "discourse_id": "14", "field": "theme.description",
            "start_char": 999, "end_char": 1012,
            "mentions": [
                {"discourse_id": "11", "chunk_index": 0, "start_char": 5, "end_char": 18},
                {"discourse_id": "14", "chunk_index": 0, "start_char": 12, "end_char": 25},
            ],
            "surface_form": "Cultural Bias",
        }
        target_uri = f"{self.etd_base}entity_span_source_check_target"
        entities = [
            Entity(uri=uri, label="Cultural Bias", entity_type="Bias", confidence=0.8, metadata=metadata),
            Entity(uri=target_uri, label="Target", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Target"}),
        ]
        relations = [{
            "source": uri, "relation": "relatedTo", "target": target_uri, "confidence": 0.9,
            "metadata": {"discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor"},
        }]
        serializer = RDFSerializer(annotations_dict=_base_annotations(relations), registry=_Registry(entities), namespaces=Config.NAMESPACES)
        subj = URIRef(uri)

        did = str(next(iter(serializer.graph.objects(subj, etd_ns.discourse_id))))
        field = str(next(iter(serializer.graph.objects(subj, etd_ns.field))))
        s = int(next(iter(serializer.graph.objects(subj, etd_ns.start_char))))
        e = int(next(iter(serializer.graph.objects(subj, etd_ns.end_char))))
        surface_form = str(next(iter(serializer.graph.objects(subj, etd_ns.surface_form))))

        source_text = self.SOURCE_TEXTS[(field, did)]
        self.assertEqual(
            source_text[s:e], surface_form,
            f"recorded span ({s},{e}) for (field={field!r}, discourse_id={did!r}) must exactly bound the surface form",
        )


if __name__ == "__main__":
    unittest.main()
