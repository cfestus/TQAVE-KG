"""Expert-profile 57 -> 36 regression.

A source dataset Expert record is one canonical Expert entity. When that
entity is first encountered through a vote / answer / question / theme
occurrence, the C.3 first-writer-wins rule for the atomic source-identity
tuple keeps that earlier occurrence as the entity's primary scalars, so
the later ``expert.profile`` occurrence never becomes the entity's primary
source -- and downstream no ``expert.profile`` DiscourseUnit was
materialised from it. Only 36 of 57 source Expert records produced an
``expert.profile`` DiscourseUnit / correct 1:1 ``hasSourceEntity`` link.

The ``expert.profile`` occurrence is span-less, so it is NOT preserved in
``metadata["mentions"]`` (both mention builders are span-gated). Fix:

  * ``process_expert`` records it in a generic
    ``metadata["source_occurrences"]`` bucket;
  * ``GlobalEntityRegistry._merge_source_occurrences`` keeps every distinct
    occurrence across canonical-entity merges, keyed on
    (field, discourse_id, origin_key) -- never displaced by first-writer-
    wins;
  * ``RDFSerializer._add_entity_metadata`` materialises each occurrence's
    own DiscourseUnit from ITS provenance tuple and asserts the 1:1
    ``hasSourceEntity`` link directly (the entity that declared the
    occurrence is its source).

No scalar-merge policy change, no primary-source-identity change, and no
canonical-URI parsing.

Synthetic ids / names only.
"""
import importlib.util
import unittest

from config import Config
from models import Entity, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]


class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


def _annotations():
    return {
        "themes": {}, "questions": {}, "answers": {}, "votes": {},
        "experts": {}, "documents": {},
        "metadata": {"datasetId": "unit_ds", "runId": "run-unit"},
    }


def _bare_pipeline():
    from pipeline import EnhancedSemanticPipeline
    from utils.helpers import GlobalEntityRegistry
    from ontology.mapper import OntologyMapper
    from utils.entity_normalizer import EntityNormalizer

    p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    p.config = Config
    p.global_registry = GlobalEntityRegistry()
    p.ontology_mapper = OntologyMapper()
    p.entity_normalizer = EntityNormalizer(p.global_registry)
    return p


def _register_prior_occurrence(pipeline, expert_id, *, field, discourse_id, context_source):
    """Register the canonical Expert entity through a NON-expert.profile
    occurrence first, exactly as the structural vote/answer/question/theme
    handlers would (same URI, so it merges into one canonical entity; C.3
    then locks these scalars first-writer-wins)."""
    ent = Entity(
        uri=f"{ETD}expert_{expert_id}", label=f"Expert {expert_id}",
        entity_type="Expert", confidence=0.95,
        metadata={
            "discourse_id": str(discourse_id), "field": field,
            "source_doc_id": str(discourse_id), "context_source": context_source,
            "source": "structural", "chunk_index": -1, "chunk_index_unknown": True,
            "origin_key": f"{field}|{discourse_id}", "surface_form": f"Expert {expert_id}",
        },
    )
    pipeline._ensure_entity_ontology_grounding(ent)
    return pipeline.global_registry.get_or_create(ent)


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class ExpertProfileFromPipelineMergeTests(unittest.TestCase):
    """Drive the real process_* handlers so the C.3 scalar merge actually
    happens, then serialize and assert on the graph."""

    def _serialize_registry(self, pipeline):
        from serializers.rdf_serializer import RDFSerializer
        ents = list(pipeline.global_registry.get_all_entities())
        s = RDFSerializer(annotations_dict=_annotations(),
                          registry=_Registry(ents), namespaces=Config.NAMESPACES)
        return s, ents

    def _expert_profile_units(self, graph):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        etd = Namespace(ETD)
        out = []
        for du in graph.subjects(RDF.type, etd.DiscourseUnit):
            fields = {str(o) for o in graph.objects(du, etd.field)}
            if "expert.profile" in fields:
                out.append(du)
        return out

    def _assert_one_to_one(self, graph, du, expected_id):
        from rdflib import Namespace, URIRef
        etd = Namespace(ETD)
        sources = [o for o in graph.objects(du, etd.hasSourceEntity) if isinstance(o, URIRef)]
        self.assertEqual(len(sources), 1, f"{du}: expected exactly one hasSourceEntity, got {sources}")
        s = str(sources[0])
        self.assertTrue(
            f"expert-{expected_id}-" in s or s.endswith(f"expert_{expected_id}")
            or f"/expert/expert-{expected_id}-" in s,
            f"{du} -> {s} is not the expert-{expected_id} record",
        )

    def test_expert_seen_first_via_vote_then_expert_profile(self):
        p = _bare_pipeline()
        _register_prior_occurrence(p, 5, field="vote.vote_value", discourse_id="900",
                                   context_source="dataset_vote")
        p.process_expert({"id": 5, "user_id": 50, "display_name": "Cohort Reviewer Five",
                          "area_of_expertise": "Research", "bio": "synthetic bio"})
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)  # no duplicate Expert entities

        s, _ = self._serialize_registry(p)
        units = self._expert_profile_units(s.graph)
        self.assertEqual(len(units), 1)
        self._assert_one_to_one(s.graph, units[0], "5")

    def test_expert_seen_first_via_answer_then_expert_profile(self):
        p = _bare_pipeline()
        # emulate an answer occurrence registering the expert first
        ans_expert = Entity(
            uri=f"{ETD}expert_7", label="Expert 7", entity_type="Expert", confidence=0.95,
            metadata={
                "discourse_id": "31", "field": "answer.description", "source_doc_id": "31",
                "context_source": "dataset_answer", "source": "structural",
                "chunk_index": -1, "chunk_index_unknown": True,
                "origin_key": "answer.description|31", "surface_form": "Expert 7",
            },
        )
        p._ensure_entity_ontology_grounding(ans_expert)
        p.global_registry.get_or_create(ans_expert)
        p.process_expert({"id": 7, "user_id": 70, "display_name": "Cohort Reviewer Seven",
                          "area_of_expertise": "NGO", "bio": "synthetic"})
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)

        s, _ = self._serialize_registry(p)
        units = self._expert_profile_units(s.graph)
        self.assertEqual(len(units), 1)
        self._assert_one_to_one(s.graph, units[0], "7")

    def test_expert_seen_only_via_expert_profile(self):
        p = _bare_pipeline()
        p.process_expert({"id": 9, "user_id": 90, "display_name": "Cohort Reviewer Nine",
                          "area_of_expertise": "Citizen", "bio": "synthetic"})
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)

        s, _ = self._serialize_registry(p)
        units = self._expert_profile_units(s.graph)
        self.assertEqual(len(units), 1)
        self._assert_one_to_one(s.graph, units[0], "9")

    def test_one_canonical_expert_many_occurrences_one_profile_unit(self):
        p = _bare_pipeline()
        _register_prior_occurrence(p, 4, field="vote.vote_value", discourse_id="901",
                                   context_source="dataset_vote")
        _register_prior_occurrence(p, 4, field="vote.vote_value", discourse_id="902",
                                   context_source="dataset_vote")
        _register_prior_occurrence(p, 4, field="question.body", discourse_id="21",
                                   context_source="dataset_question")
        p.process_expert({"id": 4, "user_id": 40, "display_name": "Cohort Reviewer Four",
                          "area_of_expertise": "Enterprise", "bio": "synthetic"})

        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)  # one canonical entity despite many occurrences

        s, _ = self._serialize_registry(p)
        units = self._expert_profile_units(s.graph)
        self.assertEqual(len(units), 1)  # exactly one expert.profile DiscourseUnit
        self._assert_one_to_one(s.graph, units[0], "4")

    def test_many_source_experts_each_get_exactly_one_profile_unit(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF
        etd = Namespace(ETD)

        p = _bare_pipeline()
        # 6 synthetic experts: half first seen via a vote, half profile-only
        for i in (11, 12, 13):
            _register_prior_occurrence(p, i, field="vote.vote_value", discourse_id=str(950 + i),
                                       context_source="dataset_vote")
        for i in (11, 12, 13, 14, 15, 16):
            p.process_expert({"id": i, "user_id": 100 + i,
                              "display_name": f"Cohort Reviewer {i}",
                              "area_of_expertise": "Research", "bio": "synthetic"})

        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 6)  # no duplicates

        s, _ = self._serialize_registry(p)
        units = self._expert_profile_units(s.graph)
        self.assertEqual(len(units), 6)  # one per source Expert record

        seen_ids = set()
        for du in units:
            expected_id = str(du).rsplit("%7C", 1)[-1]
            seen_ids.add(expected_id)
            self._assert_one_to_one(s.graph, du, expected_id)
        self.assertEqual(seen_ids, {"11", "12", "13", "14", "15", "16"})

        # every expert.profile DU actually points at a distinct Expert node
        linked = set()
        for du in units:
            for o in s.graph.objects(du, etd.hasSourceEntity):
                if isinstance(o, URIRef):
                    linked.add(str(o))
        self.assertEqual(len(linked), 6)


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class ExpertProfileSerializerUnitTests(unittest.TestCase):
    """Serializer-only: a canonical Expert entity already merged into a
    non-expert.profile primary source identity still yields its
    expert.profile DiscourseUnit."""

    def _serialize(self, entities):
        from serializers.rdf_serializer import RDFSerializer
        return RDFSerializer(annotations_dict=_annotations(),
                             registry=_Registry(entities), namespaces=Config.NAMESPACES)

    @staticmethod
    def _occ(rec_id):
        return {
            "field": "expert.profile", "discourse_id": str(rec_id),
            "source_doc_id": str(rec_id), "origin_key": f"expert.profile|{rec_id}",
            "source": "structural", "context_source": "dataset_expert",
            "chunk_index": -1, "chunk_index_unknown": True,
        }

    def _profile_units(self, g, etd):
        from rdflib.namespace import RDF
        return [d for d in g.subjects(RDF.type, etd.DiscourseUnit)
                if "expert.profile" in {str(o) for o in g.objects(d, etd.field)}]

    def test_profile_unit_created_when_primary_scalars_point_elsewhere(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF
        etd = Namespace(ETD)

        # canonical Expert whose PRIMARY source identity is a vote occurrence;
        # the expert.profile occurrence survives only in source_occurrences.
        expert = Entity(
            uri="https://alfie-project.eu/etd-hub/entity/person/expert-42-abcdef01",
            label="Expert 42", entity_type="Expert", confidence=1.0,
            metadata={
                "discourse_id": "777", "field": "vote.vote_value", "source_doc_id": "777",
                "context_source": "dataset_vote", "source": "structural",
                "chunk_index": -1, "chunk_index_unknown": True,
                "origin_key": "vote.vote_value|777", "surface_form": "Expert 42",
                "source_occurrences": [self._occ("42")],
            },
        )
        s = self._serialize([expert])
        g = s.graph
        du = URIRef(f"{ETD}discourse_expert.profile%7C42")
        self.assertIn((du, RDF.type, etd.DiscourseUnit), g)
        self.assertIn("expert.profile", {str(o) for o in g.objects(du, etd.field)})
        sources = [o for o in g.objects(du, etd.hasSourceEntity) if isinstance(o, URIRef)]
        self.assertEqual([str(x) for x in sources], [str(expert.uri)])
        # primary source identity is UNCHANGED (still the vote occurrence)
        self.assertIn("vote.vote_value", {str(o) for o in g.objects(URIRef(str(expert.uri)), etd.field)})
        self.assertIn(
            (URIRef(f"{ETD}discourse_vote.vote_value%7C777"), RDF.type, etd.DiscourseUnit), g
        )

    def test_arbitrary_non_pattern_canonical_uri_still_gets_profile_unit(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF
        etd = Namespace(ETD)

        # A canonical Expert URI with NO expert-<id> / expert_<id> shape at
        # all -- the fix must not read identity from URI syntax.
        arb_uri = "https://example.org/kg/node/7f3a9c-Q-whatever~unit"
        expert = Entity(
            uri=arb_uri, label="Panelist", entity_type="Expert", confidence=1.0,
            metadata={
                "discourse_id": "555", "field": "answer.description", "source_doc_id": "555",
                "context_source": "dataset_answer", "source": "structural",
                "chunk_index": -1, "chunk_index_unknown": True,
                "origin_key": "answer.description|555", "surface_form": "Panelist",
                "source_occurrences": [self._occ("88")],
            },
        )
        s = self._serialize([expert])
        g = s.graph
        du = URIRef(f"{ETD}discourse_expert.profile%7C88")
        self.assertIn((du, RDF.type, etd.DiscourseUnit), g)
        self.assertIn("expert.profile", {str(o) for o in g.objects(du, etd.field)})
        sources = [o for o in g.objects(du, etd.hasSourceEntity) if isinstance(o, URIRef)]
        self.assertEqual([str(x) for x in sources], [arb_uri])
        self.assertEqual(len(self._profile_units(g, etd)), 1)

    def test_external_identifier_on_canonical_entity_does_not_block_profile_unit(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF, OWL
        etd = Namespace(ETD)

        expert = Entity(
            uri="https://alfie-project.eu/etd-hub/entity/person/expert-91-deadbeef",
            label="Reviewer 91", entity_type="Expert", confidence=1.0,
            metadata={
                "discourse_id": "301", "field": "vote.vote_value", "source_doc_id": "301",
                "context_source": "dataset_vote", "source": "structural",
                "chunk_index": -1, "chunk_index_unknown": True,
                "origin_key": "vote.vote_value|301", "surface_form": "Reviewer 91",
                "wikidata_uri": "http://www.wikidata.org/entity/Q999999",
                "sameAs": "http://www.wikidata.org/entity/Q999999",
                "source_occurrences": [self._occ("91")],
            },
        )
        s = self._serialize([expert])
        g = s.graph
        du = URIRef(f"{ETD}discourse_expert.profile%7C91")
        self.assertIn((du, RDF.type, etd.DiscourseUnit), g)
        sources = [o for o in g.objects(du, etd.hasSourceEntity) if isinstance(o, URIRef)]
        self.assertEqual([str(x) for x in sources], [str(expert.uri)])
        self.assertEqual(len(self._profile_units(g, etd)), 1)

    def test_idempotent_when_primary_source_is_already_expert_profile(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF
        etd = Namespace(ETD)

        # primary scalars AND source_occurrences both point at expert.profile|43
        expert = Entity(
            uri="https://alfie-project.eu/etd-hub/entity/person/expert-43-abcdef02",
            label="Expert 43", entity_type="Expert", confidence=1.0,
            metadata={
                "discourse_id": "43", "field": "expert.profile", "source_doc_id": "43",
                "context_source": "dataset_expert", "source": "structural",
                "chunk_index": -1, "chunk_index_unknown": True,
                "origin_key": "expert.profile|43", "surface_form": "Expert 43",
                "source_occurrences": [self._occ("43")],
            },
        )
        s = self._serialize([expert])
        g = s.graph
        du = URIRef(f"{ETD}discourse_expert.profile%7C43")
        self.assertEqual(self._profile_units(g, etd), [du])  # exactly one, not duplicated
        sources = [o for o in g.objects(du, etd.hasSourceEntity) if isinstance(o, URIRef)]
        self.assertEqual([str(x) for x in sources], [str(expert.uri)])

    def test_expert_without_recorded_occurrence_gets_no_profile_unit(self):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        etd = Namespace(ETD)

        # An Expert-typed node with no source_occurrences bucket (e.g. an
        # extracted named person mistyped Expert) is occurrence-driven ->
        # nothing to materialise.
        person = Entity(
            uri="https://alfie-project.eu/etd-hub/entity/person/wikidata-q31359093-cf561c39",
            label="Some Named Person", entity_type="Expert", confidence=0.9,
            metadata={
                "discourse_id": "12", "field": "answer.description", "source_doc_id": "12",
                "context_source": "llm_extraction", "source": "llm",
                "surface_form": "Some Named Person",
            },
        )
        s = self._serialize([person])
        self.assertEqual(self._profile_units(s.graph, etd), [])


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class ExpertProfileOccurrencePreservationTests(unittest.TestCase):
    """Registry-level: the span-less expert.profile occurrence must survive
    a canonical-entity merge (it is NOT a mention), without disturbing the
    entity's primary scalar source identity."""

    def _occ_keys(self, ent):
        occ = (ent.metadata or {}).get("source_occurrences") or []
        return {(str(o.get("field")), str(o.get("discourse_id")), str(o.get("origin_key")))
                for o in occ if isinstance(o, dict)}

    def test_expert_profile_occurrence_survives_merge_after_vote(self):
        p = _bare_pipeline()
        _register_prior_occurrence(p, 5, field="vote.vote_value", discourse_id="900",
                                   context_source="dataset_vote")
        p.process_expert({"id": 5, "user_id": 50, "display_name": "Cohort Reviewer Five",
                          "area_of_expertise": "Research", "bio": "synthetic"})
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)
        ent = experts[0]

        # occurrence preserved in source_occurrences (NOT in mentions)
        self.assertIn(
            ("expert.profile", "5", "expert.profile|5"), self._occ_keys(ent),
        )
        for m in (ent.metadata or {}).get("mentions", []) or []:
            self.assertNotEqual(str(m.get("field")), "expert.profile")

        # primary scalar source identity is UNCHANGED (still the vote)
        self.assertEqual(ent.metadata.get("field"), "vote.vote_value")
        self.assertEqual(str(ent.metadata.get("discourse_id")), "900")

    def test_occurrence_survives_when_expert_profile_registered_first(self):
        p = _bare_pipeline()
        p.process_expert({"id": 6, "user_id": 60, "display_name": "Cohort Reviewer Six",
                          "area_of_expertise": "NGO", "bio": "synthetic"})
        _register_prior_occurrence(p, 6, field="vote.vote_value", discourse_id="901",
                                   context_source="dataset_vote")
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)
        self.assertIn(
            ("expert.profile", "6", "expert.profile|6"), self._occ_keys(experts[0]),
        )

    def test_occurrence_not_duplicated_across_repeated_registration(self):
        p = _bare_pipeline()
        for _ in range(3):
            p.process_expert({"id": 7, "user_id": 70, "display_name": "Cohort Reviewer Seven",
                              "area_of_expertise": "Citizen", "bio": "synthetic"})
        experts = [e for e in p.global_registry.get_all_entities() if e.entity_type == "Expert"]
        self.assertEqual(len(experts), 1)
        occ = (experts[0].metadata or {}).get("source_occurrences") or []
        self.assertEqual(len(occ), 1)


if __name__ == "__main__":
    unittest.main()
