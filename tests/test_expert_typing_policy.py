import importlib.util
import json
import unittest
from pathlib import Path

from models import AnnotationBatch, Entity, ProcessingResult, Relation


HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF integration tests")
class ExpertTypingPolicyTests(unittest.TestCase):
    def test_extracted_named_people_are_not_typed_as_expert_or_source_expert(self):
        from rdflib import Namespace
        from rdflib.namespace import FOAF, RDF

        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        expert_uri = "https://alfie-project.eu/etd-hub/entity/person/expert-12-demo"
        joy_uri = "https://alfie-project.eu/etd-hub/entity/person/wikidata-q31359093-cf561c39"
        timnit_uri = "https://alfie-project.eu/etd-hub/entity/person/wikidata-q59753117-2bef7072"
        concept_uri = f"{etd_base}entity/concept/fairness-demo"

        entities = [
            Entity(
                uri=expert_uri,
                label="Dataset Expert 12",
                entity_type="Expert",
                confidence=1.0,
                metadata={
                    "discourse_id": "12",
                    "source_doc_id": "12",
                    "field": "expert.profile",
                    "context_source": "dataset_expert",
                },
            ),
            Entity(
                uri=joy_uri,
                label="Joy Buolamwini",
                entity_type="Person",
                confidence=0.95,
                metadata={
                    "discourse_id": "12",
                    "source_doc_id": "12",
                    "field": "answer.description",
                    "context_source": "llm_extraction",
                },
            ),
            Entity(
                uri=timnit_uri,
                label="Timnit Gebru",
                entity_type="Person",
                confidence=0.95,
                metadata={
                    "discourse_id": "12",
                    "source_doc_id": "12",
                    "field": "answer.description",
                    "context_source": "llm_extraction",
                },
            ),
            Entity(uri=concept_uri, label="Fairness", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        rel = Relation(
            source=expert_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.91,
            metadata={
                "discourse_id": "12",
                "field": "expert.profile",
                "source_doc_id": "12",
                "source": "structural",
                "context_source": "dataset_expert",
            },
        )
        joy_rel = Relation(
            source=joy_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "3001",
                "field": "answer.description",
                "source_doc_id": "3001",
                "source": "llm",
                "context_source": "llm_extraction",
            },
        )
        timnit_rel = Relation(
            source=timnit_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "3002",
                "field": "answer.description",
                "source_doc_id": "3002",
                "source": "llm",
                "context_source": "llm_extraction",
            },
        )

        ann = AnnotationBatch(
            experts={"12": ProcessingResult(entities=entities[:3], relations=[rel], metadata={})},
            answers={
                "3001": ProcessingResult(entities=[entities[1], entities[3]], relations=[joy_rel], metadata={}),
                "3002": ProcessingResult(entities=[entities[2], entities[3]], relations=[timnit_rel], metadata={}),
            },
            questions={},
            themes={},
            votes={},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ai_ethics_datasets",
            run_id="run_test",
        )
        g = serializer.graph

        from rdflib import URIRef

        joy_ref = URIRef(joy_uri)
        timnit_ref = URIRef(timnit_uri)
        expert_ref = URIRef(expert_uri)

        self.assertIn((expert_ref, RDF.type, etd.Expert), g)
        self.assertNotIn((joy_ref, RDF.type, etd.Expert), g)
        self.assertNotIn((timnit_ref, RDF.type, etd.Expert), g)

        self.assertTrue(
            (joy_ref, RDF.type, etd.Person) in g or (joy_ref, RDF.type, FOAF.Person) in g,
            "Joy should remain typed as a person resource",
        )
        self.assertIn((joy_ref, RDF.type, etd.Entity), g)
        self.assertTrue(
            (timnit_ref, RDF.type, etd.Person) in g or (timnit_ref, RDF.type, FOAF.Person) in g,
            "Timnit should remain typed as a person resource",
        )
        self.assertIn((timnit_ref, RDF.type, etd.Entity), g)

        expert_profile_sources = set()
        discourse_profile_units = []
        for subject in list(g.subjects(RDF.type, etd.Relation)) + list(g.subjects(RDF.type, etd.DiscourseUnit)):
            fields = {str(o) for o in g.objects(subject, etd.field)}
            if "expert.profile" not in fields:
                continue
            source_entities = list(g.objects(subject, etd.hasSourceEntity))
            expert_profile_sources.update(source_entities)
            if (subject, RDF.type, etd.DiscourseUnit) in g:
                discourse_profile_units.append((subject, source_entities))

        self.assertNotIn(joy_ref, expert_profile_sources)
        self.assertNotIn(timnit_ref, expert_profile_sources)
        for source_entity in expert_profile_sources:
            self.assertTrue(
                "/entity/person/expert-" in str(source_entity)
                or "/entity/expert/" in str(source_entity),
                f"expert.profile source link must resolve to a source expert record, got {source_entity}",
            )
        self.assertEqual(len(discourse_profile_units), 1)
        self.assertEqual(len(discourse_profile_units[0][1]), 1)
        self.assertEqual(discourse_profile_units[0][1][0], expert_ref)

    def test_regenerated_ttl_keeps_only_dataset_experts_typed_as_expert(self):
        from rdflib import Graph, Namespace, URIRef
        from rdflib.namespace import FOAF, OWL, RDF, SKOS

        from config import Config

        ttl_path = Path("output/etd_hub_enhanced_kg.ttl")
        data_path = Path("data/ai_ethics_datasets.json")
        if not ttl_path.exists() or not data_path.exists():
            self.skipTest("Expected TTL or source dataset is missing")
        freshness_inputs = [
            Path("pipeline.py"),
            Path("serializers/rdf_serializer.py"),
            Path("utils/semantic_relation_policy.py"),
            data_path,
        ]
        ttl_mtime = ttl_path.stat().st_mtime
        if any(path.exists() and path.stat().st_mtime > ttl_mtime for path in freshness_inputs):
            self.skipTest("Output TTL is older than current code/data; regenerate graph before asserting regenerated TTL invariants")

        with data_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        expert_records = data.get("Expert", [])
        expected_expert_count = len(expert_records) if isinstance(expert_records, (list, dict)) else 0
        self.assertEqual(expected_expert_count, 57)

        g = Graph().parse(str(ttl_path), format="turtle")
        etd = Namespace(Config.NAMESPACES["etd"])

        expert_nodes = {n for n in g.subjects(RDF.type, etd.Expert)}
        self.assertEqual(len(expert_nodes), expected_expert_count)

        source_expert_nodes = {
            n for n in expert_nodes
            if "/entity/person/expert-" in str(n) or "/entity/expert/" in str(n)
        }
        self.assertEqual(len(source_expert_nodes), expected_expert_count)

        # Which specific named people the LLM extracts from free text is
        # non-deterministic across runs (a real prior run extracted "Joy
        # Buolamwini" but not the co-mentioned "Timnit Gebru" from the same
        # sentence -- a standalone-entity extraction-recall gap, not a
        # typing-policy violation). Hardcoding two specific person URIs
        # made this assertion depend on extraction luck rather than on the
        # policy it claims to test. The policy itself -- "an LLM-extracted,
        # Wikidata-identified named person is never ALSO typed Expert" --
        # is verified generically here across WHATEVER such entities this
        # run happened to produce, so the test is deterministic with
        # respect to the policy while still remaining meaningful for any
        # run's actual content.
        #
        # Identity-link predicate is NOT a fixed owl:sameAs (post B
        # identity-semantics fix, see test_wikidata_identity_semantics.py):
        # an AUTHORITATIVE Wikidata identity is asserted via owl:sameAs, a
        # heuristic/non-authoritative one via skos:closeMatch instead --
        # owl:sameAs is never emitted for a heuristic link. "Wikidata-linked"
        # therefore means either predicate pointing at a wikidata.org
        # entity URI; which one is used is an identity-CONFIDENCE signal,
        # not a typing signal, and must have no bearing on Expert typing.
        def _wikidata_linked_people(predicate):
            return {
                s for s in g.subjects(RDF.type, etd.Person)
                if any(str(o).startswith("http://www.wikidata.org/entity/") for o in g.objects(s, predicate))
            }

        llm_wikidata_people_sameas = _wikidata_linked_people(OWL.sameAs)
        llm_wikidata_people_closematch = _wikidata_linked_people(SKOS.closeMatch)
        llm_wikidata_people = llm_wikidata_people_sameas | llm_wikidata_people_closematch

        self.assertTrue(
            llm_wikidata_people,
            "expected at least one LLM-extracted, Wikidata-linked named person in this discourse "
            "(e.g. Joy Buolamwini / Timnit Gebru), linked via owl:sameAs (authoritative) or "
            "skos:closeMatch (heuristic) -- if none exist, this policy has nothing to verify",
        )
        # A heuristic link must never be required to also carry owl:sameAs:
        # a Wikidata-linked person reachable ONLY via skos:closeMatch is
        # exactly as valid evidence for this policy as one reachable via
        # owl:sameAs -- confirm at least the closeMatch predicate alone
        # already accounts for some of these people whenever it is used at
        # all in this run (i.e. the heuristic form is genuinely load-bearing
        # here, not merely tolerated).
        if llm_wikidata_people_closematch:
            self.assertTrue(
                llm_wikidata_people_closematch - llm_wikidata_people_sameas,
                "expected at least one Wikidata-linked person reachable via skos:closeMatch alone "
                "(no owl:sameAs) -- a heuristic link must not need owl:sameAs to be recognised",
            )

        canonical_dataset_expert_count = len(expert_nodes)
        self.assertEqual(canonical_dataset_expert_count, 57)

        # Expert typing must be IDENTICAL regardless of which identity-link
        # predicate (or neither) a person carries -- the predicate choice
        # is an identity-confidence signal, never a typing signal.
        external_person_nodes_typed_as_expert = llm_wikidata_people & expert_nodes
        self.assertEqual(
            len(external_person_nodes_typed_as_expert), 0,
            f"external Wikidata-linked Person nodes must not be typed Expert, got {external_person_nodes_typed_as_expert}",
        )
        self.assertEqual(len(llm_wikidata_people_sameas & expert_nodes), 0)
        self.assertEqual(len(llm_wikidata_people_closematch & expert_nodes), 0)

        for person_ref in llm_wikidata_people:
            self.assertNotIn(person_ref, expert_nodes, f"{person_ref} is an LLM-extracted named person and must not be typed Expert")
            self.assertIn((person_ref, RDF.type, etd.Entity), g)
            self.assertIn((person_ref, RDF.type, FOAF.Person), g)

        bad_expert_profile_sources = []
        discourse_profile_units = []
        for subject in list(g.subjects(RDF.type, etd.Relation)) + list(g.subjects(RDF.type, etd.DiscourseUnit)):
            fields = {str(o) for o in g.objects(subject, etd.field)}
            if "expert.profile" not in fields:
                continue
            source_entities = [o for o in g.objects(subject, etd.hasSourceEntity)]
            if (subject, RDF.type, etd.DiscourseUnit) in g:
                discourse_profile_units.append((subject, source_entities))
            for source_entity in source_entities:
                if source_entity in llm_wikidata_people:
                    bad_expert_profile_sources.append((str(subject), str(source_entity)))
        self.assertEqual(bad_expert_profile_sources, [])

        self.assertEqual(len(discourse_profile_units), expected_expert_count)
        for subject, source_entities in discourse_profile_units:
            self.assertEqual(len(source_entities), 1, f"{subject} should have exactly one source expert")
            only_source = source_entities[0]
            self.assertTrue(
                "/entity/person/expert-" in str(only_source)
                or "/entity/expert/" in str(only_source),
                f"{subject} should link only to a source expert record, got {only_source}",
            )

        du12 = next((subject for subject, _ in discourse_profile_units if str(subject).endswith("discourse_expert.profile%7C12")), None)
        self.assertIsNotNone(du12)
        du12_sources = [o for o in g.objects(du12, etd.hasSourceEntity)]
        self.assertEqual(len(du12_sources), 1)
        self.assertIn("expert-12-", str(du12_sources[0]))
