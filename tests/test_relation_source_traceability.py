import importlib.util
import unittest

from models import AnnotationBatch, Entity, ProcessingResult, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF integration tests")
class RelationSourceTraceabilityTests(unittest.TestCase):
    def test_answer_and_vote_relations_get_explicit_source_entities(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        theme_uri = f"{etd_base}theme_101"
        question_uri = f"{etd_base}question_201"
        answer_uri = f"{etd_base}answer_301"
        expert_uri = f"{etd_base}expert_401"
        vote_uri = f"{etd_base}vote_501"
        concept_uri = f"{etd_base}entity_concept_x"

        entities = [
            Entity(
                uri=theme_uri,
                label="Theme T1",
                entity_type="Theme",
                confidence=1.0,
                metadata={"discourse_id": "101", "raw_fields": {"expert_id": "401"}},
            ),
            Entity(
                uri=question_uri,
                label="Question Q1",
                entity_type="Question",
                confidence=1.0,
                metadata={"discourse_id": "201", "raw_fields": {"expert_id": "401"}},
            ),
            Entity(uri=answer_uri, label="Answer A1", entity_type="Answer", confidence=1.0, metadata={"discourse_id": "301"}),
            Entity(uri=expert_uri, label="Expert E1", entity_type="Expert", confidence=1.0, metadata={"discourse_id": "401"}),
            Entity(uri=vote_uri, label="Vote V1", entity_type="Vote", confidence=1.0, metadata={"discourse_id": "501"}),
            Entity(uri=concept_uri, label="Concept X", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        structural_rels = [
            Relation(source=question_uri, relation="belongsToTheme", target=theme_uri, confidence=0.99, metadata={"discourse_id": "201", "field": "question.body", "source_doc_id": "201", "source": "structural"}),
            Relation(source=answer_uri, relation="answersQuestion", target=question_uri, confidence=0.99, metadata={"discourse_id": "301", "field": "answer.description", "source_doc_id": "301", "source": "structural"}),
            Relation(source=answer_uri, relation="authoredBy", target=expert_uri, confidence=0.99, metadata={"discourse_id": "301", "field": "answer.description", "source_doc_id": "301", "source": "structural"}),
            Relation(source=vote_uri, relation="castBy", target=expert_uri, confidence=0.99, metadata={"discourse_id": "501", "field": "vote.vote_value", "source_doc_id": "501", "source": "structural"}),
            Relation(source=answer_uri, relation="receivesVote", target=vote_uri, confidence=0.99, metadata={"discourse_id": "501", "field": "vote.vote_value", "source_doc_id": "501", "source": "structural"}),
        ]
        claim_rels = [
            Relation(source=answer_uri, relation="relatedTo", target=concept_uri, confidence=0.9, metadata={"discourse_id": "301", "field": "answer.description", "source_doc_id": "301", "source": "llm"}),
            Relation(source=answer_uri, relation="endorses", target=concept_uri, confidence=0.9, metadata={"discourse_id": "501", "field": "vote.vote_value", "source_doc_id": "501", "source": "vote"}),
        ]

        ann = AnnotationBatch(
            themes={"101": ProcessingResult(entities=[entities[0]], relations=[], metadata={})},
            questions={"201": ProcessingResult(entities=[entities[1]], relations=[structural_rels[0]], metadata={})},
            answers={"301": ProcessingResult(entities=[entities[2], entities[3]], relations=structural_rels[1:3] + claim_rels[:1], metadata={})},
            votes={"501": ProcessingResult(entities=[entities[4], entities[2], entities[3]], relations=structural_rels[3:] + claim_rels[1:], metadata={})},
            experts={"401": ProcessingResult(entities=[entities[3]], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        answer_claim = None
        vote_claim = None
        for rel in g.subjects(RDF.type, etd.Relation):
            preds = {str(o).split("#")[-1] for o in g.objects(rel, etd.predicate)}
            fields = {str(o) for o in g.objects(rel, etd.field)}
            if "relatedTo" in preds and "answer.description" in fields:
                answer_claim = rel
            if "endorses" in preds and "vote.vote_value" in fields:
                vote_claim = rel

        self.assertIsNotNone(answer_claim)
        self.assertIsNotNone(vote_claim)

        answer_sources = {str(o) for o in g.objects(answer_claim, etd.hasSourceEntity)}
        vote_sources = {str(o) for o in g.objects(vote_claim, etd.hasSourceEntity)}

        self.assertIn(answer_uri, answer_sources)
        self.assertIn(question_uri, answer_sources)
        self.assertIn(theme_uri, answer_sources)
        self.assertIn(expert_uri, answer_sources)

        self.assertIn(vote_uri, vote_sources)
        self.assertIn(answer_uri, vote_sources)
        self.assertIn(question_uri, vote_sources)
        self.assertIn(theme_uri, vote_sources)
        self.assertIn(expert_uri, vote_sources)

    def test_theme_description_relations_resolve_theme_by_theme_id(self):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        theme_uri = f"{etd_base}entity/theme/theme-101-demo"
        concept_uri = f"{etd_base}entity/concept/demo"

        entities = [
            Entity(
                uri=theme_uri,
                label="Theme T101",
                entity_type="Theme",
                confidence=1.0,
                metadata={
                    "discourse_id": "999",
                    "theme_id": "101",
                    "field": "question.body",
                    "context_source": "dataset_question",
                },
            ),
            Entity(uri=concept_uri, label="Concept Y", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        claim_rel = Relation(
            source=theme_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "101",
                "field": "theme.description",
                "source_doc_id": "101",
                "source": "llm",
            },
        )

        ann = AnnotationBatch(
            themes={"101": ProcessingResult(entities=[entities[0]], relations=[claim_rel], metadata={})},
            questions={},
            answers={},
            votes={},
            experts={},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        themed_claim = None
        for rel in g.subjects(RDF.type, etd.Relation):
            preds = {str(o).split("#")[-1] for o in g.objects(rel, etd.predicate)}
            fields = {str(o) for o in g.objects(rel, etd.field)}
            if "relatedTo" in preds and "theme.description" in fields:
                themed_claim = rel
                break

        self.assertIsNotNone(themed_claim)
        themed_sources = {str(o) for o in g.objects(themed_claim, etd.hasSourceEntity)}
        self.assertIn(theme_uri, themed_sources)

    def test_source_entity_uri_drives_traceability_even_when_ids_drift(self):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        theme_uri = f"{etd_base}entity/theme/theme-10-demo"
        concept_uri = f"{etd_base}entity/concept/demo2"

        entities = [
            Entity(
                uri=theme_uri,
                label="Theme T10",
                entity_type="Theme",
                confidence=1.0,
                metadata={"discourse_id": "999", "field": "question.body", "context_source": "dataset_question"},
            ),
            Entity(uri=concept_uri, label="Concept Z", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        claim_rel = Relation(
            source=theme_uri,
            relation="hasRisk",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "16",
                "field": "theme.description",
                "source_doc_id": "16",
                "source": "reasoner",
                "source_entity_uri": theme_uri,
            },
        )

        ann = AnnotationBatch(
            themes={"10": ProcessingResult(entities=[entities[0]], relations=[claim_rel], metadata={})},
            questions={},
            answers={},
            votes={},
            experts={},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        themed_claim = None
        for rel in g.subjects(RDF.type, etd.Relation):
            preds = {str(o).split("#")[-1] for o in g.objects(rel, etd.predicate)}
            fields = {str(o) for o in g.objects(rel, etd.field)}
            if "hasRisk" in preds and "theme.description" in fields:
                themed_claim = rel
                break

        self.assertIsNotNone(themed_claim)
        themed_sources = {str(o) for o in g.objects(themed_claim, etd.hasSourceEntity)}
        self.assertEqual(themed_sources, {theme_uri})
        self.assertTrue((Namespace(etd_base)["entity/theme/theme-10-demo"], RDF.type, etd.Theme) in g)

    def test_theme_16_and_17_do_not_collapse_to_theme_10(self):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        theme10_uri = f"{etd_base}entity/theme/theme-10-demo"
        theme16_uri = f"{etd_base}entity/theme/theme-16-demo"
        theme17_uri = f"{etd_base}entity/theme/theme-17-demo"
        concept_uri = f"{etd_base}entity/concept/demo3"

        entities = [
            Entity(uri=theme10_uri, label="Theme T10", entity_type="Theme", confidence=1.0, metadata={"discourse_id": "10", "theme_id": "10", "field": "theme"}),
            Entity(uri=theme16_uri, label="Theme T16", entity_type="Theme", confidence=1.0, metadata={"discourse_id": "1600", "theme_id": "16", "field": "theme"}),
            Entity(uri=theme17_uri, label="Theme T17", entity_type="Theme", confidence=1.0, metadata={"discourse_id": "1700", "theme_id": "17", "field": "theme"}),
            Entity(uri=concept_uri, label="Concept A", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        rel16 = Relation(
            source=theme16_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "16",
                "field": "theme.description",
                "source_doc_id": "16",
                "source": "llm",
                "source_entity_uri": theme16_uri,
            },
        )
        rel17 = Relation(
            source=theme17_uri,
            relation="hasRisk",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "17",
                "field": "theme.description",
                "source_doc_id": "17",
                "source": "reasoner",
                "source_entity_uri": theme17_uri,
            },
        )

        ann = AnnotationBatch(
            themes={
                "10": ProcessingResult(entities=[entities[0]], relations=[], metadata={}),
                "16": ProcessingResult(entities=[entities[1]], relations=[rel16], metadata={}),
                "17": ProcessingResult(entities=[entities[2]], relations=[rel17], metadata={}),
            },
            questions={},
            answers={},
            votes={},
            experts={},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        hits = {}
        for rel in g.subjects(RDF.type, etd.Relation):
            fields = {str(o) for o in g.objects(rel, etd.field)}
            preds = {str(o).split("#")[-1] for o in g.objects(rel, etd.predicate)}
            if "theme.description" not in fields:
                continue
            sources = {str(o) for o in g.objects(rel, etd.hasSourceEntity)}
            if "relatedTo" in preds:
                hits["16"] = sources
            if "hasRisk" in preds:
                hits["17"] = sources

        self.assertEqual(hits.get("16"), {theme16_uri})
        self.assertEqual(hits.get("17"), {theme17_uri})
        self.assertNotIn(theme10_uri, hits.get("16", set()))
        self.assertNotIn(theme10_uri, hits.get("17", set()))

    def test_theme_source_resolution_ignores_polluted_theme_discourse_ids(self):
        from rdflib import Namespace
        from rdflib.namespace import RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        theme10_uri = f"{etd_base}entity/theme/theme-10-demo"
        theme16_uri = f"{etd_base}entity/theme/theme-16-demo"
        theme17_uri = f"{etd_base}entity/theme/theme-17-demo"
        concept_uri = f"{etd_base}entity/concept/demo4"

        entities = [
            Entity(
                uri=theme10_uri,
                label="Theme T10",
                entity_type="Theme",
                confidence=1.0,
                metadata={
                    "discourse_id": "10",
                    "theme_id": "10",
                    "source_doc_id": "10",
                    "field": "question.body",
                    "mentions": [
                        {"discourse_id": "16", "chunk_index": 0, "start_char": 0, "end_char": 10},
                        {"discourse_id": "17", "chunk_index": 0, "start_char": 20, "end_char": 30},
                    ],
                },
            ),
            Entity(uri=theme16_uri, label="Theme T16", entity_type="Theme", confidence=1.0, metadata={"discourse_id": "16", "theme_id": "16", "source_doc_id": "16", "field": "theme"}),
            Entity(uri=theme17_uri, label="Theme T17", entity_type="Theme", confidence=1.0, metadata={"discourse_id": "17", "theme_id": "17", "source_doc_id": "17", "field": "theme"}),
            Entity(uri=concept_uri, label="Concept B", entity_type="Concept", confidence=1.0, metadata={}),
        ]

        rel16 = Relation(
            source=concept_uri,
            relation="relatedTo",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "16",
                "field": "theme.description",
                "source_doc_id": "16",
                "source": "lightweight",
            },
        )
        rel17 = Relation(
            source=concept_uri,
            relation="hasRisk",
            target=concept_uri,
            confidence=0.9,
            metadata={
                "discourse_id": "17",
                "field": "theme.description",
                "source_doc_id": "17",
                "source": "reasoner",
            },
        )

        ann = AnnotationBatch(
            themes={
                "10": ProcessingResult(entities=[entities[0]], relations=[], metadata={}),
                "16": ProcessingResult(entities=[entities[1]], relations=[rel16], metadata={}),
                "17": ProcessingResult(entities=[entities[2]], relations=[rel17], metadata={}),
            },
            questions={},
            answers={},
            votes={},
            experts={},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        hits = {}
        for rel in g.subjects(RDF.type, etd.Relation):
            fields = {str(o) for o in g.objects(rel, etd.field)}
            preds = {str(o).split("#")[-1] for o in g.objects(rel, etd.predicate)}
            if "theme.description" not in fields:
                continue
            sources = {str(o) for o in g.objects(rel, etd.hasSourceEntity)}
            if "relatedTo" in preds:
                hits["16"] = sources
            if "hasRisk" in preds:
                hits["17"] = sources

        self.assertEqual(hits.get("16"), {theme16_uri})
        self.assertEqual(hits.get("17"), {theme17_uri})
        self.assertNotIn(theme10_uri, hits.get("16", set()))
        self.assertNotIn(theme10_uri, hits.get("17", set()))


if __name__ == "__main__":
    unittest.main()
