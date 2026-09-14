import importlib.util
import unittest

from models import AnnotationBatch, Entity, ProcessingResult, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF integration tests")
class RelationProvenanceCompletionTests(unittest.TestCase):
    def test_final_relation_provenance_completion_sets_chunk_and_run_dataset(self):
        from rdflib import Namespace, RDF, Literal
        from rdflib.namespace import XSD
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        vote_uri = f"{etd_base}vote_v1"
        expert_uri = f"{etd_base}expert_e1"
        answer_uri = f"{etd_base}answer_a1"
        question_uri = f"{etd_base}question_q1"

        entities = [
            Entity(uri=vote_uri, label="Vote v1", entity_type="Vote", confidence=1.0, metadata={"discourse_id": "v1", "chunk_index": 3}),
            Entity(uri=expert_uri, label="Expert e1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri=answer_uri, label="Answer a1", entity_type="Answer", confidence=1.0, metadata={"discourse_id": "a1", "chunk_index": 7}),
            Entity(uri=question_uri, label="Question q1", entity_type="Question", confidence=1.0, metadata={"discourse_id": "q1", "chunk_index": 5}),
        ]

        rels_votes = [
            Relation(source=vote_uri, relation="castBy", target=expert_uri, confidence=0.99, metadata={"discourse_id": "v1", "extractor": "structural"}),
            Relation(source=answer_uri, relation="receivesVote", target=vote_uri, confidence=0.99, metadata={"discourse_id": "v1", "extractor": "structural"}),
            Relation(source=expert_uri, relation="endorses", target=answer_uri, confidence=0.90, metadata={"discourse_id": "v1", "extractor": "vote"}),
        ]
        rels_answers = [
            Relation(source=answer_uri, relation="answersQuestion", target=question_uri, confidence=0.95, metadata={"discourse_id": "a1", "extractor": "structural"}),
        ]

        ann = AnnotationBatch(
            votes={"v1": ProcessingResult(entities=[entities[0], entities[1], entities[2]], relations=rels_votes, metadata={})},
            answers={"a1": ProcessingResult(entities=[entities[2], entities[3]], relations=rels_answers, metadata={})},
            experts={"e1": ProcessingResult(entities=[entities[1]], relations=[], metadata={})},
            questions={"q1": ProcessingResult(entities=[entities[3]], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        rel_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertGreater(len(rel_nodes), 0)

        for r in rel_nodes:
            chunks = list(g.objects(r, etd.chunk_index))
            self.assertEqual(len(chunks), 1, f"Relation missing/duplicate chunk_index: {r}")
            self.assertIsInstance(chunks[0], Literal)
            self.assertEqual(chunks[0].datatype, XSD.integer, f"Relation chunk_index datatype not xsd:integer: {r}")
            self.assertEqual(len(list(g.objects(r, etd.source))), 1, f"Relation source cardinality != 1: {r}")
            self.assertEqual(len(list(g.objects(r, etd.source_doc_id))), 1, f"Relation source_doc_id cardinality != 1: {r}")

            ds_vals = list(g.objects(r, etd.datasetId))
            run_vals = list(g.objects(r, etd.runId))
            self.assertEqual(len(ds_vals), 1, f"Relation datasetId cardinality != 1: {r}")
            self.assertEqual(len(run_vals), 1, f"Relation runId cardinality != 1: {r}")
            self.assertEqual(len(list(g.objects(r, etd.groundingStatus))), 1, f"Relation groundingStatus cardinality != 1: {r}")
            self.assertEqual(len(list(g.objects(r, etd.groundingMethod))), 1, f"Relation groundingMethod cardinality != 1: {r}")

        # Vote backbone relation families must carry runId/datasetId in this model extension.
        vote_predicates = {etd.castBy, etd.receivesVote, etd.castVote}
        for r in rel_nodes:
            pred = next((p for p in g.objects(r, etd.predicate)), None)
            if pred in vote_predicates:
                self.assertEqual(len(list(g.objects(r, etd.datasetId))), 1)
                self.assertEqual(len(list(g.objects(r, etd.runId))), 1)
                self.assertEqual(len(list(g.objects(r, etd.groundingStatus))), 1)
                self.assertEqual(len(list(g.objects(r, etd.groundingMethod))), 1)

    def test_multi_chunk_evidence_relation_does_not_keep_span_map_with_unknown_chunk(self):
        from rdflib import Namespace, URIRef, Literal, RDF
        from rdflib.namespace import XSD
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def get_all_entities(self):
                return []

        etd = Namespace(Config.NAMESPACES["etd"])
        serializer = RDFSerializer(
            annotations_dict={"metadata": {"datasetId": "d1", "runId": "r1"}},
            registry=_Registry(),
            namespaces=Config.NAMESPACES,
            dataset_id="d1",
            run_id="r1",
        )
        g = serializer.graph

        rel = URIRef(f"{Config.NAMESPACES['etd']}rel_test_multi_chunk")
        ev1 = URIRef(f"{Config.NAMESPACES['etd']}evidence_test_1")
        ev2 = URIRef(f"{Config.NAMESPACES['etd']}evidence_test_2")
        subj = URIRef(f"{Config.NAMESPACES['etd']}entity_s")
        obj = URIRef(f"{Config.NAMESPACES['etd']}entity_o")
        pred = etd.relatedTo

        g.add((rel, RDF.type, etd.Relation))
        g.add((rel, etd.subject, subj))
        g.add((rel, etd.predicate, pred))
        g.add((rel, etd.object, obj))
        g.add((rel, etd.groundingStatus, Literal("grounded", datatype=XSD.string)))
        g.add((rel, etd.groundingMethod, Literal("span_map", datatype=XSD.string)))

        for ev, ci in ((ev1, 0), (ev2, 1)):
            g.add((ev, RDF.type, etd.Evidence))
            g.add((rel, etd.hasEvidence, ev))
            g.add((ev, etd.chunk_index, Literal(ci, datatype=XSD.integer)))
            g.add((ev, etd.chunk_index_unknown, Literal(False, datatype=XSD.boolean)))
            g.add((ev, etd.groundingStatus, Literal("grounded", datatype=XSD.string)))
            g.add((ev, etd.groundingMethod, Literal("span_map", datatype=XSD.string)))
            g.add((ev, etd.source, Literal("reasoner", datatype=XSD.string)))
            g.add((ev, etd.source_doc_id, Literal("29", datatype=XSD.string)))
            g.add((ev, etd.datasetId, Literal("d1", datatype=XSD.string)))
            g.add((ev, etd.runId, Literal("r1", datatype=XSD.string)))

        serializer._finalize_relation_provenance(etd)

        self.assertEqual(str(next(g.objects(rel, etd.chunk_index))), "-1")
        self.assertEqual(str(next(g.objects(rel, etd.chunk_index_unknown))).lower(), "true")
        self.assertEqual(str(next(g.objects(rel, etd.groundingStatus))), "grounded")
        self.assertEqual(str(next(g.objects(rel, etd.groundingMethod))), "multi_chunk_evidence")


if __name__ == "__main__":
    unittest.main()
