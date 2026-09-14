import json
import unittest
from pathlib import Path

from config import Config
from models import AnnotationBatch
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from rdflib import Namespace, URIRef
from serializers.rdf_serializer import RDFSerializer
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline._discourse_chunk_spans = {}
    return pipeline


class VotePipelineGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_vote_fixture_materializes_backbone_only(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "ai_ethics_datasets.json"
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        votes = payload.get("Vote", [])
        self.assertEqual(len(votes), 26)

        vote_nodes = 0
        cast_by = 0
        receives_vote = 0
        semantic_vote_relations = 0

        for vote in votes:
            result = self.pipeline.process_vote(vote)
            vote_nodes += sum(1 for entity in result.entities if entity.entity_type == "Vote")
            cast_by += sum(1 for relation in result.relations if relation.relation == "castBy")
            receives_vote += sum(1 for relation in result.relations if relation.relation == "receivesVote")
            semantic_vote_relations += sum(
                1
                for relation in result.relations
                if relation.metadata.get("field") == "vote.vote_value"
                and relation.relation not in {"castBy", "receivesVote"}
            )
            self.assertTrue(all(relation.metadata.get("relation_kind") == "structural" for relation in result.relations))

        self.assertEqual(vote_nodes, 26)
        self.assertEqual(cast_by, 26)
        self.assertEqual(receives_vote, 26)
        self.assertEqual(semantic_vote_relations, 0)

    def test_processed_vote_serializes_vote_values_and_timestamps(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "ai_ethics_datasets.json"
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        vote = payload.get("Vote", [])[0]
        result = self.pipeline.process_vote(vote)
        vote_entity = next(entity for entity in result.entities if entity.entity_type == "Vote")

        ann = AnnotationBatch(votes={str(vote["id"]): result})
        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=self.pipeline.global_registry,
            namespaces=Config.NAMESPACES,
            config=Config,
        )
        graph = serializer.graph
        etd = Namespace(Config.NAMESPACES["etd"])
        vote_ref = URIRef(vote_entity.uri)
        answer_ref = URIRef(next(entity.uri for entity in result.entities if entity.entity_type == "Answer"))

        self.assertEqual(len(list(graph.objects(vote_ref, etd.rawVoteValue))), 1)
        self.assertEqual(len(list(graph.objects(vote_ref, etd.voteValue))), 1)
        self.assertEqual(len(list(graph.objects(vote_ref, etd.createdAt))), 1)
        self.assertEqual(len(list(graph.objects(vote_ref, etd.updatedAt))), 1)
        self.assertEqual(len(list(graph.objects(answer_ref, etd.receivesVote))), 1)


if __name__ == "__main__":
    unittest.main()
