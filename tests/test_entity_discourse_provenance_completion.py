import importlib.util
import unittest

from models import AnnotationBatch, Entity, ProcessingResult

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF integration tests")
class EntityDiscourseProvenanceCompletionTests(unittest.TestCase):
    def test_entities_and_discourse_units_receive_minimum_provenance(self):
        from rdflib import Namespace, RDF
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        class _Cfg:
            NAMESPACES = Config.NAMESPACES
            PRUNE_ISOLATED_ENTITIES = False
            STRICT_ETD_PROPERTY_REGISTRY = True
            MATERIALIZE_RAW_FIELDS = False
            MATERIALIZE_SENSITIVE_FIELDS = False
            PIPELINE_VERSION = "test"

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        entity_uri = f"{etd_base}entity_bias"
        ent = Entity(
            uri=entity_uri,
            label="Bias",
            entity_type="Bias",
            confidence=0.9,
            metadata={
                "discourse_id": "14",
                "field": "answer.description",
                "mentions": [{"discourse_id": "14", "chunk_index": 0, "start_char": 5, "end_char": 9}],
            },
        )
        ann = AnnotationBatch(
            answers={
                "14": ProcessingResult(
                    entities=[ent],
                    relations=[],
                    metadata={"text": "Bias in answer.", "discourse_id": "14", "field": "answer.description"},
                )
            }
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry([ent]),
            namespaces=Config.NAMESPACES,
            config=_Cfg,
            dataset_id="ds_test",
            run_id="run_test",
        )
        g = serializer.graph

        ent_ref = Namespace(etd_base).entity_bias
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.source)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.extractedFrom)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.chunk_index)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.chunk_index_unknown)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.surface_form)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.discourse_id)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.context_source)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.datasetId)))
        self.assertTrue(any(True for _ in g.objects(ent_ref, etd.runId)))

        discourse_units = list(g.subjects(RDF.type, etd.DiscourseUnit))
        self.assertGreaterEqual(len(discourse_units), 1)
        for du in discourse_units:
            self.assertTrue(any(True for _ in g.objects(du, etd.source_doc_id)))
            self.assertTrue(any(True for _ in g.objects(du, etd.chunk_index)))
            self.assertTrue(any(True for _ in g.objects(du, etd.chunk_index_unknown)))
            self.assertTrue(any(True for _ in g.objects(du, etd.field)))
            self.assertTrue(any(True for _ in g.objects(du, etd.extractedFrom)))
            self.assertTrue(any(True for _ in g.objects(du, etd.chunk_id)))
            self.assertTrue(any(True for _ in g.objects(du, etd.discourse_id)))
            self.assertTrue(any(True for _ in g.objects(du, etd.datasetId)))
            self.assertTrue(any(True for _ in g.objects(du, etd.runId)))


if __name__ == "__main__":
    unittest.main()
