import unittest

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from serializers.rdf_serializer import enforce_single_en_labels


class ExportLabelEnforcementTests(unittest.TestCase):
    def test_export_node_gets_single_en_label_from_dataset_and_run(self):
        etd = Namespace("https://alfie-project.eu/etd-hub#")
        g = Graph()

        export_node = URIRef(f"{etd}export_abc")
        g.add((export_node, etd.datasetId, Literal("ai_ethics_datasets")))
        g.add((export_node, etd.runId, Literal("run-123")))
        g.add((export_node, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), etd.Export))

        enforce_single_en_labels(g, etd)

        labels = list(g.objects(export_node, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].language, "en")
        self.assertEqual(str(labels[0]), "Export: ai_ethics_datasets (run-123)")

    def test_serializer_build_path_emits_export_label_en(self):
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def get_all_entities(self):
                return []

        ann = {
            "themes": {"t1": {"entities": [], "relations": []}},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {"datasetId": "ai_ethics_datasets", "runId": "run-xyz"},
        }

        s = RDFSerializer(annotations_dict=ann, registry=_Registry(), namespaces=Config.NAMESPACES, config=Config)
        g = s.graph
        etd = Namespace(Config.NAMESPACES["etd"])

        export_nodes = list(g.subjects(RDF.type, etd.Export))
        self.assertEqual(len(export_nodes), 1)
        labels = list(g.objects(export_nodes[0], RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].language, "en")


if __name__ == "__main__":
    unittest.main()

