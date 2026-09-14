import unittest

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from models import Entity
from serializers.rdf_serializer import RDFSerializer


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


class _Config:
    NAMESPACES = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "prov": "http://www.w3.org/ns/prov#",
        "etd": "https://alfie-project.eu/etd-hub#",
        "onto": "https://alfie-project.eu/ontology#",
    }
    PRUNE_ISOLATED_ENTITIES = False
    MATERIALIZE_RAW_FIELDS = True
    MATERIALIZE_SENSITIVE_FIELDS = False
    MEDIA_BASE_URL = ""


class RawFieldMaterializationTests(unittest.TestCase):
    def _serializer_with(self, entities):
        ann = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {"datasetId": "d1", "runId": "r1"},
        }
        return RDFSerializer(
            annotations_dict=ann,
            registry=_Registry(entities),
            namespaces=_Config.NAMESPACES,
            config=_Config,
        )

    def test_expert_materialization_respects_sensitive_flag_and_types(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#expert_1",
            label="Expert 1",
            entity_type="Expert",
            metadata={
                "raw_fields": {
                    "area_of_expertise": "AI Governance",
                    "date_joined": "2025-07-20T08:34:34Z",
                    "profile_picture": "imgs/experts/1.jpeg",
                    "bio": "Sensitive text",
                }
            },
        )
        s = self._serializer_with([ent])
        g = s.graph
        etd = Namespace(_Config.NAMESPACES["etd"])
        u = URIRef(ent.uri)

        self.assertTrue((u, etd.areaOfExpertise, Literal("AI Governance", datatype=XSD.string)) in g)
        joined = list(g.objects(u, etd.dateJoined))
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].datatype, XSD.dateTime)
        self.assertTrue((u, etd.profilePicture, Literal("imgs/experts/1.jpeg", datatype=XSD.string)) in g)
        self.assertEqual(len(list(g.objects(u, etd.bio))), 0)

        field_nodes = list(g.objects(u, etd.hasFieldAssertion))
        self.assertGreaterEqual(len(field_nodes), 3)
        area_field = None
        for fn in field_nodes:
            if (fn, etd.fieldName, Literal("area_of_expertise", datatype=XSD.string)) in g:
                area_field = fn
                break
        self.assertIsNotNone(area_field)
        self.assertTrue((area_field, etd.assertedPredicate, etd.areaOfExpertise) in g)
        self.assertTrue((area_field, etd.assertedValue, Literal("AI Governance", datatype=XSD.string)) in g)
        self.assertTrue((area_field, etd.datasetId, Literal("d1", datatype=XSD.string)) in g)
        self.assertTrue((area_field, etd.runId, Literal("r1", datatype=XSD.string)) in g)

        bio_fields = [
            fn for fn in field_nodes
            if (fn, etd.fieldName, Literal("bio", datatype=XSD.string)) in g
        ]
        self.assertEqual(len(bio_fields), 0)

    def test_vote_created_updated_are_datetime(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#vote_11",
            label="Vote 11",
            entity_type="Vote",
            metadata={
                "raw_fields": {
                    "created_at": "2025-07-20T08:34:34Z",
                    "updated_at": "2025-07-21T09:10:11Z",
                }
            },
        )
        s = self._serializer_with([ent])
        g = s.graph
        etd = Namespace(_Config.NAMESPACES["etd"])
        u = URIRef(ent.uri)

        created_vals = list(g.objects(u, etd.createdAt))
        updated_vals = list(g.objects(u, etd.updatedAt))
        self.assertEqual(len(created_vals), 1)
        self.assertEqual(len(updated_vals), 1)
        self.assertEqual(created_vals[0].datatype, XSD.dateTime)
        self.assertEqual(updated_vals[0].datatype, XSD.dateTime)

        created_field = None
        updated_field = None
        for fn in g.objects(u, etd.hasFieldAssertion):
            if (fn, etd.fieldName, Literal("created_at", datatype=XSD.string)) in g:
                created_field = fn
            if (fn, etd.fieldName, Literal("updated_at", datatype=XSD.string)) in g:
                updated_field = fn
        self.assertIsNotNone(created_field)
        self.assertIsNotNone(updated_field)
        self.assertTrue((created_field, etd.assertedPredicate, etd.createdAt) in g)
        self.assertTrue((updated_field, etd.assertedPredicate, etd.updatedAt) in g)

    def test_theme_created_at_and_views_datatypes(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#theme_9",
            label="Theme 9",
            entity_type="Theme",
            metadata={
                "raw_fields": {
                    "created_at": "2025-07-20T08:42:30Z",
                    "views": "12",
                }
            },
        )
        s = self._serializer_with([ent])
        g = s.graph
        etd = Namespace(_Config.NAMESPACES["etd"])
        u = URIRef(ent.uri)

        created_vals = list(g.objects(u, etd.createdAt))
        views_vals = list(g.objects(u, etd.views))
        self.assertEqual(len(created_vals), 1)
        self.assertEqual(created_vals[0].datatype, XSD.dateTime)
        self.assertEqual(len(views_vals), 1)
        self.assertEqual(views_vals[0].datatype, XSD.integer)
        self.assertEqual(str(views_vals[0]), "12")

    def test_field_assertion_iris_are_deterministic(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#expert_2",
            label="Expert 2",
            entity_type="Expert",
            metadata={
                "raw_fields": {
                    "id": 2,
                    "area_of_expertise": "Policy",
                    "date_joined": "2025-01-01T00:00:00Z",
                }
            },
        )
        s1 = self._serializer_with([ent])
        s2 = self._serializer_with([ent])
        etd = Namespace(_Config.NAMESPACES["etd"])

        f1 = sorted(
            [str(fn) for fn in s1.graph.subjects(RDF.type, etd.FieldAssertion)],
            key=str,
        )
        f2 = sorted(
            [str(fn) for fn in s2.graph.subjects(RDF.type, etd.FieldAssertion)],
            key=str,
        )
        self.assertEqual(f1, f2)
        self.assertEqual(len(f1), len(f2))

    def test_field_assertions_have_exactly_one_en_label(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#expert_3",
            label="Expert 3",
            entity_type="Expert",
            metadata={
                "raw_fields": {
                    "id": 3,
                    "area_of_expertise": "Responsible AI",
                    "date_joined": "2025-01-01T00:00:00Z",
                    "profile_picture": "imgs/experts/3.jpeg",
                }
            },
        )
        s = self._serializer_with([ent])
        g = s.graph
        etd = Namespace(_Config.NAMESPACES["etd"])

        field_nodes = list(g.subjects(RDF.type, etd.FieldAssertion))
        self.assertGreater(len(field_nodes), 0)
        for fn in field_nodes:
            labels = list(g.objects(fn, RDFS.label))
            self.assertEqual(len(labels), 1, f"FieldAssertion label cardinality != 1: {fn}")
            self.assertIsInstance(labels[0], Literal)
            self.assertEqual(labels[0].language, "en", f"FieldAssertion label lang != en: {fn}")

    def test_field_assertion_negative_chunk_sets_unknown_and_origin(self):
        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#vote_52",
            label="Vote 52",
            entity_type="Vote",
            metadata={
                "discourse_id": "52",
                "field": "vote.vote_value",
                "source": "structural",
                "source_doc_id": "52",
                "chunk_index": -1,
                "chunk_index_unknown": True,
                "raw_fields": {
                    "vote_value": 1,
                },
            },
        )
        s = self._serializer_with([ent])
        g = s.graph
        etd = Namespace(_Config.NAMESPACES["etd"])

        field_nodes = list(g.subjects(RDF.type, etd.FieldAssertion))
        self.assertGreater(len(field_nodes), 0)
        target = None
        for fn in field_nodes:
            if (fn, etd.fieldName, Literal("vote_value", datatype=XSD.string)) in g:
                target = fn
                break
        self.assertIsNotNone(target)
        chunk_vals = list(g.objects(target, etd.chunk_index))
        self.assertEqual(len(chunk_vals), 1)
        self.assertEqual(str(chunk_vals[0]), "-1")
        unknown_vals = list(g.objects(target, etd.chunk_index_unknown))
        self.assertEqual(len(unknown_vals), 1)
        self.assertEqual(str(unknown_vals[0]).lower(), "true")
        self.assertTrue(any(True for _ in g.objects(target, etd.extractedFrom)))


if __name__ == "__main__":
    unittest.main()

