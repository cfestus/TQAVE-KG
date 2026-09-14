import importlib.util
import unittest

from models import AnnotationBatch, Entity, ProcessingResult, Relation


HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF integration tests")
class BackboneContractIntegrationTests(unittest.TestCase):
    def test_backbone_relations_use_reified_model_and_no_dangling_endpoints(self):
        from rdflib import Namespace, RDF, URIRef
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        q_uri = f"{etd_base}question_q1"
        t_uri = f"{etd_base}theme_t1"
        a_uri = f"{etd_base}answer_a1"
        e_uri = f"{etd_base}expert_e1"
        v_uri = f"{etd_base}vote_v1"

        entities = [
            Entity(uri=q_uri, label="Question q1", entity_type="Question", confidence=1.0, metadata={}),
            Entity(uri=t_uri, label="Theme t1", entity_type="Theme", confidence=1.0, metadata={}),
            Entity(uri=a_uri, label="Answer a1", entity_type="Answer", confidence=1.0, metadata={}),
            Entity(uri=e_uri, label="Expert e1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri=v_uri, label="Vote v1", entity_type="Vote", confidence=1.0, metadata={}),
        ]

        rels = [
            Relation(source=q_uri, relation="belongsToTheme", target=t_uri, confidence=0.99, metadata={"discourse_id": "q1"}),
            Relation(source=q_uri, relation="askedBy", target=e_uri, confidence=0.99, metadata={"discourse_id": "q1"}),
            Relation(source=a_uri, relation="answersQuestion", target=q_uri, confidence=0.99, metadata={"discourse_id": "a1"}),
            Relation(source=a_uri, relation="authoredBy", target=e_uri, confidence=0.99, metadata={"discourse_id": "a1"}),
            Relation(source=t_uri, relation="createdBy", target=e_uri, confidence=0.99, metadata={"discourse_id": "t1"}),
            Relation(source=v_uri, relation="castBy", target=e_uri, confidence=0.99, metadata={"discourse_id": "v1"}),
            Relation(source=a_uri, relation="receivesVote", target=v_uri, confidence=0.99, metadata={"discourse_id": "v1"}),
        ]

        ann = AnnotationBatch(
            questions={"q1": ProcessingResult(entities=[entities[0]], relations=rels[0:2], metadata={})},
            answers={"a1": ProcessingResult(entities=[entities[2]], relations=rels[2:4], metadata={})},
            votes={"v1": ProcessingResult(entities=[entities[4]], relations=rels[5:], metadata={})},
            themes={"t1": ProcessingResult(entities=[entities[1]], relations=[rels[4]], metadata={})},
            experts={"e1": ProcessingResult(entities=[entities[3]], relations=[], metadata={})},
        )

        # Explicit graph construction happens in RDFSerializer initialization.
        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        graph = serializer.graph
        self.assertGreater(len(graph), 0)

        def assert_reified(source_uri: str, predicate_local: str, target_uri: str) -> None:
            source_ref = URIRef(source_uri)
            pred_ref = URIRef(f"{etd_base}{predicate_local}")
            target_ref = URIRef(target_uri)

            found = False
            for rel_node in graph.subjects(RDF.type, etd.Relation):
                if (
                    (rel_node, etd.subject, source_ref) in graph
                    and (rel_node, etd.predicate, pred_ref) in graph
                    and (rel_node, etd.object, target_ref) in graph
                ):
                    found = True
                    break
            self.assertTrue(
                found,
                f"Missing reified relation for {source_uri} --{predicate_local}--> {target_uri}",
            )

        assert_reified(q_uri, "belongsToTheme", t_uri)
        assert_reified(q_uri, "askedBy", e_uri)
        assert_reified(a_uri, "answersQuestion", q_uri)
        assert_reified(a_uri, "authoredBy", e_uri)
        assert_reified(t_uri, "createdBy", e_uri)
        assert_reified(v_uri, "castBy", e_uri)
        assert_reified(a_uri, "receivesVote", v_uri)

        self.assertIn((URIRef(q_uri), etd.askedBy, URIRef(e_uri)), graph)
        self.assertIn((URIRef(t_uri), etd.createdBy, URIRef(e_uri)), graph)

        # Hard backbone range invariants:
        # - castBy object must be Expert/Person
        # - receivesVote object must be Vote
        for obj in graph.objects(URIRef(v_uri), etd.castBy):
            self.assertTrue(
                (obj, RDF.type, etd.Expert) in graph or (obj, RDF.type, etd.Person) in graph,
                f"castBy object is not Expert/Person: {obj}",
            )
        for obj in graph.objects(URIRef(a_uri), etd.receivesVote):
            self.assertTrue(
                (obj, RDF.type, etd.Vote) in graph,
                f"receivesVote object is not Vote: {obj}",
            )
        self.assertEqual(
            len(list(graph.objects(URIRef(v_uri), etd.receivesVote))),
            0,
            "Legacy Vote -> Answer receivesVote direct triple must not remain",
        )

        # Every direct vote backbone edge must also be reified.
        pred_ref = URIRef(f"{etd_base}castBy")
        direct_edges = list(graph.triples((URIRef(v_uri), pred_ref, None)))
        reified_count = 0
        for rel_node in graph.subjects(RDF.type, etd.Relation):
            if (
                (rel_node, etd.subject, URIRef(v_uri)) in graph
                and (rel_node, etd.predicate, pred_ref) in graph
            ):
                reified_count += 1
        self.assertEqual(
            len(direct_edges),
            reified_count,
            f"Direct vs reified count mismatch for castBy: {len(direct_edges)} vs {reified_count}",
        )

        pred_ref = URIRef(f"{etd_base}receivesVote")
        direct_edges = list(graph.triples((URIRef(a_uri), pred_ref, None)))
        reified_count = 0
        for rel_node in graph.subjects(RDF.type, etd.Relation):
            if (
                (rel_node, etd.subject, URIRef(a_uri)) in graph
                and (rel_node, etd.predicate, pred_ref) in graph
            ):
                reified_count += 1
        self.assertEqual(
            len(direct_edges),
            reified_count,
            f"Direct vs reified count mismatch for receivesVote: {len(direct_edges)} vs {reified_count}",
        )
        legacy_reified = [
            rel_node
            for rel_node in graph.subjects(RDF.type, etd.Relation)
            if (rel_node, etd.predicate, URIRef(f"{etd_base}receivesVote")) in graph
            and (rel_node, etd.subject, URIRef(v_uri)) in graph
            and (rel_node, etd.object, URIRef(a_uri)) in graph
        ]
        self.assertEqual(legacy_reified, [], "Legacy Vote -> Answer receivesVote reified relation must not remain")

        # RDF-level no-dangling-endpoints check:
        entity_uris = {str(e.uri) for e in entities}
        for rel_node in graph.subjects(RDF.type, etd.Relation):
            subjects = list(graph.objects(rel_node, etd.subject))
            objects = list(graph.objects(rel_node, etd.object))
            self.assertGreaterEqual(len(subjects), 1, f"Relation {rel_node} missing etd:subject")
            self.assertGreaterEqual(len(objects), 1, f"Relation {rel_node} missing etd:object")
            for s in subjects:
                if isinstance(s, URIRef):
                    self.assertIn(str(s), entity_uris, f"Dangling relation subject: {s}")
            for o in objects:
                if isinstance(o, URIRef):
                    self.assertIn(str(o), entity_uris, f"Dangling relation object: {o}")

    def test_backbone_type_invariant_raises_on_invalid_vote_edge(self):
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        vote_uri = f"{etd_base}vote_v1"
        bad_uri = f"{etd_base}vote_v2"

        entities = [
            Entity(uri=vote_uri, label="Vote v1", entity_type="Vote", confidence=1.0, metadata={}),
            Entity(uri=bad_uri, label="Vote v2", entity_type="Vote", confidence=1.0, metadata={}),
        ]

        ann = AnnotationBatch(
            votes={
                "v1": ProcessingResult(
                    entities=entities,
                    relations=[
                        Relation(
                            source=vote_uri,
                            relation="castBy",
                            target=bad_uri,
                            confidence=0.99,
                            metadata={"discourse_id": "v1"},
                        )
                    ],
                    metadata={},
                )
            }
        )

        with self.assertRaises(RuntimeError):
            RDFSerializer(
                annotations_dict=ann.to_dict(),
                registry=_Registry(entities),
                namespaces=Config.NAMESPACES,
            )

    def test_semantic_vote_nodes_have_deterministic_labels(self):
        from rdflib import Namespace, RDF, URIRef
        from rdflib.namespace import RDFS, SKOS
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        e_uri = f"{etd_base}expert_e1"
        a_uri = f"{etd_base}answer_a1"
        entities = [
            Entity(uri=e_uri, label="Expert e1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri=a_uri, label="Answer a1", entity_type="Answer", confidence=1.0, metadata={}),
        ]

        rel = Relation(
            source=e_uri,
            relation="endorses",
            target=a_uri,
            confidence=0.9,
            metadata={"extractor": "vote", "context_source": "dataset_vote", "discourse_id": "v1"},
        )
        ann = AnnotationBatch(
            votes={"v1": ProcessingResult(entities=[], relations=[rel], metadata={})},
            experts={"e1": ProcessingResult(entities=[entities[0]], relations=[], metadata={})},
            answers={"a1": ProcessingResult(entities=[entities[1]], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        graph = serializer.graph

        vote_nodes = list(graph.subjects(RDF.type, etd.Vote))
        self.assertGreaterEqual(len(vote_nodes), 1, "Expected at least one materialized Vote node")
        for v in vote_nodes:
            self.assertFalse(
                str(v).startswith(f"{etd_base}vote_"),
                f"Vote node must not use legacy local vote_* URI: {v}",
            )
            labels = list(graph.objects(v, RDFS.label))
            pref_labels = list(graph.objects(v, SKOS.prefLabel))
            self.assertGreaterEqual(len(labels), 1, f"Vote node missing rdfs:label: {v}")
            self.assertGreaterEqual(len(pref_labels), 1, f"Vote node missing skos:prefLabel: {v}")
            self.assertIn("Vote by", str(labels[0]))

    def test_semantic_vote_backbone_edges_are_reified_1_to_1(self):
        from rdflib import Namespace, RDF, URIRef
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        e_uri = f"{etd_base}expert_e1"
        a_uri = f"{etd_base}answer_a1"
        entities = [
            Entity(uri=e_uri, label="Expert e1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri=a_uri, label="Answer a1", entity_type="Answer", confidence=1.0, metadata={}),
        ]
        rel = Relation(
            source=e_uri,
            relation="endorses",
            target=a_uri,
            confidence=0.9,
            metadata={"extractor": "vote", "context_source": "dataset_vote", "discourse_id": "v1"},
        )
        ann = AnnotationBatch(
            votes={"v1": ProcessingResult(entities=[], relations=[rel], metadata={})},
            experts={"e1": ProcessingResult(entities=[entities[0]], relations=[], metadata={})},
            answers={"a1": ProcessingResult(entities=[entities[1]], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        graph = serializer.graph

        for pred_local in ("castBy", "castVote", "receivesVote"):
            pred_ref = URIRef(f"{etd_base}{pred_local}")
            direct = list(graph.triples((None, pred_ref, None)))
            reified = 0
            for rel_node in graph.subjects(RDF.type, etd.Relation):
                if (rel_node, etd.predicate, pred_ref) in graph:
                    reified += 1
            self.assertEqual(
                len(direct),
                reified,
                f"{pred_local} direct/reified mismatch: {len(direct)} vs {reified}",
            )

    def test_all_votes_have_exactly_one_castBy_and_shacl_castby_constraint_passes(self):
        from rdflib import Namespace, RDF, URIRef
        from config import Config
        from serializers.rdf_serializer import RDFSerializer

        class _Registry:
            def __init__(self, entities):
                self._entities = entities

            def get_all_entities(self):
                return self._entities

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        e_uri = f"{etd_base}expert_e1"
        a_uri = f"{etd_base}answer_a1"
        entities = [
            Entity(uri=e_uri, label="Expert e1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri=a_uri, label="Answer a1", entity_type="Answer", confidence=1.0, metadata={}),
        ]
        # Semantic vote relation triggers vote-node materialization; castBy must be
        # materialized from inverse castVote and enforced to exactly one per Vote.
        rel = Relation(
            source=e_uri,
            relation="endorses",
            target=a_uri,
            confidence=0.9,
            metadata={"extractor": "vote", "context_source": "dataset_vote", "discourse_id": "v1"},
        )
        ann = AnnotationBatch(
            votes={"v1": ProcessingResult(entities=[], relations=[rel], metadata={})},
            experts={"e1": ProcessingResult(entities=[entities[0]], relations=[], metadata={})},
            answers={"a1": ProcessingResult(entities=[entities[1]], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            config=Config,
        )
        graph = serializer.graph

        vote_nodes = list(graph.subjects(RDF.type, etd.Vote))
        self.assertGreaterEqual(len(vote_nodes), 1, "Expected at least one Vote node")

        for vote in vote_nodes:
            self.assertFalse(
                str(vote).startswith(f"{etd_base}vote_"),
                f"Vote node must not use legacy local vote_* URI: {vote}",
            )
            cast_by_vals = list(graph.objects(vote, etd.castBy))
            self.assertEqual(
                len(cast_by_vals),
                1,
                f"Vote must have exactly one castBy; vote={vote} values={cast_by_vals}",
            )
            self.assertTrue(
                (cast_by_vals[0], RDF.type, etd.Expert) in graph or (cast_by_vals[0], RDF.type, etd.Person) in graph,
                f"castBy object must be Expert/Person; vote={vote} object={cast_by_vals[0]}",
            )

        # Optional SHACL check if pyshacl is present.
        if importlib.util.find_spec("pyshacl") is not None:
            import tempfile
            from validation.shacl_validator import SHACLValidator

            validator = SHACLValidator(output_dir=tempfile.mkdtemp(prefix="skg_shacl_"))
            shacl = validator.validate_graph(graph)
            self.assertIn("results_text", shacl)
            self.assertNotIn(
                "Vote MUST have exactly one castBy (Person IRI).",
                shacl.get("results_text", ""),
                "SHACL castBy violation still present",
            )


if __name__ == "__main__":
    unittest.main()
