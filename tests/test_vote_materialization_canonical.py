import unittest

from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from config import Config
from models import AnnotationBatch, Entity, ProcessingResult, Relation
from serializers.rdf_serializer import RDFSerializer


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


class VoteMaterializationCanonicalTests(unittest.TestCase):
    def test_semantic_vote_relation_reuses_canonical_vote_entity(self):
        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        prov = Namespace(Config.NAMESPACES["prov"])

        vote_uri = f"{etd_base.rstrip('#/')}/entity/vote/vote-59-a1842491"
        expert_uri = f"{etd_base}entity/person/expert-31-e1c8b0ad"
        answer_uri = f"{etd_base}entity/answer/answer-11-e9cce9fe"

        vote_entity = Entity(
            uri=vote_uri,
            label="Vote 59",
            entity_type="Vote",
            confidence=1.0,
            ontology_uri=Config.NAMESPACES["prov"] + "Activity",
            metadata={
                "discourse_id": "59",
                "context_source": "dataset_vote",
                "raw_fields": {
                    "id": 59,
                    "expert_id": 31,
                    "answer_id": 11,
                    "vote_value": 1,
                },
            },
        )
        expert_entity = Entity(uri=expert_uri, label="Expert 31", entity_type="Expert", confidence=1.0, metadata={})
        answer_entity = Entity(uri=answer_uri, label="Answer 11", entity_type="Answer", confidence=1.0, metadata={})

        rel = Relation(
            source=expert_uri,
            relation="endorses",
            target=answer_uri,
            confidence=0.9,
            metadata={
                "extractor": "vote",
                "context_source": "dataset_vote",
                "discourse_id": "59",
                "field": "vote.vote_value",
                "vote_value": "1",
                "raw_vote_value": 1,
                "vote_value_concept": f"{etd_base}VoteValueSchemeUpvote",
                "origin_key": "vote.vote_value|59",
                "source_doc_id": "59",
            },
        )

        ann = AnnotationBatch(
            votes={"59": ProcessingResult(entities=[vote_entity], relations=[rel], metadata={"vote_id": 59})},
            experts={"31": ProcessingResult(entities=[expert_entity], relations=[], metadata={})},
            answers={"11": ProcessingResult(entities=[answer_entity], relations=[], metadata={})},
        )

        serializer = RDFSerializer(
            annotations_dict=ann.to_dict(),
            registry=_Registry([vote_entity, expert_entity, answer_entity]),
            namespaces=Config.NAMESPACES,
            config=Config,
        )
        graph = serializer.graph

        vote_nodes = sorted(graph.subjects(RDF.type, etd.Vote), key=str)
        self.assertEqual(len(vote_nodes), 1)
        self.assertEqual(vote_nodes[0], URIRef(vote_uri))

        local_vote_nodes = [
            node for node in vote_nodes
            if str(node).startswith(f"{etd_base}vote_")
        ]
        self.assertEqual(local_vote_nodes, [])

        self.assertIn((URIRef(vote_uri), RDF.type, etd.Entity), graph)
        self.assertIn((URIRef(vote_uri), RDF.type, prov.Activity), graph)

        cast_by_vals = list(graph.objects(URIRef(vote_uri), etd.castBy))
        receives_vals = list(graph.objects(URIRef(answer_uri), etd.receivesVote))
        raw_vals = list(graph.objects(URIRef(vote_uri), etd.rawVoteValue))
        scheme_vals = list(graph.objects(URIRef(vote_uri), etd.voteValue))
        polarity_vals = list(graph.objects(URIRef(vote_uri), etd.polarity))
        field_vals = list(graph.objects(URIRef(vote_uri), etd.field))
        fa_vals = list(graph.objects(URIRef(vote_uri), etd.hasFieldAssertion))
        did_vals = list(graph.objects(URIRef(vote_uri), etd.discourse_id))

        self.assertEqual(len(cast_by_vals), 1)
        self.assertEqual(len(receives_vals), 1)
        self.assertEqual(len(raw_vals), 1)
        self.assertEqual(len(scheme_vals), 1)
        self.assertEqual(len(polarity_vals), 1)
        self.assertEqual(len(field_vals), 1)
        self.assertGreaterEqual(len(fa_vals), 1)
        self.assertEqual(len(did_vals), 1)
        self.assertEqual(len(list(graph.objects(URIRef(vote_uri), etd.receivesVote))), 0)

        self.assertEqual(cast_by_vals[0], URIRef(expert_uri))
        self.assertEqual(receives_vals[0], URIRef(vote_uri))
        self.assertEqual(str(raw_vals[0]), "1")
        self.assertEqual(scheme_vals[0], URIRef(f"{etd_base}VoteValueSchemeUpvote"))
        self.assertEqual(str(polarity_vals[0]), "1")
        self.assertEqual(str(field_vals[0]), "vote.vote_value")
        self.assertEqual(str(did_vals[0]), "59")
        self.assertTrue(
            any((fa, etd.assertedPredicate, etd.rawVoteValue) in graph for fa in fa_vals),
            "Canonical vote should preserve raw-field provenance using etd:rawVoteValue",
        )

        labels = list(graph.objects(URIRef(vote_uri), RDFS.label))
        pref_labels = list(graph.objects(URIRef(vote_uri), SKOS.prefLabel))
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(pref_labels), 1)
        self.assertIn("Vote by Expert 31 for Answer 11", str(labels[0]))
        self.assertIn("Vote by Expert 31 for Answer 11", str(pref_labels[0]))

        self.assertEqual(len(list(graph.objects(URIRef(vote_uri), etd.voteValueRaw))), 0)

        for pred_local in ("castBy", "castVote", "receivesVote"):
            pred_ref = URIRef(f"{etd_base}{pred_local}")
            direct = list(graph.triples((None, pred_ref, None)))
            reified = 0
            for rel_node in graph.subjects(RDF.type, etd.Relation):
                if (rel_node, etd.predicate, pred_ref) in graph:
                    reified += 1
                    for endpoint in list(graph.objects(rel_node, etd.subject)) + list(graph.objects(rel_node, etd.object)):
                        self.assertFalse(
                            str(endpoint).startswith(f"{etd_base}vote_"),
                            f"Vote backbone relation still points to legacy local vote node: {endpoint}",
                        )
            self.assertEqual(len(direct), reified)

        receives_reified = [
            rel_node for rel_node in graph.subjects(RDF.type, etd.Relation)
            if (rel_node, etd.predicate, etd.receivesVote) in graph
        ]
        self.assertEqual(len(receives_reified), 1)
        self.assertIn((receives_reified[0], etd.subject, URIRef(answer_uri)), graph)
        self.assertIn((receives_reified[0], etd.object, URIRef(vote_uri)), graph)


if __name__ == "__main__":
    unittest.main()
