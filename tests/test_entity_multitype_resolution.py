import importlib.util
import unittest

from config import Config
from models import Entity

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self.entities = {e.uri: e for e in entities}

    def get_all_entities(self):
        return list(self.entities.values())


def _make_graph(entities):
    from serializers.rdf_serializer import RDFSerializer

    reg = _Registry(entities)
    ann = {
        "themes": {}, "questions": {}, "answers": {},
        "votes": {}, "experts": {}, "documents": {},
    }
    s = RDFSerializer(
        annotations_dict=ann, registry=reg, namespaces=Config.NAMESPACES,
        prune_isolated_entities=False,
    )
    return s.graph, s


def _entity_types(g, etd, uri):
    from rdflib import URIRef, RDF
    return {
        str(t).rsplit("#", 1)[-1]
        for t in g.objects(URIRef(uri), RDF.type)
    } - {"Entity"}


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for entity-multitype tests")
class SemanticTyperDirectTests(unittest.TestCase):
    """Exercise ontology.semantic_typing.SemanticTyper._add_type directly."""

    def _typer(self):
        from rdflib import Graph, Namespace
        from ontology.add_schema import add_schema
        from ontology.semantic_typing import SemanticTyper

        g = Graph()
        etd = Namespace(Config.NAMESPACES["etd"])
        allowed = add_schema(g, etd)
        return g, etd, SemanticTyper(g, etd, allowed_classes=allowed)

    def _types(self, g, etd, uri):
        from rdflib import RDF
        return {str(t).rsplit("#", 1)[-1] for t in g.objects(uri, RDF.type)} - {"Entity"}

    def test_subclass_and_superclass_typing_remains_valid(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_subclass_superclass"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.DemographicGroup))
        st.classify(uri, "black")
        types = self._types(g, etd, uri)
        self.assertIn("DemographicGroup", types)
        self.assertIn("PopulationGroup", types)

    def test_retyping_removes_obsolete_sibling_type(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_retyped_analysis"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.Analysis))
        st.classify(uri, "analysis of the underlying dataset")
        types = self._types(g, etd, uri)
        self.assertIn("Analysis", types)
        self.assertNotIn("Dataset", types)

    def test_model_does_not_accumulate_aisystem_from_earlier_proposal(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_model_no_accum"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.Model))
        st.classify(uri, "commercial pricing model")
        types = self._types(g, etd, uri)
        self.assertIn("Model", types)
        self.assertNotIn("AISystem", types)
        self.assertNotIn("SystemComponent", types)

    def test_process_retyping_removes_incompatible_system_type(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_process_no_system"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.Process))
        st.classify(uri, "the underlying algorithmic system audit process")
        types = self._types(g, etd, uri)
        self.assertIn("Process", types)
        self.assertNotIn("AISystem", types)
        self.assertNotIn("SystemComponent", types)

    def test_orthogonal_compatible_typing_can_survive(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_auditprocess_evaluation"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.AuditProcess))
        st.classify(uri, "audit assessment")
        types = self._types(g, etd, uri)
        self.assertIn("AuditProcess", types)
        self.assertIn("Evaluation", types, "Evaluation is an ontology ancestor of AuditProcess and may legitimately co-occur")

    def test_no_primary_type_lets_fallback_lexical_typing_fire(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_untyped_fallback"]
        g.add((uri, RDF.type, etd.Entity))
        st.classify(uri, "fairness")
        types = self._types(g, etd, uri)
        self.assertIn("EthicalPrinciple", types)

    def test_superseded_proposed_types_remain_traceable(self):
        from rdflib import RDF
        g, etd, st = self._typer()
        uri = etd["e_traceable"]
        g.add((uri, RDF.type, etd.Entity))
        g.add((uri, RDF.type, etd.Model))
        st.classify(uri, "risk model")
        superseded = {str(o) for o in g.objects(uri, etd.supersededProposedType)}
        self.assertTrue(superseded & {"AISystem", "Bias", "Risk", "Harm", "Impact"})
        types = self._types(g, etd, uri)
        self.assertNotIn("Bias", types)
        self.assertNotIn("Risk", types)

    def test_ontology_inference_still_produces_legitimate_supertypes(self):
        from rdflib import RDFS
        from ontology.add_schema import add_schema
        from rdflib import Graph, Namespace
        g = Graph()
        etd = Namespace(Config.NAMESPACES["etd"])
        add_schema(g, etd)
        self.assertIn((etd.DemographicGroup, RDFS.subClassOf, etd.PopulationGroup), g)
        self.assertIn((etd.AuditProcess, RDFS.subClassOf, etd.Evaluation), g)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for entity-multitype tests")
class FullSerializerMultiTypeTests(unittest.TestCase):
    """Exercise the fix through the real RDFSerializer entity-metadata path."""

    def test_canonical_uri_type_and_ontology_state_remain_consistent(self):
        etd_base = Config.NAMESPACES["etd"]
        ent = Entity(
            uri=f"{etd_base}entity_model/coherence-check",
            label="commercial risk model",
            entity_type="Model", confidence=0.9,
            metadata={"ontology_uri": "https://w3id.org/aipo#Model"},
        )
        g, s = _make_graph([ent])
        etd_ns = __import__("rdflib").Namespace(etd_base)
        uri = __import__("rdflib").URIRef(ent.uri)
        types = _entity_types(g, etd_ns, ent.uri)
        self.assertIn("Model", types)
        self.assertNotIn("AISystem", types)
        ontology_uris = list(g.objects(uri, etd_ns.ontology_uri))
        self.assertTrue(ontology_uris, "Model entity must still carry its ontology_uri grounding")

    def test_entity_with_no_hidden_labels_is_still_classified_exactly_once(self):
        # Regression test for the pre-existing indentation defect: an entity
        # with zero hidden labels must still be run through SemanticTyper
        # exactly once. Entity.entity_type cannot be empty, so this uses the
        # pipeline's own generic fallback bucket ("Concept") as the primary
        # type; since Concept is not ontology-compatible with EthicalPrinciple,
        # the proposal is correctly suppressed from rdf:type but must still
        # be recorded as a superseded proposal -- proving classify() actually
        # ran despite zero hidden labels (under the old bug it would not
        # have run at all, leaving no trace either way).
        etd_base = Config.NAMESPACES["etd"]
        ent = Entity(
            uri=f"{etd_base}concept/fairness-notion",
            label="fairness", entity_type="Concept", confidence=0.9, metadata={},
        )
        g, s = _make_graph([ent])
        etd_ns = __import__("rdflib").Namespace(etd_base)
        uri = __import__("rdflib").URIRef(ent.uri)
        superseded = {str(o) for o in g.objects(uri, etd_ns.supersededProposedType)}
        self.assertIn("EthicalPrinciple", superseded)

    def test_entity_with_multiple_hidden_labels_is_not_multiply_classified(self):
        etd_base = Config.NAMESPACES["etd"]
        ent = Entity(
            uri=f"{etd_base}model/repeat-classify-check",
            label="commercial risk model", entity_type="Model", confidence=0.9,
            metadata={"aliases": ["risk model alias one", "risk model alias two", "risk model alias three"]},
        )
        g, s = _make_graph([ent])
        etd_ns = __import__("rdflib").Namespace(etd_base)
        types = _entity_types(g, etd_ns, ent.uri)
        self.assertEqual(types, {"Model"})


class _CanonicalLinkerRegistry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return list(self.entities.values())

    def replace_entity(self, old_uri, new_uri):
        if old_uri in self.entities and new_uri not in self.entities:
            self.entities[new_uri] = self.entities[old_uri]


class Task941RegressionTests(unittest.TestCase):
    """Confirm the external-ID conflict protection is unaffected."""

    def test_incompatible_same_external_id_entities_are_not_unioned(self):
        import tempfile

        from utils.canonical_linker import CanonicalLinker

        reg = _CanonicalLinkerRegistry()
        external = {"wikidata_uri": "https://www.wikidata.org/entity/Qtest12345"}
        e1 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_model_one",
            label="Some Mention", entity_type="Model", confidence=0.9,
            metadata=dict(external),
        )
        e2 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_bias_two",
            label="Some Mention", entity_type="Bias", confidence=0.9,
            metadata=dict(external),
        )
        reg.entities = {e1.uri: e1, e2.uri: e2}

        # Test-isolation fix: a literal "output" resolved to the real
        # production output/ directory, overwriting its live
        # canonical_linking_log.jsonl with this test's synthetic data.
        with tempfile.TemporaryDirectory() as tmpdir:
            linker = CanonicalLinker(registry=reg, output_dir=tmpdir)
            out = linker.run_full_registry()

        # Incompatible confident types must not be silently merged/unioned
        # onto one canonical entity -- both remain, each with its own
        # original type intact.
        self.assertEqual(len(out), 2)
        types = {o.entity_type for o in out}
        self.assertEqual(types, {"Model", "Bias"})


if __name__ == "__main__":
    unittest.main()
