"""
E -- predicate / schema cleanup.

Three deterministic, domain-agnostic mechanisms, all applied ONLY to
predicates with no canonical schema mapping (kept as local_unmapped):

  1. utils.predicate_map.normalize_local_predicate -- casing / punctuation
     canonicalisation + a single conservative regular '-s'/'-es'/'-ies'
     inflection fold on the final word. No '-ed', no '-ing', no stemming,
     never merges predicates differing by a content word.
  2. utils.predicate_map.is_malformed_local_predicate -- True only when
     EVERY token is an English function word (or the predicate is empty).
     PredicatePrecisionLayer drops such relations; anything with a content
     token is left unresolved.
  3. serializers.rdf_serializer._declare_local_relation_predicates -- every
     local predicate actually used gets rdf:type
     owl:DatatypeProperty | owl:ObjectProperty (from OBSERVED object usage:
     all-literal -> datatype; any resource -> object), an rdfs:label, and a
     generic rdfs:comment. No domain/range invented.

Fixtures are synthetic and domain-neutral.
"""
import importlib.util
import unittest

from config import Config
from models import Entity, Relation
from utils.predicate_map import (
    is_malformed_local_predicate,
    normalize_local_predicate,
    predicate_label_from_local_name,
)
from utils.predicate_precision import PredicatePrecisionLayer

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = "https://alfie-project.eu/etd-hub#"


# ----------------------------------------------------------------------
# 1. pure normalization / malformed helpers
# ----------------------------------------------------------------------

class NormalizeLocalPredicateTests(unittest.TestCase):
    def test_casing_and_punctuation_variants_collapse(self):
        for variant in ("shieldFrom", "shield_from", "shield-from", "Shield From", "shield  from"):
            self.assertEqual(normalize_local_predicate(variant), "shieldFrom")

    def test_safe_plural_third_person_fold(self):
        self.assertEqual(normalize_local_predicate("blocks"), "block")
        self.assertEqual(normalize_local_predicate("wardsOffThreats"), "wardOffThreat")  # every word folded
        self.assertEqual(normalize_local_predicate("relies"), "rely")          # -ies -> -y
        self.assertEqual(normalize_local_predicate("touches"), "touch")        # -ches -> drop es
        self.assertEqual(normalize_local_predicate("shieldFrom"), "shieldFrom")  # nothing to fold

    def test_no_aggressive_stemming(self):
        # '-ed' and '-ing' are NOT folded (cannot be reversed deterministically)
        self.assertEqual(normalize_local_predicate("used"), "used")
        self.assertEqual(normalize_local_predicate("analyzed"), "analyzed")
        self.assertEqual(normalize_local_predicate("showing"), "showing")
        # a content-word difference is NEVER merged away
        self.assertNotEqual(normalize_local_predicate("guards"), normalize_local_predicate("bodyGuards"))
        # singular nouns ending in 's' are not truncated
        self.assertEqual(normalize_local_predicate("bias"), "bias")
        self.assertEqual(normalize_local_predicate("analysis"), "analysis")

    def test_label_is_readable(self):
        self.assertEqual(predicate_label_from_local_name("wardsOffThreats"), "wards off threats")
        self.assertEqual(predicate_label_from_local_name("hasWidgetCount"), "has widget count")


class MalformedLocalPredicateTests(unittest.TestCase):
    def test_function_word_only_predicates_are_malformed(self):
        for p in ("is", "are", "has", "from", "in", "during", "without", "against",
                  "isNot", "ofThe", "toTheAndFrom", ""):
            self.assertTrue(is_malformed_local_predicate(p), p)

    def test_any_content_token_is_left_unresolved_not_malformed(self):
        for p in ("wardsOff", "hasWidgetCount", "isEnhancedBy", "reinforcement",
                  "comparison", "leadsToOutcome", "canTrigger"):
            self.assertFalse(is_malformed_local_predicate(p), p)

    def test_a_clause_shaped_phrase_with_a_content_word_is_not_deleted(self):
        # deterministic detection of "this is a clause fragment" is not
        # attempted -- if it carries content it stays unresolved.
        self.assertFalse(is_malformed_local_predicate("areBeingForcedToRetain"))


# ----------------------------------------------------------------------
# 2. PredicatePrecisionLayer integration
# ----------------------------------------------------------------------

class PrecisionLayerLocalPredicateTests(unittest.TestCase):
    def setUp(self):
        self.layer = PredicatePrecisionLayer()
        self.a = Entity(uri=f"{ETD}entity_a", label="Widget A", entity_type="Concept", confidence=1.0, metadata={})
        self.b = Entity(uri=f"{ETD}entity_b", label="Widget B", entity_type="Concept", confidence=1.0, metadata={})
        self.ents = [self.a, self.b]

    def _rel(self, predicate, *, literal=None, allow_local=True):
        md = {"allow_local_predicate": allow_local} if allow_local else {}
        if literal is not None:
            return Relation(source=self.a.uri, relation=predicate, target=str(literal),
                            target_literal=str(literal), target_is_literal=True, confidence=0.7, metadata=md)
        return Relation(source=self.a.uri, relation=predicate, target=self.b.uri, confidence=0.7, metadata=md)

    def test_casing_variant_local_predicates_normalize_to_one(self):
        out = self.layer.refine_relations(
            [self._rel("shield_from", literal="threats"), self._rel("ShieldFrom", literal="threats")],
            self.ents,
        )
        self.assertEqual({r.relation for r in out}, {"shieldFrom"})
        for r in out:
            self.assertEqual(r.metadata.get("predicate_mapping_status"), "local_unmapped")

    def test_safe_morphological_variant_local_predicates_normalize_to_one(self):
        out = self.layer.refine_relations(
            [self._rel("wardOff", literal="x"), self._rel("wardsOff", literal="x")],
            self.ents,
        )
        self.assertEqual({r.relation for r in out}, {"wardOff"})

    def test_function_word_only_predicate_relation_is_dropped(self):
        out = self.layer.refine_relations(
            [self._rel("is", literal="important"),
             self._rel("shieldFrom", literal="threats")],
            self.ents,
        )
        self.assertEqual([r.relation for r in out], ["shieldFrom"])

    def test_malformed_predicate_is_not_silently_downgraded_to_relatedTo(self):
        out = self.layer.refine_relations([self._rel("during", literal="setup")], self.ents)
        self.assertEqual(out, [])  # dropped, not turned into relatedTo

    def test_ambiguous_content_predicate_is_retained_unresolved(self):
        out = self.layer.refine_relations([self._rel("reinforcement", literal="patterns")], self.ents)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].relation, "reinforcement")
        self.assertEqual(out[0].metadata.get("predicate_mapping_status"), "local_unmapped")

    def test_local_predicate_without_allow_flag_still_downgrades_to_relatedTo(self):
        out = self.layer.refine_relations([self._rel("wardsOff", allow_local=False)], self.ents)
        self.assertEqual(out[0].relation, "relatedTo")

    def test_mapped_predicates_are_untouched(self):
        out = self.layer.refine_relations(
            [Relation(source=self.a.uri, relation="causes", target=self.b.uri, confidence=0.8, metadata={})],
            self.ents,
        )
        self.assertEqual(out[0].relation, "causes")


# ----------------------------------------------------------------------
# 3. serializer local-predicate declaration
# ----------------------------------------------------------------------

class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


def _annotations(relations, discourse_id="d1"):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations, "metadata": {"text": "some source text"}}},
        "votes": {}, "experts": {}, "documents": {},
        "metadata": {"datasetId": "unit_ds", "runId": "run-unit"},
    }


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class LocalPredicateDeclarationTests(unittest.TestCase):
    def _serialize(self, relations, entities):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        s = RDFSerializer(
            annotations_dict=_annotations(relations),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        return s, Namespace(ETD)

    def _entity(self, slug, label):
        return Entity(uri=f"{ETD}{slug}", label=label, entity_type="Concept", confidence=0.9,
                      metadata={"surface_form": label})

    def test_resource_valued_local_predicate_declared_object_property(self):
        from rdflib import RDF, OWL, RDFS
        a, b = self._entity("d_a", "Alpha"), self._entity("d_b", "Beta")
        rel = {"source": a.uri, "relation": "shieldsFrom", "target": b.uri, "confidence": 0.8,
               "metadata": {"discourse_id": "d1", "field": "answer.description",
                            "predicate_mapping_status": "local_unmapped", "relation_kind": "extracted_semantic"}}
        s, ns = self._serialize([rel], [a, b])
        p = ns.shieldsFrom
        self.assertIn((p, RDF.type, OWL.ObjectProperty), s.graph)
        self.assertNotIn((p, RDF.type, OWL.DatatypeProperty), s.graph)
        self.assertTrue(list(s.graph.objects(p, RDFS.label)))
        self.assertTrue(list(s.graph.objects(p, RDFS.comment)))
        self.assertEqual(str(next(s.graph.objects(p, RDFS.label))), "shields from")

    def test_literal_valued_local_predicate_declared_datatype_property(self):
        from rdflib import RDF, OWL, RDFS
        a = self._entity("d_c", "Gamma")
        rel = {"source": a.uri, "relation": "carriesLabel", "target": "restricted use",
               "target_literal": "restricted use", "target_is_literal": True, "confidence": 0.8,
               "metadata": {"discourse_id": "d1", "field": "answer.description",
                            "predicate_mapping_status": "local_unmapped", "relation_kind": "extracted_semantic"}}
        s, ns = self._serialize([rel], [a])
        p = ns.carriesLabel
        self.assertIn((p, RDF.type, OWL.DatatypeProperty), s.graph)
        self.assertNotIn((p, RDF.type, OWL.ObjectProperty), s.graph)
        self.assertTrue(list(s.graph.objects(p, RDFS.label)))
        self.assertTrue(list(s.graph.objects(p, RDFS.comment)))

    def test_mixed_object_and_literal_usage_declared_object_property_and_reported(self):
        from rdflib import RDF, OWL
        a, b = self._entity("m_a", "A"), self._entity("m_b", "B")
        rels = [
            {"source": a.uri, "relation": "notesAbout", "target": b.uri, "confidence": 0.8,
             "metadata": {"discourse_id": "d1", "field": "answer.description",
                          "predicate_mapping_status": "local_unmapped", "relation_kind": "extracted_semantic"}},
            {"source": b.uri, "relation": "notesAbout", "target": "a caveat", "target_literal": "a caveat",
             "target_is_literal": True, "confidence": 0.8,
             "metadata": {"discourse_id": "d1", "field": "answer.description",
                          "predicate_mapping_status": "local_unmapped", "relation_kind": "extracted_semantic"}},
        ]
        s, ns = self._serialize(rels, [a, b])
        self.assertIn((ns.notesAbout, RDF.type, OWL.ObjectProperty), s.graph)
        self.assertIn("notesAbout", s.local_predicate_declaration["mixed_use"])

    def test_existing_schema_predicate_is_not_redeclared(self):
        from rdflib import RDF, OWL
        a, b = self._entity("x_a", "A"), self._entity("x_b", "B")
        rel = {"source": a.uri, "relation": "causes", "target": b.uri, "confidence": 0.8,
               "metadata": {"discourse_id": "d1", "field": "answer.description",
                            "relation_kind": "extracted_semantic"}}
        s, ns = self._serialize([rel], [a, b])
        # causes is already an ObjectProperty from the schema; not counted as newly declared
        self.assertNotIn("causes", s.local_predicate_declaration["declared"])


# ----------------------------------------------------------------------
# E gate side effects: lifecycle metadata + class/property local-name
# collision on DYNAMICALLY declared local relation predicates.
# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class DynamicPredicateGovernanceTests(unittest.TestCase):
    def _serialize(self, relations, entities):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer(
            annotations_dict=_annotations(relations),
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        return s, Namespace(ETD)

    def _entity(self, slug, label):
        return Entity(uri=f"{ETD}{slug}", label=label, entity_type="Concept", confidence=0.9,
                      metadata={"surface_form": label})

    def _rel(self, pred, src, tgt, *, literal=False):
        md = {"discourse_id": "d1", "field": "answer.description",
              "predicate_mapping_status": "local_unmapped", "relation_kind": "extracted_semantic"}
        r = {"source": src, "relation": pred, "confidence": 0.8, "metadata": md}
        if literal:
            r.update(target=tgt, target_literal=tgt, target_is_literal=True)
        else:
            r.update(target=tgt)
        return r

    def _lifecycle_ok(self, g, ns, p):
        from ontology.add_schema import LIFECYCLE_STATUSES
        statuses = [str(o) for o in g.objects(p, ns.lifecycleStatus)]
        self.assertEqual(len(statuses), 1, f"{p} must carry exactly one lifecycleStatus, got {statuses}")
        self.assertIn(statuses[0], LIFECYCLE_STATUSES)
        # non-active status must carry a rationale note (the gate requires it)
        if statuses[0] != "active":
            self.assertTrue(list(g.objects(p, ns.lifecycleNote)),
                            f"{p} status {statuses[0]} needs a lifecycleNote")

    def test_dynamic_object_property_gets_lifecycle_metadata(self):
        from rdflib import RDF, OWL
        a, b = self._entity("g_a", "Alpha"), self._entity("g_b", "Beta")
        s, ns = self._serialize([self._rel("shieldsFrom", a.uri, b.uri)], [a, b])
        p = ns.shieldsFrom
        self.assertIn((p, RDF.type, OWL.ObjectProperty), s.graph)
        self._lifecycle_ok(s.graph, ns, p)

    def test_dynamic_datatype_property_gets_lifecycle_metadata(self):
        from rdflib import RDF, OWL
        a = self._entity("g_c", "Gamma")
        s, ns = self._serialize([self._rel("carriesTag", a.uri, "restricted", literal=True)], [a])
        p = ns.carriesTag
        self.assertIn((p, RDF.type, OWL.DatatypeProperty), s.graph)
        self._lifecycle_ok(s.graph, ns, p)

    def test_mixed_use_local_predicate_gets_lifecycle_metadata(self):
        from rdflib import RDF, OWL
        a, b = self._entity("g_d", "Delta"), self._entity("g_e", "Epsilon")
        rels = [self._rel("notesAbout", a.uri, b.uri),
                self._rel("notesAbout", b.uri, "a caveat", literal=True)]
        s, ns = self._serialize(rels, [a, b])
        p = ns.notesAbout
        self.assertIn((p, RDF.type, OWL.ObjectProperty), s.graph)
        self.assertIn("notesAbout", s.local_predicate_declaration["mixed_use"])
        self._lifecycle_ok(s.graph, ns, p)

    def test_dynamic_predicate_colliding_with_existing_class_is_not_declared(self):
        from rdflib import RDF, OWL
        # 'harm' normalises to the same local name as the declared ETD class
        # 'Harm' -> must NOT be emitted as a colliding ETD property.
        a, b = self._entity("g_f", "Zeta"), self._entity("g_g", "Eta")
        s, ns = self._serialize([self._rel("harm", a.uri, b.uri)], [a, b])
        p = ns.harm
        for t in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
            self.assertNotIn((p, RDF.type, t), s.graph)
        self.assertNotIn("harm", s.local_predicate_declaration["declared"])
        self.assertIn("harm", s.local_predicate_declaration["collision_skipped"])
        # the class stays; no class/property local-name collision exists
        self.assertIn((ns.Harm, RDF.type, OWL.Class), s.graph)
        self._assert_no_unexpected_local_name_collision(s.graph, ns)

    def test_non_colliding_dynamic_predicate_declared_normally_with_lifecycle(self):
        from rdflib import RDF, OWL
        a, b = self._entity("g_h", "Theta"), self._entity("g_i", "Iota")
        s, ns = self._serialize([self._rel("mitigatesExposure", a.uri, b.uri)], [a, b])
        p = ns.mitigatesExposure
        self.assertIn((p, RDF.type, OWL.ObjectProperty), s.graph)
        self.assertIn("mitigatesExposure", s.local_predicate_declaration["declared"])
        self.assertNotIn("mitigatesExposure", s.local_predicate_declaration["collision_skipped"])
        self._lifecycle_ok(s.graph, ns, p)

    def test_static_schema_properties_unchanged(self):
        from rdflib import RDF, OWL
        from ontology.add_schema import _lifecycle_for_property
        a, b = self._entity("g_j", "Kappa"), self._entity("g_k", "Lambda")
        s, ns = self._serialize([self._rel("shieldsFrom", a.uri, b.uri)], [a, b])
        g = s.graph
        for local in ("causes", "belongsToTheme", "castBy"):
            p = ns[local]
            self.assertIn((p, RDF.type, OWL.ObjectProperty), g)
            self.assertNotIn(local, s.local_predicate_declaration["declared"])
            statuses = sorted({str(o) for o in g.objects(p, ns.lifecycleStatus)})
            self.assertEqual(statuses, [_lifecycle_for_property(local)[0]])

    def test_whole_graph_satisfies_gate_lifecycle_and_collision_invariants(self):
        from rdflib import RDF, OWL, RDFS
        from ontology.add_schema import LIFECYCLE_STATUSES
        a, b = self._entity("g_l", "Mu"), self._entity("g_m", "Nu")
        rels = [
            self._rel("shieldsFrom", a.uri, b.uri),
            self._rel("carriesTag", a.uri, "x", literal=True),
            self._rel("notesAbout", a.uri, b.uri),
            self._rel("notesAbout", b.uri, "c", literal=True),
            self._rel("harm", a.uri, b.uri),          # collides with class Harm
            self._rel("bias", a.uri, b.uri),          # collides with class Bias
            self._rel("mitigatesExposure", a.uri, b.uri),
        ]
        s, ns = self._serialize(rels, [a, b])
        g = s.graph
        etd_pref = str(ns)
        # (1) every declared ETD property has a valid lifecycleStatus
        declared_props = set()
        for t in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
            for p in g.subjects(RDF.type, t):
                if str(p).startswith(etd_pref):
                    declared_props.add(p)
        for p in declared_props:
            vals = [str(o) for o in g.objects(p, ns.lifecycleStatus)]
            self.assertTrue(vals and vals[0] in LIFECYCLE_STATUSES,
                            f"{p} missing/invalid lifecycleStatus: {vals}")
        # (2) no unexpected class/property local-name collision
        self._assert_no_unexpected_local_name_collision(g, ns)
        # collisions were skipped, not declared
        self.assertEqual(sorted(s.local_predicate_declaration["collision_skipped"]), ["bias", "harm"])

    def _assert_no_unexpected_local_name_collision(self, g, ns):
        from rdflib import RDF, OWL
        from ontology.add_schema import normalized_etd_local_name
        pref = str(ns)
        buckets = {}
        for kind, t in (("class", OWL.Class), ("property", OWL.ObjectProperty),
                        ("property", OWL.DatatypeProperty), ("property", RDF.Property)):
            for term in g.subjects(RDF.type, t):
                if not str(term).startswith(pref):
                    continue
                local = str(term)[len(pref):]
                buckets.setdefault(normalized_etd_local_name(local), set()).add((kind, local))
        allow = {"source": {("class", "Source"), ("property", "source")}}
        for key, vals in buckets.items():
            if len(vals) <= 1:
                continue
            if key in allow and vals == allow[key]:
                continue
            self.fail(f"unexpected ETD local-name collision {key!r}: {sorted(vals)}")


if __name__ == "__main__":
    unittest.main()
