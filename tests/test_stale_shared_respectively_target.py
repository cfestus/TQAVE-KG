"""
Defect A -- stale shared target identity after an upstream respectively/
value split.

Sibling per-subject relations already carry DISTINCT single-value
``object_text`` values, but their ``target`` still points at ONE shared
compound value entity because the "respectively" cue was stripped from
their own evidence before ``_split_shared_respectively_values`` could see
it. ``_rebind_stale_shared_respectively_targets`` re-establishes distinct
targets, pairing subject -> value strictly by SOURCE-TEXT ORDER.

No live LLM/network call; all extractor output is hand-built. The named
sentence ("Black and Latino borrowers pay 7.9 and 3.6 basis points more
respectively") is a regression fixture only -- the correction logic keys
on structure (a compound-of-values label, a respectively cue in the
source, distinct single scalar object values), never on any label,
value, discourse id or domain term.
"""
import importlib.util
import unittest

from config import Config
from models import Entity, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]


class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


def _base_annotations(relations, discourse_id="9"):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations}},
        "votes": {}, "experts": {}, "documents": {},
        "metadata": {},
    }


def _bare_pipeline():
    from pipeline import EnhancedSemanticPipeline
    from utils.helpers import GlobalEntityRegistry
    from ontology.mapper import OntologyMapper
    from utils.entity_normalizer import EntityNormalizer

    p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    p.config = Config
    p.global_registry = GlobalEntityRegistry()
    p.ontology_mapper = OntologyMapper()
    p.entity_normalizer = EntityNormalizer(p.global_registry)
    return p


def _register(pipeline, label, entity_type="Concept", discourse_id="9"):
    ent = Entity(
        uri=f"{ETD}seed-{label}", label=label, entity_type=entity_type,
        confidence=0.9, metadata={"discourse_id": discourse_id, "surface_form": label},
    )
    pipeline._ensure_entity_ontology_grounding(ent)
    ent.uri = pipeline.entity_normalizer.canonical_uri(
        label=ent.label, entity_type=ent.entity_type, ontology_uri=getattr(ent, "ontology_uri", None),
    )
    return pipeline.global_registry.get_or_create(ent)


# The confirmed defect-A shape: object_text ALREADY split to single
# values, evidence carries NO "respectively" cue, but both relations still
# point at the one compound entity.
_SOURCE = "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively."
_STRIPPED_EVIDENCE = "Black and Latino borrowers pay 7.9 and 3.6 basis points more."


def _sibling_relations(compound_uri, subj_a_uri, subj_b_uri, *, evidence=_STRIPPED_EVIDENCE,
                       obj_a="7.9 basis points more", obj_b="3.6 basis points more"):
    rel_a = Relation(
        source=subj_a_uri, relation="pay", target=str(compound_uri), confidence=0.92,
        metadata={
            "subject_text": "Black borrowers", "object_text": obj_a,
            "discourse_id": "9", "field": "theme.description",
            "evidence": evidence, "evidence_quote": evidence,
        },
    )
    rel_b = Relation(
        source=subj_b_uri, relation="pay", target=str(compound_uri), confidence=0.92,
        metadata={
            "subject_text": "Latino borrowers", "object_text": obj_b,
            "discourse_id": "9", "field": "theme.description",
            "evidence": evidence, "evidence_quote": evidence,
        },
    )
    return rel_a, rel_b


class RebindUnitTests(unittest.TestCase):
    def _setup(self, **rel_kwargs):
        p = _bare_pipeline()
        a = _register(p, "Black Borrowers", "PopulationGroup")
        b = _register(p, "Latino Borrowers", "PopulationGroup")
        compound = _register(p, "7.9 and 3.6 basis points more", "Metric")
        local = [a, b, compound]
        rel_a, rel_b = _sibling_relations(compound.uri, a.uri, b.uri, **rel_kwargs)
        return p, local, compound, a, b, rel_a, rel_b

    def test_stale_shared_target_is_rebound_in_source_order(self):
        p, local, compound, a, b, rel_a, rel_b = self._setup()
        self.assertEqual(rel_a.target, rel_b.target, "sanity: both start on the one compound entity")

        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=_SOURCE, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": _SOURCE}, unit_text=_SOURCE,
        )

        self.assertNotEqual(rel_a.target, rel_b.target, "targets must now be distinct")
        self.assertNotEqual(rel_a.target, str(compound.uri))
        self.assertNotEqual(rel_b.target, str(compound.uri))
        self.assertTrue(rel_a.target.startswith("http") and rel_b.target.startswith("http"))
        self.assertFalse(rel_a.target_is_literal)
        self.assertIsNone(rel_a.target_literal)
        # source order: "Black ... 7.9", "Latino ... 3.6"
        self.assertEqual(rel_a.metadata["object_text"], "7.9 basis points more")
        self.assertEqual(rel_b.metadata["object_text"], "3.6 basis points more")
        self.assertEqual(rel_a.metadata["object_surface_form"], "7.9 basis points more")
        events = [e for e in p.quality_events if e["category"] == "shared_compound_respectively_rebind"]
        self.assertEqual(len(events), 2)
        # the stale compound entity is dropped from the local pool
        self.assertNotIn(str(compound.uri), [str(getattr(e, "uri", "")) for e in local])

    def test_pairing_follows_source_order_not_relation_list_order(self):
        # object_text values swapped on the relations; source order must win.
        p, local, compound, a, b, rel_a, rel_b = self._setup(
            obj_a="3.6 basis points more", obj_b="7.9 basis points more",
        )
        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=_SOURCE, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": _SOURCE}, unit_text=_SOURCE,
        )
        # "Black" precedes "Latino"; "7.9" precedes "3.6" -> Black gets 7.9.
        self.assertEqual(rel_a.metadata["object_text"], "7.9 basis points more")
        self.assertEqual(rel_b.metadata["object_text"], "3.6 basis points more")
        self.assertNotEqual(rel_a.target, rel_b.target)

    def test_no_correction_without_a_respectively_cue(self):
        p, local, compound, a, b, rel_a, rel_b = self._setup()
        no_cue = "Black and Latino borrowers pay 7.9 and 3.6 basis points more."
        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=no_cue, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": no_cue}, unit_text=no_cue,
        )
        self.assertEqual(rel_a.target, str(compound.uri))
        self.assertEqual(rel_b.target, str(compound.uri))
        self.assertEqual(p.quality_events, [])

    def test_no_correction_when_source_order_cannot_be_established(self):
        # A value appears BEFORE both subject mentions -> subjects do not
        # all precede all values -> order unsafe -> leave untouched.
        p, local, compound, a, b, rel_a, rel_b = self._setup()
        jumbled = "At 7.9 and 3.6 basis points more respectively, Black and Latino borrowers pay extra."
        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=jumbled, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": jumbled}, unit_text=jumbled,
        )
        self.assertEqual(rel_a.target, str(compound.uri))
        self.assertEqual(rel_b.target, str(compound.uri))
        self.assertEqual(p.quality_events, [])

    def test_no_correction_when_shared_target_is_legitimately_shared(self):
        # Two relations genuinely sharing one ordinary concept target
        # (label is not a compound of values; object_text is not scalar).
        p = _bare_pipeline()
        a = _register(p, "Automated Screening", "AISystem")
        b = _register(p, "Manual Review", "Process")
        shared = _register(p, "Hiring Outcomes", "Concept")
        local = [a, b, shared]
        src = "Both automated screening and manual review shape hiring outcomes respectively."
        rel_a = Relation(
            source=a.uri, relation="affects", target=str(shared.uri), confidence=0.9,
            metadata={"subject_text": "automated screening", "object_text": "hiring outcomes",
                      "discourse_id": "9", "field": "theme.description",
                      "evidence": src, "evidence_quote": src},
        )
        rel_b = Relation(
            source=b.uri, relation="affects", target=str(shared.uri), confidence=0.9,
            metadata={"subject_text": "manual review", "object_text": "hiring outcomes",
                      "discourse_id": "9", "field": "theme.description",
                      "evidence": src, "evidence_quote": src},
        )
        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=src, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": src}, unit_text=src,
        )
        self.assertEqual(rel_a.target, str(shared.uri))
        self.assertEqual(rel_b.target, str(shared.uri))
        self.assertEqual(p.quality_events, [])

    def test_no_correction_when_object_values_are_not_distinct(self):
        # Both siblings still carry the SAME single value -> this is not
        # the "already split" shape; leave it to the other pass.
        p, local, compound, a, b, rel_a, rel_b = self._setup(
            obj_a="7.9 basis points more", obj_b="7.9 basis points more",
        )
        p._rebind_stale_shared_respectively_targets(
            [rel_a, rel_b], local, discourse_text=_SOURCE, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": _SOURCE}, unit_text=_SOURCE,
        )
        self.assertEqual(rel_a.target, str(compound.uri))
        self.assertEqual(rel_b.target, str(compound.uri))
        self.assertEqual(p.quality_events, [])


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class RebindEndToEndSerializationTests(unittest.TestCase):
    """Production path: hand-built entities/relations in the confirmed
    defect-A shape -> the real ``_resolve_semantic_unit_relations`` ->
    ``RDFSerializer`` -> assertions on the SERIALIZED graph's actual
    ``etd:subject``/``etd:object`` bindings."""

    def test_serialized_graph_has_distinct_ordered_value_targets(self):
        from rdflib import Namespace, RDF, RDFS, URIRef
        from serializers.rdf_serializer import RDFSerializer

        p = _bare_pipeline()
        black = _register(p, "Black Borrowers", "PopulationGroup")
        latino = _register(p, "Latino Borrowers", "PopulationGroup")
        compound = _register(p, "7.9 and 3.6 basis points more", "Metric")
        local = [black, latino, compound]

        rel_a, rel_b = _sibling_relations(compound.uri, black.uri, latino.uri)
        unit = {"start_offset": 0, "text": _SOURCE, "section": "Ethical Challenges"}
        resolved = p._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=local, discourse_text=_SOURCE,
            discourse_id="9", field="theme.description", origin_key="theme.description|9",
            unit=unit, unit_text=_SOURCE,
        )

        def _rel_dict(r):
            return {
                "uri": r.uri, "source": r.source, "relation": r.relation, "target": r.target,
                "confidence": r.confidence, "target_literal": r.target_literal,
                "target_is_literal": r.target_is_literal, "metadata": r.metadata,
            }

        ann = _base_annotations([_rel_dict(r) for r in resolved], discourse_id="9")
        registry_entities = list(p.global_registry.get_all_entities())
        serializer = RDFSerializer(
            annotations_dict=ann, registry=_Registry(registry_entities), namespaces=Config.NAMESPACES,
        )
        g = serializer.graph
        etd_ns = Namespace(ETD)

        pay_rels = [r for r in g.subjects(RDF.type, etd_ns.Relation)
                    if (r, etd_ns.predicate, etd_ns.pay) in g]
        self.assertEqual(len(pay_rels), 2, "exactly two 'pay' relations must survive")

        by_subject = {}
        for rel in pay_rels:
            subj = list(g.objects(rel, etd_ns.subject))[0]
            obj = list(g.objects(rel, etd_ns.object))[0]
            label = None
            if isinstance(obj, URIRef):
                labels = [str(v) for v in g.objects(obj, RDFS.label)]
                label = labels[0] if labels else None
            by_subject[str(subj)] = {"object": obj, "label": label}

        black_o = by_subject[str(black.uri)]
        latino_o = by_subject[str(latino.uri)]

        # distinct targets, neither is the stale compound
        self.assertNotEqual(black_o["object"], latino_o["object"])
        self.assertNotEqual(str(black_o["object"]), str(compound.uri))
        self.assertNotEqual(str(latino_o["object"]), str(compound.uri))
        self.assertIsInstance(black_o["object"], URIRef)
        self.assertIsInstance(latino_o["object"], URIRef)

        # correct source order: Black -> 7.9, Latino -> 3.6, not swapped
        black_repr = str(black_o["label"] or black_o["object"])
        latino_repr = str(latino_o["label"] or latino_o["object"])
        self.assertIn("7.9", black_repr)
        self.assertNotIn("3.6", black_repr)
        self.assertIn("3.6", latino_repr)
        self.assertNotIn("7.9", latino_repr)

        # no duplicate / orphan value entity: exactly one entity node per
        # value, and the stale compound is gone from the graph entirely.
        def _entity_nodes_mentioning(token):
            hits = set()
            for s in g.subjects(RDF.type, etd_ns.Entity):
                labels = [str(v) for v in g.objects(s, RDFS.label)]
                if any(token in lab for lab in labels):
                    hits.add(s)
            return hits

        self.assertEqual(len(_entity_nodes_mentioning("7.9")), 1)
        self.assertEqual(len(_entity_nodes_mentioning("3.6")), 1)
        compound_still_present = (
            (URIRef(str(compound.uri)), RDF.type, etd_ns.Entity) in g
            or any(o == URIRef(str(compound.uri)) for _, _, o in g.triples((None, etd_ns.object, None)))
        )
        self.assertFalse(compound_still_present, "stale compound value entity must not survive")


class ProductionOrderRebindTests(unittest.TestCase):
    """Defect A -- PRODUCTION ORDER.

    The confirmed production shape: the extractor hands both sibling
    relations RAW OBJECT TEXT (already split per subject), NOT a resolved
    compound-entity URI. The pre-resolution rebind pass cannot act (target
    is not yet a resource), and normal endpoint resolution then maps every
    split value back onto the ONE pre-existing compound value entity via
    permissive surface matching. The mirrored POST-resolution rebind call
    must then separate them.

    Synthetic, domain-neutral fixture: a compound value label
    "<n1> <unit> and <n2> <unit>" (two scalar cores) split into
    "<n1> <unit>" / "<n2> <unit>", two subjects whose first tokens differ,
    a source sentence with a 'respectively' cue and every subject before
    every value. The unit phrase is deliberately not a bare quantity, so
    the values fall through to surface resolution (where they collide on
    the compound) rather than being literalised.
    """

    _SRC = ("Alpha cohort and Beta cohort each recorded "
            "12 units lost and 8 units lost respectively.")
    _COMPOUND_LABEL = "12 units lost and 8 units lost"
    _VAL_A = "12 units lost"
    _VAL_B = "8 units lost"

    def _raw_text_siblings(self, subj_a_uri, subj_b_uri):
        common = dict(
            discourse_id="9", field="theme.description",
            source_field="theme.description", origin_key="theme.description|9",
            evidence="Alpha cohort and Beta cohort each recorded 12 units lost and 8 units lost",
            evidence_quote=self._SRC,
        )
        rel_a = Relation(
            source=subj_a_uri, relation="pay", target=self._VAL_A, confidence=0.9,
            metadata={**common, "subject_text": "Alpha cohort", "object_text": self._VAL_A},
        )
        rel_b = Relation(
            source=subj_b_uri, relation="pay", target=self._VAL_B, confidence=0.9,
            metadata={**common, "subject_text": "Beta cohort", "object_text": self._VAL_B},
        )
        return rel_a, rel_b

    def test_normal_resolution_first_collapses_both_onto_the_compound(self):
        # Establishes the precondition the production fix targets: with the
        # compound entity already in the pool, ordinary surface resolution
        # of the two split values BOTH select it.
        p = _bare_pipeline()
        compound = _register(p, self._COMPOUND_LABEL, "Metric")
        unit = {"start_offset": 0, "text": self._SRC}
        got = []
        for val in (self._VAL_A, self._VAL_B):
            ve = p._resolve_surface_entity(
                surface=val, preferred_type="", local_entities=[compound],
                discourse_id="9", field="theme.description",
                origin_key="theme.description|9", unit=unit, unit_text=self._SRC,
            )
            got.append(str(getattr(ve, "uri", "")))
        self.assertEqual(got[0], str(compound.uri))
        self.assertEqual(got[1], str(compound.uri))

    def test_post_resolution_rebind_separates_raw_text_targets(self):
        p = _bare_pipeline()
        a = _register(p, "Alpha Cohort", "PopulationGroup")
        b = _register(p, "Beta Cohort", "PopulationGroup")
        compound = _register(p, self._COMPOUND_LABEL, "Metric")
        local = [a, b, compound]
        rel_a, rel_b = self._raw_text_siblings(a.uri, b.uri)
        self.assertFalse(rel_a.target.startswith("http"))  # raw text on entry

        unit = {"start_offset": 0, "text": self._SRC, "section": "Findings"}
        resolved = p._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=local, discourse_text=self._SRC,
            discourse_id="9", field="theme.description", origin_key="theme.description|9",
            unit=unit, unit_text=self._SRC,
        )
        self.assertEqual(len(resolved), 2)
        by_src = {r.source: r for r in resolved}
        ra, rb = by_src[str(a.uri)], by_src[str(b.uri)]

        # distinct targets, both real resources, neither the stale compound
        self.assertTrue(ra.target.startswith("http") and rb.target.startswith("http"))
        self.assertNotEqual(ra.target, rb.target)
        self.assertNotEqual(ra.target, str(compound.uri))
        self.assertNotEqual(rb.target, str(compound.uri))
        self.assertFalse(ra.target_is_literal)
        self.assertIsNone(ra.target_literal)

        # source order preserved: Alpha -> 12 units lost, Beta -> 8 units lost
        self.assertEqual(ra.metadata["object_text"], self._VAL_A)
        self.assertEqual(rb.metadata["object_text"], self._VAL_B)
        self.assertEqual(ra.metadata["object_surface_form"], self._VAL_A)
        self.assertEqual(rb.metadata["object_surface_form"], self._VAL_B)

        # corrected exactly once: one rebind event per sibling, no more
        events = [e for e in p.quality_events
                  if e["category"] == "shared_compound_respectively_rebind"]
        self.assertEqual(len(events), 2)
        # stale compound dropped from the working pool (no duplicate entity)
        self.assertNotIn(str(compound.uri), [str(getattr(e, "uri", "")) for e in local])

    def test_idempotent_when_target_already_resolved_pre_resolution(self):
        # The pre-resolution call still handles the URI-already-present
        # shape; the added post-resolution call must not double-correct it
        # or emit extra events.
        p = _bare_pipeline()
        a = _register(p, "Alpha Cohort", "PopulationGroup")
        b = _register(p, "Beta Cohort", "PopulationGroup")
        compound = _register(p, self._COMPOUND_LABEL, "Metric")
        local = [a, b, compound]
        common = dict(
            discourse_id="9", field="theme.description",
            source_field="theme.description", origin_key="theme.description|9",
            evidence="Alpha cohort and Beta cohort each recorded 12 units lost and 8 units lost",
            evidence_quote=self._SRC,
        )
        rel_a = Relation(source=a.uri, relation="pay", target=str(compound.uri),
                         confidence=0.9,
                         metadata={**common, "subject_text": "Alpha cohort", "object_text": self._VAL_A})
        rel_b = Relation(source=b.uri, relation="pay", target=str(compound.uri),
                         confidence=0.9,
                         metadata={**common, "subject_text": "Beta cohort", "object_text": self._VAL_B})
        unit = {"start_offset": 0, "text": self._SRC, "section": "Findings"}
        resolved = p._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=local, discourse_text=self._SRC,
            discourse_id="9", field="theme.description", origin_key="theme.description|9",
            unit=unit, unit_text=self._SRC,
        )
        by_src = {r.source: r for r in resolved}
        ra, rb = by_src[str(a.uri)], by_src[str(b.uri)]
        self.assertNotEqual(ra.target, rb.target)
        self.assertNotEqual(ra.target, str(compound.uri))
        self.assertNotEqual(rb.target, str(compound.uri))
        events = [e for e in p.quality_events
                  if e["category"] == "shared_compound_respectively_rebind"]
        self.assertEqual(len(events), 2)  # exactly one correction, not two


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class ProductionOrderRebindSerializationTests(unittest.TestCase):
    """Raw-text-target production shape -> real
    ``_resolve_semantic_unit_relations`` -> ``RDFSerializer`` -> the
    serialized graph must carry two distinct, source-ordered value targets
    and no surviving compound value entity."""

    _SRC = ("Alpha cohort and Beta cohort each recorded "
            "12 units lost and 8 units lost respectively.")
    _COMPOUND_LABEL = "12 units lost and 8 units lost"

    def test_serialized_graph_separates_raw_text_targets(self):
        from rdflib import Namespace, RDF, RDFS, URIRef
        from serializers.rdf_serializer import RDFSerializer

        p = _bare_pipeline()
        a = _register(p, "Alpha Cohort", "PopulationGroup")
        b = _register(p, "Beta Cohort", "PopulationGroup")
        compound = _register(p, self._COMPOUND_LABEL, "Metric")
        local = [a, b, compound]

        common = dict(
            discourse_id="9", field="theme.description",
            source_field="theme.description", origin_key="theme.description|9",
            evidence="Alpha cohort and Beta cohort each recorded 12 units lost and 8 units lost",
            evidence_quote=self._SRC,
        )
        rel_a = Relation(source=a.uri, relation="pay", target="12 units lost", confidence=0.9,
                         metadata={**common, "subject_text": "Alpha cohort", "object_text": "12 units lost"})
        rel_b = Relation(source=b.uri, relation="pay", target="8 units lost", confidence=0.9,
                         metadata={**common, "subject_text": "Beta cohort", "object_text": "8 units lost"})
        unit = {"start_offset": 0, "text": self._SRC, "section": "Findings"}
        resolved = p._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=local, discourse_text=self._SRC,
            discourse_id="9", field="theme.description", origin_key="theme.description|9",
            unit=unit, unit_text=self._SRC,
        )

        def _rel_dict(r):
            return {
                "uri": r.uri, "source": r.source, "relation": r.relation, "target": r.target,
                "confidence": r.confidence, "target_literal": r.target_literal,
                "target_is_literal": r.target_is_literal, "metadata": r.metadata,
            }

        ann = _base_annotations([_rel_dict(r) for r in resolved], discourse_id="9")
        registry_entities = list(p.global_registry.get_all_entities())
        serializer = RDFSerializer(
            annotations_dict=ann, registry=_Registry(registry_entities), namespaces=Config.NAMESPACES,
        )
        g = serializer.graph
        etd_ns = Namespace(ETD)

        pay_rels = [r for r in g.subjects(RDF.type, etd_ns.Relation)
                    if (r, etd_ns.predicate, etd_ns.pay) in g]
        self.assertEqual(len(pay_rels), 2)

        by_subject = {}
        for rel in pay_rels:
            subj = list(g.objects(rel, etd_ns.subject))[0]
            obj = list(g.objects(rel, etd_ns.object))[0]
            label = None
            if isinstance(obj, URIRef):
                labels = [str(v) for v in g.objects(obj, RDFS.label)]
                label = labels[0] if labels else None
            by_subject[str(subj)] = {"object": obj, "label": label}

        a_o = by_subject[str(a.uri)]
        b_o = by_subject[str(b.uri)]
        self.assertNotEqual(a_o["object"], b_o["object"])
        self.assertNotEqual(str(a_o["object"]), str(compound.uri))
        self.assertNotEqual(str(b_o["object"]), str(compound.uri))
        self.assertIsInstance(a_o["object"], URIRef)
        self.assertIsInstance(b_o["object"], URIRef)

        # source order: Alpha -> "12 units lost", Beta -> "8 units lost"
        a_repr = str(a_o["label"] or a_o["object"])
        b_repr = str(b_o["label"] or b_o["object"])
        self.assertIn("12", a_repr)
        self.assertNotIn("8", a_repr)
        self.assertIn("8", b_repr)
        self.assertNotIn("12", b_repr)

        # the stale compound value entity does not survive
        compound_present = (
            (URIRef(str(compound.uri)), RDF.type, etd_ns.Entity) in g
            or any(o == URIRef(str(compound.uri)) for _, _, o in g.triples((None, etd_ns.object, None)))
        )
        self.assertFalse(compound_present)


if __name__ == "__main__":
    unittest.main()
