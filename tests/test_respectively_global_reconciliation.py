"""Defect A -- global-scope respectively reconciliation.

The generic detector ``_rebind_stale_shared_respectively_targets`` is
correct, but when the two sibling relations of a
"N1 and N2 <unit> respectively" statistic are produced by SEPARATE
``_resolve_semantic_unit_relations`` calls, they are never co-present as a
>=2 group at either per-unit call site (``len(group) < 2``).

Fix: re-run the SAME detector once per discourse in
``validate_semantic_relations`` -- where ``pr.relations`` holds every
surviving relation for the record, endpoints are canonical URIs, and the
registry is live -- immediately before
``_supersede_stale_respectively_duplicates``. Registry-only
(``local_entities=None``). All per-unit calls are preserved.

Synthetic ids / fields only; no LLM / network.
"""
import importlib.util
import unittest

from config import Config
from models import AnnotationBatch, Entity, ProcessingResult, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]
COMP_URI = f"{ETD.rstrip('#')}/entity/metric/n1-and-n2-units-more-abc123"

_DTEXT = (
    "**Findings**:\r\n"
    "* **Disparity**: Alpha cohort and Beta cohort record 12 and 8 units more respectively  \r\n"
    "* **Restatement**: the 12 and 8 units more gap persists\r\n"
)
_COMPOUND_LABEL = "12 And 8 Units More"


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
    p.quality_events = []
    return p


def _reg(p, uri, label, etype, md=None):
    e = Entity(uri=uri, label=label, entity_type=etype, confidence=0.9,
               metadata={"discourse_id": "9", "surface_form": label, **(md or {})})
    p._ensure_entity_ontology_grounding(e)
    return p.global_registry.get_or_create(e)


def _setup(p):
    a = _reg(p, f"{ETD.rstrip('#')}/entity/populationgroup/alpha-cohort-1", "Alpha Cohort", "PopulationGroup")
    b = _reg(p, f"{ETD.rstrip('#')}/entity/populationgroup/beta-cohort-2", "Beta Cohort", "PopulationGroup")
    comp = _reg(p, COMP_URI, _COMPOUND_LABEL, "Metric",
                md={"context_source": "dataset_theme", "field": "theme.description", "discourse_id": "9"})
    return a, b, comp


def _common_meta(*, section="Findings", **extra):
    return dict(
        discourse_id="9", field="theme.description", source_field="theme.description",
        origin_key="theme.description|9", section=section,
        evidence="Alpha cohort and Beta cohort record 12 and 8 units more",
        evidence_quote="Alpha cohort and Beta cohort record 12 and 8 units more respectively",
        **extra,
    )


def _siblings(a_uri, b_uri):
    ra = Relation(source=a_uri, relation="pay", target=COMP_URI, confidence=0.92,
                  metadata=_common_meta(subject_text="Alpha cohort", object_text="12 units more"))
    rb = Relation(source=b_uri, relation="pay", target=COMP_URI, confidence=0.92,
                  metadata=_common_meta(subject_text="Beta cohort", object_text="8 units more"))
    return ra, rb


def _stale_duplicate(subj_uri):
    # another section's unsplit relation whose object_text == the compound
    # label -> the existing supersession pass must remove it once the
    # siblings are corrected.
    return Relation(source=subj_uri, relation="pay", target="12 and 8 units more", confidence=0.9,
                    metadata=_common_meta(section="Restatement",
                                          subject_text="Alpha and Beta cohorts",
                                          object_text="12 and 8 units more"))


class PerUnitCannotGroupSeparateSiblingsTests(unittest.TestCase):
    def test_per_unit_rebind_on_one_sibling_is_a_noop(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        # each sibling reaches its own per-unit call ALONE
        for solo in ([ra], [rb]):
            p._rebind_stale_shared_respectively_targets(
                solo, [a, b, comp], discourse_text=_DTEXT, discourse_id="9",
                field="theme.description", origin_key="theme.description|9",
                unit={"start_offset": 0, "text": _DTEXT}, unit_text=_DTEXT)
        self.assertEqual(ra.target, COMP_URI)
        self.assertEqual(rb.target, COMP_URI)
        self.assertEqual([e for e in p.quality_events
                          if e["category"] == "shared_compound_respectively_rebind"], [])

    def test_per_unit_rebind_still_fires_when_both_siblings_in_one_call(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        p._rebind_stale_shared_respectively_targets(
            [ra, rb], [a, b, comp], discourse_text=_DTEXT, discourse_id="9",
            field="theme.description", origin_key="theme.description|9",
            unit={"start_offset": 0, "text": _DTEXT}, unit_text=_DTEXT)
        self.assertNotEqual(ra.target, rb.target)
        self.assertEqual(len([e for e in p.quality_events
                              if e["category"] == "shared_compound_respectively_rebind"]), 2)


class RegistryOnlyRebindTests(unittest.TestCase):
    def test_registry_only_resolves_compound_and_mints_distinct_values(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        self.assertEqual(ra.target, rb.target)  # before: same compound

        p._rebind_stale_shared_respectively_targets(
            [ra, rb], local_entities=None, discourse_text=_DTEXT, discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit_text=_DTEXT)

        self.assertTrue(ra.target.startswith("http") and rb.target.startswith("http"))
        self.assertNotEqual(ra.target, rb.target)
        self.assertNotEqual(ra.target, COMP_URI)
        self.assertNotEqual(rb.target, COMP_URI)
        # source order: "Alpha" before "Beta"; "12" before "8"
        self.assertEqual(ra.metadata["object_text"], "12 units more")
        self.assertEqual(rb.metadata["object_text"], "8 units more")
        self.assertEqual(len([e for e in p.quality_events
                              if e["category"] == "shared_compound_respectively_rebind"]), 2)
        self.assertTrue(all(r.metadata.get("respectively_split_from") for r in (ra, rb)))
        # compound value entity is no longer a live registry entity
        self.assertFalse(any(getattr(e, "uri", "") == COMP_URI
                             for e in p.global_registry.get_all_entities()))


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class ValidateSemanticRelationsGlobalReconciliationTests(unittest.TestCase):
    def _batch(self, relations):
        ann = AnnotationBatch()
        ann.themes["9"] = ProcessingResult(entities=[], relations=list(relations),
                                           metadata={"text": _DTEXT})
        return ann

    def test_before_global_reconciliation_siblings_share_the_compound(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        self.assertEqual(ra.target, rb.target)
        self.assertEqual(ra.target, COMP_URI)

    def test_validate_semantic_relations_reconciles_and_supersedes(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        rc = _stale_duplicate(a.uri)
        ann = self._batch([ra, rb, rc])

        p.validate_semantic_relations(ann)
        kept = ann.themes["9"].relations

        # distinct targets, neither the compound, correct subject->value order
        self.assertNotEqual(ra.target, rb.target)
        self.assertNotEqual(ra.target, COMP_URI)
        self.assertNotEqual(rb.target, COMP_URI)
        self.assertEqual(ra.metadata["object_text"], "12 units more")
        self.assertEqual(rb.metadata["object_text"], "8 units more")

        # exactly one correction event per sibling
        rebind_evs = [e for e in p.quality_events
                      if e["category"] == "shared_compound_respectively_rebind"]
        self.assertEqual(len(rebind_evs), 2)

        # the stale compound-value duplicate is superseded by the EXISTING
        # _supersede_stale_respectively_duplicates pass
        self.assertNotIn(rc, kept)
        self.assertEqual(len([e for e in p.quality_events
                              if e["category"] == "respectively_split_supersession"]), 1)
        self.assertIn(ra, kept)
        self.assertIn(rb, kept)

    def test_repeated_validation_does_not_double_correct(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        ra, rb = _siblings(a.uri, b.uri)
        ann = self._batch([ra, rb])

        p.validate_semantic_relations(ann)
        first = len([e for e in p.quality_events
                     if e["category"] == "shared_compound_respectively_rebind"])
        p.validate_semantic_relations(ann)
        second = len([e for e in p.quality_events
                      if e["category"] == "shared_compound_respectively_rebind"])
        self.assertEqual(first, 2)
        self.assertEqual(second, 2)  # no new correction on re-run

    def test_single_occurrence_without_split_siblings_is_untouched(self):
        p = _bare_pipeline()
        a, b, comp = _setup(p)
        # one ordinary relation sharing the compound is NOT a >=2 split group
        lone = Relation(source=a.uri, relation="pay", target=COMP_URI, confidence=0.9,
                        metadata=_common_meta(subject_text="Alpha cohort", object_text="12 units more"))
        ann = self._batch([lone])
        p.validate_semantic_relations(ann)
        self.assertEqual(lone.target, COMP_URI)
        self.assertEqual([e for e in p.quality_events
                          if e["category"] == "shared_compound_respectively_rebind"], [])


if __name__ == "__main__":
    unittest.main()
