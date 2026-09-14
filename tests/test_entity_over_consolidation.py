"""
C.1 -- entity identity / over-consolidation.

Root cause: EntityConsolidator._should_merge_fuzzy accepted a merge whenever
one normalized label was a substring of the other (`ka in kb or kb in ka`)
and both were >= 4 chars -- containment / lexical overlap treated as
sufficient identity evidence. That merged broader vs narrower terms,
populations vs subgroups, negated/opposite modifiers, and generic vs
qualified nouns.

Fix: a containment pair now merges only when the extra material is
word-order / an English function word / plural-possessive inflection
(spelling variants), OR an independent SAME-REFERENT signal is present
(shared external id, a recorded alias, a genuine initialism, or the exact
same extracted mention span). CanonicalLinker._can_fuzzy_merge carries the
same guard.

Fixtures are synthetic and domain-neutral. The handful of named
current-graph pairs at the end are regression *checks*, never logic.
"""
import tempfile
import unittest
from pathlib import Path

from models import Entity
from utils.entity_consolidator import EntityConsolidator
from utils.entity_normalizer import (
    EntityNormalizer,
    containment_adds_material_token,
    inflection_equivalent,
    is_initialism_pair,
)

ETD = "https://alfie-project.eu/etd-hub#"


class _Reg:
    def replace_entity(self, *a, **k):
        pass


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.norm = EntityNormalizer()

    def _cons(self):
        return EntityConsolidator(registry=_Reg(), normalizer=self.norm, output_dir=Path(self._tmp.name))

    def _ent(self, slug, label, *, etype="Concept", conf=0.8, meta=None):
        return Entity(uri=f"{ETD}{slug}", label=label, entity_type=etype, confidence=conf, metadata=dict(meta or {}))

    def _merged(self, a, b):
        """True iff consolidate() collapses the two into one entity."""
        out = self._cons().consolidate([a, b])
        return len(out) == 1


# ---------------------------------------------------------------------------
# helper unit checks
# ---------------------------------------------------------------------------

class HelperTests(_Base):
    def test_inflection_equivalent_is_relational_not_stemming(self):
        self.assertTrue(inflection_equivalent("prediction", "predictions"))
        self.assertTrue(inflection_equivalent("policy", "policies"))
        self.assertTrue(inflection_equivalent("bias", "biases"))
        self.assertTrue(inflection_equivalent("model's", "model"))
        # a derivational/modifier change is NOT an inflection
        self.assertFalse(inflection_equivalent("bias", "biased"))
        self.assertFalse(inflection_equivalent("member", "non-member"))
        self.assertFalse(inflection_equivalent("care", "healthcare"))

    def test_containment_material_token_detection(self):
        self.assertFalse(containment_adds_material_token("beacon signal", "signal beacon"))     # word order
        self.assertFalse(containment_adds_material_token("beacon", "the beacon"))               # function word
        self.assertFalse(containment_adds_material_token("beacon", "beacons"))                  # inflection
        self.assertTrue(containment_adds_material_token("beacon", "signal beacon"))             # added noun
        self.assertTrue(containment_adds_material_token("members", "non members"))              # opposite modifier
        self.assertTrue(containment_adds_material_token("planners", "urban planners"))          # subgroup

    def test_initialism_pair(self):
        self.assertTrue(is_initialism_pair("GRC", "Global Robotics Corp"))
        self.assertTrue(is_initialism_pair("Universal Basic Income", "UBI"))
        self.assertFalse(is_initialism_pair("Cat", "Category"))
        self.assertFalse(is_initialism_pair("GRC", "Global Robotics Holdings"))


# ---------------------------------------------------------------------------
# valid merges that must be preserved
# ---------------------------------------------------------------------------

class ValidMergeTests(_Base):
    def test_exact_case_punctuation_variant_merges(self):
        a = self._ent("v1", "Acme Widget Co.", meta={"ontology_uri": "ex:One"})
        b = self._ent("v2", "acme widget co", meta={"ontology_uri": "ex:Two"})  # forced into the fuzzy pass
        self.assertTrue(self._merged(a, b))

    def test_recorded_alias_merges_even_with_a_material_extra_token(self):
        full = self._ent("al1", "Globex Corporation", etype="Organisation")
        clip = self._ent("al2", "Globex", etype="Organisation",
                         meta={"aliases": ["Globex Corporation"]})
        self.assertTrue(self._merged(full, clip))

    def test_shared_authoritative_external_id_merges_broader_looking_pair(self):
        # A shared AUTHORITATIVE identifier (DOI here) is asserted identity
        # -> it overrides the material-token containment guard.
        a = self._ent("x1", "Northwind", etype="Organisation",
                      meta={"doi": "10.1234/northwind-2020"})
        b = self._ent("x2", "Northwind Trading Group", etype="Organisation",
                      meta={"doi": "10.1234/northwind-2020"})
        self.assertTrue(self._merged(a, b))

    def test_shared_authoritative_wikidata_id_merges_broader_looking_pair(self):
        # A Wikidata QID explicitly flagged wikidata_authoritative=True is
        # treated as an asserted identifier, same as a DOI.
        a = self._ent("xw1", "Northwind", etype="Organisation",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q999001",
                            "wikidata_authoritative": True})
        b = self._ent("xw2", "Northwind Trading Group", etype="Organisation",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q999001",
                            "wikidata_authoritative": True})
        self.assertTrue(self._merged(a, b))

    def test_same_extracted_span_merges(self):
        span = {"field": "answer.description", "discourse_id": "3", "start_char": 10, "end_char": 24}
        a = self._ent("s1", "Sensor", meta={"mentions": [span]})
        b = self._ent("s2", "Proximity Sensor", meta={"mentions": [dict(span)]})
        self.assertTrue(self._merged(a, b))

    def test_repeated_same_named_entity_across_discourses_merges(self):
        a = self._ent("r1", "Zenith Motors", etype="Organisation", meta={"discourse_id": "4"})
        b = self._ent("r2", "Zenith Motors", etype="Organisation", meta={"discourse_id": "9"})
        self.assertTrue(self._merged(a, b))

    def test_plural_inflection_merges(self):
        a = self._ent("p1", "Coastal Beacon", meta={"ontology_uri": "ex:A"})
        b = self._ent("p2", "Coastal Beacons", meta={"ontology_uri": "ex:B"})
        self.assertTrue(self._merged(a, b))


# ---------------------------------------------------------------------------
# invalid merges that must now be blocked
# ---------------------------------------------------------------------------

class InvalidMergeTests(_Base):
    def test_broader_vs_narrower_containment_not_merged(self):
        broad = self._ent("b1", "Coastal Erosion")
        narrow = self._ent("b2", "Coastal Erosion Acceleration")
        self.assertFalse(self._merged(broad, narrow))

    def test_generic_noun_vs_qualified_noun_not_merged(self):
        generic = self._ent("g1", "Beacon")
        qualified = self._ent("g2", "Emergency Signal Beacon")
        self.assertFalse(self._merged(generic, qualified))

    def test_population_vs_subgroup_not_merged(self):
        pop = self._ent("s1", "Residents", etype="PopulationGroup")
        sub = self._ent("s2", "Coastal Residents", etype="PopulationGroup")
        self.assertFalse(self._merged(pop, sub))

    def test_negated_opposite_modifier_not_merged(self):
        base = self._ent("n1", "Members", etype="PopulationGroup")
        opp = self._ent("n2", "Non-Members", etype="PopulationGroup")
        self.assertFalse(self._merged(base, opp))

    def test_same_lexical_form_in_unrelated_type_contexts_not_merged(self):
        planet = self._ent("m1", "Mercury", etype="Planet")
        element = self._ent("m2", "Mercury", etype="ChemicalElement")
        self.assertFalse(self._merged(planet, element))

    def test_alias_on_an_unrelated_entity_does_not_leak(self):
        # B records "Globex Holdings" as an alias -- that must not license a
        # merge with a *different* containment partner "Globex Systems".
        a = self._ent("u1", "Globex Systems Division", etype="Organisation")
        b = self._ent("u2", "Globex", etype="Organisation",
                      meta={"aliases": ["Globex Holdings"]})
        self.assertFalse(self._merged(a, b))

    def test_shared_heuristic_wikidata_qid_alone_is_insufficient(self):
        # Both entities heuristically aligned by B to the SAME Wikidata QID
        # (wikidata_authoritative=False). That alone must NOT override the
        # material-token containment guard (broader vs narrower).
        a = self._ent("hw1", "Coastal Erosion",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q555001",
                            "wikidata_authoritative": False,
                            "wikidata_identity_evidence": "exact_label"})
        b = self._ent("hw2", "Coastal Erosion Acceleration",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q555001",
                            "wikidata_authoritative": False,
                            "wikidata_identity_evidence": "exact_label"})
        self.assertFalse(self._merged(a, b))

    def test_missing_authoritative_flag_defaults_to_heuristic(self):
        # No wikidata_authoritative key at all -> treated as heuristic ->
        # insufficient by itself.
        a = self._ent("hw3", "Beacon",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q555002"})
        b = self._ent("hw4", "Emergency Signal Beacon",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q555002"})
        self.assertFalse(self._merged(a, b))


class HeuristicWikidataWithCorroborationTests(_Base):
    def test_heuristic_wikidata_plus_recorded_alias_merges(self):
        # Heuristic shared QID is not enough alone, but WITH an independent
        # same-referent signal (one label recorded as the other's alias)
        # the merge is allowed.
        a = self._ent("c1", "Globex Corporation", etype="Organisation",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q777001"})
        b = self._ent("c2", "Globex", etype="Organisation",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q777001",
                            "aliases": ["Globex Corporation"]})
        self.assertTrue(self._merged(a, b))

    def test_heuristic_wikidata_plus_same_extracted_span_merges(self):
        span = {"field": "answer.description", "discourse_id": "5", "start_char": 4, "end_char": 22}
        a = self._ent("c3", "Sensor",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q777002", "mentions": [span]})
        b = self._ent("c4", "Proximity Sensor",
                      meta={"wikidata_uri": "http://www.wikidata.org/entity/Q777002", "mentions": [dict(span)]})
        self.assertTrue(self._merged(a, b))


# ---------------------------------------------------------------------------
# regression CHECKS against confirmed current-graph over-merges (no logic)
# ---------------------------------------------------------------------------

class ConfirmedRegressionChecks(_Base):
    CASES = [
        ("Algorithms", "Biased Algorithms", "Concept"),
        ("Non-Smokers", "Smokers", "PopulationGroup"),
        ("Women", "African American Women", "PopulationGroup"),
        ("Communities", "Marginalized Communities", "PopulationGroup"),
        ("Historical Bias", "Historical Bias Perpetuation", "Concept"),
        ("Resumes", "CVs/Resumes", "Concept"),
        ("Users", "End-Users", "PopulationGroup"),
        ("Gender", "Gender Bias", "Concept"),
        ("Outcomes", "Two-Year Recidivism Outcomes", "Concept"),
    ]

    def test_none_of_the_confirmed_over_merges_recur(self):
        for shorter, longer, etype in self.CASES:
            a = self._ent("cr_a", shorter, etype=etype)
            b = self._ent("cr_b", longer, etype=etype)
            self.assertFalse(
                self._merged(a, b),
                f"{shorter!r} must not consolidate into {longer!r}",
            )

    def test_confirmed_valid_inflection_merges_still_hold(self):
        for shorter, longer in [("Prediction", "Predictions"), ("AI Model", "AI Models")]:
            a = self._ent("cv_a", shorter, meta={"ontology_uri": "ex:A"})
            b = self._ent("cv_b", longer, meta={"ontology_uri": "ex:B"})
            self.assertTrue(self._merged(a, b), f"{shorter!r} / {longer!r} should still merge")


if __name__ == "__main__":
    unittest.main()
