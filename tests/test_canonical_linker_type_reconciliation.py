import tempfile
import unittest

from config import Config
from models import Entity
from utils.canonical_linker import CanonicalLinker

# Test-isolation fix: CanonicalLinker(output_dir=...) writes
# canonical_linking_log.jsonl into that directory. A literal "output" here
# resolved to the real production output/ directory and silently
# overwrote its live log with this module's synthetic test data.
_TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="skg_test_canonical_linker_type_reconciliation_")


class _Registry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return list(self.entities.values())

    def replace_entity(self, old_uri: str, new_uri: str):
        if old_uri in self.entities and new_uri not in self.entities:
            self.entities[new_uri] = self.entities[old_uri]


def _mk(uri, label, entity_type, confidence=0.9, metadata=None):
    return Entity(uri=uri, label=label, entity_type=entity_type, confidence=confidence, metadata=metadata or {})


def _external(qid="Q999999"):
    return {"wikidata_uri": f"https://www.wikidata.org/entity/{qid}"}


class CompatibleSpecificityDifferenceTests(unittest.TestCase):
    def test_subclass_and_superclass_types_merge_into_one_canonical(self):
        # Dataset rdfs:subClassOf Resource in the declared ETD schema --
        # a compatible specificity difference, not a disagreement.
        reg = _Registry()
        e1 = _mk("https://alfie-project.eu/etd-hub#r1", "Some Repository", "Resource", 0.7, dict(_external()))
        e2 = _mk("https://alfie-project.eu/etd-hub#r2", "Some Repository", "Dataset", 0.9, dict(_external()))
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        canonical = out[0]
        self.assertEqual(canonical.entity_type, "Dataset")
        self.assertIn("dataset", canonical.uri.lower())

    def test_both_members_promoted_to_most_specific_type(self):
        reg = _Registry()
        e1 = _mk("https://alfie-project.eu/etd-hub#r1", "Some Repository", "Resource", 0.7, dict(_external()))
        e2 = _mk("https://alfie-project.eu/etd-hub#r2", "Some Repository", "Dataset", 0.9, dict(_external()))
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        linker.run_full_registry()

        self.assertEqual(e1.entity_type, "Dataset")
        self.assertEqual(e2.entity_type, "Dataset")
        self.assertEqual(e1.ontology_uri, e2.ontology_uri)


class AmbiguousFallbackVsConfidentTests(unittest.TestCase):
    def test_ambiguous_fallback_promoted_to_confident_specific_type(self):
        reg = _Registry()
        confident = _mk(
            "https://alfie-project.eu/etd-hub#g1", "Some Cohort", "DemographicGroup", 0.9,
            {**_external(), "population_group_typing_reason": "llm_classification_confident"},
        )
        ambiguous_fallback = _mk(
            "https://alfie-project.eu/etd-hub#g2", "Some Cohort", "Concept", 0.4,
            {**_external(), "population_group_typing_reason": "llm_classification_ambiguous"},
        )
        reg.entities = {confident.uri: confident, ambiguous_fallback.uri: ambiguous_fallback}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "DemographicGroup")
        self.assertEqual(ambiguous_fallback.entity_type, "DemographicGroup")
        self.assertEqual(
            ambiguous_fallback.metadata.get("external_id_type_reconciled_from"), "Concept"
        )

    def test_all_members_uncertain_leaves_type_unresolved_but_still_merges(self):
        reg = _Registry()
        e1 = _mk(
            "https://alfie-project.eu/etd-hub#g1", "Some Cohort", "Concept", 0.4,
            {**_external(), "population_group_typing_reason": "llm_classification_ambiguous"},
        )
        e2 = _mk(
            "https://alfie-project.eu/etd-hub#g2", "Some Cohort", "Concept", 0.4,
            {**_external(), "population_group_typing_reason": "llm_classification_ambiguous"},
        )
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Concept")


class IncompatibleTypeDisagreementTests(unittest.TestCase):
    def test_incompatible_confident_types_do_not_silently_merge(self):
        reg = _Registry()
        e1 = _mk("https://alfie-project.eu/etd-hub#a1", "Some Mention", "Concept", 0.85, dict(_external()))
        e2 = _mk("https://alfie-project.eu/etd-hub#a2", "Some Mention", "DemographicGroup", 0.9, dict(_external()))
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        self.assertNotEqual(out[0].uri, out[1].uri)
        # Original types are preserved -- neither silently overwritten.
        types = {o.entity_type for o in out}
        self.assertEqual(types, {"Concept", "DemographicGroup"})

    def test_incompatible_group_flagged_as_suspect(self):
        reg = _Registry()
        e1 = _mk("https://alfie-project.eu/etd-hub#a1", "Some Mention", "SoftwareProduct", 0.85, dict(_external()))
        e2 = _mk("https://alfie-project.eu/etd-hub#a2", "Some Mention", "Dataset", 0.9, dict(_external()))
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        linker.run_full_registry()

        self.assertTrue(e1.metadata.get("external_link_suspect"))
        self.assertTrue(e2.metadata.get("external_id_type_conflict"))
        self.assertEqual(set(e1.metadata.get("external_id_type_conflict_types", [])), {"SoftwareProduct", "Dataset"})

    def test_wrong_external_link_cannot_override_strong_contextual_type(self):
        # A confident, specific, incompatible type must survive even
        # though it happens to share an (evidently wrong) external ID with
        # something else entirely.
        reg = _Registry()
        person = _mk("https://alfie-project.eu/etd-hub#p1", "Some Org", "Person", 0.9, dict(_external()))
        org = _mk("https://alfie-project.eu/etd-hub#p2", "Some Org", "Organization", 0.92, dict(_external()))
        reg.entities = {person.uri: person, org.uri: org}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        self.assertEqual(person.entity_type, "Person")
        self.assertEqual(org.entity_type, "Organization")


class UriTypeOntologyAtomicityTests(unittest.TestCase):
    def test_promoted_canonical_has_consistent_uri_type_ontology(self):
        reg = _Registry()
        e1 = _mk("https://alfie-project.eu/etd-hub#r1", "Some Repository", "Resource", 0.7, dict(_external()))
        e2 = _mk("https://alfie-project.eu/etd-hub#r2", "Some Repository", "Dataset", 0.9, dict(_external()))
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        canonical = out[0]
        self.assertIn("dataset", canonical.uri.lower())
        self.assertEqual(canonical.entity_type, "Dataset")
        self.assertIn("Dataset", str(canonical.ontology_uri))


class OntologyPreservationOnPromotionTests(unittest.TestCase):
    def test_promotion_to_type_without_own_mapping_preserves_richer_existing_grounding(self):
        # "Governance" has no dedicated ENTITY_TYPE_TO_ONTOLOGY entry --
        # promoting a generic "Entity" fallback member to it must not
        # clobber a MORE SPECIFIC ontology_uri that member already carries
        # (e.g. from earlier lexical concept-matching) with the generic
        # prov:Entity fallback.
        reg = _Registry()
        specific = _mk(
            "https://alfie-project.eu/etd-hub#g1", "Some Governance Concept", "Governance", 0.85,
            dict(_external()),
        )
        generic_with_richer_grounding = _mk(
            "https://alfie-project.eu/etd-hub#g2", "Some Governance Concept", "Entity", 0.5,
            dict(_external()),
        )
        generic_with_richer_grounding.ontology_uri = "https://w3id.org/aipo#Accountability"
        reg.entities = {specific.uri: specific, generic_with_richer_grounding.uri: generic_with_richer_grounding}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        linker.run_full_registry()

        self.assertEqual(generic_with_richer_grounding.entity_type, "Governance")
        self.assertEqual(generic_with_richer_grounding.ontology_uri, "https://w3id.org/aipo#Accountability")


class ProvenanceTraceabilityTests(unittest.TestCase):
    def test_member_retains_traceable_original_proposed_type(self):
        reg = _Registry()
        confident = _mk(
            "https://alfie-project.eu/etd-hub#g1", "Some Cohort", "DemographicGroup", 0.9,
            {**_external(), "population_group_typing_reason": "llm_classification_confident"},
        )
        ambiguous = _mk(
            "https://alfie-project.eu/etd-hub#g2", "Some Cohort", "Concept", 0.4,
            {**_external(), "population_group_typing_reason": "llm_classification_ambiguous"},
        )
        reg.entities = {confident.uri: confident, ambiguous.uri: ambiguous}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        linker.run_full_registry()

        self.assertEqual(ambiguous.metadata.get("external_id_type_reconciled_from"), "Concept")
        self.assertEqual(ambiguous.metadata.get("external_id_type_reconciled_to"), "DemographicGroup")


class DiscourseScopingRegressionTests(unittest.TestCase):
    def test_discourse_scoped_entities_without_external_id_still_kept_separate(self):
        # Identity-scope protection: unresolved referential mentions scoped
        # to different discourses must never collapse into one canonical
        # entity, regardless of external-ID changes (these entities carry
        # no external ID, so they go through Pass 2, entirely untouched by
        # external-ID reconciliation).
        reg = _Registry()
        e1 = _mk(
            "https://alfie-project.eu/etd-hub#u1", "This", "Concept", 0.6,
            {"discourse_scoped_identity": True, "origin_key": "answer.description|d1"},
        )
        e2 = _mk(
            "https://alfie-project.eu/etd-hub#u2", "This", "Concept", 0.6,
            {"discourse_scoped_identity": True, "origin_key": "answer.description|d2"},
        )
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        self.assertNotEqual(out[0].uri, out[1].uri)

    def test_no_cross_discourse_contamination_via_external_id_path(self):
        # Two genuinely different discourses' mentions that DO share a
        # real external ID and a compatible type must still merge (this
        # is intended cross-discourse deduplication, unaffected by identity
        # scoping, which only applies to UNRESOLVED referential mentions).
        reg = _Registry()
        e1 = _mk(
            "https://alfie-project.eu/etd-hub#d1", "Some Cohort", "PopulationGroup", 0.8,
            {**_external(), "discourse_id": "1"},
        )
        e2 = _mk(
            "https://alfie-project.eu/etd-hub#d2", "Some Cohort", "DemographicGroup", 0.85,
            {**_external(), "discourse_id": "2"},
        )
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "DemographicGroup")


if __name__ == "__main__":
    unittest.main()
