import tempfile
import unittest

from config import Config
from models import Entity, Relation
from utils.canonical_linker import CanonicalLinker

# Test-isolation fix: CanonicalLinker(output_dir=...) writes
# canonical_linking_log.jsonl into that directory. A literal "output" here
# resolved to the real production output/ directory and silently
# overwrote its live log with this module's synthetic test data.
_TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="skg_test_canonical_linker_")


class _Registry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return list(self.entities.values())

    def replace_entity(self, old_uri: str, new_uri: str):
        if old_uri in self.entities and new_uri not in self.entities:
            self.entities[new_uri] = self.entities[old_uri]


class CanonicalLinkerTests(unittest.TestCase):
    def _mk(
        self,
        uri: str,
        label: str,
        entity_type: str = "PolicyFramework",
        confidence: float = 0.9,
        metadata=None,
    ):
        return Entity(uri=uri, label=label, entity_type=entity_type, confidence=confidence, metadata=metadata or {})

    def test_hmda_merges_to_same_canonical_uri(self):
        reg = _Registry()
        original_1 = "https://alfie-project.eu/etd-hub#e1"
        original_2 = "https://alfie-project.eu/etd-hub#e2"
        e1 = self._mk(original_1, "HMDA", "Regulation", 0.8)
        e2 = self._mk(original_2, "Home Mortgage Disclosure Act", "Regulation", 0.9)
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        canonical_uri = out[0].uri
        self.assertNotEqual(canonical_uri, original_1)
        self.assertNotEqual(canonical_uri, original_2)
        self.assertTrue(canonical_uri.startswith(Config.NAMESPACES["etd"].rstrip("#/")))

    def test_ai_not_merged_with_ai_act(self):
        reg = _Registry()
        e1 = self._mk("https://alfie-project.eu/etd-hub#a1", "AI", "PolicyFramework", 0.8)
        e2 = self._mk("https://alfie-project.eu/etd-hub#a2", "AI Act", "PolicyFramework", 0.9)
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        uris = {x.uri for x in out}
        self.assertEqual(len(uris), 2)
        self.assertNotIn(e1.uri, linker.redirects)
        self.assertNotIn(e2.uri, linker.redirects)

    def test_generic_dataset_not_merged_with_specific_dataset(self):
        reg = _Registry()
        e1 = self._mk("https://alfie-project.eu/etd-hub#d1", "Dataset", "Dataset", 0.8)
        e2 = self._mk("https://alfie-project.eu/etd-hub#d2", "NIH Chest X-ray Dataset", "Dataset", 0.9)
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        self.assertNotIn(e1.uri, linker.redirects)
        self.assertNotIn(e2.uri, linker.redirects)

    def test_relation_endpoints_rewritten_to_canonical(self):
        reg = _Registry()
        original_source = "https://alfie-project.eu/etd-hub#u1"
        original_target = "https://alfie-project.eu/etd-hub#u2"
        e1 = self._mk(original_source, "UNESCO AI Ethics Recommendation", "PolicyFramework", 0.8)
        e2 = self._mk(original_target, "UNESCO Recommendation on the Ethics of Artificial Intelligence", "PolicyFramework", 0.9)
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()
        self.assertEqual(len(out), 1)
        canonical_uri = out[0].uri

        rel = Relation(source=original_source, relation="supports", target=original_target, confidence=0.8, metadata={})
        linker.rewrite_relation_endpoints([rel])

        self.assertEqual(rel.source, canonical_uri)
        self.assertEqual(rel.target, canonical_uri)

    def test_canonical_uri_is_stable_across_input_order(self):
        e1 = self._mk("https://alfie-project.eu/etd-hub#h1", "HMDA", "Regulation", 0.8)
        e2 = self._mk("https://alfie-project.eu/etd-hub#h2", "Home Mortgage Disclosure Act", "Regulation", 0.9)

        reg_a = _Registry()
        reg_a.entities = {e1.uri: e1, e2.uri: e2}
        linker_a = CanonicalLinker(registry=reg_a, output_dir=_TEST_OUTPUT_DIR)
        out_a = linker_a.run_full_registry()
        self.assertEqual(len(out_a), 1)
        uri_a = out_a[0].uri

        reg_b = _Registry()
        reg_b.entities = {e2.uri: e2, e1.uri: e1}
        linker_b = CanonicalLinker(registry=reg_b, output_dir=_TEST_OUTPUT_DIR)
        out_b = linker_b.run_full_registry()
        self.assertEqual(len(out_b), 1)
        uri_b = out_b[0].uri

        self.assertEqual(uri_a, uri_b)

    def test_same_wikidata_id_collapses_across_types_and_discourses(self):
        reg = _Registry()
        uri_a = "https://alfie-project.eu/etd-hub/entity/domain/http-www-wikidata-org-entity-q27726790-aaaa1111"
        uri_b = "https://alfie-project.eu/etd-hub/entity/healthcare/http-www-wikidata-org-entity-q27726790-bbbb2222"

        e1 = Entity(
            uri=uri_a,
            label="Healthcare",
            entity_type="Domain",
            ontology_uri="https://alfie-project.eu/ontology#vair:HealthcareDomain",
            confidence=0.8,
            metadata={
                "wikidata_uri": "http://www.wikidata.org/entity/Q27726790",
                "discourse_id": "8",
            },
        )
        e2 = Entity(
            uri=uri_b,
            label="Healthcare",
            entity_type="Healthcare",
            ontology_uri="https://alfie-project.eu/ontology#vair:HealthcareDomain",
            confidence=0.9,
            metadata={
                "wikidata_uri": "https://www.wikidata.org/entity/Q27726790",
                "discourse_id": "7",
            },
        )
        reg.entities = {e1.uri: e1, e2.uri: e2}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        canonical_uri = out[0].uri
        self.assertNotEqual(canonical_uri, uri_a)
        self.assertNotEqual(canonical_uri, uri_b)
        self.assertEqual(linker._resolve_redirect(uri_a), canonical_uri)
        self.assertEqual(linker._resolve_redirect(uri_b), canonical_uri)

    def test_clean_shorter_label_beats_clause_like_alias_for_same_external_id(self):
        reg = _Registry()
        external_id = {"wikidata_uri": "https://www.wikidata.org/entity/Q467"}
        clean = self._mk(
            "https://alfie-project.eu/etd-hub#women_clean",
            "Women",
            "DemographicGroup",
            0.55,
            metadata=dict(external_id),
        )
        malformed = self._mk(
            "https://alfie-project.eu/etd-hub#women_bad",
            "UW study found names associated with Black women",
            "DemographicGroup",
            0.98,
            metadata=dict(external_id),
        )
        reg.entities = {clean.uri: clean, malformed.uri: malformed}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "Women")
        self.assertIn("UW study found names associated with Black women", out[0].metadata.get("aliases", []))

    def test_clean_specific_aliases_still_compete_normally(self):
        reg = _Registry()
        external_id = {"wikidata_uri": "https://www.wikidata.org/entity/Q467"}
        generic = self._mk(
            "https://alfie-project.eu/etd-hub#women_generic",
            "Women",
            "DemographicGroup",
            0.60,
            metadata=dict(external_id),
        )
        specific = self._mk(
            "https://alfie-project.eu/etd-hub#women_specific",
            "Black women",
            "DemographicGroup",
            0.95,
            metadata=dict(external_id),
        )
        reg.entities = {generic.uri: generic, specific.uri: specific}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "Black women")
        self.assertIn("Women", out[0].metadata.get("aliases", []))


if __name__ == "__main__":
    unittest.main()


