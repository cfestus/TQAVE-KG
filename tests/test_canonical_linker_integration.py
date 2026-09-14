import tempfile
import unittest

from models import Entity, Relation
from utils.canonical_linker import CanonicalLinker

# Test-isolation fix: CanonicalLinker(output_dir=...) writes
# canonical_linking_log.jsonl into that directory. A literal "output" here
# resolved to the real production output/ directory and silently
# overwrote its live log with this module's synthetic test data.
_TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="skg_test_canonical_linker_integration_")


class _Registry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return list(self.entities.values())

    def replace_entity(self, old_uri: str, new_uri: str):
        if old_uri in self.entities and new_uri not in self.entities:
            self.entities[new_uri] = self.entities[old_uri]


class CanonicalLinkerIntegrationTests(unittest.TestCase):
    def _entity(self, uri: str, label: str, entity_type: str):
        return Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={})

    def test_fixture_cross_chunk_unification_and_relation_endpoint_rewrite(self):
        reg = _Registry()
        entities = [
            self._entity("https://alfie-project.eu/etd-hub#t_unesco", "UNESCO AI Ethics Recommendation", "PolicyFramework"),
            self._entity("https://alfie-project.eu/etd-hub#q_unesco", "UNESCO Recommendation on the Ethics of Artificial Intelligence", "PolicyFramework"),
            self._entity("https://alfie-project.eu/etd-hub#a_oecd", "OECD AI Principles", "PolicyFramework"),
            self._entity("https://alfie-project.eu/etd-hub#a_oecd2", "OECD Principles on Artificial Intelligence", "PolicyFramework"),
            self._entity("https://alfie-project.eu/etd-hub#a_hmda", "HMDA", "Regulation"),
            self._entity("https://alfie-project.eu/etd-hub#q_hmda", "Home Mortgage Disclosure Act", "Regulation"),
            self._entity("https://alfie-project.eu/etd-hub#t_compas", "COMPAS", "Dataset"),
            self._entity("https://alfie-project.eu/etd-hub#a_compas", "Correctional Offender Management Profiling for Alternative Sanctions", "Dataset"),
        ]
        reg.entities = {e.uri: e for e in entities}

        rels = [
            Relation(source="https://alfie-project.eu/etd-hub#t_unesco", relation="mentions", target="https://alfie-project.eu/etd-hub#a_oecd", confidence=0.8, metadata={}),
            Relation(source="https://alfie-project.eu/etd-hub#q_unesco", relation="mentions", target="https://alfie-project.eu/etd-hub#q_hmda", confidence=0.8, metadata={}),
            Relation(source="https://alfie-project.eu/etd-hub#a_compas", relation="mentions", target="https://alfie-project.eu/etd-hub#a_hmda", confidence=0.8, metadata={}),
        ]

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        canonical_entities = linker.run_full_registry()

        self.assertLess(len(canonical_entities), len(entities))

        linker.rewrite_relation_endpoints(rels)
        old_uris = set(linker.redirects.keys())
        for r in rels:
            self.assertNotIn(r.source, old_uris)
            self.assertNotIn(r.target, old_uris)


if __name__ == "__main__":
    unittest.main()

