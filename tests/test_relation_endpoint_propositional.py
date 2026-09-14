import importlib.util
import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
HAS_PYSHACL = importlib.util.find_spec("pyshacl") is not None


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.wikidata_linker = None
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


def _unit(unit_id, text, start_offset, *, section_category="semantic", unit_type="sentence"):
    return {
        "unit_id": unit_id,
        "text": text,
        "start_offset": start_offset,
        "end_offset": start_offset + len(text),
        "section_category": section_category,
        "unit_type": unit_type,
    }


class PropositionalEndpointResolutionTests(unittest.TestCase):
    """Bare sentence-initial 'This'/'That'/'It' resolved to a
    Proposition entity representing the immediately preceding semantic
    unit's content -- distinct from nominal and possessive
    endpoint resolution."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_clear_preceding_proposition_deixis_creates_and_rewrites(self):
        prev_text = "The review process omitted several required verification steps."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This undermines confidence in the resulting decisions."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        prev_relation = Relation(
            source="https://kg/e/review_process", relation="omitted", target="verification steps",
            target_is_literal=True, uri="https://kg/r/rel_prev",
            metadata={"evidence_quote": prev_text},
        )
        unit_relations_by_id = {"Body:1": [prev_relation]}

        relation = Relation(
            source="This", relation="undermines", target="confidence in the resulting decisions",
            target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )

        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d1",
            origin_key="answer.description|d1", field="answer.description",
        )

        self.assertEqual(len(extra), 1)
        proposition = extra[0]
        self.assertEqual(proposition.entity_type, "Proposition")
        self.assertEqual(resolved[0].source, proposition.uri)
        self.assertNotEqual(resolved[0].source, "This")
        self.assertEqual(
            resolved[0].metadata.get("subject_endpoint_coref_resolution_method"),
            "propositional_preceding_unit_reference",
        )
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_original_surface"), "This")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_coref_resolved"))
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_antecedent_unit_id"), "Body:1")

    def test_same_proposition_reused_by_multiple_references_in_same_unit(self):
        # Two separate relations extracted from the SAME sentence (a common
        # LLM output shape for compound predicates), both anchored to the
        # SAME sentence-initial "This" occurrence -- both must resolve to
        # the identical Proposition entity for the preceding unit.
        prev_text = "The audit found no evidence of the required documentation."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This raises concerns and delays the rollout."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]
        unit_relations_by_id: dict = {}  # preceding unit produced zero relations -- still eligible

        first = Relation(
            source="This", relation="raises", target="concerns", target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )
        second = Relation(
            source="This", relation="delays", target="the rollout", target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )

        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[first, second], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d2",
            origin_key="answer.description|d2", field="answer.description",
        )

        self.assertEqual(len(extra), 1, "both references must resolve to the SAME Proposition entity")
        self.assertEqual(resolved[0].source, resolved[1].source)
        self.assertEqual(resolved[0].source, extra[0].uri)

    def test_ordinary_entity_coreference_still_resolves_to_entity_not_proposition(self):
        unit_start = 100
        unit_text = "The Screening Tool flagged several applicants, and reviewers later overruled it."
        local_start = unit_text.index("Screening Tool")
        tool = Entity(
            uri="https://kg/e/screening_tool", label="Screening Tool", entity_type="AISystem", confidence=0.9,
            metadata={
                "discourse_id": "d3", "section_category": "semantic",
                "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                "mentions": [{
                    "discourse_id": "d3", "chunk_index": 0,
                    "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                }],
            },
        )
        relation = Relation(
            source="reviewers", relation="overruled", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d3", "evidence_quote": unit_text},
        )
        unit = _unit("Body:1", unit_text, unit_start)
        semantic_units = [unit]

        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool], unit=unit, discourse_id="d3",
        )
        self.assertEqual(resolved[0].target, tool.uri, "nominal resolution must already have fired")

        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=resolved, unit=unit, unit_index=0, semantic_units=semantic_units,
            unit_relations_by_id={}, discourse_id="d3",
            origin_key="answer.description|d3", field="answer.description",
        )
        self.assertEqual(resolved[0].target, tool.uri, "propositional step must not override a resolved entity")
        self.assertEqual(extra, [], "no Proposition should be created when nominal coreference already succeeded")

    def test_ambiguous_preceding_unit_with_multiple_subjects_remains_unresolved(self):
        prev_text = "Team Alpha missed the deadline. Team Beta shipped early."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This complicates the retrospective."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]
        prev_relations = [
            Relation(source="https://kg/e/team_alpha", relation="missed", target="the deadline",
                     target_is_literal=True, uri="https://kg/r/rel_a"),
            Relation(source="https://kg/e/team_beta", relation="shipped", target="early",
                     target_is_literal=True, uri="https://kg/r/rel_b"),
        ]
        unit_relations_by_id = {"Body:1": prev_relations}

        relation = Relation(
            source="This", relation="complicates", target="the retrospective", target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d4",
            origin_key="answer.description|d4", field="answer.description",
        )
        self.assertEqual(resolved[0].source, "This")
        self.assertEqual(extra, [])

    def test_no_preceding_unit_remains_unresolved(self):
        cur_text = "This is the first sentence in the field."
        cur = _unit("Body:1", cur_text, 0)
        semantic_units = [cur]
        relation = Relation(
            source="This", relation="describes", target="the field", target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=0, semantic_units=semantic_units,
            unit_relations_by_id={}, discourse_id="d5",
            origin_key="answer.description|d5", field="answer.description",
        )
        self.assertEqual(resolved[0].source, "This")
        self.assertEqual(extra, [])

    def test_expletive_it_construction_does_not_create_proposition(self):
        prev_text = "The team ran a full compliance review."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "It is important to document every exception that arises."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]
        unit_relations_by_id = {"Body:1": [
            Relation(source="https://kg/e/team", relation="ran", target="a full compliance review",
                     target_is_literal=True, uri="https://kg/r/rel_c"),
        ]}
        relation = Relation(
            source="It", relation="requires", target="documentation", target_is_literal=True,
            metadata={"subject_text": "It", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d6",
            origin_key="answer.description|d6", field="answer.description",
        )
        self.assertEqual(resolved[0].source, "It")
        self.assertEqual(extra, [], "expletive 'It' must never mint a Proposition")

    def test_non_expletive_it_is_still_eligible_no_lexical_exclusion(self):
        # Same lexical item ("It"), same sentence-initial position, but a
        # genuine referential (non-expletive) construction -- must NOT be
        # blocked, proving the guard is syntactic, not a lexical ban on "it".
        prev_text = "The pilot deployment skipped the standard review board."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "It concerns several members of the oversight committee."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]
        unit_relations_by_id = {"Body:1": [
            Relation(source="https://kg/e/pilot", relation="skipped", target="the standard review board",
                     target_is_literal=True, uri="https://kg/r/rel_d"),
        ]}
        relation = Relation(
            source="It", relation="concerns", target="several members of the oversight committee",
            target_is_literal=True,
            metadata={"subject_text": "It", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d7",
            origin_key="answer.description|d7", field="answer.description",
        )
        self.assertEqual(len(extra), 1)
        self.assertEqual(resolved[0].source, extra[0].uri)
        self.assertEqual(
            resolved[0].metadata.get("subject_endpoint_coref_resolution_method"),
            "propositional_preceding_unit_reference",
        )

    def test_proposition_identity_stable_across_repeated_processing(self):
        prev_text = "The dataset excluded several relevant subgroups."
        prev = _unit("Body:1", prev_text, 0)

        def _run_once():
            pipeline = _make_pipeline()
            return pipeline._get_or_create_unit_proposition_entity(
                unit=prev, discourse_id="d8", origin_key="answer.description|d8",
                field="answer.description", derived_from_relation_uris=[],
            )

        first = _run_once()
        second = _run_once()
        self.assertEqual(first.uri, second.uri)

    def test_proposition_provenance_points_to_correct_semantic_unit(self):
        prev_text = "The dataset excluded several relevant subgroups."
        prev = _unit("Body:1", prev_text, 42)
        supporting = Relation(
            source="https://kg/e/dataset", relation="excluded", target="several relevant subgroups",
            target_is_literal=True, uri="https://kg/r/rel_support",
        )
        proposition = self.pipeline._get_or_create_unit_proposition_entity(
            unit=prev, discourse_id="d9", origin_key="answer.description|d9",
            field="answer.description", derived_from_relation_uris=["https://kg/r/rel_support"],
        )
        self.assertEqual(proposition.metadata.get("discourse_id"), "d9")
        self.assertEqual(proposition.metadata.get("semantic_unit_id"), "Body:1")
        self.assertEqual(proposition.metadata.get("text"), prev_text)
        self.assertEqual(proposition.metadata.get("start_char"), 42)
        self.assertEqual(proposition.metadata.get("end_char"), 42 + len(prev_text))
        self.assertEqual(proposition.metadata.get("derived_from_relation_uris"), ["https://kg/r/rel_support"])


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for the Proposition SHACL test")
class PropositionShaclConformanceTests(unittest.TestCase):
    class _Registry:
        def __init__(self, entities):
            self._entities = entities

        def get_all_entities(self):
            return self._entities

    @unittest.skipUnless(HAS_PYSHACL, "pyshacl is required for SHACL validation")
    def test_proposition_subject_relation_passes_shacl(self):
        import tempfile
        from pathlib import Path

        from serializers.rdf_serializer import RDFSerializer
        from validation.shacl_validator import SHACLValidator

        etd_base = Config.NAMESPACES["etd"]
        prop_uri = f"{etd_base}proposition_answerdescription_d10_body1_0_50"
        target_uri = f"{etd_base}entity_legal_principles"

        entities = [
            Entity(
                uri=prop_uri, label="Proposition Body:1", entity_type="Proposition", confidence=1.0,
                metadata={
                    "discourse_id": "d10", "origin_key": "answer.description|d10", "field": "answer.description",
                    "text": "The review process omitted several required verification steps.",
                    "start_char": 0, "end_char": 50, "semantic_unit_id": "Body:1",
                    "derived_from_relation_uris": [f"{etd_base}rel_support_prop"],
                },
            ),
            Entity(
                uri=target_uri, label="Legal Principles", entity_type="Concept", confidence=0.9,
                metadata={"discourse_id": "d10", "field": "answer.description"},
            ),
        ]

        ann = {
            "themes": {}, "questions": {},
            "answers": {
                "d10": {
                    "relations": [
                        {
                            "source": prop_uri,
                            "relation": "undermines",
                            "target": target_uri,
                            "confidence": 0.85,
                            "metadata": {
                                "discourse_id": "d10",
                                "field": "answer.description",
                                "subject_endpoint_coref_resolved": True,
                                "subject_endpoint_coref_resolution_method": "propositional_preceding_unit_reference",
                                "subject_endpoint_original_surface": "This",
                            },
                        }
                    ]
                }
            },
            "votes": {}, "experts": {}, "documents": {},
            "metadata": {},
        }

        serializer = RDFSerializer(
            annotations_dict=ann,
            registry=self._Registry(entities),
            namespaces=Config.NAMESPACES,
        )

        from rdflib import Namespace, RDF
        etd_ns = Namespace(etd_base)
        prop_node = None
        for s in serializer.graph.subjects(RDF.type, etd_ns.Proposition):
            prop_node = s
            break
        self.assertIsNotNone(prop_node, "Proposition entity must be typed etd:Proposition in RDF")
        self.assertIn((prop_node, RDF.type, etd_ns.Entity), serializer.graph)

        # Test-isolation fix: Config.OUTPUT_DIR here resolved to the real
        # production output/ directory, silently overwriting its live
        # shacl_report.ttl with this test's synthetic graph.
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SHACLValidator(output_dir=Path(tmpdir))
            result = validator.validate_graph(serializer.graph)
        self.assertTrue(result.get("success", False))
        self.assertTrue(result.get("conforms", False), result.get("results_text", "SHACL did not conform"))


class AnswersElevenLiveShapeReplayTest(unittest.TestCase):
    """End-to-end replay of the real answers/11 shape (generic content, not
    the literal dataset text): a sentence-initial 'This undermines...'
    immediately following a single-assertion preceding sentence."""

    def test_this_undermines_resolves_to_preceding_unit_proposition(self):
        pipeline = _make_pipeline()
        did = "A11"
        text = (
            "The review process omitted several required verification steps. "
            "This undermines confidence in the resulting decisions."
        )

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "omitted" in unit_text:
                return [
                    Relation(
                        source="the review process", relation="omitted", target="verification steps",
                        target_is_literal=True, confidence=0.9,
                        metadata={
                            "extractor": "llm", "context_source": "llm_relation",
                            "subject_text": "the review process", "object_text": "verification steps",
                            "evidence_quote": unit_text, "evidence": unit_text,
                        },
                    )
                ]
            if "undermines" in unit_text:
                return [
                    Relation(
                        source="This", relation="undermines", target="confidence in the resulting decisions",
                        target_is_literal=True, confidence=0.9,
                        metadata={
                            "extractor": "llm", "context_source": "llm_relation",
                            "subject_text": "This", "object_text": "confidence in the resulting decisions",
                            "evidence_quote": unit_text, "evidence": unit_text,
                        },
                    )
                ]
            return []

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [[]], [did], provenance_list=[{"field": "answer.description", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "undermines"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        self.assertTrue(relation.metadata.get("subject_endpoint_coref_resolved"))
        self.assertEqual(
            relation.metadata.get("subject_endpoint_coref_resolution_method"),
            "propositional_preceding_unit_reference",
        )
        self.assertEqual(relation.metadata.get("subject_endpoint_original_surface"), "This")
        self.assertNotEqual(relation.source, "This")

        propositions = [
            e for e in pipeline.global_registry.get_all_entities() if getattr(e, "entity_type", "") == "Proposition"
        ]
        self.assertEqual(len(propositions), 1, f"expected exactly one Proposition entity, got: {propositions}")
        self.assertEqual(relation.source, propositions[0].uri)
        self.assertIn("omitted", str(propositions[0].metadata.get("text") or ""))


if __name__ == "__main__":
    unittest.main()
