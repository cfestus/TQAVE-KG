import tempfile
import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.canonical_linker import CanonicalLinker
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_relation_policy import SemanticRelationPolicy

# Test-isolation fix: CanonicalLinker(output_dir=...) writes
# canonical_linking_log.jsonl into that directory. A literal "output" here
# resolved to the real production output/ directory and silently
# overwrote its live log with this module's synthetic test data.
_TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="skg_test_relation_target_state_invariant_")


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


def _unit(unit_id, text, start_offset):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": "semantic", "unit_type": "sentence",
    }


class SetTargetEntityInvariantTests(unittest.TestCase):
    """Relation.set_target_entity is the single enforcement point
    -- whenever target is (re)assigned to an internal Entity/Proposition
    URI, target_is_literal/target_literal/target_datatype/object_type must
    end up coherent, regardless of what they held before."""

    def test_raw_numeric_comparison_target_becomes_entity_state_cleared(self):
        rel = Relation(
            source="https://kg/e/subj", relation="pay", target="7.9 basis points more",
            target_is_literal=True, target_literal="7.9 basis points more", target_datatype=None,
            object_type="quantity", metadata={"object_type": "quantity"},
        )
        rel.set_target_entity("https://kg/e/metric_a")
        self.assertEqual(rel.target, "https://kg/e/metric_a")
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)
        self.assertIsNone(rel.target_datatype)
        self.assertIsNone(rel.object_type)
        self.assertNotIn("object_type", rel.metadata)

    def test_literal_monetary_value_converted_to_entity_state_coherent(self):
        rel = Relation(
            source="https://kg/e/subj", relation="causes", target="$765M+ annually",
            target_is_literal=True, target_literal="$765M+ annually", target_datatype=None,
            object_type="money", metadata={"object_type": "money"},
        )
        rel.set_target_entity("https://kg/e/metric_money")
        self.assertEqual(rel.target, "https://kg/e/metric_money")
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)
        self.assertIsNone(rel.object_type)

    def test_date_literal_that_remains_literal_is_preserved(self):
        rel = Relation(
            source="https://kg/e/subj", relation="publishedon", target="September 19, 2019",
            target_is_literal=True, target_literal="September 19, 2019", target_datatype="xsd:date",
            object_type="date", metadata={"object_type": "date"},
        )
        # No reassignment happens -- genuine literal state must be untouched.
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "September 19, 2019")
        self.assertEqual(rel.target_datatype, "xsd:date")
        self.assertEqual(rel.object_type, "date")

    def test_date_value_subsequently_converted_to_entity_state_cleared(self):
        rel = Relation(
            source="https://kg/e/subj", relation="publishedon", target="September 19, 2019",
            target_is_literal=True, target_literal="September 19, 2019", target_datatype="xsd:date",
            object_type="date", metadata={"object_type": "date"},
        )
        rel.set_target_entity("https://kg/e/date_20190919")
        self.assertEqual(rel.target, "https://kg/e/date_20190919")
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)
        self.assertIsNone(rel.target_datatype)
        self.assertIsNone(rel.object_type)

    def test_canonical_replacement_of_already_entity_target_stays_coherent(self):
        rel = Relation(
            source="https://kg/e/subj", relation="uses", target="https://kg/e/old_target",
            target_is_literal=False, target_literal=None, target_datatype=None,
            object_type=None, metadata={},
        )
        rel.set_target_entity("https://kg/e/canonical_target")
        self.assertEqual(rel.target, "https://kg/e/canonical_target")
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)
        self.assertIsNone(rel.target_datatype)

    def test_ordinary_percentage_literal_untouched_by_construction(self):
        rel = Relation(
            source="https://kg/e/subj", relation="varied", target="up to 34.7%",
            target_is_literal=False, target_literal="up to 34.7%", target_datatype="percentage",
            object_type="literal", metadata={},
        )
        # Nothing calls set_target_entity on this relation -- simulating a
        # relation that never gets converted to an entity target. State
        # must be whatever the caller left it as (no implicit mutation).
        self.assertEqual(rel.target, "up to 34.7%")
        self.assertEqual(rel.target_literal, "up to 34.7%")


class CanonicalLinkerRedirectStateTests(unittest.TestCase):
    """Reproduces the exact themes/9 shape -- a literal-typed
    relation whose target string coincidentally equals a pre-canonical
    entity URI gets redirected by CanonicalLinker; the stale literal state
    must not survive the redirect."""

    class _Registry:
        def __init__(self):
            self.entities = {}

        def get_all_entities(self):
            return list(self.entities.values())

        def replace_entity(self, old_uri, new_uri):
            if old_uri in self.entities and new_uri not in self.entities:
                self.entities[new_uri] = self.entities[old_uri]

    def test_redirect_of_literal_marked_target_clears_stale_state(self):
        reg = self._Registry()
        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        old_uri = "https://alfie-project.eu/etd-hub#metric_7_9_and_3_6_basis_points_584c957c2a"
        new_uri = "https://alfie-project.eu/etd-hub/entity/metric/7-9-and-3-6-basis-points-1c8505b0"
        linker.redirects = {old_uri: new_uri}

        rel = Relation(
            source="https://kg/e/black_borrowers", relation="pay", target=old_uri,
            target_is_literal=True, target_literal=old_uri, target_datatype=None,
            object_type="quantity", metadata={"object_type": "quantity"},
        )
        linker.rewrite_relation_endpoints([rel])

        self.assertEqual(rel.target, new_uri)
        self.assertFalse(rel.target_is_literal, "target_is_literal must be cleared once target is a redirected entity URI")
        self.assertIsNone(rel.target_literal, "stale target_literal must not survive the redirect")
        self.assertIsNone(rel.object_type)

    def test_genuine_external_url_literal_untouched_by_redirect(self):
        reg = self._Registry()
        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        # An external URL is never a key in `redirects` (redirects only
        # ever maps internal entity URIs), so a genuine xsd:anyURI literal
        # target must pass through completely unaffected.
        linker.redirects = {"https://alfie-project.eu/etd-hub#unrelated": "https://alfie-project.eu/etd-hub/entity/x"}

        rel = Relation(
            source="https://kg/e/dataset", relation="referencesSource",
            target="https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis",
            target_is_literal=True,
            target_literal="https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis",
            target_datatype="xsd:anyURI", object_type="literal", metadata={},
        )
        linker.rewrite_relation_endpoints([rel])

        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, rel.target)
        self.assertEqual(rel.target_datatype, "xsd:anyURI")

    def test_already_entity_target_redirect_stays_coherent(self):
        reg = self._Registry()
        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        old_uri = "https://alfie-project.eu/etd-hub#e_old"
        new_uri = "https://alfie-project.eu/etd-hub/entity/concept/old-6f"
        linker.redirects = {old_uri: new_uri}

        rel = Relation(
            source="https://kg/e/subj", relation="uses", target=old_uri,
            target_is_literal=False, target_literal=None, target_datatype=None,
            object_type=None, metadata={},
        )
        linker.rewrite_relation_endpoints([rel])
        self.assertEqual(rel.target, new_uri)
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)


class DirectionSwapEndpointStateTests(unittest.TestCase):
    """Direction-swap audit: subject/object swap operations must
    leave the NEW target's literal-state fields coherent, not carried over
    stale from before the swap."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_impact_relation_swap_leaves_new_target_state_coherent(self):
        source = Entity(uri="https://kg/e/dataset_x", label="Dataset X", entity_type="Dataset", confidence=0.9, metadata={})
        target = Entity(uri="https://kg/e/group_y", label="Group Y", entity_type="Population", confidence=0.9, metadata={})
        entities_by_uri = {source.uri: source, target.uri: target}
        evidence = "Group Y is affected by Dataset X due to biased sampling."

        rel = Relation(
            source=target.uri, relation="affectedBy", target=source.uri,
            target_is_literal=False, target_literal=None, metadata={},
        )
        # Simulate stale leftover state on the relation object (as could
        # occur if an earlier stage left it inconsistent) to prove the
        # swap normalizes rather than blindly carrying it across.
        rel.target_literal = "some stale text"

        def _fake_locally_entailed(pred, s, t, ev):
            return True

        self.pipeline._predicate_locally_entailed = _fake_locally_entailed
        self.pipeline._entity_semantic_family = lambda e: "group" if e is target else "dataset"
        self.pipeline._is_impact_actor_entity = lambda e: True
        self.pipeline._entity_surface_variants = lambda e: [getattr(e, "label", "")]
        self.pipeline._is_disparate_impact_text = lambda ev: False

        changed = self.pipeline._canonicalize_impact_relation(rel, entities_by_uri=entities_by_uri, evidence_text=evidence)
        self.assertTrue(changed)
        # Active-voice normalization: "Group Y affectedBy Dataset X" (as
        # constructed above) becomes "Dataset X affects Group Y".
        self.assertEqual(rel.source, source.uri)
        self.assertEqual(rel.target, target.uri)
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal, "stale literal state must not survive onto the new target after the swap")

    def test_policy_normalize_direction_swap_leaves_new_target_state_coherent(self):
        policy = SemanticRelationPolicy()
        src = Entity(uri="https://kg/e/group_z", label="Group Z", entity_type="Population", confidence=0.9, metadata={})
        tgt = Entity(uri="https://kg/e/system_w", label="System W", entity_type="AISystem", confidence=0.9, metadata={})
        entities_by_uri = {src.uri: src, tgt.uri: tgt}

        rel = Relation(source=src.uri, relation="affectedBy", target=tgt.uri, metadata={})
        rel.target_literal = "leftover stale text"
        rel.target_is_literal = False

        policy.predicate_locally_entailed = lambda *a, **k: True
        policy._is_disparate_impact_text = lambda ev: False

        direction = policy.normalize_direction(rel, entities_by_uri, "System W affects Group Z.")
        self.assertEqual(direction, "normalized_inverse")
        self.assertEqual(rel.source, tgt.uri)
        self.assertEqual(rel.target, src.uri)
        self.assertFalse(rel.target_is_literal)
        self.assertIsNone(rel.target_literal)


class NoCompetingLocalAgentCorrectionStateTests(unittest.TestCase):
    """Reproduces the exact themes/12/themes/13 shape -- a
    group-family entity that was extracted as the SUBJECT of a
    literal-valued relation gets moved to the OBJECT position by the
    passive/agent-elliptical correction fallback; the relation's previous
    (unrelated) literal value must not survive as a stale target_literal."""

    def test_group_entity_moved_to_object_clears_stale_literal(self):
        pipeline = _make_pipeline()
        unit_text = "Error rates varied dramatically: 0.8% for light-skinned men vs. up to 34.7% for dark-skinned women"
        unit = _unit("Ethical Challenges:0", unit_text, 0)

        group_entity = Entity(
            uri="https://kg/e/dark_skinned_women", label="Dark-Skinned Women", entity_type="DemographicGroup",
            confidence=0.9, metadata={"discourse_id": "d1"},
        )
        carried_entity = Entity(
            uri="https://kg/e/facial_recognition_systems", label="Commercial Facial Recognition Systems",
            entity_type="AISystem", confidence=0.9, metadata={"discourse_id": "d1"},
        )

        relation = Relation(
            source=group_entity.uri, relation="varied", target="up to 34.7%",
            target_is_literal=True, target_literal="up to 34.7%", target_datatype="percentage",
            object_type="literal",
            metadata={"evidence_quote": unit_text, "evidence": unit_text, "discourse_id": "d1"},
        )

        extracted_relations = [relation]
        extraction_entities = [group_entity]
        no_competing_local_agent = True
        carried_uri = carried_entity.uri
        carried_label = carried_entity.label

        self.assertTrue(no_competing_local_agent)
        corrected_relations = []
        for rel in extracted_relations:
            rel_source_uri = str(getattr(rel, "source", "") or "")
            subject_entity = next((e for e in extraction_entities if str(getattr(e, "uri", "") or "") == rel_source_uri), None)
            if (
                subject_entity is None
                or pipeline._entity_semantic_family(subject_entity) != "group"
                or not getattr(rel, "target_is_literal", False)
            ):
                corrected_relations.append(rel)
                continue
            rel_meta = getattr(rel, "metadata", None) or {}
            evidence_anchor = str(rel_meta.get("evidence_quote") or rel_meta.get("evidence") or "").strip()
            if not evidence_anchor or evidence_anchor not in unit_text:
                corrected_relations.append(rel)
                continue
            rel.source = carried_uri
            rel.set_target_entity(str(getattr(subject_entity, "uri", "") or ""))
            widened_context = f"{carried_label}, {unit_text}"
            rel_meta["local_context"] = widened_context
            rel_meta["subject_resolved_from_field_context"] = True
            rel.metadata = rel_meta
            rel.context = widened_context
            corrected_relations.append(rel)

        result = corrected_relations[0]
        self.assertEqual(result.source, carried_uri)
        self.assertEqual(result.target, group_entity.uri)
        self.assertFalse(result.target_is_literal)
        self.assertIsNone(result.target_literal, "must not retain the original 'up to 34.7%' literal as stale state")
        self.assertIsNone(result.target_datatype)


class GenuineLiteralAndReferentialRegressionTests(unittest.TestCase):
    """Regression: genuine literal endpoints and the
    referential-literal rejection guard are unaffected by the state-transition
    invariant changes."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_answers_29_approach_possessive_remains_correct(self):
        team = Entity(
            uri="https://alfie-project.eu/etd-hub#concept_organisations_teams_56bdf53a37",
            label="Organisations Teams", entity_type="Concept", confidence=0.9, metadata={"discourse_id": "29"},
        )
        unit_text = "I also think that if organisations/teams do this, share their approach publically."
        unit = _unit("Body:28", unit_text, 3691)
        relation = Relation(
            source=team.uri, relation="share", target="their approach",
            metadata={
                "object_text": "their approach", "discourse_id": "29", "evidence_quote": unit_text,
                "allow_local_predicate": True,
            },
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[team], discourse_text=unit_text, discourse_id="29",
            field="answer.description", origin_key="answer.description|29", unit=unit, unit_text=unit_text,
        )
        final = self.pipeline._apply_possessive_endpoint_semantics(
            relations=resolved, entities=[team], local_pool_entities=[team], unit=unit, discourse_id="29",
        )
        precision = PredicatePrecisionLayer()
        refined = precision.refine_relations(final, entities=[team], discourse_text=unit_text)
        r = refined[0]
        self.assertEqual(r.target, "approach")
        self.assertTrue(r.target_is_literal)
        self.assertEqual(r.target_literal, "approach")

    def test_referential_literal_object_still_rejected_task_6_3(self):
        entity = Entity(
            uri="https://kg/e/team", label="Team", entity_type="Organisation", confidence=0.9, metadata={},
        )
        unit_text = "...knowledge graph on top of this?"
        unit = _unit("Body:1", unit_text, 0)
        relation = Relation(
            source=entity.uri, relation="buildOnTopOf", target="this",
            target_is_literal=True, target_literal="this",
            metadata={"object_text": "this", "discourse_id": "d1", "evidence_quote": unit_text},
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[entity], discourse_text=unit_text, discourse_id="d1",
            field="answer.description", origin_key="answer.description|d1", unit=unit, unit_text=unit_text,
        )
        r = resolved[0]
        self.assertFalse(r.target_is_literal)
        self.assertIsNone(r.target_literal)
        self.assertIsNone(r.object_type)
        self.assertTrue(r.metadata.get("object_endpoint_unresolved_coreference"))


if __name__ == "__main__":
    unittest.main()
