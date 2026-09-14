"""Regression tests for the three KG-quality-gate root-cause fixes:

Defect 1 (entity provenance / genuine surface_form): serializers/rdf_serializer.py
Defect 2 (predicate-kind mismatches): pipeline.py, utils/predicate_map.py,
    utils/semantic_relation_policy.py, ontology/add_schema.py
Defect 3 (same-chunk canonicalization / unresolved-pronoun fabrication):
    pipeline.py, utils/entity_consolidator.py

Each test targets the specific failure mode observed in the frozen canonical
TTL before these fixes, not just the generic mechanism.
"""

import importlib.util
import unittest

from config import Config
from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from utils.entity_consolidator import EntityConsolidator
from utils.helpers import GlobalEntityRegistry
from utils.predicate_map import normalize_predicate
from utils.semantic_relation_policy import PREDICATE_REGISTRY

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for entity provenance tests")
class EntityProvenanceGenuineSurfaceFormTests(unittest.TestCase):
    """Defect 1: an entity minted only as an LLM relation's endpoint (no
    independent extraction ever set its metadata) must end up with a
    surface_form taken from the relation's own captured mention text, never
    from its own label/URI slug -- reproduces the concept_algorithms_*/
    concept_features_* shape from the frozen canonical TTL."""

    def test_endpoint_only_entity_gets_genuine_surface_form_not_label(self):
        from rdflib import Namespace, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_base = Config.NAMESPACES["etd"]
        etd_ns = Namespace(etd_base)
        source_uri = f"{etd_base}concept_some_uri_slug_deadbeef00"
        target_uri = f"{etd_base}entity_target_deadbeef01"

        entities = [
            # No metadata at all -- exactly the "minted only as a relation
            # endpoint" shape. label is URI-slug-shaped on purpose: this is
            # what _add_entities() would otherwise silently reuse.
            Entity(uri=source_uri, label="some uri slug", entity_type="Concept", confidence=0.8, metadata={}),
            Entity(
                uri=target_uri, label="Target Thing", entity_type="Concept", confidence=0.8,
                metadata={"discourse_id": "d1", "field": "answer.description"},
            ),
        ]
        ann = {
            "themes": {}, "questions": {},
            "answers": {
                "d1": {
                    "relations": [
                        {
                            "source": source_uri,
                            "relation": "relatedTo",
                            "target": target_uri,
                            "confidence": 0.9,
                            "metadata": {
                                "discourse_id": "d1",
                                "field": "answer.description",
                                "context_source": "llm_json",
                                "chunk_index": 0,
                                "source": "llm_relation_extractor",
                                # genuine LLM-captured mention text --
                                # written by llm_relation_extractor.py as
                                # subject_text/object_text.
                                "subject_text": "the credit scoring model",
                                "object_text": "Target Thing",
                            },
                        }
                    ]
                }
            },
            "votes": {}, "experts": {}, "documents": {},
            "metadata": {},
        }

        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        source_node = None
        for s in g.subjects(RDF.type, etd_ns.Entity):
            if str(s) == source_uri:
                source_node = s
        self.assertIsNotNone(source_node)

        surface_forms = [str(v) for v in g.objects(source_node, etd_ns.surface_form)]
        self.assertEqual(surface_forms, ["the credit scoring model"])
        self.assertNotIn("some uri slug", surface_forms)

        # Relation-level key reconciliation: subject_text must reach the
        # graph as etd:subject_surface_form (this is what the entity-level
        # backfill above reads from).
        rel_node = next(iter(g.subjects(RDF.type, etd_ns.Relation)))
        rel_surface = [str(v) for v in g.objects(rel_node, etd_ns.subject_surface_form)]
        self.assertEqual(rel_surface, ["the credit scoring model"])

        # Every field Defect 1 requires must be present (genuinely missing
        # ones -- discourse_id, extractedFrom -- backfilled from the
        # relation; chunk_index/chunk_index_unknown/source/context_source
        # always populated by _add_entity_metadata's own honest-unknown
        # fallback when the entity has no directly recorded span).
        for pred in (etd_ns.source, etd_ns.extractedFrom, etd_ns.chunk_index,
                     etd_ns.chunk_index_unknown, etd_ns.discourse_id, etd_ns.context_source):
            self.assertTrue(list(g.objects(source_node, pred)), f"missing {pred}")

    def test_genuinely_provided_surface_form_is_not_disturbed(self):
        """An entity that DID come with its own genuine surface_form must
        keep it -- this fix must only replace the ent.label placeholder,
        never a real independently-sourced value."""
        from rdflib import Namespace, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_base = Config.NAMESPACES["etd"]
        etd_ns = Namespace(etd_base)
        uri = f"{etd_base}entity_well_formed_deadbeef02"

        entities = [
            Entity(
                uri=uri, label="COMPAS", entity_type="AISystem", confidence=0.9,
                metadata={
                    "surface_form": "the COMPAS tool",
                    "discourse_id": "d2", "field": "answer.description",
                    "chunk_index": 0, "context_source": "llm", "source": "llm",
                },
            ),
        ]
        ann = {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}, "metadata": {}}

        serializer = RDFSerializer(
            annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES,
            prune_isolated_entities=False,
        )
        g = serializer.graph
        node = next(s for s in g.subjects(RDF.type, etd_ns.Entity) if str(s) == uri)
        self.assertEqual([str(v) for v in g.objects(node, etd_ns.surface_form)], ["the COMPAS tool"])


class PredicateKindReconciliationTests(unittest.TestCase):
    """Defect 2: hasRisk/mitigatedBy/reflects/relatedTo/requires are
    declared object-range in the ontology; a proposed literal object for
    one of these must be redirected to entity resolution instead of
    accepted verbatim -- reproduces the etd_declared_vs_observed_kind_
    mismatch failures from the frozen canonical TTL. containsSubject's
    former alias phrases must no longer collide with its own distinct
    structural predicate of the same name."""

    def test_entity_only_predicates_are_registered_literal_disallowed(self):
        for pred in ("hasRisk", "mitigatedBy", "reflects", "relatedTo", "requires"):
            entry = PREDICATE_REGISTRY.get(pred)
            self.assertIsNotNone(entry, f"{pred} missing from PREDICATE_REGISTRY")
            self.assertFalse(
                entry.literal_allowed,
                f"{pred} must remain literal_allowed=False -- the fix routes proposed "
                "literals to entity resolution, it must not just relax the policy",
            )

    def test_has_subject_description_is_registered_literal_allowed(self):
        entry = PREDICATE_REGISTRY.get("hasSubjectDescription")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.literal_allowed)

    def test_containssubject_alias_phrases_no_longer_collide(self):
        # These raw phrases used to map onto "containsSubject", colliding
        # with that name's own distinct structural (DiscourseUnit->Entity)
        # declaration. They must now map to the new, distinct, literal-
        # carrying predicate instead.
        for phrase in ("contains", "contains subject", "has subject"):
            self.assertEqual(normalize_predicate(phrase), "hasSubjectDescription")

    def test_predicate_self_reference_survives_normalization(self):
        # Regression for the _alias_to_predicate clobbering bug this fix
        # introduced and then had to correct: normalize_predicate() applied
        # to a predicate's OWN camelCase name must return itself, not fall
        # through to "relatedTo" (which would silently hijack the shared
        # relatedTo self-mapping used across the whole registry).
        self.assertEqual(normalize_predicate("hasSubjectDescription"), "hasSubjectDescription")
        self.assertEqual(normalize_predicate("relatedTo"), "relatedTo")

    def test_declared_entity_only_predicate_redirects_proposed_literal(self):
        """The actual pipeline-level fix: _resolve_semantic_unit_relations
        must not trust target_is_literal=True for a predicate registered
        literal_allowed=False -- it must route the object through entity
        resolution instead (same as an unflagged target), reproducing the
        requires -> "credentials" shape from the frozen TTL."""
        pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        pipeline.config = Config
        pipeline.global_registry = GlobalEntityRegistry()

        from models import Relation
        from ontology.mapper import OntologyMapper
        from utils.entity_normalizer import EntityNormalizer

        pipeline.ontology_mapper = OntologyMapper()
        pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)

        subject = Entity(
            uri=f"{Config.NAMESPACES['etd']}organisation_physionet_test",
            label="PhysioNet", entity_type="Organisation", confidence=0.9,
            metadata={"discourse_id": "d1"},
        )
        pipeline._ensure_entity_ontology_grounding(subject)
        subject = pipeline.global_registry.get_or_create(subject)

        relation = Relation(
            source=subject.uri, relation="requires", target="credentials",
            target_is_literal=True, target_literal="credentials",
            metadata={"object_text": "credentials", "discourse_id": "d1", "evidence_quote": "Access requires credentials."},
        )
        unit = {"start_offset": 0, "text": "Access requires credentials."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation],
            local_entities=[subject],
            discourse_text=unit["text"],
            discourse_id="d1",
            field="answer.description",
            origin_key="answer.description|d1",
            unit=unit,
            unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        out = resolved[0]
        # requires is declared object-range -- the proposed literal must
        # have been redirected to a materialised entity, not accepted as-is.
        self.assertFalse(out.target_is_literal, "requires must not accept a bare literal object")
        self.assertTrue(str(out.target).startswith("http"), "requires' object must resolve to an entity URI")


class SameChunkCanonicalizationTests(unittest.TestCase):
    """Defect 3: (a) 1st/2nd-person pronouns ("you"/"we"/"I") must never be
    fabricated into a persistent entity, matching how 3rd-person pronouns
    are already handled, but WITHOUT being fed to antecedent search (which
    has no valid target for them); (b) same-context mentions typed at two
    levels of a declared ontology subclass hierarchy (e.g. PopulationGroup /
    DemographicGroup) must consolidate into one entity instead of staying
    split by type -- reproduces the "you" and "black defendants" duplicate-
    entity pairs from the frozen canonical TTL."""

    def test_person_deictic_pronouns_blocked_from_fabrication(self):
        for pronoun in ("you", "we", "I", "your", "our"):
            self.assertTrue(
                EnhancedSemanticPipeline._is_unresolved_coreference_candidate(pronoun),
                f"{pronoun!r} must be blocked from persistent-entity fabrication",
            )

    def test_person_deictic_pronouns_excluded_from_antecedent_search(self):
        # Regression for the test_coordinated_fragment_ambiguity_no_false_
        # unique_resolution failure this fix caused and then had to correct:
        # "you" latched onto an unrelated nearby entity ("Biases") via
        # antecedent search once it was (wrongly) added to the same set
        # antecedent search also consults.
        for pronoun in ("you", "we", "I"):
            self.assertFalse(
                EnhancedSemanticPipeline._is_referential_endpoint_text(pronoun),
                f"{pronoun!r} must not be antecedent-search-eligible (no textual antecedent exists for it)",
            )

    def test_third_person_pronouns_still_antecedent_search_eligible(self):
        for pronoun in ("it", "this", "they"):
            self.assertTrue(EnhancedSemanticPipeline._is_referential_endpoint_text(pronoun))

    def test_declared_subclass_pair_consolidates_same_context_duplicate(self):
        registry = GlobalEntityRegistry()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidator = EntityConsolidator(registry=registry, output_dir=tmpdir)
            e_population = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/populationgroup/black-defendants-aaaa",
                label="black defendants", entity_type="PopulationGroup", confidence=0.8,
                metadata={"discourse_id": "d1", "chunk_index": 0},
            )
            e_demographic = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/demographicgroup/black-defendants-bbbb",
                label="black defendants", entity_type="DemographicGroup", confidence=0.85,
                metadata={"discourse_id": "d1", "chunk_index": 0},
            )
            result = consolidator.consolidate([e_population, e_demographic])

        self.assertEqual(len(result), 1, "same-label, subclass-related-type duplicates must merge")
        # The more specific declared subclass (DemographicGroup) must
        # survive as the canonical type -- it carries strictly more
        # information than its own declared parent (PopulationGroup).
        self.assertEqual(result[0].entity_type, "DemographicGroup")

    def test_unrelated_types_are_not_merged(self):
        registry = GlobalEntityRegistry()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidator = EntityConsolidator(registry=registry, output_dir=tmpdir)
            e1 = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/dataset/bias-aaaa",
                label="Bias", entity_type="Dataset", confidence=0.8, metadata={},
            )
            e2 = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/ethicalprinciple/bias-bbbb",
                label="Bias", entity_type="EthicalPrinciple", confidence=0.8, metadata={},
            )
            result = consolidator.consolidate([e1, e2])
        self.assertEqual(len(result), 2, "types with no declared subclass relation must not merge")

    def test_generic_entity_vs_organisation_consolidates(self):
        """Reproduces the "openai" shape from the latest generated TTL:
        one mention typed only generic Entity (no more specific type was
        ever determined), the other typed Organisation, same label, same
        chunk. Bare Entity is not a real, independently-determined type of
        its own -- it must be treated as compatible with any specific
        type, with the specific type surviving as canonical."""
        registry = GlobalEntityRegistry()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidator = EntityConsolidator(registry=registry, output_dir=tmpdir)
            e_generic = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/entity/openai-29cc8680",
                label="OpenAI", entity_type="Entity", confidence=0.7,
                metadata={"discourse_id": "23", "chunk_index": 0},
            )
            e_specific = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/organisation/wikidata-q21708200-1686dac8",
                label="OpenAI", entity_type="Organisation", confidence=0.9,
                metadata={"discourse_id": "23", "chunk_index": 0},
            )
            result = consolidator.consolidate([e_generic, e_specific])
        self.assertEqual(len(result), 1, "generic Entity and specific Organisation, same label, must merge")
        self.assertEqual(result[0].entity_type, "Organisation")

    def test_generic_entity_vs_model_consolidates(self):
        """Reproduces the "gpt-4o" shape from the latest generated TTL."""
        registry = GlobalEntityRegistry()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidator = EntityConsolidator(registry=registry, output_dir=tmpdir)
            e_generic = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/entity/gpt-4o-56b852ec",
                label="GPT-4o", entity_type="Entity", confidence=0.7,
                metadata={"discourse_id": "11", "chunk_index": 0},
            )
            e_specific = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/model/wikidata-q125919502-c9a1b270",
                label="GPT-4o", entity_type="Model", confidence=0.9,
                metadata={"discourse_id": "11", "chunk_index": 0},
            )
            result = consolidator.consolidate([e_generic, e_specific])
        self.assertEqual(len(result), 1, "generic Entity and specific Model, same label, must merge")
        self.assertEqual(result[0].entity_type, "Model")

    def test_generic_entity_does_not_enable_broad_label_only_merging(self):
        """Guardrail: the bare-Entity compatibility rule must not become a
        backdoor for merging unrelated SPECIFIC types that merely share
        Entity as a distant ancestor. Two different, non-generic, non-
        subclass-related types with the same label must still NOT merge --
        only a genuinely generic ("Entity") side may bridge to a specific
        type."""
        registry = GlobalEntityRegistry()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidator = EntityConsolidator(registry=registry, output_dir=tmpdir)
            e1 = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/dataset/x-aaaa",
                label="X", entity_type="Dataset", confidence=0.8,
                metadata={"discourse_id": "1", "chunk_index": 0},
            )
            e2 = Entity(
                uri="https://alfie-project.eu/etd-hub/entity/ethicalprinciple/x-bbbb",
                label="X", entity_type="EthicalPrinciple", confidence=0.8,
                metadata={"discourse_id": "1", "chunk_index": 0},
            )
            result = consolidator.consolidate([e1, e2])
        self.assertEqual(len(result), 2, "two specific, unrelated types must not merge even though both descend from Entity")


class PredicateReassignmentRevalidationTests(unittest.TestCase):
    """Defect-2 residual fix: utils/predicate_precision.py's
    PredicatePrecisionLayer.refine_relations() can reassign a relation's
    predicate (type-guard downgrade to relatedTo/mentions, or an evidence-
    based upgrade) without revalidating whether the relation's object
    representation (in particular, a literal-valued object) is still
    permitted under the NEWLY assigned predicate -- reproducing the
    reflects/relatedTo literal-object mismatches that survived the first
    Defect-2 fix. Invariant under test: whenever a relation ends up with a
    given predicate (whether reassigned here or not), a literal object may
    only reach the output if that predicate is registered
    literal_allowed=True."""

    @staticmethod
    def _layer():
        from utils.predicate_precision import PredicatePrecisionLayer
        return PredicatePrecisionLayer()

    def test_literal_object_cannot_survive_under_declared_object_only_predicate(self):
        from models import Relation
        layer = self._layer()
        src = Entity(uri="https://kg/e/dataset1", label="Some Dataset", entity_type="Dataset", confidence=0.9, metadata={})
        rel = Relation(
            source=src.uri, relation="reflects", target="systematic discrimination",
            target_is_literal=True, target_literal="systematic discrimination",
            metadata={"evidence": "This dataset reflects systematic discrimination."},
        )
        out = layer.refine_relations([rel], [src], discourse_text=rel.metadata["evidence"])
        self.assertEqual(out, [], "a literal object must never survive under reflects (or whatever it gets reassigned to)")

    def test_type_guard_downgrade_no_longer_leaks_a_literal_into_relatedto(self):
        """End-to-end reproduction of the exact failure mode: a relation
        whose object is literal-valued gets type-guard-downgraded to
        relatedTo (declared literal_allowed=False) and must not reach
        serialization with that literal still attached."""
        from models import Relation
        layer = self._layer()
        src = Entity(uri="https://kg/e/org1", label="Some Org", entity_type="Organisation", confidence=0.9, metadata={})
        # particularlyAffects has a narrow domain/range spec (population-
        # group target) that this literal object will always fail,
        # guaranteeing the type-guard downgrade path fires.
        rel = Relation(
            source=src.uri, relation="particularlyAffects", target="a vague description",
            target_is_literal=True, target_literal="a vague description",
            metadata={"evidence": "x"},
        )
        out = layer.refine_relations([rel], [src], discourse_text="x")
        self.assertEqual(out, [], "a downgraded relation must not carry its old literal into the new predicate")

    def test_literal_allowed_predicate_survives_type_guard_revalidation(self):
        """Regression for the bug the fix itself introduced and had to
        correct: _type_valid() used to return False for ANY predicate not
        in self.specs whenever the object was literal (target_entity is
        always None for a literal), which incorrectly downgraded genuinely
        literal-permitting predicates like hasDescription to relatedTo,
        carrying their still-valid literal into an object-only predicate."""
        from models import Relation
        layer = self._layer()
        src = Entity(uri="https://kg/e/dataset2", label="Some Dataset", entity_type="Dataset", confidence=0.9, metadata={})
        rel = Relation(
            source=src.uri, relation="hasDescription", target="a large tabular dataset",
            target_is_literal=True, target_literal="a large tabular dataset",
            metadata={"evidence": "This is a large tabular dataset."},
        )
        out = layer.refine_relations([rel], [src], discourse_text=rel.metadata["evidence"])
        self.assertEqual(len(out), 1, "hasDescription must not be incorrectly downgraded merely for having a literal object")
        self.assertEqual(out[0].relation, "hasDescription")
        self.assertTrue(out[0].target_is_literal)
        self.assertEqual(out[0].target_literal, "a large tabular dataset")

    def test_entity_object_relation_unaffected_by_revalidation(self):
        from models import Relation
        layer = self._layer()
        src = Entity(uri="https://kg/e/dataset3", label="Some Dataset", entity_type="Dataset", confidence=0.9, metadata={})
        tgt = Entity(uri="https://kg/e/risk1", label="Bias Risk", entity_type="Risk", confidence=0.9, metadata={})
        rel = Relation(source=src.uri, relation="hasRisk", target=tgt.uri, target_is_literal=False, metadata={"evidence": "x"})
        out = layer.refine_relations([rel], [src, tgt], discourse_text="x")
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].target_is_literal)

    def test_reflects_and_relatedto_remain_literal_disallowed(self):
        # This fix must not clear the gate by relaxing the policy.
        for pred in ("reflects", "relatedTo"):
            entry = PREDICATE_REGISTRY.get(pred)
            self.assertIsNotNone(entry)
            self.assertFalse(entry.literal_allowed, f"{pred} must remain literal_allowed=False")


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


def _make_recovery_pipeline():
    from ontology.mapper import OntologyMapper
    from utils.entity_normalizer import EntityNormalizer

    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.coref_resolver = _EchoResolver()
    pipeline._discourse_chunk_spans = {}
    return pipeline


def _coverage_unit(unit_id, text, start_offset):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": "semantic", "unit_type": "bullet",
    }


class CoverageRecoveryDuplicationTests(unittest.TestCase):
    """Reproduces the exact "$765M+ annually in excessive interest
    payments" duplicate from the latest generated TTL, and the revised
    fix for it. The ORIGINAL fix (span-overlap-anywhere-in-discourse) was
    rejected as too broad: one sentence can legitimately support several
    distinct relations, and rejecting on mere span overlap would have
    dropped genuinely valid ones. The invariant enforced now is narrower:
    a relation produced by _recover_semantic_coverage_for_unit must
    itself relate to the segment/unit that triggered that specific
    recovery call (checked via the SAME textual-relatedness test already
    used for gap-detection, applied to the recovered relation's own
    evidence against its own trigger) -- a relation whose evidence
    describes something else entirely (e.g. hallucinated from the
    document-wide Context the "Recovery focus" LLM call is handed) is
    rejected regardless of any span/overlap bookkeeping. Overlap/duplicate
    checking is retained ONLY as a secondary, narrower check: a candidate
    that already passed the in-target check is rejected only when it ALSO
    names the same (source, target) endpoint pair as an existing relation
    with related evidence -- a genuine re-statement of the same fact, not
    merely text sharing a sentence."""

    def test_cross_unit_hallucinated_relation_is_rejected(self):
        """The real failure mode: coverage-recovery for an unrelated
        "Public Datasets" bullet (no relations of its own yet -- a
        genuine gap) hallucinates a relation whose evidence is actually
        about a completely different bullet's dollar figure elsewhere in
        the document (available to the LLM only via the broad `context`
        it was handed, not via the segment it was asked about)."""
        pipeline = _make_recovery_pipeline()

        def fake_extract_hallucinated(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            borrowers = Entity(
                uri="https://kg/e/borrowers", label="Black and Latino borrowers", entity_type="PopulationGroup",
                confidence=0.9, metadata={},
            )
            rel = Relation(
                source=borrowers.uri, relation="pay", target="$765M+ annually in excessive interest payments",
                target_is_literal=True, target_literal="$765M+ annually in excessive interest payments",
                metadata={
                    "evidence_quote": "Results in $765M+ annually",
                    "discourse_id": discourse_id, "object_text": "$765M+ annually in excessive interest payments",
                },
            )
            return [rel]

        pipeline._extract_relations_with_pair_focus = fake_extract_hallucinated.__get__(pipeline, EnhancedSemanticPipeline)

        unit = _coverage_unit(
            "Public Datasets:5",
            "**Content**: 88-90% of all U.S. mortgage loans since 1975, with enhanced data since 2017", 1026,
        )
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="full discourse placeholder", discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit=unit,
        )
        self.assertEqual(relations, [], "a relation whose evidence has nothing to do with its own trigger segment/unit must be rejected")
        self.assertEqual(meta["unresolved_coverage_gaps"], 1, "the gap must be reported as genuinely unresolved, not silently papered over")

    def test_relation_grounded_outside_target_unit_is_rejected(self):
        """General case of the same rule: any recovered relation whose
        evidence does not relate to either the specific segment or the
        wider unit that triggered it is rejected, independent of the
        $765M+ specifics."""
        pipeline = _make_recovery_pipeline()

        def fake_extract_unrelated(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            rel = Relation(
                source="https://kg/e/somesystem", relation="causes", target="https://kg/e/someoutcome",
                metadata={"evidence_quote": "an entirely unrelated sentence from somewhere else in the document",
                          "discourse_id": discourse_id},
            )
            return [rel]

        pipeline._extract_relations_with_pair_focus = fake_extract_unrelated.__get__(pipeline, EnhancedSemanticPipeline)
        unit = _coverage_unit("Ethical Challenges:2", "**Digital Redlining**: Algorithmic discrimination in historically disadvantaged neighborhoods", 460)
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="placeholder", discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit=unit,
        )
        self.assertEqual(relations, [])
        self.assertEqual(meta["unresolved_coverage_gaps"], 1)

    def test_two_distinct_relations_from_same_sentence_are_both_allowed(self):
        """The exact scenario the original (rejected) fix would have
        broken: "Black and Latino borrowers pay 7.9 and 3.6 basis points
        more respectively" legitimately supports TWO distinct relations
        (one per figure), both grounded in the SAME sentence/segment.
        Neither may be dropped merely for sharing evidence text with the
        other -- their (source, target) pairs differ."""
        pipeline = _make_recovery_pipeline()

        def fake_extract_two_facts(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            borrowers = Entity(
                uri="https://kg/e/borrowers2", label="Black and Latino borrowers", entity_type="PopulationGroup",
                confidence=0.9, metadata={},
            )
            shared_evidence = "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively"
            rel1 = Relation(
                source=borrowers.uri, relation="pay", target="7.9 basis points more",
                target_is_literal=True, target_literal="7.9 basis points more",
                metadata={"evidence_quote": shared_evidence, "discourse_id": discourse_id, "object_text": "7.9 basis points more"},
            )
            rel2 = Relation(
                source=borrowers.uri, relation="pay", target="3.6 basis points more",
                target_is_literal=True, target_literal="3.6 basis points more",
                metadata={"evidence_quote": shared_evidence, "discourse_id": discourse_id, "object_text": "3.6 basis points more"},
            )
            return [rel1, rel2]

        pipeline._extract_relations_with_pair_focus = fake_extract_two_facts.__get__(pipeline, EnhancedSemanticPipeline)
        unit = _coverage_unit("Ethical Challenges:0", "**Racial/Ethnic Bias**: Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively", 275)
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="placeholder", discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit=unit,
        )
        self.assertEqual(len(relations), 2, "two distinct relations supported by the same sentence must both be kept")
        self.assertEqual(meta["coverage_gaps_recovered"], 2)

    def test_genuinely_new_in_unit_recovery_relation_is_retained(self):
        pipeline = _make_recovery_pipeline()

        def fake_extract_genuine_new_fact(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            rel = Relation(
                source="https://kg/e/org2", relation="publishes", target="https://kg/e/hmda_dataset",
                metadata={"evidence_quote": "88-90% of all U.S. mortgage loans since 1975", "discourse_id": discourse_id},
            )
            return [rel]

        pipeline._extract_relations_with_pair_focus = fake_extract_genuine_new_fact.__get__(pipeline, EnhancedSemanticPipeline)
        unit = _coverage_unit("Public Datasets:5", "**Content**: 88-90% of all U.S. mortgage loans since 1975, with enhanced data since 2017", 1026)
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="placeholder", discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit=unit,
        )
        self.assertEqual(len(relations), 1, "a genuinely new, in-target recovered fact must not be dropped")
        self.assertEqual(meta["coverage_gaps_recovered"], 1)

    def test_true_in_unit_duplicate_same_endpoints_is_rejected(self):
        """Secondary-gate sanity check: a recovered relation that DOES
        relate to its own unit but re-states an already-known fact (same
        source/target pair, related evidence) is still rejected as a true
        duplicate -- the narrower secondary rule this fix retains."""
        pipeline = _make_recovery_pipeline()
        existing = Relation(
            source="https://kg/e/orgX", relation="publishes", target="https://kg/e/hmda_dataset",
            metadata={"evidence_quote": "88-90% of all U.S. mortgage loans since 1975", "discourse_id": "9"},
        )

        def fake_extract_true_duplicate(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            rel = Relation(
                source="https://kg/e/orgX", relation="publishes", target="https://kg/e/hmda_dataset",
                metadata={"evidence_quote": "88-90% of all U.S. mortgage loans since 1975, with enhanced data since 2017",
                          "discourse_id": discourse_id},
            )
            return [rel]

        pipeline._extract_relations_with_pair_focus = fake_extract_true_duplicate.__get__(pipeline, EnhancedSemanticPipeline)
        unit = _coverage_unit("Public Datasets:5", "**Content**: 88-90% of all U.S. mortgage loans since 1975, with enhanced data since 2017", 1026)
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[existing], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="placeholder", discourse_id="9",
            field="theme.description", origin_key="theme.description|9", unit=unit,
        )
        self.assertEqual(len(relations), 1, "a true duplicate (same endpoints, related evidence) must not be added again")

    def test_relation_with_no_evidence_text_is_not_blocked_by_containment_check(self):
        """A relation with no evidence_quote/evidence/context at all has
        nothing for the in-target check to compare -- it must survive on
        the existing gap logic alone, not be swept up by this check."""
        pipeline = _make_recovery_pipeline()

        def fake_extract_no_evidence(self, *, unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            rel = Relation(
                source="https://kg/e/org3", relation="publishes", target="https://kg/e/some_dataset",
                metadata={"discourse_id": discourse_id},
            )
            return [rel]

        pipeline._extract_relations_with_pair_focus = fake_extract_no_evidence.__get__(pipeline, EnhancedSemanticPipeline)
        unit = _coverage_unit("Section:0", "some fact with a % in it", 0)
        relations, meta = pipeline._recover_semantic_coverage_for_unit(
            relations=[], unit_text=unit["text"], support_context=unit["text"],
            local_entities=[], discourse_text="placeholder", discourse_id="42",
            field="theme.description", origin_key="theme.description|42", unit=unit,
        )
        self.assertEqual(len(relations), 1)


if __name__ == "__main__":
    unittest.main()
