"""
Identity-evidence integration check 1 -- source-grounded alias safety.

entity.metadata["aliases"] is a MIXED collection: EntityNormalizer's
pre-normalization label (which is the LLM `resolved_label` when the mention
was context-resolved), plus -- added by later stages --
coreference/consolidation/canonical-linker forms. Passing the whole
collection to WikidataLinker as `source_grounded_aliases` would let an
inferred name become trusted identity evidence.

Fix: an explicit allow-list metadata["source_grounded_aliases"], written
ONLY by EntityNormalizer (for a real extracted mention, never a
resolved_label) and _attach_local_surface_aliases (verbatim discourse
slices at the entity's own mention span). The pipeline linker call passes
surface_form + that allow-list, never raw aliases.
"""
import unittest
from unittest.mock import patch

from models import Entity
from utils.entity_normalizer import EntityNormalizer
from utils.wikidata_linker import WikidataLinker


class NormalizerSourceGroundedAliasTests(unittest.TestCase):
    def test_real_extracted_mention_becomes_source_grounded(self):
        # not context-resolved: the pre-normalization label IS the mention.
        e = Entity(uri="https://ex.org/e1", label="ACME corp", entity_type="Organisation",
                   confidence=0.9, metadata={"surface_form": "ACME corp"})
        EntityNormalizer().normalize_entity(e)
        self.assertNotEqual(e.label, "ACME corp")  # normalized
        self.assertIn("ACME corp", e.metadata.get("aliases", []))
        self.assertIn("ACME corp", e.metadata.get("source_grounded_aliases", []))

    def test_llm_resolved_label_does_not_become_source_grounded(self):
        # the entity's label is a context-derived name (resolved_label),
        # NOT a verbatim mention -- it may sit in `aliases` (retrieval help)
        # but must never enter source_grounded_aliases.
        e = Entity(
            uri="https://ex.org/e2", label="the recidivism scoring tool", entity_type="AISystem",
            confidence=0.9,
            metadata={
                "surface_form": "the tool",
                "resolved_label": "the recidivism scoring tool",
                "mention_resolved_from_context": True,
            },
        )
        EntityNormalizer().normalize_entity(e)
        # it can help retrieval via `aliases`
        self.assertIn("the recidivism scoring tool", e.metadata.get("aliases", []))
        # but is NOT trusted identity evidence
        self.assertNotIn("the recidivism scoring tool",
                         e.metadata.get("source_grounded_aliases", []))

    def test_consolidation_style_alias_is_not_source_grounded(self):
        # simulate a later stage appending a merged-entity label to aliases
        e = Entity(uri="https://ex.org/e3", label="Widget", entity_type="Concept", confidence=0.9,
                   metadata={"surface_form": "Widget"})
        EntityNormalizer().normalize_entity(e)
        e.metadata.setdefault("aliases", []).append("Widget Assembly Line")  # consolidation-derived
        self.assertIn("Widget Assembly Line", e.metadata["aliases"])
        self.assertNotIn("Widget Assembly Line", e.metadata.get("source_grounded_aliases", []))


class AttachLocalSurfaceAliasSourceGroundedTests(unittest.TestCase):
    def _bare_pipeline(self):
        from pipeline import EnhancedSemanticPipeline
        return EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)

    def test_verbatim_discourse_slice_is_source_grounded(self):
        p = self._bare_pipeline()
        text = "aaaa Acme Widget Co bbbb the Acme Widget Corporation cccc"
        s1 = text.index("Acme Widget Co")
        ent = Entity(
            uri="https://ex.org/e1", label="Acme Widget Corporation", entity_type="Organisation",
            confidence=0.9,
            metadata={"discourse_id": "7",
                      "mentions": [{"discourse_id": "7", "start_char": s1, "end_char": s1 + len("Acme Widget Co")}]},
        )
        p._attach_local_surface_aliases(
            entities=[ent], discourse_text=text, discourse_id="7",
            unit_start=0, unit_end=len(text),
        )
        self.assertIn("Acme Widget Co", ent.metadata.get("aliases", []))
        self.assertIn("Acme Widget Co", ent.metadata.get("source_grounded_aliases", []))


class LinkerUnverifiedAliasCannotGainTrustTests(unittest.TestCase):
    """An alias that is only in metadata['aliases'] (inferred/consolidated)
    and is NOT passed as a source_grounded_alias cannot confer
    'source_grounded_alias' identity evidence, nor bypass the B ambiguity
    gate."""

    @staticmethod
    def _cand(qid, label, description="", aliases=None):
        return {"qid": qid, "label": label, "description": description,
                "aliases": list(aliases or []), "match_type": "label", "match_text": label}

    def _search(self, catalog):
        def _impl(_self, label, limit=25):
            return list(catalog.get(label.strip(), []))
        return _impl

    def test_source_grounded_alias_is_not_a_free_pass_for_an_ambiguous_homonym(self):
        # A source_grounded_alias that matches NEITHER a candidate label
        # near-exactly must not rescue an ambiguous exact-label homonym.
        cands = [
            self._cand("Q_A", "Vault", "2000s comic book series"),
            self._cand("Q_B", "Vault", "financial institution room"),
        ]
        with patch.object(WikidataLinker, "_search_api", self._search({"Vault": cands})):
            linker = WikidataLinker()
            r = linker.link("Vault", entity_type="Concept",
                            context_text="the team stored keys in the vault before the review",
                            source_grounded_aliases=["the vault"])  # verified but non-discriminating
        self.assertIsNone(r)

    def test_wikidata_candidate_alias_match_is_never_source_grounded(self):
        # A candidate matched via its OWN Wikidata alias (not anything the
        # caller verified) must be labelled 'wikidata_alias'/'fuzzy_label',
        # never 'source_grounded_alias'.
        cands = [
            self._cand("Q1", "Some Canonical Name", "a widget maker", aliases=["Widget Co"]),
        ]
        with patch.object(WikidataLinker, "_search_api", self._search({"Widget Co": cands})):
            linker = WikidataLinker()
            r = linker.link("Widget Co", entity_type="Organisation",
                            context_text="Widget Co, a widget maker, announced a product")
        if r is not None:
            self.assertNotEqual(r["identity_evidence"].split("+")[0], "source_grounded_alias")
            self.assertIn(r["identity_evidence"].split("+")[0], {"wikidata_alias", "fuzzy_label"})

    def test_genuinely_verified_source_grounded_alias_is_trusted(self):
        # The caller HAS verified "GPT-4" appears in the entity's own
        # source text -> a candidate whose primary label is "GPT-4" is a
        # trusted match, and this corroboration lets the ambiguity gate
        # accept it over a bare exact-label homonym.
        cands = [
            self._cand("Q_RIGHT", "GPT-4", "large language model developed by OpenAI"),
            self._cand("Q_WRONG", "GPT4", "unrelated exact-label item", aliases=["GPT4"]),
        ]
        with patch.object(WikidataLinker, "_search_api", self._search({"GPT4": cands})):
            linker = WikidataLinker()
            r = linker.link("GPT4", entity_type="AISystem",
                            context_text="the system is a large language model developed for text tasks",
                            source_grounded_aliases=["GPT-4"])
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q_RIGHT")
        self.assertEqual(r["identity_evidence"].split("+")[0], "source_grounded_alias")

    def test_pipeline_call_site_passes_only_surface_form_and_allowlist(self):
        # The real pipeline call site must not forward raw metadata["aliases"].
        import inspect
        from pipeline import EnhancedSemanticPipeline
        src = inspect.getsource(EnhancedSemanticPipeline._process_entities_batch)
        self.assertIn("source_grounded_aliases", src)
        self.assertNotIn("source_grounded_aliases=_sg_aliases if isinstance(_sg_aliases, list) else None", src)
        self.assertIn('e.metadata.get("source_grounded_aliases")', src)


if __name__ == "__main__":
    unittest.main()
