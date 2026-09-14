"""B production fix -- two parts.

A. Acceptance safety: a lexical/alias homonym with strong rivals must not
   be accepted just because 2-3 GENERIC shared context tokens overlap. The
   corroboration test now counts only DISCRIMINATING shared tokens --
   those not also in the strongest rival's description and not generic
   across the retrieved candidate pool (present in > 1/3 of the other
   candidates' descriptions). No global threshold change, no word list.

B. RDF identity semantics: owl:sameAs is emitted ONLY for an authoritative
   Wikidata alignment (wikidata_authoritative True). A heuristic best-guess
   is serialized as skos:closeMatch (non-transitive, not identity). QID,
   score, identity evidence, authority flag and decision reason are always
   preserved on the entity.

Frozen candidates / synthetic ids only; no network.
"""
import importlib.util
import unittest
from unittest.mock import patch

from config import Config
from models import Entity
from utils.wikidata_linker import WikidataLinker

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]


def _cand(qid, label, description="", aliases=None):
    return {"qid": qid, "label": label, "description": description,
            "aliases": list(aliases or []), "match_type": "label", "match_text": label}


def _link(catalog, query, **kw):
    def _impl(self, label, limit=25):
        return list(catalog.get(label.strip(), []))
    with patch.object(WikidataLinker, "_search_api", _impl):
        return WikidataLinker().link(query, **kw)


# ----------------------------------------------------------------------
# A -- strengthened corroboration
# ----------------------------------------------------------------------

class GenericTokenCorroborationTests(unittest.TestCase):
    """Two exact 'Signal' homonyms; only the winner's description shares
    tokens with the context. When those shared tokens are GENERIC across
    the pool -> abstain. When they are DISCRIMINATING -> accept. Same
    counts either way, so only the generic-vs-discriminating distinction
    can flip it."""

    QUERY = "Beacon"
    CTX = ("the team deployed a Beacon to broadcast a periodic radio pulse "
           "from the lighthouse tower to passing ships at night")

    def _pool(self, *, generic):
        # winner: shares exactly two tokens with CTX ("radio", "pulse")
        winner = _cand("Q_WIN", "Beacon",
                       "device that emits a radio pulse for navigation")
        if generic:
            # both rivals' descriptions ALSO contain 'radio' and 'pulse'
            # -> pool-generic, not discriminating
            r1 = _cand("Q_R1", "Beacon", "radio pulse transmitter used in aviation")
            r2 = _cand("Q_R2", "Beacon", "radio pulse marker on a map")
        else:
            r1 = _cand("Q_R1", "Beacon", "surname of Welsh origin")
            r2 = _cand("Q_R2", "Beacon", "2007 studio album")
        return [winner, r1, r2]

    def test_generic_shared_tokens_do_not_corroborate(self):
        r = _link({self.QUERY: self._pool(generic=True)}, self.QUERY,
                  entity_type="Concept", context_text=self.CTX)
        self.assertIsNone(r)

    def test_discriminating_shared_tokens_still_corroborate(self):
        r = _link({self.QUERY: self._pool(generic=False)}, self.QUERY,
                  entity_type="Concept", context_text=self.CTX)
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q_WIN")

    def test_tokens_shared_with_strongest_rival_do_not_corroborate(self):
        # winner and the strong rival have the SAME two context-shared
        # tokens in their descriptions -> cannot separate them -> abstain.
        winner = _cand("Q_WIN", "Beacon", "device that emits a radio pulse for navigation")
        rival = _cand("Q_RIV", "Beacon", "unit that emits a radio pulse for tracking")
        far = _cand("Q_FAR", "Beacon", "surname")
        r = _link({self.QUERY: [winner, rival, far]}, self.QUERY,
                  entity_type="Concept", context_text=self.CTX)
        self.assertIsNone(r)


# ----------------------------------------------------------------------
# B -- RDF identity semantics
# ----------------------------------------------------------------------

class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


def _annotations():
    return {
        "themes": {}, "questions": {}, "answers": {}, "votes": {},
        "experts": {}, "documents": {},
        "metadata": {"datasetId": "unit_ds", "runId": "run-unit"},
    }


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class WikidataRDFIdentitySemanticsTests(unittest.TestCase):
    def _serialize(self, entities):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer(annotations_dict=_annotations(),
                          registry=_Registry(entities), namespaces=Config.NAMESPACES,
                          prune_isolated_entities=False)
        return s, Namespace(ETD)

    _BASE = ETD.rstrip("#")

    def _entity(self, slug, wd):
        return Entity(uri=f"{self._BASE}/entity/concept/{slug}", label=slug.title(),
                      entity_type="Concept", confidence=0.9,
                      metadata={"surface_form": slug, "discourse_id": "d1",
                                "field": "answer.description",
                                "wikidata": wd,
                                "wikidata_identity_evidence": wd.get("identity_evidence"),
                                "wikidata_authoritative": bool(wd.get("authoritative")),
                                "wikidataScore": wd.get("score")})

    _HEURISTIC = {"qid": "Q111111", "label": "Some Item", "description": "a thing",
                  "score": 0.71, "identity_evidence": "wikidata_alias+corroborated",
                  "authoritative": False, "decision_reason": "accepted"}
    _AUTH = {"qid": "Q222222", "label": "Named Org", "description": "an organisation",
             "score": 0.97, "identity_evidence": "exact_label",
             "authoritative": True, "decision_reason": "accepted"}

    def test_heuristic_link_does_not_emit_owl_sameas(self):
        from rdflib import RDF, OWL, SKOS, URIRef
        e = self._entity("heuristic-x", self._HEURISTIC)
        s, ns = self._serialize([e])
        ent = URIRef(str(e.uri))
        wd = URIRef("http://www.wikidata.org/entity/Q111111")
        self.assertNotIn((ent, OWL.sameAs, wd), s.graph)
        self.assertIn((ent, SKOS.closeMatch, wd), s.graph)
        self.assertIn((ent, ns.wikidata, wd), s.graph)

    def test_authoritative_link_emits_owl_sameas(self):
        from rdflib import RDF, OWL, SKOS, URIRef
        e = self._entity("auth-x", self._AUTH)
        s, ns = self._serialize([e])
        ent = URIRef(str(e.uri))
        wd = URIRef("http://www.wikidata.org/entity/Q222222")
        self.assertIn((ent, OWL.sameAs, wd), s.graph)
        self.assertNotIn((ent, SKOS.closeMatch, wd), s.graph)
        self.assertIn((ent, ns.wikidata, wd), s.graph)

    def test_identity_and_authority_metadata_survives_serialization(self):
        from rdflib import URIRef, Literal, XSD
        for e, want_auth in ((self._entity("h2", self._HEURISTIC), False),
                             (self._entity("a2", self._AUTH), True)):
            s, ns = self._serialize([e])
            ent = URIRef(str(e.uri))
            auth = list(s.graph.objects(ent, ns.wikidataAuthoritative))
            self.assertEqual([bool(x.toPython()) for x in auth], [want_auth])
            ev = [str(x) for x in s.graph.objects(ent, ns.wikidataIdentityEvidence)]
            self.assertEqual(ev, [e.metadata["wikidata"]["identity_evidence"]])
            reason = [str(x) for x in s.graph.objects(ent, ns.wikidataDecisionReason)]
            self.assertEqual(reason, ["accepted"])
            score = list(s.graph.objects(ent, ns.wikidataScore))
            self.assertEqual(len(score), 1)
            qid_ref = list(s.graph.objects(ent, ns.wikidata))
            self.assertEqual([str(x).rsplit("/", 1)[-1] for x in qid_ref],
                             [e.metadata["wikidata"]["qid"]])

    def test_declared_alignment_predicates_have_lifecycle_and_no_collision(self):
        from rdflib import RDF, OWL
        from ontology.add_schema import LIFECYCLE_STATUSES, normalized_etd_local_name
        s, ns = self._serialize([self._entity("h3", self._HEURISTIC)])
        g = s.graph
        for local in ("wikidataAuthoritative", "wikidataIdentityEvidence", "wikidataDecisionReason"):
            p = ns[local]
            self.assertIn((p, RDF.type, OWL.DatatypeProperty), g)
            st = [str(o) for o in g.objects(p, ns.lifecycleStatus)]
            self.assertTrue(st and st[0] in LIFECYCLE_STATUSES)
        # no class shares a normalised local name with these properties
        cls_norms = {normalized_etd_local_name(str(c)[len(str(ns)):])
                     for c in g.subjects(RDF.type, OWL.Class) if str(c).startswith(str(ns))}
        for local in ("wikidataAuthoritative", "wikidataIdentityEvidence", "wikidataDecisionReason"):
            self.assertNotIn(normalized_etd_local_name(local), cls_norms)


if __name__ == "__main__":
    unittest.main()
