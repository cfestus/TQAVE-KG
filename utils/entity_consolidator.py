import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union

from config import Config
from utils.entity_normalizer import (
    EntityNormalizer,
    containment_adds_material_token,
    is_initialism_pair,
)
from utils.helpers import SOURCE_IDENTITY_SCALAR_KEYS, dedup_mentions, mention_from_source_identity
from ontology.add_schema import SUBCLASS_MAP


class EntityConsolidator:
    _STRUCTURAL_LABEL_RE = re.compile(
        r"^(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+$",
        flags=re.IGNORECASE,
    )

    def __init__(self, registry, normalizer: Optional[EntityNormalizer] = None, output_dir: Optional[Union[Path, str]] = None):
        self.registry = registry
        self.normalizer = normalizer or EntityNormalizer(registry=registry)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(getattr(Config, "OUTPUT_DIR", Path("output")))
        self.redirects: Dict[str, str] = {}
        self.merge_events: List[Dict[str, Any]] = []

    def _normalize_mention(self, mention: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(mention, dict):
            return None
        did_raw = mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if not did:
            return None
        try:
            start = int(mention.get("start_char"))
            end = int(mention.get("end_char"))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        out: Dict[str, Any] = {
            "discourse_id": did,
            "start_char": int(start),
            "end_char": int(end),
        }
        field = str(mention.get("field") or mention.get("source_field") or "").strip()
        if field:
            out["field"] = field
        try:
            ci = int(mention.get("chunk_index"))
            if ci >= 0:
                out["chunk_index"] = int(ci)
        except (TypeError, ValueError):
            pass
        return out

    def _merge_mentions(self, canonical_meta: Dict[str, Any], source_meta: Dict[str, Any]) -> None:
        merged: List[Dict[str, Any]] = []
        for meta in (canonical_meta, source_meta):
            bucket = meta.get("mentions")
            if isinstance(bucket, list):
                for item in bucket:
                    norm = self._normalize_mention(item)
                    if norm is not None:
                        merged.append(norm)
            scalar_mention = mention_from_source_identity(meta)
            if scalar_mention is not None:
                merged.append(scalar_mention)
        if not merged:
            return
        canonical_meta["mentions"] = dedup_mentions(merged)

    @staticmethod
    def _entity_scope(e: Any) -> Optional[str]:
        """Identity scope for entities not yet resolved to a
        genuine antecedent -- e.g. unresolved referential mentions,
        unit-scoped Propositions -- driven purely by a generic
        metadata flag set by the producer, never by label vocabulary. None
        for genuine entities, which keep merging by label as before."""
        meta = getattr(e, "metadata", None) or {}
        if not meta.get("discourse_scoped_identity"):
            return None
        return str(meta.get("origin_key") or meta.get("discourse_id") or "")

    def consolidate(self, entities: List[Any]) -> List[Any]:
        if not entities:
            return []

        groups: Dict[Tuple[str, str, str, Optional[str]], List[Any]] = defaultdict(list)
        alias_map: Dict[str, str] = {}

        for e in entities:
            if e is None:
                continue

            label = getattr(e, "label", "") or ""
            norm_label = self.normalizer.normalize_label(label)
            etype = self.normalizer.normalize_type(getattr(e, "entity_type", "Entity") or "Entity")

            ont = ""
            if hasattr(e, "metadata") and e.metadata:
                ont = e.metadata.get("ontology_uri") or ""
            if not ont:
                ont = getattr(e, "ontology_uri", "") or ""

            key = (self.normalizer.match_key(norm_label), etype, str(ont).strip(), self._entity_scope(e))
            groups[key].append(e)
            alias_map[label] = norm_label

        consolidated: List[Any] = []

        for _, group in groups.items():
            canonical = self._pick_canonical(group)
            for other in group:
                if other is canonical:
                    continue
                self._merge_entity_into_canonical(canonical, other, reason="exact_key", score=1.0)
            consolidated.append(canonical)

        # Conservative fuzzy pass within type buckets only
        by_type: Dict[str, List[Any]] = defaultdict(list)
        for e in consolidated:
            by_type[getattr(e, "entity_type", "Entity")].append(e)

        final_entities: List[Any] = []
        consumed: Set[str] = set()
        all_types = list(by_type.keys())

        for etype, ents in by_type.items():
            # Compare within this type bucket, plus against buckets that are
            # directly subclass/superclass-compatible (one hop only, see
            # _type_compatible) -- bounded to the immediate type family
            # rather than a global O(n^2) sweep across the whole entity set.
            related_pool: List[Any] = []
            for other_type in all_types:
                if other_type == etype:
                    continue
                if self._type_compatible(etype, other_type):
                    related_pool.extend(by_type[other_type])

            for i, a in enumerate(ents):
                a_uri = str(getattr(a, "uri", ""))
                if a_uri in consumed:
                    continue

                for j in range(i + 1, len(ents)):
                    b = ents[j]
                    b_uri = str(getattr(b, "uri", ""))
                    if b_uri in consumed:
                        continue
                    if self._should_merge_fuzzy(a, b):
                        score = self._fuzzy_score(getattr(a, "label", ""), getattr(b, "label", ""))
                        self._merge_entity_into_canonical(a, b, reason="fuzzy", score=score)
                        consumed.add(b_uri)

                for b in related_pool:
                    b_uri = str(getattr(b, "uri", ""))
                    if b_uri in consumed or b_uri == a_uri:
                        continue
                    if self._should_merge_fuzzy(a, b):
                        score = self._fuzzy_score(getattr(a, "label", ""), getattr(b, "label", ""))
                        self._merge_entity_into_canonical(a, b, reason="fuzzy", score=score)
                        consumed.add(b_uri)

                final_entities.append(a)
                consumed.add(a_uri)

        self._write_artifacts(alias_map)
        return final_entities

    @staticmethod
    def _type_compatible(type_a: str, type_b: str) -> bool:
        """True when type_a == type_b, one is the OTHER's direct declared
        rdfs:subClassOf parent per ontology/add_schema.py's SUBCLASS_MAP
        (e.g. DemographicGroup and its declared parent PopulationGroup), or
        one side is the bare generic "Entity" type (meaning no more
        specific type was ever determined for that mention) while the
        other is any specific type. A same-context mention typed at two
        adjacent specificity levels of a KNOWN hierarchy relation, or typed
        generically on one side and specifically on the other, is still the
        same referent -- strict entity_type string equality alone
        previously left such pairs (e.g. "black defendants" typed once as
        PopulationGroup and once as DemographicGroup, or "openai"/"gpt-4o"
        typed once as bare Entity and once as Organisation/Model, all
        within the same chunk) permanently unmerged.

        The bare-Entity case is deliberately its own rule, not folded into
        the one-hop SUBCLASS_MAP walk: "Entity" is the root of the ENTIRE
        hierarchy, so a naive transitive climb from it would match every
        declared type in the ontology, merging distant-cousin types that
        merely share that universal ancestor (e.g. Dataset and
        EthicalPrinciple). It is safe here specifically because bare
        "Entity" never represents a genuine, independently-determined type
        of its own -- it only ever means "no more specific type was
        assigned" -- so there is no real distinct type on that side to
        protect. The SUBCLASS_MAP walk itself stays one-hop only, for the
        same distant-cousin reason, for every OTHER pair."""
        a = str(type_a or "")
        b = str(type_b or "")
        if a == b:
            return True
        if a == "Entity" or b == "Entity":
            return True
        return SUBCLASS_MAP.get(a) == b or SUBCLASS_MAP.get(b) == a

    def _pick_canonical(self, group: List[Any]) -> Any:
        return max(group, key=lambda x: float(getattr(x, "confidence", 0.0) or 0.0))

    def _fuzzy_score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, self.normalizer.match_key(a), self.normalizer.match_key(b)).ratio()

    def _is_structural_id_like(self, entity: Any) -> bool:
        etype = str(getattr(entity, "entity_type", "") or "")
        if etype in {"Theme", "Question", "Answer", "Vote", "Expert", "Document", "DiscourseUnit"}:
            return True
        label = str(getattr(entity, "label", "") or "").strip()
        if label and self._STRUCTURAL_LABEL_RE.match(label):
            return True
        uri = str(getattr(entity, "uri", "") or "").strip().lower()
        if any(tok in uri for tok in ("/theme/", "/question/", "/answer/", "/vote/", "/expert/", "discourse_")):
            return True
        return False

    def _should_merge_fuzzy(self, a: Any, b: Any) -> bool:
        la = getattr(a, "label", "") or ""
        lb = getattr(b, "label", "") or ""
        if self._is_structural_id_like(a) or self._is_structural_id_like(b):
            return False
        if self._entity_scope(a) != self._entity_scope(b):
            return False
        if self.normalizer.should_avoid_fuzzy_merge(la) or self.normalizer.should_avoid_fuzzy_merge(lb):
            return False
        if not self._type_compatible(getattr(a, "entity_type", None), getattr(b, "entity_type", None)):
            return False

        ka = self.normalizer.match_key(la)
        kb = self.normalizer.match_key(lb)
        if not ka or not kb:
            return False
        if self._is_coordinated_superset_pair(ka, kb):
            return False

        # Identical after normalization: a spelling / case / punctuation
        # variant, or the same named entity re-registered under a
        # differently-hashed URI -- always the same referent.
        if ka == kb:
            return True

        is_containment = ka in kb or kb in ka
        score = self._fuzzy_score(la, lb)

        if is_containment:
            # Containment ALONE is not identity evidence. A qualifier that
            # adds a content token (broader vs narrower, population vs
            # subgroup, "non-"/opposite modifier, generic vs qualified
            # noun) is a DIFFERENT referent -- even at a high lexical
            # score. Merge only when the extra material is inflection /
            # word-order / an English function word, OR when an
            # independent signal says SAME REFERENT.
            short_key, long_key = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
            if len(short_key) < 4 or len(long_key) < 4:
                return False
            if not containment_adds_material_token(short_key, long_key):
                return True
            return self._has_same_referent_evidence(a, b, la, lb)

        # Not a containment pair: pure lexical similarity, strict threshold.
        return score >= 0.93

    def _entity_external_id(self, e: Any) -> str:
        meta = getattr(e, "metadata", None) or {}
        for key in ("external_id", "wikidata_uri", "wikidata_id", "qid", "doi", "url"):
            v = str(meta.get(key) or "").strip().lower()
            if v:
                return v
        return ""

    @staticmethod
    def _authoritative_external_id(e: Any) -> str:
        """An external identifier that is an ASSERTED identity, not a
        heuristic best-guess alignment: a DOI, an authoritative URL, an
        explicit external_id, or a Wikidata QID that was flagged
        wikidata_authoritative=True upstream. A Wikidata QID from the
        heuristic API linker (B) -- wikidata_authoritative=False, the
        default -- is deliberately NOT returned here: it is useful
        supporting evidence, but on its own must not override the
        material-token containment guard."""
        meta = getattr(e, "metadata", None) or {}
        for key in ("doi", "url", "external_id"):
            v = str(meta.get(key) or "").strip().lower()
            if v:
                return v
        if bool(meta.get("wikidata_authoritative")):
            for key in ("wikidata_uri", "wikidata_id", "qid"):
                v = str(meta.get(key) or "").strip().lower()
                if v:
                    return v
        return ""

    def _entity_alias_keys(self, e: Any, own_label: str) -> Set[str]:
        meta = getattr(e, "metadata", None) or {}
        raw = meta.get("aliases")
        keys: Set[str] = set()
        if isinstance(raw, list):
            for a in raw:
                if isinstance(a, str) and a.strip():
                    keys.add(self.normalizer.match_key(a))
        keys.discard(self.normalizer.match_key(own_label))
        keys.discard("")
        return keys

    def _mention_span_keys(self, e: Any) -> Set[Tuple[str, str, int, int]]:
        meta = getattr(e, "metadata", None) or {}
        out: Set[Tuple[str, str, int, int]] = set()
        for m in meta.get("mentions") or []:
            if not isinstance(m, dict):
                continue
            try:
                s = int(m.get("start_char"))
                en = int(m.get("end_char"))
            except (TypeError, ValueError):
                continue
            did = str(m.get("discourse_id") or "").strip()
            if not did or en <= s:
                continue
            out.add((str(m.get("field") or "").strip(), did, s, en))
        return out

    def _has_same_referent_evidence(self, a: Any, b: Any, la: str, lb: str) -> bool:
        """Independent (non-lexical) evidence that two containment-related
        labels denote ONE referent, drawn only from signals the pipeline
        already produces."""
        # Shared AUTHORITATIVE external identity (DOI / authoritative URL /
        # explicit external_id / wikidata_authoritative=True). A shared
        # heuristic Wikidata QID (wikidata_authoritative=False) is NOT
        # sufficient on its own -- it only counts alongside one of the
        # independent same-referent signals below.
        auth_a, auth_b = self._authoritative_external_id(a), self._authoritative_external_id(b)
        if auth_a and auth_b and auth_a == auth_b:
            return True
        # One label is a recorded alias of the other entity.
        a_aliases = self._entity_alias_keys(a, la)
        b_aliases = self._entity_alias_keys(b, lb)
        if self.normalizer.match_key(lb) in a_aliases or self.normalizer.match_key(la) in b_aliases:
            return True
        if a_aliases & b_aliases:
            return True
        # A genuine initialism / abbreviation of the other.
        if is_initialism_pair(la, lb):
            return True
        # Literally the same extracted span (same field+discourse+offsets).
        if self._mention_span_keys(a) & self._mention_span_keys(b):
            return True
        return False

    @staticmethod
    def _is_coordinated_superset_pair(left_key: str, right_key: str) -> bool:
        left = str(left_key or "").strip()
        right = str(right_key or "").strip()
        if not left or not right:
            return False
        longer, shorter = (left, right) if len(left) >= len(right) else (right, left)
        if not re.search(r"\b(?:and|or)\b", longer):
            return False
        parts = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"\s+(?:and|or)\s+", longer)
            if part.strip()
        ]
        return shorter in parts

    def _merge_entity_into_canonical(self, canonical: Any, source: Any, *, reason: str, score: float) -> None:
        canonical.metadata = canonical.metadata or {}
        source.metadata = source.metadata or {}

        # When this merge crosses a declared parent/child subclass pair
        # (e.g. PopulationGroup <-> DemographicGroup), or one side is the
        # bare generic "Entity" type, keep the MORE SPECIFIC type on the
        # surviving canonical entity, regardless of which side happened to
        # be picked as canonical -- the more specific type is strictly
        # more informative (bare "Entity" carries none at all) and is what
        # the ontology's own class definitions recommend using.
        c_type = str(getattr(canonical, "entity_type", "") or "")
        s_type = str(getattr(source, "entity_type", "") or "")
        if c_type != s_type and (SUBCLASS_MAP.get(s_type) == c_type or (c_type == "Entity" and s_type)):
            canonical.entity_type = s_type

        self._merge_mentions(canonical.metadata, source.metadata)

        # aliases
        aliases = canonical.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        src_label = getattr(source, "label", None)
        if src_label and src_label not in aliases and src_label != getattr(canonical, "label", ""):
            aliases.append(src_label)
        if aliases:
            canonical.metadata["aliases"] = aliases

        if not getattr(canonical, "ontology_uri", None) and getattr(source, "ontology_uri", None):
            canonical.ontology_uri = source.ontology_uri

        for k, v in source.metadata.items():
            if k == "wikidata":
                continue
            if k == "mentions":
                continue
            if k in SOURCE_IDENTITY_SCALAR_KEYS:
                # C.3: never merge one key of the atomic (field,
                # discourse_id, offsets, chunk) tuple across two occurrences
                # -- the canonical entity keeps its own coherent tuple; the
                # source entity's occurrence is already preserved in
                # canonical.metadata["mentions"] via _merge_mentions above.
                if not str(canonical.metadata.get(k) or "").strip():
                    canonical.metadata[k] = v
                continue
            if k not in canonical.metadata:
                canonical.metadata[k] = v
                continue
            existing = canonical.metadata[k]
            if not isinstance(existing, list):
                existing = [existing]
            incoming = v if isinstance(v, list) else [v]
            seen = set()
            merged = []
            for item in existing + incoming:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                if marker not in seen:
                    seen.add(marker)
                    merged.append(item)
            canonical.metadata[k] = merged

        c_conf = float(getattr(canonical, "confidence", 0.0) or 0.0)
        s_conf = float(getattr(source, "confidence", 0.0) or 0.0)
        if s_conf > c_conf:
            canonical.confidence = s_conf

        old_uri = str(getattr(source, "uri", ""))
        new_uri = str(getattr(canonical, "uri", ""))
        if old_uri and new_uri and old_uri != new_uri:
            # Keep the redirect map FLAT. A consolidation run can merge the
            # same entity in two steps (A -> B via exact key, then B -> C via
            # the later fuzzy pass). If we stored the raw hop, a consumer that
            # only does a single dict lookup (see
            # pipeline._apply_entity_redirects_to_annotations) would land on
            # the intermediate URI B -- which no surviving entity holds and
            # which the later CanonicalLinker has no redirect for -- leaving a
            # relation endpoint dangling on a provenance-less bare URI. Store
            # the terminal target instead, and collapse any existing chains
            # that were pointing at this hop.
            final_uri = self._terminal_redirect_target(new_uri)
            if final_uri and final_uri != old_uri:
                self.redirects[old_uri] = final_uri
                for key, value in list(self.redirects.items()):
                    if value == old_uri:
                        self.redirects[key] = final_uri
                if hasattr(self.registry, "replace_entity"):
                    self.registry.replace_entity(old_uri, final_uri)

    def _terminal_redirect_target(self, uri: str) -> str:
        """Follow ``self.redirects`` from ``uri`` to the end of the chain
        (cycle-guarded). Returns ``uri`` unchanged when it is not a redirect
        key."""
        seen: Set[str] = set()
        cur = str(uri or "")
        while cur in self.redirects and cur not in seen:
            seen.add(cur)
            nxt = str(self.redirects[cur] or "")
            if not nxt or nxt == cur:
                break
            cur = nxt
        return cur

        self.merge_events.append({
            "from_uri": old_uri,
            "to_uri": new_uri,
            "from_label": src_label,
            "to_label": getattr(canonical, "label", ""),
            "reason": reason,
            "score": round(float(score), 4),
            "entity_type": getattr(canonical, "entity_type", "Entity"),
        })

    def _write_artifacts(self, alias_map: Dict[str, str]) -> None:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        alias_path = out_dir / "alias_map.json"
        alias_path.write_text(json.dumps(dict(sorted(alias_map.items())), ensure_ascii=False, indent=2), encoding="utf-8")

        merge_path = out_dir / "merge_log.jsonl"
        with merge_path.open("w", encoding="utf-8") as f:
            for ev in sorted(
                self.merge_events,
                key=lambda x: (
                    str(x.get("to_uri", "")),
                    str(x.get("from_uri", "")),
                    str(x.get("reason", "")),
                    str(x.get("from_label", "")),
                ),
            ):
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
