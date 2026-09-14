"""
models.py
---------
Core data models for the Semantic Knowledge Graph Pipeline.
Defines Entity, Relation, ProcessingResult, AnnotationBatch, and metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

def _looks_like_iri(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"))

# ============================================================
# ENTITY MODEL (UPDATED FOR LLM + ONTOLOGY MAPPING PIPELINE)
# ============================================================

@dataclass
class Entity:
    """
    Represents an extracted entity with:
    - canonical URI
    - LLM semantic type
    - ontology mapping
    - provenance (doc ID + text span)
    - extra metadata
    """

    uri: str                               # Canonical entity URI (etd:xxxxx)
    label: str                             # Human-readable surface form
    entity_type: str                       # LLM semantic type (Dataset, BiasType…)
    confidence: float = 0.85

    ontology_uri: Optional[str] = None     # Primary ontology class (VAIR / AIPO / etc.)

    # Provenance
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    doc_id: Optional[str] = None
    source: str = "llm"                    # "llm", "regex", "manual", etc.

    # For cross-ontology alignment (AIPO, VAIR, FMO…)
    additional_ontologies: Dict[str, str] = field(default_factory=dict)

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Validation ---
    def __post_init__(self):
        try:
            self.confidence = float(self.confidence)
        except Exception as e:
            raise ValueError("confidence must be a float") from e
        if not self.uri:
            raise ValueError("Entity.uri must be a non-empty string")
        if not _looks_like_iri(self.uri):
            raise ValueError(f"Entity.uri must look like an IRI, got: {self.uri}")        
        if not self.label:
            raise ValueError("Entity.label must be a non-empty string")
        if not self.entity_type:
            raise ValueError("Entity.entity_type cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Entity.confidence must be between 0 and 1")

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "label": self.label,
            "entity_type": self.entity_type,
            "ontology_uri": self.ontology_uri,
            "confidence": round(self.confidence, 3),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "doc_id": self.doc_id,
            "source": self.source,
            "additional_ontologies": self.additional_ontologies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            uri=data["uri"],
            label=data["label"],
            entity_type=data["entity_type"],
            ontology_uri=data.get("ontology_uri"),
            confidence=data.get("confidence", 0.85),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            doc_id=data.get("doc_id"),
            source=data.get("source", "llm"),
            additional_ontologies=data.get("additional_ontologies", {}),
            metadata=data.get("metadata", {})
        )


# ============================================================
# RELATION MODEL (COMPATIBLE WITH LLM RELATION EXTRACTION)
# ============================================================

# Object types representing literal-valued relation targets rather than
# entity references. Kept consistent with the literal-object policy in
# utils/semantic_relation_policy.py's _relation_has_literal_object, so a
# value in this set is never left describing a target that has since
# become a genuine internal (Entity/Proposition) URI.
_LITERAL_OBJECT_TYPE_MARKERS = {
    "literal", "quantity", "money", "date", "percentage", "count",
    "temporal_range", "comparison",
}


@dataclass
class Relation:
    """
    Represents a semantic relation between two entities.
    Uses canonical URIs, supports ontology alignment, and inference flags.
    """

    # --- REQUIRED (non-default) fields FIRST ---
    source: str                            # entity URI (subject)
    relation: str                          # LLM relation label
    target: str                            # entity URI (object)

    # --- OPTIONAL (defaulted) fields AFTER ---
    uri: Optional[str] = None              # canonical URI for the relation triple
    confidence: float = 0.75
    ontology_uri: Optional[str] = None     # mapped ontology class (e.g. rel:causes)
    target_literal: Optional[Any] = None   # literal-valued object when no entity target exists
    target_datatype: Optional[str] = None  # explicit datatype IRI / CURIE / local marker
    target_is_literal: bool = False
    object_type: Optional[str] = None      # entity, literal, quantity, money, date, percentage, comparison...
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    predicate_text: Optional[str] = None
    predicate_mapping_status: str = "mapped"
    negated: bool = False
    temporal_information: Dict[str, Any] = field(default_factory=dict)
    quantitative_information: Dict[str, Any] = field(default_factory=dict)

    is_inferred: bool = False
    context: Optional[str] = None          # sentence or doc segment
    doc_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)



    # --- Validation ---
    def __post_init__(self):
        try:
            self.confidence = float(self.confidence)
        except Exception as e:
            raise ValueError("confidence must be a float") from e        
        
        if not self.uri:
            # Allow late binding (pipeline may set uri after constructing the object)
            # Serializer/graph layer must ensure a URI exists before RDF export.
            self.uri = "" 
        if not self.source or not self.target:
            raise ValueError("Relation.source and Relation.target must be non-empty")
        if not self.relation:
            raise ValueError("Relation.relation must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Relation.confidence must be between 0 and 1")

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "source": self.source,
            "relation": self.relation,
            "ontology_uri": self.ontology_uri,
            "target": self.target,
            "confidence": round(self.confidence, 3),
            "is_inferred": self.is_inferred,
            "context": self.context,
            "doc_id": self.doc_id,
            "target_literal": self.target_literal,
            "target_datatype": self.target_datatype,
            "target_is_literal": self.target_is_literal,
            "object_type": self.object_type,
            "qualifiers": self.qualifiers,
            "predicate_text": self.predicate_text,
            "predicate_mapping_status": self.predicate_mapping_status,
            "negated": self.negated,
            "temporal_information": self.temporal_information,
            "quantitative_information": self.quantitative_information,
            "metadata": self.metadata,
        }

    def set_target_entity(self, uri: str) -> None:
        """Single enforcement point for the endpoint-state
        invariant -- whenever `target` is (re)assigned to point at an
        internal semantic node URI (an Entity or a Proposition), the
        literal-value fields must never be left describing a different,
        now-stale value. Every pipeline stage that reassigns an already-
        constructed Relation's target to a resolved/canonical/redirected
        entity URI (coreference resolution, canonical-linking redirects,
        direction-swap corrections, ...) should call this instead of
        setting `target`/`target_is_literal` by hand, so the invariant
        holds regardless of which stage performs the reassignment.

        Deliberately does not touch `source` or any subject-side metadata
        -- this is the object/target side of the model only."""
        self.target = str(uri)
        self.target_is_literal = False
        self.target_literal = None
        self.target_datatype = None
        if str(self.object_type or "").strip().lower() in _LITERAL_OBJECT_TYPE_MARKERS:
            self.object_type = None
        if isinstance(self.metadata, dict) and str(self.metadata.get("object_type") or "").strip().lower() in _LITERAL_OBJECT_TYPE_MARKERS:
            self.metadata.pop("object_type", None)

    def set_target_literal(
        self,
        value: Any,
        *,
        datatype: Optional[str] = None,
        object_type: str = "literal",
        qualifiers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Symmetric counterpart to set_target_entity -- single
        enforcement point for the endpoint-state invariant in the reverse
        direction, whenever `target` is (re)assigned to a literal value
        instead of an internal entity/proposition URI (e.g. a quantity/
        statistic candidate that was resolved as a scalar value rather than
        minted as a graph node). No stale entity-URI-shaped `target` may
        survive this call: `target` becomes the literal's string form, and
        `target_is_literal`/`target_literal`/`target_datatype`/`object_type`
        are set together so no combination of these fields can be read as
        "this endpoint is still an entity reference" afterwards.

        `qualifiers`, when given, are merged into (not replacing) any
        existing qualifiers already on the relation -- consistent with how
        the LLM relation extractor already populates this dict."""
        literal_str = str(value)
        self.target = literal_str
        self.target_is_literal = True
        self.target_literal = value
        self.target_datatype = datatype
        self.object_type = object_type
        if isinstance(self.metadata, dict):
            self.metadata["object_type"] = object_type
        if qualifiers:
            if not isinstance(self.qualifiers, dict):
                self.qualifiers = {}
            self.qualifiers.update(qualifiers)

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(
            uri=data["uri"],
            source=data["source"],
            relation=data["relation"],
            target=data["target"],
            confidence=data.get("confidence", 0.75),
            ontology_uri=data.get("ontology_uri"),
            target_literal=data.get("target_literal"),
            target_datatype=data.get("target_datatype"),
            target_is_literal=data.get("target_is_literal", False),
            object_type=data.get("object_type"),
            qualifiers=data.get("qualifiers", {}) or {},
            predicate_text=data.get("predicate_text"),
            predicate_mapping_status=data.get("predicate_mapping_status", "mapped"),
            negated=data.get("negated", False),
            temporal_information=data.get("temporal_information", {}) or {},
            quantitative_information=data.get("quantitative_information", {}) or {},
            is_inferred=data.get("is_inferred", False),
            context=data.get("context"),
            doc_id=data.get("doc_id"),
            metadata=data.get("metadata", {})
        )


# ============================================================
# PROCESSING RESULT
# ============================================================

@dataclass
class ProcessingResult:
    """Result of processing one document/theme/question."""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "metadata": self.metadata,
            "errors": self.errors
        }


# ============================================================
# BATCH ANNOTATION MODEL
# ============================================================

@dataclass
class AnnotationBatch:
    """Holds all processing results for all document categories."""

    themes: Dict[str, ProcessingResult] = field(default_factory=dict)
    questions: Dict[str, ProcessingResult] = field(default_factory=dict)
    answers: Dict[str, ProcessingResult] = field(default_factory=dict)
    votes: Dict[str, ProcessingResult] = field(default_factory=dict)
    experts: Dict[str, ProcessingResult] = field(default_factory=dict)
    documents: Dict[str, ProcessingResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "themes": {k: v.to_dict() for k, v in self.themes.items()},
            "questions": {k: v.to_dict() for k, v in self.questions.items()},
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "votes": {k: v.to_dict() for k, v in self.votes.items()},
            "experts": {k: v.to_dict() for k, v in self.experts.items()},
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
        }

    def get_all_entities(self) -> List[Entity]:
        all_e = []
        for block in [self.themes, self.questions, self.answers, self.votes, self.experts, self.documents]:
            for result in block.values():
                all_e.extend(result.entities)
        return all_e

    def get_all_relations(self) -> List[Relation]:
        all_r = []
        for block in [self.themes, self.questions, self.answers, self.votes, self.experts, self.documents]:
            for result in block.values():
                all_r.extend(result.relations)
        return all_r


# ============================================================
# QUALITY METRICS
# ============================================================

@dataclass
class QualityMetrics:
    """Aggregated quality metrics for the entire knowledge graph."""

    total_entities: int = 0
    unique_entities: int = 0
    total_relations: int = 0
    entity_coverage: float = 0.0
    relation_coverage: float = 0.0
    avg_confidence: float = 0.0
    linking_rate: float = 0.0
    overall_quality_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_entities": self.total_entities,
            "unique_entities": self.unique_entities,
            "total_relations": self.total_relations,
            "entity_coverage": round(self.entity_coverage, 3),
            "relation_coverage": round(self.relation_coverage, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "linking_rate": round(self.linking_rate, 3),
            "overall_quality_score": round(self.overall_quality_score, 2),
            "timestamp": self.timestamp
        }
