from __future__ import annotations

from typing import Dict, List

from rdflib.namespace import RDF, RDFS, XSD


# Authoritative ETD predicate registry for serializer-emitted predicates.
# Serializer code must not emit ETD predicates from dynamic keys unless the local
# name appears in this registry.
ETD_PROPERTY_REGISTRY: Dict[str, Dict[str, object]] = {
    # --- Export/build metadata (operational metadata, broad domain by design) ---
    "datasetId": {
        "kind": "datatype",
        "label": "dataset identifier",
        "comment": "Identifier of the source dataset used for this export/run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "runId": {
        "kind": "datatype",
        "label": "run identifier",
        "comment": "Identifier of the pipeline execution run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "pipelineVersion": {
        "kind": "datatype",
        "label": "pipeline version",
        "comment": "Version identifier of the pipeline code/config used for export.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "gitCommit": {
        "kind": "datatype",
        "label": "git commit",
        "comment": "Source code commit hash associated with the export run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "shaclProfileVersion": {
        "kind": "datatype",
        "label": "SHACL profile version",
        "comment": "Version or identifier of SHACL profile used for validation.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "exportMode": {
        "kind": "datatype",
        "label": "export mode",
        "comment": "Export mode identifier (for example: ai).",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    # --- Reified relation metadata ---
    "confidence": {
        "kind": "datatype",
        "label": "confidence",
        "comment": "Confidence score for an entity/relation/evidence assertion.",
        "domains": ["Entity"],
        "range": XSD.float,
    },
    "isInferred": {
        "kind": "datatype",
        "label": "is inferred",
        "comment": "True when relation is inferred rather than directly extracted/structural.",
        "domains": ["Relation"],
        "range": XSD.boolean,
    },
    "inferenceRule": {
        "kind": "datatype",
        "label": "inference rule",
        "comment": "Deterministic identifier of the inference rule/path that produced a relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "relation_kind": {
        "kind": "datatype",
        "label": "relation kind",
        "comment": "Layer marker for relations such as structural, extracted_semantic, or inferred_semantic.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "section": {
        "kind": "datatype",
        "label": "section",
        "comment": "Structured source section associated with the relation or evidence.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "section_category": {
        "kind": "datatype",
        "label": "section category",
        "comment": "Section category such as semantic, resource, or metadata.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "semantic_validation_status": {
        "kind": "datatype",
        "label": "semantic validation status",
        "comment": "Outcome of semantic validation for a candidate relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "semantic_validation_reason": {
        "kind": "datatype",
        "label": "semantic validation reason",
        "comment": "Reason recorded by semantic validation for accepting a relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "semantic_validation_code": {
        "kind": "datatype",
        "label": "semantic validation code",
        "comment": "Machine-readable semantic validation decision code.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "relation_origin": {
        "kind": "datatype",
        "label": "relation origin",
        "comment": "Primary origin path for a semantic relation such as LLM, lexical recovery, canonical mapping, entailed mapping, or reasoner.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "predicate_policy_source": {
        "kind": "datatype",
        "label": "predicate policy source",
        "comment": "Predicate precedence tier used by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "predicate_surface_phrase": {
        "kind": "datatype",
        "label": "predicate surface phrase",
        "comment": "Local lexical predicate phrase supporting the canonical semantic relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "canonical_predicate": {
        "kind": "datatype",
        "label": "canonical predicate",
        "comment": "Canonical ontology predicate chosen by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "supersededProposedType": {
        "kind": "datatype",
        "label": "superseded proposed type",
        "comment": "Local ETD class name that was proposed for this entity by a supplementary/lexical typing signal but not asserted as a substantive rdf:type because it was ontology-incompatible with a type the entity already carries. Preserved for traceability only -- not current semantic truth.",
        "domains": ["Entity"],
        "range": XSD.string,
    },
    "predicate_mapping_status": {
        "kind": "datatype",
        "label": "predicate mapping status",
        "comment": "Mapping status for the predicate, such as mapped or local_unmapped.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "direction_normalized": {
        "kind": "datatype",
        "label": "direction normalized",
        "comment": "Direction normalization outcome recorded by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "locality_scope": {
        "kind": "datatype",
        "label": "locality scope",
        "comment": "Local proposition scope used to validate the semantic relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "discourse_mode": {
        "kind": "datatype",
        "label": "discourse mode",
        "comment": "Discourse mode preserved during semantic validation, such as declarative or interrogative.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "epistemic_status": {
        "kind": "datatype",
        "label": "epistemic status",
        "comment": "Primary epistemic modality preserved during semantic validation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "subject_surface_form": {
        "kind": "datatype",
        "label": "subject surface form",
        "comment": "Local subject surface form used by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "object_surface_form": {
        "kind": "datatype",
        "label": "object surface form",
        "comment": "Local object surface form used by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "subject_canonical_entity": {
        "kind": "datatype",
        "label": "subject canonical entity",
        "comment": "Canonical subject entity URI selected by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "object_canonical_entity": {
        "kind": "datatype",
        "label": "object canonical entity",
        "comment": "Canonical object entity URI selected by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "subject_family": {
        "kind": "datatype",
        "label": "subject semantic family",
        "comment": "Semantic endpoint family assigned to the subject.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "object_family": {
        "kind": "datatype",
        "label": "object semantic family",
        "comment": "Semantic endpoint family assigned to the object.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "object_type": {
        "kind": "datatype",
        "label": "object type",
        "comment": "Assertion object kind such as entity, literal, money, quantity, or date.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "entity_role_check": {
        "kind": "datatype",
        "label": "entity role check",
        "comment": "Entity-role validation outcome recorded by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "domain_range_check": {
        "kind": "datatype",
        "label": "domain range check",
        "comment": "Domain/range validation outcome recorded by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "entailment_check": {
        "kind": "datatype",
        "label": "entailment check",
        "comment": "Entailment validation outcome recorded by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "modality_check": {
        "kind": "datatype",
        "label": "modality check",
        "comment": "Modality preservation outcome recorded by the semantic relation policy.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "modality_tags": {
        "kind": "datatype",
        "label": "modality tags",
        "comment": "Pipe-delimited modality qualifiers attached to the semantic relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "target_is_literal": {
        "kind": "datatype",
        "label": "target is literal",
        "comment": "True when the semantic assertion object is a literal rather than an entity resource.",
        "domains": ["Relation"],
        "range": XSD.boolean,
    },
    "target_datatype": {
        "kind": "datatype",
        "label": "target datatype",
        "comment": "Datatype or literal-kind marker for literal-valued semantic assertion objects.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "qualifiers_json": {
        "kind": "datatype",
        "label": "qualifiers json",
        "comment": "JSON-encoded qualifiers attached to a semantic assertion.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "temporal_information_json": {
        "kind": "datatype",
        "label": "temporal information json",
        "comment": "JSON-encoded temporal qualifiers attached to a semantic assertion.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "quantitative_information_json": {
        "kind": "datatype",
        "label": "quantitative information json",
        "comment": "JSON-encoded quantitative qualifiers attached to a semantic assertion.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    "coverage_recovery": {
        "kind": "datatype",
        "label": "coverage recovery",
        "comment": "True when an assertion was recovered during semantic coverage recovery.",
        "domains": ["Relation"],
        "range": XSD.boolean,
    },
    "negated": {
        "kind": "datatype",
        "label": "negated",
        "comment": "True when the proposition is explicitly negated in the source text.",
        "domains": ["Relation"],
        "range": XSD.boolean,
    },
    # --- Raw field materialisation provenance ---
    "hasFieldAssertion": {
        "kind": "object",
        "label": "has field assertion",
        "comment": "Links a subject node to a field-level provenance assertion.",
        "domains": ["Entity"],
        "range": None,
    },
    "fieldName": {
        "kind": "datatype",
        "label": "field name",
        "comment": "Original source field name from raw dataset JSON.",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "recordId": {
        "kind": "datatype",
        "label": "record identifier",
        "comment": "Original source record identifier from raw dataset JSON.",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "sourceSection": {
        "kind": "datatype",
        "label": "source section",
        "comment": "Source section name (themes/questions/answers/votes/experts).",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "assertedPredicate": {
        "kind": "object",
        "label": "asserted predicate",
        "comment": "Predicate used by the corresponding materialised raw-field triple.",
        "domains": ["FieldAssertion"],
        "range": RDF.Property,
    },
    "assertedValue": {
        "kind": "rdf_property",
        "label": "asserted value",
        "comment": "Object used by the corresponding materialised raw-field triple.",
        "domains": ["FieldAssertion"],
        "range": RDFS.Resource,
    },
    # --- Evidence / grounding / offsets ---
    "source": {
        "kind": "datatype",
        "label": "source",
        "comment": "Extractor/provenance source identifier literal attached to relation/evidence metadata.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "field": {
        "kind": "datatype",
        "label": "field",
        "comment": "Source field identifier (for example theme.description).",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    # The larger sentence/semantic-unit text needed to interpret
    # a relation's full assertion (subject, antecedent, qualifiers,
    # surrounding context) -- distinct from Evidence.evidenceText, which
    # remains the exact minimal supporting span. Attached to the
    # DiscourseUnit a relation's Evidence node already links to via
    # etd:extractedFrom, so relations sharing one discourse/semantic unit
    # share one contextText literal rather than duplicating it per relation.
    "contextText": {
        "kind": "datatype",
        "label": "context text",
        "comment": "Full sentence/semantic-unit source text needed to interpret an assertion, as distinct from the minimal evidence span.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "discourse_id": {
        "kind": "datatype",
        "label": "discourse identifier",
        "comment": "Discourse unit identifier associated with provenance.",
        "domains": ["Resource"],
        "range": RDFS.Literal,
    },
    "chunk_index": {
        "kind": "datatype",
        "label": "chunk index",
        "comment": "Chunk index of source span; -1 denotes unknown.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "chunk_index_unknown": {
        "kind": "datatype",
        "label": "chunk index unknown",
        "comment": "Flag indicating unknown chunk index provenance.",
        "domains": ["Resource"],
        "range": XSD.boolean,
    },
    "chunk_total": {
        "kind": "datatype",
        "label": "chunk total",
        "comment": "Total number of chunks in the source discourse.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "chunk_start": {
        "kind": "datatype",
        "label": "chunk start",
        "comment": "Chunk start character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "start_char": {
        "kind": "datatype",
        "label": "start character",
        "comment": "Start character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "end_char": {
        "kind": "datatype",
        "label": "end character",
        "comment": "End character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "char_start": {
        "kind": "datatype",
        "label": "character start",
        "comment": "Start character offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "char_end": {
        "kind": "datatype",
        "label": "character end",
        "comment": "End character offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "start_offset": {
        "kind": "datatype",
        "label": "start offset",
        "comment": "Normalised start offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "end_offset": {
        "kind": "datatype",
        "label": "end offset",
        "comment": "Normalised end offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "context_char_start": {
        "kind": "datatype",
        "label": "context character start",
        "comment": "Start character offset of evidenceText within the linked DiscourseUnit's contextText, set only when evidence_alignment_status is not unresolved. Distinct from char_start/end_offset, which reflect the original extraction-time chunk coordinate system.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "context_char_end": {
        "kind": "datatype",
        "label": "context character end",
        "comment": "End character offset of evidenceText within the linked DiscourseUnit's contextText, set only when evidence_alignment_status is not unresolved.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "evidence_alignment_status": {
        "kind": "datatype",
        "label": "evidence alignment status",
        "comment": "Outcome of resolving the LLM evidence candidate against Source_Context: exact, normalized_exact, fuzzy_recovered, or unresolved.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "evidence_alignment_method": {
        "kind": "datatype",
        "label": "evidence alignment method",
        "comment": "Short description of how evidenceText was resolved against Source_Context (verbatim match, which normalization rules applied, fuzzy similarity ratio, or why resolution failed).",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "original_evidence_candidate": {
        "kind": "datatype",
        "label": "original evidence candidate",
        "comment": "The LLM-produced evidence_quote as originally returned, preserved for traceability whenever the serialized evidenceText differs from it (recovered raw-source span, or left as an unresolved candidate). Not itself asserted as verbatim source text.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "source_doc_id": {
        "kind": "datatype",
        "label": "source document id",
        "comment": "Identifier of the source discourse/document.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "chunk_id": {
        "kind": "datatype",
        "label": "chunk id",
        "comment": "Stable chunk identifier.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "quote_hash": {
        "kind": "datatype",
        "label": "quote hash",
        "comment": "Deterministic hash of stored evidence text.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "span_hash": {
        "kind": "datatype",
        "label": "span hash",
        "comment": "Deterministic hash of source/chunk/span provenance key.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "groundingStatus": {
        "kind": "datatype",
        "label": "grounding status",
        "comment": "Grounding status marker.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "groundingMethod": {
        "kind": "datatype",
        "label": "grounding method",
        "comment": "Grounding method marker.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "batch_id": {
        "kind": "datatype",
        "label": "batch id",
        "comment": "Processing batch identifier.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "ingestedAt": {
        "kind": "datatype",
        "label": "ingested at",
        "comment": "Ingestion timestamp.",
        "domains": ["Resource"],
        "range": XSD.dateTime,
    },
    "surface_form": {
        "kind": "datatype",
        "label": "surface form",
        "comment": "Surface lexical form.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "context_source": {
        "kind": "datatype",
        "label": "context source",
        "comment": "Extractor/source context.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "text": {
        "kind": "datatype",
        "label": "text",
        "comment": "Text literal carried in metadata.",
        "domains": ["Entity"],
        "range": XSD.string,
    },
    # --- Raw materialised source fields ---
    "createdAt": {"kind": "datatype", "label": "created at", "comment": "Creation timestamp in source data.", "domains": ["Entity"], "range": XSD.dateTime},
    "updatedAt": {"kind": "datatype", "label": "updated at", "comment": "Update timestamp in source data.", "domains": ["Entity"], "range": XSD.dateTime},
    "views": {"kind": "datatype", "label": "views", "comment": "View count in source data.", "domains": ["Entity"], "range": XSD.integer},
    "isDeleted": {"kind": "datatype", "label": "is deleted", "comment": "Deletion marker in source data.", "domains": ["Entity"], "range": XSD.boolean},
    "dateJoined": {"kind": "datatype", "label": "date joined", "comment": "Joined timestamp in source profile data.", "domains": ["Entity"], "range": XSD.dateTime},
    "areaOfExpertise": {"kind": "datatype", "label": "area of expertise", "comment": "Declared area of expertise.", "domains": ["Entity"], "range": XSD.string},
    "bio": {"kind": "datatype", "label": "bio", "comment": "Biographical text.", "domains": ["Entity"], "range": XSD.string},
    "profilePicture": {"kind": "rdf_property", "label": "profile picture", "comment": "Profile picture IRI or literal path.", "domains": ["Entity"], "range": RDFS.Resource},
    "modelCategory": {"kind": "datatype", "label": "model category", "comment": "Model category from source data.", "domains": ["Entity"], "range": XSD.string},
    "domainCategory": {"kind": "datatype", "label": "domain category", "comment": "Domain category from source data.", "domains": ["Entity"], "range": XSD.string},
    "problemCategory": {"kind": "datatype", "label": "problem category", "comment": "Problem category from source data.", "domains": ["Entity"], "range": XSD.string},
    "name": {"kind": "datatype", "label": "name", "comment": "Name from source data.", "domains": ["Entity"], "range": XSD.string},
    "title": {"kind": "datatype", "label": "title", "comment": "Title from source data.", "domains": ["Entity"], "range": XSD.string},
    "body": {"kind": "datatype", "label": "body", "comment": "Body text from source data.", "domains": ["Entity"], "range": XSD.string},
    "description": {"kind": "datatype", "label": "description", "comment": "Description text from source data.", "domains": ["Entity"], "range": XSD.string},
    "userId": {"kind": "datatype", "label": "user identifier", "comment": "User identifier from source data.", "domains": ["Person"], "range": XSD.integer},
    "expertId": {"kind": "datatype", "label": "expert identifier", "comment": "Expert identifier from source data.", "domains": ["Entity"], "range": XSD.integer},
    "themeId": {"kind": "datatype", "label": "theme identifier", "comment": "Theme foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "questionId": {"kind": "datatype", "label": "question identifier", "comment": "Question foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "answerId": {"kind": "datatype", "label": "answer identifier", "comment": "Answer foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "parentId": {"kind": "datatype", "label": "parent identifier", "comment": "Parent answer identifier from source data.", "domains": ["Entity"], "range": XSD.integer},
    "rawVoteValue": {"kind": "datatype", "label": "raw vote value", "comment": "Original numeric vote encoding from source data.", "domains": ["Vote"], "range": XSD.integer},
    "voteValueRaw": {"kind": "datatype", "label": "vote value raw", "comment": "Deprecated legacy name for rawVoteValue.", "domains": ["Entity"], "range": XSD.integer},
}


# Optional governance metadata used by ontology quality checks.
_LIFECYCLE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "createdAt": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "updatedAt": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "views": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "isDeleted": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "dateJoined": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "areaOfExpertise": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "bio": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when sensitive-field materialisation is enabled.",
    },
    "profilePicture": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
}

for _local_name, _spec in ETD_PROPERTY_REGISTRY.items():
    _spec.setdefault("lifecycle", "active")
    _spec.setdefault("lifecycle_note", "")
    _override = _LIFECYCLE_OVERRIDES.get(_local_name)
    if _override:
        _spec.update(_override)


def is_registered_etd_predicate(local_name: str) -> bool:
    return str(local_name) in ETD_PROPERTY_REGISTRY


def get_registered_etd_predicate(local_name: str) -> Dict[str, object]:
    return ETD_PROPERTY_REGISTRY[str(local_name)]


def registered_etd_predicates() -> List[str]:
    return sorted(ETD_PROPERTY_REGISTRY.keys())
