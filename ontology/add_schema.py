import re

from rdflib import RDF, RDFS, OWL, Literal, URIRef
from rdflib.namespace import XSD, SKOS, DCTERMS
from typing import Dict, List, Tuple
from ontology.property_registry import ETD_PROPERTY_REGISTRY


# Canonical ETD class local-names (single source of truth for variant normalisation).
CLASS_CANONICAL_ALIASES: Dict[str, str] = {
    "aisystem": "AISystem",
    "algorithmicbias": "AlgorithmicBias",
    "personaldata": "PersonalData",
    # "PopulationGroup"/"DemographicGroup" are compound camelCase
    # names with no internal separator, so Python's str.title() (used by
    # entity_normalizer._normalize_entity_type) mis-cases them to
    # "Populationgroup"/"Demographicgroup". The alnum-only key here also
    # transparently absorbs slash/hyphen/space variants ("Population/Group",
    # "Population-Group", "Population Group") since those characters are
    # stripped before lookup. "Population" alone is treated as the same
    # abbreviated variant of PopulationGroup.
    "populationgroup": "PopulationGroup",
    "population": "PopulationGroup",
    "demographicgroup": "DemographicGroup",
}


# Lifecycle statuses for ontology governance.
LIFECYCLE_STATUSES = {"active", "conditional", "reserved", "deprecated"}

# Compatibility deprecations (no destructive rename in this patch).
DEPRECATED_PROPERTY_REPLACEMENTS: Dict[str, str] = {
    "hasTheme": ("deprecated", "Use etd:belongsToTheme for canonical backbone output."),
    "hasAnswer": ("active", "Canonical forward relation linking a Question to its Answer instances."),
    "votesForAnswer": ("deprecated", "Use etd:receivesVote for canonical vote backbone output."),
    "voteValueRaw": "rawVoteValue",
}

# Properties that are intentionally conditional on input/config or reserved for future coverage.
PROPERTY_LIFECYCLE_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "hasTheme": ("deprecated", "Use etd:belongsToTheme for canonical backbone output."),
    "votesForAnswer": ("deprecated", "Use etd:receivesVote for canonical vote backbone output."),
    "voteValueRaw": ("deprecated", "Use etd:rawVoteValue for canonical raw vote encoding output."),
    "createdAt": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "updatedAt": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "views": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "isDeleted": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "dateJoined": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "areaOfExpertise": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "askedBy": ("active", "Canonical structural relation linking a Question record to its source Expert record."),
    "createdBy": ("active", "Canonical structural relation linking a Theme record to its source Expert record."),
    "receivesVote": ("active", "Canonical structural relation linking an Answer record to each Vote cast on it."),
    "bio": ("conditional", "Materialised only when sensitive-field materialisation is enabled."),
    "profilePicture": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "modelCategory": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "domainCategory": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "problemCategory": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "name": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "title": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "body": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "description": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "userId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "expertId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "themeId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "questionId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "answerId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "parentId": ("conditional", "Materialised only when raw source fields are enabled and present."),
    "wikidataAuthoritative": ("active", "Whether the Wikidata alignment is an authoritative asserted identity (owl:sameAs) or a heuristic semantic match (skos:closeMatch)."),
    "wikidataIdentityEvidence": ("active", "Evidence class behind the Wikidata alignment decision (e.g. exact_label, wikidata_alias, source_grounded_alias, fuzzy_label, +corroborated)."),
    "wikidataDecisionReason": ("active", "Linker accept/abstain reason string recorded with the Wikidata alignment."),
    "hasLikelihood": ("reserved", "Reserved for structured risk-assessment workflows."),
    "hasRiskLevel": ("reserved", "Reserved for structured risk-assessment workflows."),
    "hasSeverity": ("reserved", "Reserved for structured risk-assessment workflows."),
    "assessesRisk": ("reserved", "Reserved for structured risk-assessment workflows."),
    "assessedBy": ("reserved", "Reserved for structured risk-assessment workflows."),
    "usesEvidence": ("reserved", "Reserved for structured risk-assessment workflows."),
    "hasRegulatoryReference": ("reserved", "Reserved for governance instrument linking workflows."),
    "referencesInstrument": ("reserved", "Reserved for governance instrument linking workflows."),
    "referencesSource": ("reserved", "Reserved for governance instrument linking workflows."),
    "cites": ("reserved", "Reserved for governance instrument citation workflows."),
}

# Clarify semantically overloaded case-insensitive local names.
TERM_COMMENT_OVERRIDES: Dict[str, str] = {
    "Source": "Class of source artefacts/documents represented as entities in the knowledge graph.",
    "source": "Extractor/provenance source identifier literal attached to relation or evidence metadata.",
}


def _split_local_name(local: str) -> str:
    s = str(local or "").strip()
    if not s:
        return "term"
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip() or "term"


def _default_label(local: str, is_class: bool) -> str:
    words = _split_local_name(local)
    if is_class:
        return " ".join(w.capitalize() for w in words.split())
    return words.lower()


def _default_comment(local: str, kind: str) -> str:
    human = _split_local_name(local).lower()
    if local in TERM_COMMENT_OVERRIDES:
        return TERM_COMMENT_OVERRIDES[local]
    if kind == "class":
        return f"ETD class representing {human} concepts in the semantic knowledge graph."
    if kind == "object_property":
        return f"ETD object property linking resources using {human} semantics."
    if kind == "datatype_property":
        return f"ETD datatype property carrying {human} metadata values."
    return f"ETD property supporting {human} metadata."


def _lifecycle_for_property(local: str) -> Tuple[str, str]:
    override = PROPERTY_LIFECYCLE_OVERRIDES.get(local)
    if override:
        return override

    if local in ETD_PROPERTY_REGISTRY:
        spec = ETD_PROPERTY_REGISTRY.get(local, {})
        status = str(spec.get("lifecycle", "active")).strip().lower() or "active"
        note = str(spec.get("lifecycle_note", "")).strip()
        if status not in LIFECYCLE_STATUSES:
            status = "active"
        return status, note

    return (
        "reserved",
        "Declared in ontology for compatibility/future modelling; may be unused in current exports.",
    )


def normalized_etd_local_name(local: str) -> str:
    """Case/separator-insensitive local-name key.

    Identical semantics to scripts/kg_quality_gate.py
    ``_etd_local_name_collisions._norm`` -- the single normalisation used to
    decide whether two ETD local names (class vs property, or two
    properties) collide. Kept here so every ETD-term emitter can consult
    the SAME rule the gate enforces.
    """
    return "".join(ch for ch in str(local or "").lower() if ch.isalnum())


def apply_property_lifecycle(g, etd_ns, pred, local: str) -> Tuple[str, str]:
    """Stamp on one declared ETD property the governance lifecycle metadata
    the quality gate requires, using the SINGLE shared lifecycle policy
    (``_lifecycle_for_property`` / ``LIFECYCLE_STATUSES`` /
    ``DEPRECATED_PROPERTY_REPLACEMENTS``). Safe to call again -- existing
    ``etd:lifecycleStatus`` / ``etd:lifecycleNote`` values are left in
    place. Returns the ``(status, note)`` applied. Used by both
    ``add_schema``'s static stamping loop and the serializer's dynamic
    local-predicate declaration so they can never diverge.
    """
    status, note = _lifecycle_for_property(local)
    if status not in LIFECYCLE_STATUSES:
        status = "active"
    if not any(True for _ in g.objects(pred, etd_ns.lifecycleStatus)):
        g.add((pred, etd_ns.lifecycleStatus, Literal(status, datatype=XSD.string)))
    if note and not any(True for _ in g.objects(pred, etd_ns.lifecycleNote)):
        g.add((pred, etd_ns.lifecycleNote, Literal(note, lang="en")))
    if status == "deprecated":
        g.add((pred, OWL.deprecated, Literal(True, datatype=XSD.boolean)))
        replacement = DEPRECATED_PROPERTY_REPLACEMENTS.get(local)
        if isinstance(replacement, str) and replacement:
            g.add((pred, RDFS.seeAlso, etd_ns[replacement]))
    return status, note


# Canonical rdfs:subClassOf hierarchy (child -> parent local name), also
# consumed outside schema emission (e.g. utils/entity_consolidator.py) so
# merge/dedup decisions stay aware of declared subclass relationships
# (e.g. DemographicGroup is-a PopulationGroup) instead of only comparing
# entity_type strings for exact equality.
SUBCLASS_MAP: Dict[str, str] = {
    # ---- Discourse structure
    "DiscourseUnit": "Entity",
    "Theme": "DiscourseUnit",
    "Message": "DiscourseUnit",
    "Question": "DiscourseUnit",
    "Answer": "DiscourseUnit",

    # ---- Interaction
    "InteractionRecord": "Entity",
    "Vote": "InteractionRecord",

    # ---- Actors
    "Person": "Entity",
    "Expert": "Person",
    "Organization": "Entity",

    # ---- AI & data artefacts
    "Resource": "Entity",
    "Dataset": "Resource",

    "System": "Entity",
    "AISystem": "System",

    "AlgorithmicProcess": "Entity",
    "MachineLearning": "AlgorithmicProcess",

    "Model": "System",
    "Service": "System",
    "SoftwarePackage": "Resource",
    "SoftwareProduct": "System",

    # ---- Research & information resources (fills the
    # "research study/analysis typed as Dataset" gap identified in the
    # ontology-alignment audit -- Dataset remains reserved for
    # the underlying data resource itself).
    "ResearchOutput": "Resource",
    "ResearchStudy": "ResearchOutput",
    "Analysis": "ResearchOutput",
    "Publication": "ResearchOutput",

    # ---- Risk & governance
    "Risk": "Entity",
    "HighRisk": "Risk",
    "Governance": "Entity",
    "Evaluation": "Governance",
    "RiskAssessment": "Evaluation",
    "AuditProcess": "Evaluation",
    "RegulatoryReference": "Entity",
    "EthicalPrinciple": "Entity",
    "Fairness": "EthicalPrinciple",
    "Transparency": "EthicalPrinciple",
    "Privacy": "EthicalPrinciple",
    "DemographicParity": "EthicalPrinciple",

    # ---- Evidence / provenance
    "Evidence": "Entity",
    "RiskEvidence": "Evidence",
    "Relation": "Entity",
    "Proposition": "Entity",

    # ---- Population & demographics (FIXED direction)
    "PopulationGroup": "Entity",
    "DemographicGroup": "PopulationGroup",

    # ---- Bias family (these were declared classes with no
    # rdfs:subClassOf at all -- wiring them under the
    # existing "Bias" phenomenon class the ontology-alignment audit found
    # to be the best-performing, correctly-grounded category).
    "AlgorithmicBias": "Bias",
    "RacialBias": "Bias",

    # ---- Generic reusable concept ("Concept" was used
    # pervasively as an entity_type in practice but was never a
    # declared ETD class, so it fell back entirely to the generic
    # external prov:Entity grounding).
    "Concept": "Entity",

    # ---- Context
    "Domain": "Entity",
    "Healthcare": "Domain",
    "CriminalJustice": "Domain",
    "Employment": "Domain",
}


def canonicalize_class_local(name: str) -> str:
    """
    Canonicalize ETD class local names to stable publication-grade spellings.
    Keeps unknown names unchanged.
    """
    raw = str(name or "").strip()
    if not raw:
        return "Entity"
    key = "".join(ch for ch in raw.lower() if ch.isalnum())
    return CLASS_CANONICAL_ALIASES.get(key, raw)



def add_schema(graph, etd_ns):
    """
    Add schema declarations (classes, properties, domain/range) to the RDF graph
    for the ETD-Hub AI Ethics knowledge graph.
    """

    g = graph

    REGISTRY_MANAGED = set(ETD_PROPERTY_REGISTRY.keys())

    def _apply_property_registry() -> None:
        """
        Enforce authoritative ETD predicate declarations from registry.

        Rule:
        - If a predicate exists in ETD_PROPERTY_REGISTRY, its RDF type, label,
          comment, domain, and range should be governed there.
        - Manual declarations in this file should avoid re-declaring domain/range
          for registry-managed predicates unless intentionally additive.
        """
                
        for local_name in sorted(ETD_PROPERTY_REGISTRY.keys()):
            spec = ETD_PROPERTY_REGISTRY[local_name]
            pred = etd_ns[local_name]
            kind = str(spec.get("kind", "datatype")).strip().lower()
            if kind == "object":
                g.add((pred, RDF.type, OWL.ObjectProperty))
            elif kind == "rdf_property":
                g.add((pred, RDF.type, RDF.Property))
            else:
                g.add((pred, RDF.type, OWL.DatatypeProperty))

            label = str(spec.get("label", local_name)).strip()
            if label:
                g.add((pred, RDFS.label, Literal(label, lang="en")))
            comment = str(spec.get("comment", "")).strip()
            if comment:
                g.add((pred, RDFS.comment, Literal(comment, lang="en")))

            for d in spec.get("domains", []) or []:
                d_str = str(d).strip()
                if not d_str:
                    continue
                if d_str == "Resource":
                    g.add((pred, RDFS.domain, RDFS.Resource))
                elif d_str.startswith("http://") or d_str.startswith("https://"):
                    g.add((pred, RDFS.domain, URIRef(d_str)))
                else:
                    g.add((pred, RDFS.domain, etd_ns[d_str]))

            rng = spec.get("range")
            if rng is not None:
                if isinstance(rng, URIRef):
                    g.add((pred, RDFS.range, rng))
                else:
                    g.add((pred, RDFS.range, rng))


    def _assert_not_registry_managed(*local_names: str) -> None:
        overlaps = sorted(set(local_names) & REGISTRY_MANAGED)
        if overlaps:
            raise ValueError(
                "Manual schema declarations overlap with ETD_PROPERTY_REGISTRY: "
                + ", ".join(overlaps)
            )

    def _ensure_term_docs_and_lifecycle() -> None:
        """
        Production hardening:
        - Ensure every declared ETD class/property has label+comment.
        - Attach lifecycle metadata to every declared ETD property.
        """
        # Annotation properties for lifecycle governance.
        g.add((etd_ns.lifecycleStatus, RDF.type, OWL.DatatypeProperty))
        g.add((etd_ns.lifecycleStatus, RDFS.label, Literal("lifecycle status", lang="en")))
        g.add((etd_ns.lifecycleStatus, RDFS.comment, Literal(
            "Governance lifecycle status for ETD ontology predicates (active/conditional/reserved/deprecated).",
            lang="en",
        )))
        g.add((etd_ns.lifecycleStatus, RDFS.domain, RDF.Property))
        g.add((etd_ns.lifecycleStatus, RDFS.range, XSD.string))

        g.add((etd_ns.lifecycleNote, RDF.type, OWL.DatatypeProperty))
        g.add((etd_ns.lifecycleNote, RDFS.label, Literal("lifecycle note", lang="en")))
        g.add((etd_ns.lifecycleNote, RDFS.comment, Literal(
            "Human-readable rationale for lifecycle status assignment.",
            lang="en",
        )))
        g.add((etd_ns.lifecycleNote, RDFS.domain, RDF.Property))
        g.add((etd_ns.lifecycleNote, RDFS.range, XSD.string))

        declared_classes = set()
        for cls in g.subjects(RDF.type, OWL.Class):
            if isinstance(cls, URIRef) and str(cls).startswith(str(etd_ns)):
                declared_classes.add(cls)

        declared_props = set()
        for ptype in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
            for pred in g.subjects(RDF.type, ptype):
                if isinstance(pred, URIRef) and str(pred).startswith(str(etd_ns)):
                    declared_props.add(pred)

        for cls in sorted(declared_classes, key=str):
            local = str(cls)[len(str(etd_ns)) :]
            if not any(True for _ in g.objects(cls, RDFS.label)):
                g.add((cls, RDFS.label, Literal(_default_label(local, is_class=True), lang="en")))
            if not any(True for _ in g.objects(cls, RDFS.comment)):
                g.add((cls, RDFS.comment, Literal(_default_comment(local, kind="class"), lang="en")))

        for pred in sorted(declared_props, key=str):
            local = str(pred)[len(str(etd_ns)) :]
            is_obj = (pred, RDF.type, OWL.ObjectProperty) in g
            is_data = (pred, RDF.type, OWL.DatatypeProperty) in g
            kind = "object_property" if is_obj else "datatype_property" if is_data else "property"
            if not any(True for _ in g.objects(pred, RDFS.label)):
                g.add((pred, RDFS.label, Literal(_default_label(local, is_class=False), lang="en")))
            if not any(True for _ in g.objects(pred, RDFS.comment)):
                g.add((pred, RDFS.comment, Literal(_default_comment(local, kind=kind), lang="en")))

            apply_property_lifecycle(g, etd_ns, pred, local)

    # ======================================================
    # ONTOLOGY HEADER
    # ======================================================
    
    ONT = etd_ns["ontology"]  # ontology IRI (local name; change if you prefer 'ontology'/'schema')

    g.add((ONT, RDF.type, OWL.Ontology))
    g.add((ONT, DCTERMS.title, Literal("TQAVE Ontology", lang="en")))
    g.add((ONT, DCTERMS.description, Literal(
        "Ontology for generating a discourse-driven AI ethics semantic knowledge graph from Q&A datasets, "
        "supporting reified relations with evidence, provenance metadata, and vote-based endorsements.",
        lang="en"
    )))
    g.add((ONT, OWL.versionInfo, Literal("0.1.0", datatype=XSD.string)))

    # Explicit formal vocabulary reuse (publication-grade ontology metadata).
    # Use ontology document IRIs (not just namespace prefixes) for owl:imports.
    imported_ontologies = [
        # Core external ontologies actively reused in class/property mappings
        "https://w3id.org/aipo",
        "https://w3id.org/airo",
        "https://w3id.org/dpv",
        "https://w3id.org/vair",
        "http://purl.org/fairness-metrics-ontology/",
        # W3C / community vocabularies reused in KG assertions
        "http://www.w3.org/ns/dcat",
        "http://www.w3.org/ns/prov-o",
        "http://www.w3.org/2004/02/skos/core",
        "http://xmlns.com/foaf/0.1/",
        "http://www.w3.org/ns/org#",
        "http://rdfs.org/sioc/ns#",
        "http://schema.org/",
        "http://purl.org/ontology/bibo/",
        "http://purl.org/dc/terms/",
    ]
    for iri in imported_ontologies:
        g.add((ONT, OWL.imports, URIRef(iri)))
    # Optional but recommended (fill in as appropriate)
    # g.add((ONT, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    # g.add((ONT, DCTERMS.creator, Literal("Chukwudi Uwasomba", "Yannis Korkontzelos", "Nonso Nnamoko", lang="en")))
    # g.add((ONT, DCTERMS.created, Literal("2026-09-11", datatype=XSD.date)))

    # ======================================================
    # KEY TERM COMMENTS (minimal, high-value only)
    # ======================================================
    key_comments = {
        "Entity": "Top-level class for all materialised nodes in the graph, including discourse units, domain entities, and reified assertions.",
        "DiscourseUnit": "A unit of discourse from the source dataset (Theme, Question, Answer, Message) used as provenance for extracted knowledge.",
        "Risk": "A potential (not yet realised), forward-looking adverse condition or exposure associated with an AI system, dataset, or deployment context. Distinguish from Harm (the realised consequence if the risk materialises) and from Bias (a specific distortion mechanism that may be one source of risk).",
        "Bias": "A systematic distortion in data, model behaviour, or decision outcomes that produces unequal treatment or unequal error rates across groups or cases. Names the MECHANISM/property (e.g. measurement bias, racial bias) rather than its consequence -- see Harm/Outcome for the resulting effect and Risk for the potential of that effect occurring.",
        "Impact": "Broader societal, organisational, or user-level consequence linked to AI use or governance decisions.",
        "Harm": "A realised (actually occurring, not merely potential) negative consequence experienced by a person, group, system, or organisation, typically as a Risk or Bias materialising in practice.",
        "Outcome": "A neutral, general term for any observed or expected result of a process, decision, or intervention, which may be positive, negative, or mixed. Use Harm specifically when the result is negative and its adverse character is the salient fact being asserted; use Outcome when valence is not the point being made or the result is not purely negative.",
        "Governance": "Policies, controls, accountability structures, and oversight practices guiding AI lifecycle decisions.",
        "AuditProcess": "A systematic, typically formalised assessment or audit activity applied to an AI system, dataset, or organisational practice to check compliance, quality, or fairness. A specific kind of Evaluation; use Evaluation itself for assessment activity that is not specifically audit-shaped.",
        "ResearchOutput": "Umbrella class for a work-product of research activity (a study, an analysis, or a publication reporting findings) -- distinct from Dataset, which is the underlying data resource a ResearchOutput may describe, use, or report on.",
        "ResearchStudy": "An empirical or investigative research undertaking (a project, experiment, or study), which may produce its own dataset, analysis, and publications. Use Dataset for the data it produced, Analysis for a specific analytical pass over data, and Publication for a resulting document.",
        "Analysis": "A specific analytical or evaluative pass applied to existing data or a dataset (e.g. 'analysis of the HMDA dataset revealed...'), narrower in scope than a full ResearchStudy and not itself a data resource.",
        "Publication": "A published document (paper, article, or report) that reports research findings, an analysis, or study results.",
        "SoftwareProduct": "A general-purpose software application, tool, or package used in a workflow, which is not itself an AI/ML system (e.g. a spreadsheet application, a fairness-auditing toolkit). Use AISystem for systems whose core function is AI/ML-driven decision-making or prediction.",
        "Concept": "A general reusable abstract notion, topic, or idea referenced in the discourse (e.g. a named criterion, value, or subject matter) that is not itself a concrete entity, process, or artefact. Not intended to absorb full clauses, findings, or sentence fragments -- those are extraction-recall concerns handled upstream of typing, not a semantic category of their own.",
        "Dataset": "A structured data resource (a named collection of records, images, measurements, or similar) available for use, analysis, or training. Use ResearchStudy/Analysis/Publication for the research activity or output that uses or describes a dataset, not for the dataset itself.",
        "AISystem": "A deployed, operational AI/ML-driven system used in a real decision-making or prediction context (e.g. a named product, service, or platform). Use Model for the underlying computational/statistical artefact it embeds, and SoftwareProduct for general-purpose software that is not itself AI/ML-driven.",
        "Model": "An underlying computational, statistical, or machine-learning model or algorithm -- the component-level artefact (e.g. 'the model', 'the algorithm used'), as distinct from AISystem, which is the deployed system that embeds and operates one or more models.",
        "Organization": "A formally identifiable organisation, company, agency, or institution acting as a collective entity (e.g. a company, regulator, research lab, or university). Aligned with schema:Organization.",
        "Person": "An individual human being, named or referenced as an individual (as distinct from Organization, a collective actor). Aligned with foaf:Person.",
        "PopulationGroup": "Any group of people considered collectively as the subject or object of a discourse claim (e.g. 'applicants', 'end-users', 'recruiters'). Use DemographicGroup specifically when the group is defined by a shared demographic or legally/ethically salient characteristic (race, gender, age, disability, etc.); use PopulationGroup for broader or role-defined groupings that are not demographic categories per se. Do not use for a finding, statistic, or clause ABOUT a group -- that remains a literal relation value, not a group entity in its own right.",
        "DemographicGroup": "A PopulationGroup specifically defined by a shared demographic or protected characteristic (e.g. 'Black defendants', 'dark-skinned women', 'non-native speakers'). A more specific subclass of PopulationGroup, not a synonym for it. Do not use for a finding, percentage, or descriptive clause about such a group.",
        "Relation": "Reified assertion connecting a subject and object via a predicate, enabling provenance, confidence scores, and evidence attachments.",
        "Evidence": "Provenance node recording extractor source, discourse identifiers, and optional text spans supporting an assertion.",
        "Proposition": "The semantic content/state expressed by a single source semantic unit, distinct from the DiscourseUnit (source text/provenance) and Relation (reified assertion) that may derive from it; used as the antecedent for discourse-level (propositional) deixis such as a sentence-initial 'This'/'That'/'It' referring back to a preceding statement rather than to a named entity.",
        "Vote": "Interaction record representing an expert endorsement/disagreement on an Answer, including raw encoding and SKOS vote value.",
        "Export": "A specific knowledge graph export instance generated by the pipeline.",
        "subject": "Links a reified Relation to its subject entity node (IRI).",
        "predicate": "Links a reified Relation to the predicate IRI used in the assertion (object/datatype property).",
        "object": "Links a reified Relation to its object; typically an entity IRI, but may be a literal for value/rating patterns.",
        "hasEvidence": "Links a reified Relation to one or more Evidence nodes supporting the assertion.",
        "chunk_index": "Chunk index of the source text segment; -1 means unknown/unavailable chunk index.",
        "voteValue": "Links a Vote to a SKOS Concept (e.g., Upvote/Downvote/Neutral).",
        "rawVoteValue": "Original numeric vote encoding (e.g., 1, 0, -1) stored as an integer literal.",
        "FieldAssertion": "Provenance record for a raw JSON field materialised as an RDF triple.",
        "hasFieldAssertion": "Links an entity to a field-level provenance assertion node.",
        "hasSourceEntity": "Links a provenance-bearing node to one or more explicit source discourse entities such as Answer, Question, Theme, Vote, or Expert.",
        "fieldName": "Original raw JSON field name.",
        "recordId": "Source dataset record identifier.",
        "sourceSection": "Source dataset section name (themes/questions/answers/votes/experts).",
        "assertedPredicate": "Predicate used by a materialised raw-field triple.",
        "assertedValue": "Object value used by a materialised raw-field triple.",
        "createdAt": "Creation timestamp in source data.",
        "updatedAt": "Update timestamp in source data.",
        "views": "View count from source data.",
        "isDeleted": "Deletion marker from source data.",
        "dateJoined": "Date joined in source profile.",
        "areaOfExpertise": "Declared area of expertise.",
        "bio": "Profile biography text.",
        "profilePicture": "Profile picture URL or relative path.",
        "modelCategory": "Model category from source data.",
        "domainCategory": "Domain category from source data.",
        "problemCategory": "Problem category from source data.",
        "name": "Display name from source data.",
        "title": "Title from source data.",
        "body": "Body text from source data.",
        "description": "Description text from source data.",
        "userId": "Source user identifier.",
        "themeId": "Foreign key to Theme.",
        "questionId": "Foreign key to Question.",
        "answerId": "Foreign key to Answer.",
        "parentId": "Foreign key to parent Answer.",
    }

    # Note: etd_ns[...] works for both classes and properties since we mint them in one namespace.
    for term, comment in key_comments.items():
        g.add((etd_ns[term], RDFS.comment, Literal(comment, lang="en")))


    # ======================================================
    # CLASSES
    # ======================================================
    classes = {
        # ---- Core KG primitives
        "Entity": "Generic entity",
        "Relation": "Reified relation",
        "Evidence": "Evidence/provenance node for an extracted assertion or relation",
        "Proposition": "Proposition",
        "FieldAssertion": "Raw-field provenance assertion",

        # ---- Discourse layer
        "DiscourseUnit": "Discourse unit",
        "Theme": "Theme",
        "Question": "Question",
        "Answer": "Answer",
        "Message": "Message",
        "InteractionRecord": "Interaction record",
        "Vote": "Vote",

        # ---- Actors
        "Person": "Person",
        "Expert": "Expert",
        "Organization": "Organisation",

        # ---- AI & data artefacts
        "Dataset": "Dataset",
        "Export": "Export",
        "System": "System",
        "AISystem": "AI system",
        "Model": "Model",
        "Service": "Service",
        "SoftwarePackage": "Software package",
        "SoftwareProduct": "Software product",
        "AlgorithmicProcess": "Algorithmic process",
        "MachineLearning": "Machine learning process",

        # ---- Research & information resources
        "ResearchOutput": "Research output",
        "ResearchStudy": "Research study",
        "Analysis": "Analysis",
        "Publication": "Publication",

        # ---- Risk & governance
        "Risk": "Risk",
        "HighRisk": "High-risk AI",
        "RiskAssessment": "Risk assessment",
        "RiskEvidence": "Risk evidence",
        "Evaluation": "Assessment or audit activity",
        "AuditProcess": "Audit process",
        "Governance": "Governance concept",
        "RegulatoryReference": "Regulatory reference",

        # ---- Fairness / ethics concepts
        "Fairness": "Fairness",
        "Transparency": "Transparency",
        "Privacy": "Privacy",
        "AlgorithmicBias": "Algorithmic bias",
        "RacialBias": "Racial bias",
        "DemographicParity": "Demographic parity",
        "EthicalPrinciple": "Ethical principle, Normative principle guiding responsible AI behaviour (e.g., fairness, transparency, accountability)",
        "Concept": "Concept",

        # ---- Data & population
        "PersonalData": "Personal data",
        "DataType": "Data type",
        "PopulationGroup": "Population group",
        "DemographicGroup": "Demographic group",

        # ---- Context
        "Domain": "Domain",
        "Healthcare": "Healthcare domain",
        "CriminalJustice": "Criminal justice domain",
        "Employment": "Employment domain",

        # ---- Misc resources
        "Resource": "Generic resource",
        "Source": "Source",
        "Url": "URL",
        "Location": "Location",
        "TimePeriod": "Time period",
    }

    for name, label in classes.items():
        g.add((etd_ns[name], RDF.type, OWL.Class))
        g.add((etd_ns[name], RDFS.label, Literal(label, lang="en")))

    # ------------------------------------------------------------
    # Conceptual umbrella classes (future-proof layer)
    # ------------------------------------------------------------
    conceptual_classes = {
        "ConceptualEntity": "General conceptual entity",
        "Outcome": "Observed or expected result; may be positive or negative.",
        "SystemComponent": "Component or subsystem of an AI system",

        # Data-oriented
        "DataElement": "Variable, attribute, or field within a dataset",
        "DataIssue": "Problem with data representation, bias or quality",

        # Harm / Impact
        "Harm": "Negative consequence or adverse effect",
        "Impact": "Broader societal or systemic effect",
        "Bias": "Distortion or unfairness in data or systems",

        # Stakeholders
        "Stakeholder": "Actor affected by a system",
        "ProtectedGroup": "Legally protected demographic group",

        # Ethical & Governance
        "AiGovernanceFramework": "Governance instrument/framework used to guide, constrain, or evaluate AI practice (law, standards, best practices).",
        "LegalRequirement": "Binding legal or regulatory obligation applicable to AI systems or organisations.",
        "Standard": "Published technical or governance standard issued by a standards body (e.g., ISO/IEC, IEEE, NIST).",
        "BestPractice": "Non-binding recommended governance or operational practice (guidance, playbooks, organisational policies).",
        "Regulation": "Binding legal instrument (e.g., EU AI Act such as Regulation (EU) 2024/1689).",
        "CodeOfConduct": "Formal organisational or professional code of conduct",

        # Processes / Actions
        "Process": "Governance or computational process",
        "Action": "Specific system or organizational action",
               
        # Assessment
        # Using SKOS.Concept
        #"VoteValue": "Enumerated vote value (Yes, No, Abstain)",  
        
    }

    for name, label in conceptual_classes.items():
        g.add((etd_ns[name], RDF.type, OWL.Class))
        g.add((etd_ns[name], RDFS.label, Literal(label, lang="en")))
        g.add((etd_ns[name], RDFS.subClassOf, etd_ns.Entity))

    # ------------------------------------------------------------
    # Production: expose declared ETD class keys for allow-listing
    # ------------------------------------------------------------
    allowed_classes = set(classes.keys()) | set(conceptual_classes.keys())

    # ------------------------------------------------------
    # GovernanceFramework umbrella (no OWL unions needed)
    # ------------------------------------------------------
    g.add((etd_ns.LegalRequirement, RDFS.subClassOf, etd_ns.AiGovernanceFramework))
    g.add((etd_ns.Standard, RDFS.subClassOf, etd_ns.AiGovernanceFramework))
    g.add((etd_ns.BestPractice, RDFS.subClassOf, etd_ns.AiGovernanceFramework))

    g.add((etd_ns.Regulation, RDFS.subClassOf, etd_ns.LegalRequirement))
    g.add((etd_ns.CodeOfConduct, RDFS.subClassOf, etd_ns.BestPractice))

    # ------------------------------------------------------
    # Subclass structure
    # ------------------------------------------------------
    # Discourse unit serves as a source of extracted knowledge, assertions, or relations within the knowledge graph.
    # InteractionRecord is conceptual model that sets up for additional interaction artefacts later (ratings, flags, endorsements).
    subclass_map = SUBCLASS_MAP

    for child, parent in subclass_map.items():
        g.add((etd_ns[child], RDFS.subClassOf, etd_ns[parent]))

    # ------------------------------------------------------
    # External ontology alignment
    #
    # Formalises, at the schema level, external-class alignments that were
    # previously only implicit at the instance level (via each Entity's own
    # ontology_uri metadata) or entirely unused despite the corresponding
    # ontology already being declared in owl:imports above. Alignment is
    # added only where the external class's semantics clearly match --
    # rdfs:subClassOf where the ETD class may be narrower or extraction-
    # derived (safer, non-committal direction), owl:equivalentClass only
    # for Person, where the match is essentially exact and already the de
    # facto grounding used throughout the pipeline's instance data.
    # ------------------------------------------------------
    PROV = URIRef("http://www.w3.org/ns/prov#")
    FOAF = URIRef("http://xmlns.com/foaf/0.1/")
    SCHEMA = URIRef("http://schema.org/")
    DCAT = URIRef("http://www.w3.org/ns/dcat#")
    BIBO = URIRef("http://purl.org/ontology/bibo/")

    def _ext(ns: URIRef, local: str) -> URIRef:
        return URIRef(f"{ns}{local}")

    external_subclass_alignments = {
        # Every graph node ultimately derives from prov:Entity in practice
        # (it is the fallback ontology_uri for anything without a more
        # specific external mapping) -- formalise that as the top-level
        # alignment rather than leaving it implicit.
        "Entity": _ext(PROV, "Entity"),
        "Person": _ext(PROV, "Agent"),
        "Organization": _ext(PROV, "Agent"),
        "Dataset": _ext(DCAT, "Dataset"),
        # Resource is a heterogeneous information-hosting/
        # -providing class (repository, platform, catalogue, portal,
        # collection host) -- not itself necessarily a written work, so it
        # must not align to foaf:Document. dcat:Resource is DCAT's own
        # broad superclass for this category (dcat:Dataset/dcat:Catalog/
        # dcat:DataService all subclass it in the real DCAT ontology), and
        # subClassOf (not equivalentClass) is used since etd:Resource's
        # extension is not guaranteed identical to dcat:Resource's.
        "Resource": _ext(DCAT, "Resource"),
        "Publication": _ext(BIBO, "Document"),
        "SoftwareProduct": _ext(SCHEMA, "SoftwareApplication"),
        "Concept": SKOS.Concept,
    }
    for local, external in external_subclass_alignments.items():
        if (etd_ns[local], RDF.type, OWL.Class) not in g:
            continue
        g.add((etd_ns[local], RDFS.subClassOf, external))

    # Person <-> foaf:Person is treated as an exact match: it is already
    # the consistent, de facto grounding for every Person instance the
    # pipeline produces (unlike Organization/Dataset/etc., which only ever
    # partially align with their external counterpart).
    g.add((etd_ns.Person, OWL.equivalentClass, _ext(FOAF, "Person")))
    g.add((etd_ns.Organization, RDFS.subClassOf, _ext(SCHEMA, "Organization")))

    _assert_not_registry_managed(
        "mentions",
        "answersQuestion",
        "hasAnswer",
        "hasTheme",
        "authoredBy",
        "askedBy",
        "createdBy",
        "hasVote",
        "receivesVote",
        "votesForAnswer",
        "castBy",
        "castVote",
        "relatedTo",
        "reflects",
        "supports",
        "criticizes",
        "endorses",
        "disagreesWith",
        "uses",
        "causes",
        "affects",
        "includes",
        "mitigatedBy",
        "trainedOn",
        "hasQuestion",
        "partOfDomain",
        "containsSubject",
        "exhibitsRisk",
        "belongsToTheme",
        "hasSourceEntity",
        "hasRisk",
        "particularlyAffects",
        "associatedWith",
        "causedBy",
        "affectedBy",
        "repliesTo",
        "extractedFrom",
        "assessesRisk",
        "assessedBy",
        "usesEvidence",
        "hasSeverity",
        "hasLikelihood",
        "hasRiskLevel",
        "hasRegulatoryReference",
        "referencesInstrument",
        "referencesSource",
        "cites",
        "compliesWith",
        "requiresComplianceWith",
        "requires",
        "violates",
        "mitigates",
        "opposes",
        "hasDescription",
        "hasAttribute",
        "hasSubjectDescription",
        "hostedBy",
    )

    # ======================================================
    # OBJECT PROPERTIES
    # ======================================================
    object_properties = {
        "mentions": "mentions",
        "answersQuestion": "answers question",
        "hasTheme": "has theme",
        "hasAnswer": "has answer",
        "authoredBy": "authored by",
        "askedBy": "asked by",
        "createdBy": "created by",
        "hasVote": "has vote",
        "receivesVote": "receives vote",
        "votesForAnswer": "votes for answer",
        "castBy": "cast by",
        "castVote": "cast vote",
        "relatedTo": "related to",
        "reflects": "reflects",
        "supports": "supports",
        "criticizes": "criticizes",
        "endorses": "endorses",
        "disagreesWith": "disagrees with",
        "uses": "uses",
        "causes": "causes",
        "affects": "affects",
        "includes": "includes",
        "mitigatedBy": "mitigated by",
        "hasEvidence": "has evidence",
        "trainedOn": "trained on",
        "hasQuestion": "theme has question",
        "partOfDomain": "part of domain",
        "containsSubject": "contains subject",
        "exhibitsRisk": "exhibits risk",
        "belongsToTheme": "belongs to theme",
        "hasSourceEntity": "has source entity",
        "hasRisk": "has risk",
        "particularlyAffects": "particularly affects",
        "associatedWith": "associated with",
        "causedBy": "caused by",
        "affectedBy": "affected by",
        "subject": "relation subject",
        #"predicate": "relation predicate",
        #"object": "relation object",
        "repliesTo": "replies to",
        "extractedFrom": "extracted from",
        # Risk assessment bundle
        "assessesRisk": "assesses risk",
        "assessedBy": "assessed by",
        "usesEvidence": "uses evidence",
        "hasSeverity": "has severity",
        "hasLikelihood": "has likelihood",
        "hasRiskLevel": "has risk level",
        # Regulatory reference bundle
        "hasRegulatoryReference": "has regulatory reference",
        "referencesInstrument": "references governance instrument",
        "referencesSource": "references source document",
        "cites": "cites source",
        # Ontology / Wikidata links are also object properties:
        #"ontology_uri": "ontology URI",
        #"wikidata_uri": "wikidata URI",
        #"wikidata": "wikidata link",
        # --- Governance reasoning (core)
        "compliesWith": "complies with",
        "requiresComplianceWith": "requires compliance with",
        "requires": "requires",
        "violates": "violates",
        "mitigates": "mitigates",
        "opposes": "opposes",

        # Two new predicates justified by repeated is/has
        # fallback audit evidence -- see utils/semantic_relation_policy.py.
        "hasDescription": "has description",
        "hasAttribute": "has attribute",

        # Defect-2 root-cause fix: the raw LLM phrases "contains"/"has
        # subject"/etc. used to map onto containsSubject, colliding with
        # that name's OWN distinct, pre-existing structural declaration
        # (DiscourseUnit -> Entity, see the domain/range axiom below).
        # Actual usage is descriptive/quantitative content-of-a-resource
        # text (e.g. a Dataset's row count or subject-matter summary), not
        # an entity reference -- a genuinely different relation that needs
        # its own name and, like hasDescription/hasAttribute above, is
        # policy-registered literal_allowed=True in
        # utils/semantic_relation_policy.py rather than declared
        # owl:DatatypeProperty here, matching the established pattern for
        # every other literal-carrying predicate in this schema.
        "hasSubjectDescription": "has subject-matter description",

        # Category G (manual semantic-QA pass): a dataset/resource being
        # hosted/available at a repository, platform, or portal (e.g. "UCI
        # Machine Learning Repository", "PhysioNet") is not an identity
        # ("is") or free-text description ("hasDescription") relation --
        # it names a distinct hosting entity. literal_allowed=False in
        # utils/semantic_relation_policy.py (entity-range), matching every
        # other entity-referencing object property in this schema.
        "hostedBy": "hosted by",

    }

    for name, label in object_properties.items():
        g.add((etd_ns[name], RDF.type, OWL.ObjectProperty))
        g.add((etd_ns[name], RDFS.label, Literal(label, lang="en")))

    g.add((etd_ns.ontology_uri, RDF.type, OWL.ObjectProperty))
    g.add((etd_ns.ontology_uri, RDFS.range, RDFS.Resource))
    g.add((etd_ns.ontology_uri, RDFS.label, Literal("ontology URI", lang="en")))

    g.add((etd_ns.wikidata_uri, RDF.type, OWL.ObjectProperty))
    g.add((etd_ns.wikidata_uri, RDFS.range, RDFS.Resource))
    g.add((etd_ns.wikidata_uri, RDFS.label, Literal("wikidata URI", lang="en")))

    g.add((etd_ns.wikidata, RDF.type, OWL.ObjectProperty))
    g.add((etd_ns.wikidata, RDFS.range, RDFS.Resource))
    g.add((etd_ns.wikidata, RDFS.label, Literal("wikidata link", lang="en")))

    # Reified relation slots:
    # predicate points to a property (rdf:Property), object may be IRI or literal → keep as rdf:Property
    g.add((etd_ns.predicate, RDF.type, RDF.Property))
    g.add((etd_ns.predicate, RDFS.label, Literal("relation predicate", lang="en")))

    g.add((etd_ns.object, RDF.type, RDF.Property))
    g.add((etd_ns.object, RDFS.label, Literal("relation object", lang="en")))

    # ======================================================
    # DATATYPE PROPERTIES
    # ======================================================


    # ======================================================
    # SKOS: Rating scales as ConceptSchemes
    # ======================================================

    def add_rating_scheme(scheme_local: str, scheme_label: str, values: List[str]):
        scheme = etd_ns[scheme_local]
        g.add((scheme, RDF.type, SKOS.ConceptScheme))
        g.add((scheme, RDFS.label, Literal(scheme_label, lang="en")))

        for idx, val in enumerate(values, start=1):
            # scheme-specific concept URI to avoid ambiguity
            concept = etd_ns[f"{scheme_local}{val}"]  # e.g., SeverityScaleHigh
            g.add((concept, RDF.type, SKOS.Concept))
            g.add((concept, SKOS.inScheme, scheme))
            g.add((concept, SKOS.prefLabel, Literal(val, lang="en")))
            g.add((scheme, SKOS.hasTopConcept, concept))
            g.add((concept, SKOS.topConceptOf, scheme))
            g.add((concept, SKOS.notation, Literal(str(idx), datatype=XSD.string)))

    add_rating_scheme("SeverityScale", "Risk severity scale", ["High", "Medium", "Low"])
    add_rating_scheme("LikelihoodScale", "Risk likelihood scale", ["High", "Medium", "Low"])
    add_rating_scheme("RiskLevelScale", "Overall risk level scale", ["High", "Medium", "Low"])
    add_rating_scheme("VoteValueScheme", "Vote value scheme", ["Upvote", "Downvote", "Neutral"])

    # ======================================================
    # DOMAIN / RANGE AXIOMS
    # ======================================================

    # ------------------------------------------------------
    # 1. Direct semantic relations (Entity ↔ Entity)
    # ------------------------------------------------------
    # canonical generic predicate
    g.add((etd_ns.relatedTo, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.relatedTo, RDFS.range, etd_ns.Entity))

    # other semantic relations between entities
    for p in [
        "supports",
        "criticizes",
        "endorses",
        "disagreesWith",
        "uses",
        "trainedOn",
        "reflects",
        "partOfDomain",
        "exhibitsRisk",
        "hasRisk",
        "particularlyAffects",
        "associatedWith",
        "causedBy",
        "affectedBy",
        "causes",
        "affects",
        "mitigatedBy",
        "compliesWith",
        "requiresComplianceWith",
        "requires",
        "violates",
        "mitigates",
        "opposes",
    ]:
        g.add((etd_ns[p], RDFS.domain, etd_ns.Entity))
        g.add((etd_ns[p], RDFS.range, etd_ns.Entity))


    # ------------------------------------------------------
    # 2. Document - (DiscourseUnit) ↔ Question / Answer / Person ----- Vote to InteractionRecord
    # ------------------------------------------------------
    # domain/range axioms

    # DiscourseUnit mentions Entity
    g.add((etd_ns.mentions, RDFS.domain, etd_ns.DiscourseUnit))
    g.add((etd_ns.mentions, RDFS.range, etd_ns.Entity))

    # Answer answersQuestion Question
    g.add((etd_ns.answersQuestion, RDFS.domain, etd_ns.Answer))
    g.add((etd_ns.answersQuestion, RDFS.range, etd_ns.Question))

    # Question hasAnswer Answer
    g.add((etd_ns.hasAnswer, RDFS.domain, etd_ns.Question))
    g.add((etd_ns.hasAnswer, RDFS.range, etd_ns.Answer))

    # Inverse relation hint for question-answer structure
    g.add((etd_ns.hasAnswer, OWL.inverseOf, etd_ns.answersQuestion))


    # Theme hasQuestion Question
    g.add((etd_ns.hasQuestion, RDFS.domain, etd_ns.Theme))
    g.add((etd_ns.hasQuestion, RDFS.range, etd_ns.Question))
    
    # Question belongsToTheme Theme (inverse structural link)
    g.add((etd_ns.belongsToTheme, RDFS.domain, etd_ns.Question))
    g.add((etd_ns.belongsToTheme, RDFS.range, etd_ns.Theme))
    g.add((etd_ns.hasTheme, RDFS.domain, etd_ns.Question))
    g.add((etd_ns.hasTheme, RDFS.range, etd_ns.Theme))

    # Inverse relation hint (optional but useful)
    g.add((etd_ns.belongsToTheme, OWL.inverseOf, etd_ns.hasQuestion))

    # DiscourseUnit authoredBy Person
    g.add((etd_ns.authoredBy, RDFS.domain, etd_ns.DiscourseUnit))
    g.add((etd_ns.authoredBy, RDFS.range, etd_ns.Person))

    # Question askedBy Expert
    g.add((etd_ns.askedBy, RDFS.domain, etd_ns.Question))
    g.add((etd_ns.askedBy, RDFS.range, etd_ns.Expert))

    # Theme createdBy Expert
    g.add((etd_ns.createdBy, RDFS.domain, etd_ns.Theme))
    g.add((etd_ns.createdBy, RDFS.range, etd_ns.Expert))

    # Answer repliesTo Answer
    g.add((etd_ns.repliesTo, RDFS.domain, etd_ns.Answer))
    g.add((etd_ns.repliesTo, RDFS.range, etd_ns.Answer))

    # DiscourseUnit hasVote Vote
    g.add((etd_ns.hasVote, RDFS.domain, etd_ns.DiscourseUnit))
    g.add((etd_ns.hasVote, RDFS.range, etd_ns.Vote))

    # Answer receivesVote Vote
    g.add((etd_ns.receivesVote, RDFS.domain, etd_ns.Answer))
    g.add((etd_ns.receivesVote, RDFS.range, etd_ns.Vote))
    g.add((etd_ns.votesForAnswer, RDFS.domain, etd_ns.Vote))
    g.add((etd_ns.votesForAnswer, RDFS.range, etd_ns.Answer))

    # Vote castBy Person
    g.add((etd_ns.castBy, RDFS.domain, etd_ns.Vote))
    g.add((etd_ns.castBy, RDFS.range, etd_ns.Person))

    # Person castVote Vote
    g.add((etd_ns.castVote, RDFS.domain, etd_ns.Person))
    g.add((etd_ns.castVote, RDFS.range, etd_ns.Vote))

    # Vote rating metadata
    g.add((etd_ns.voteValue, RDF.type, OWL.ObjectProperty))
    g.add((etd_ns.voteValue, RDFS.domain, etd_ns.Vote))
    g.add((etd_ns.voteValue, RDFS.range, SKOS.Concept))

    g.add((etd_ns.polarity, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.polarity, RDFS.domain, etd_ns.Vote))
    g.add((etd_ns.polarity, RDFS.range, XSD.string))

    # DiscourseUnit containsSubject Entity (e.g. dataset, system, risk)
    g.add((etd_ns.containsSubject, RDFS.domain, etd_ns.DiscourseUnit))
    g.add((etd_ns.containsSubject, RDFS.range, etd_ns.Entity))

    # provenance-bearing nodes can link explicitly to the source entity/entities
    g.add((etd_ns.hasSourceEntity, RDFS.domain, RDFS.Resource))
    g.add((etd_ns.hasSourceEntity, RDFS.range, etd_ns.Entity))

    # extractedFrom links provenance-bearing nodes to their origin discourse.
    # Use neutral domain to avoid unintended RDFS intersection-typing
    # (multiple rdfs:domain axioms would infer Evidence as Relation).
    g.add((etd_ns.extractedFrom, RDFS.domain, RDFS.Resource))
    g.add((etd_ns.extractedFrom, RDFS.range, etd_ns.DiscourseUnit))
    
    # --------------------------
    # Risk assessment
    # --------------------------
    g.add((etd_ns.assessesRisk, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.assessesRisk, RDFS.range, etd_ns.Risk))

    g.add((etd_ns.assessedBy, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.assessedBy, RDFS.range, etd_ns.Person))  # or etd_ns.Expert if strict

    g.add((etd_ns.usesEvidence, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.usesEvidence, RDFS.range, etd_ns.RiskEvidence))

    g.add((etd_ns.hasSeverity, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.hasSeverity, RDFS.range, SKOS.Concept))

    g.add((etd_ns.hasLikelihood, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.hasLikelihood, RDFS.range, SKOS.Concept))

    g.add((etd_ns.hasRiskLevel, RDFS.domain, etd_ns.RiskAssessment))
    g.add((etd_ns.hasRiskLevel, RDFS.range, SKOS.Concept))

    # Evidence attachment
    g.add((etd_ns.hasEvidence, RDFS.domain, etd_ns.Relation))
    g.add((etd_ns.hasEvidence, RDFS.range, etd_ns.Evidence))

    g.add((etd_ns.evidenceText, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.evidenceText, RDFS.domain, etd_ns.Evidence))
    g.add((etd_ns.evidenceText, RDFS.range, XSD.string))

    # --------------------------
    # Governance instrument references
    # --------------------------

    # DiscourseUnit -> RegulatoryReference
    g.add((etd_ns.hasRegulatoryReference, RDFS.domain, etd_ns.DiscourseUnit))
    g.add((etd_ns.hasRegulatoryReference, RDFS.range, etd_ns.RegulatoryReference))

    # RegulatoryReference -> governance instrument (LegalRequirement OR Standard OR BestPractice)
    # RDFS can't express union cleanly, so we use AiGovernanceFramework umbrella
    g.add((etd_ns.referencesInstrument, RDFS.domain, etd_ns.RegulatoryReference))
    g.add((etd_ns.referencesInstrument, RDFS.range, etd_ns.AiGovernanceFramework))

    # RegulatoryReference -> source-bearing evidence or reference node
    g.add((etd_ns.referencesSource, RDFS.domain, etd_ns.RegulatoryReference))
    g.add((etd_ns.referencesSource, RDFS.range, etd_ns.Evidence))

    # RegulatoryReference -> cited regulatory/source reference
    g.add((etd_ns.cites, RDFS.domain, etd_ns.RegulatoryReference))
    g.add((etd_ns.cites, RDFS.range, etd_ns.RegulatoryReference))

    # RiskEvidence evidenceSpan
    g.add((etd_ns.evidenceSpan, RDFS.domain, etd_ns.RiskEvidence))
    g.add((etd_ns.evidenceSpan, RDFS.range, XSD.string))

    # RegulatoryReference referenceSpan
    g.add((etd_ns.referenceSpan, RDFS.domain, etd_ns.RegulatoryReference))
    g.add((etd_ns.referenceSpan, RDFS.range, XSD.string))
 
    #Raw vote
    g.add((etd_ns.rawVoteValue, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.rawVoteValue, RDFS.domain, etd_ns.Vote))
    g.add((etd_ns.rawVoteValue, RDFS.range, XSD.integer))

           
    # ------------------------------------------------------
    # 3. Reified Relation structure
    #    etd:Relation with etd:subject / etd:predicate / etd:object
    # ------------------------------------------------------
    # All three attach to a Relation node
    g.add((etd_ns.subject, RDFS.domain, etd_ns.Relation))
    g.add((etd_ns.predicate, RDFS.domain, etd_ns.Relation))
    g.add((etd_ns.object, RDFS.domain, etd_ns.Relation))

    # subject, object point to Entity
    g.add((etd_ns.subject, RDFS.range, etd_ns.Entity))
    #g.add((etd_ns.object, RDFS.range, etd_ns.Entity))

    # predicate points to an RDF property (object or datatype property)
    g.add((etd_ns.predicate, RDFS.range, RDF.Property))    
    
    # ------------------------------------------------------
    # 4. Ontology URI / Wikidata URI (Entity → Resource)
    # ------------------------------------------------------
    # ontology_uri: link Entity to external ontology resource
    g.add((etd_ns.ontology_uri, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.ontology_uri, RDFS.range, RDFS.Resource))
    g.add((etd_ns.ontologyMappingSource, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.ontologyMappingSource, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.ontologyMappingSource, RDFS.range, XSD.string))
    g.add((etd_ns.ontologyConfidence, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.ontologyConfidence, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.ontologyConfidence, RDFS.range, XSD.float))

    # wikidata_uri: link Entity to Wikidata resource
    g.add((etd_ns.wikidata_uri, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidata_uri, RDFS.range, RDFS.Resource))

    g.add((etd_ns.wikidata, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidata, RDFS.range, RDFS.Resource))
    
    g.add((etd_ns.wikidataLabel, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataLabel, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataLabel, RDFS.range, RDF.langString))

    g.add((etd_ns.wikidataDescription, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataDescription, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataDescription, RDFS.range, RDF.langString))
    g.add((etd_ns.wikidataScore, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataScore, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataScore, RDFS.range, XSD.float))

    # Identity-provenance of a Wikidata alignment: whether it is an
    # authoritative asserted identifier (-> owl:sameAs) or a heuristic
    # best-guess semantic match (-> skos:closeMatch), plus the evidence
    # class and decision reason behind it.
    g.add((etd_ns.wikidataAuthoritative, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataAuthoritative, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataAuthoritative, RDFS.range, XSD.boolean))
    g.add((etd_ns.wikidataAuthoritative, RDFS.label, Literal("wikidata authoritative", lang="en")))
    g.add((etd_ns.wikidataIdentityEvidence, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataIdentityEvidence, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataIdentityEvidence, RDFS.range, XSD.string))
    g.add((etd_ns.wikidataIdentityEvidence, RDFS.label, Literal("wikidata identity evidence", lang="en")))
    g.add((etd_ns.wikidataDecisionReason, RDF.type, OWL.DatatypeProperty))
    g.add((etd_ns.wikidataDecisionReason, RDFS.domain, etd_ns.Entity))
    g.add((etd_ns.wikidataDecisionReason, RDFS.range, XSD.string))
    g.add((etd_ns.wikidataDecisionReason, RDFS.label, Literal("wikidata decision reason", lang="en")))
    # ------------------------------------------------------
    # 5. Datatype property domains / ranges
    # ------------------------------------------------------

    # Generic entity-level / relation-level metadata
    for p in [
        "entityType",
    ]:
        g.add((etd_ns[p], RDFS.domain, etd_ns.Entity))
        g.add((etd_ns[p], RDFS.range, XSD.string))

   
    # Apply authoritative registry-managed predicate declarations last.
    # Avoid adding new manual domain/range axioms above for predicates already
    # present in ETD_PROPERTY_REGISTRY unless the additive effect is intentional.
    _apply_property_registry()
    _ensure_term_docs_and_lifecycle()
    return allowed_classes
