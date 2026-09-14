from typing import Dict, List

# ============================================================
# ORIGINAL FULL ONTOLOGY MAPPINGS (required by OntologyMapper)
# ============================================================

CONCEPT_TO_ONTOLOGY_CLASSES: Dict[str, List[str]] = {
    "bias": ["airo:BiasRisk"],
    "racial bias": ["airo:RacialBiasRisk"],
    "historical bias": ["fmo:HistoricalBias"],
    "algorithmic bias": ["fmo:Bias"],
    "fairness": ["fmo:Fairness"],
    "transparency": ["aipo:Transparency"],
    "explainability": ["aipo:Explainability"],
    "privacy": ["dpv:Privacy"],
    "personal data": ["dpv:PersonalData"],
    "consent": ["dpv:Consent"],
    "accountability": ["aipo:Accountability"],
    "risk": ["airo:Risk"],
    "safety": ["airo:SafetyRisk"],
    "governance": ["aipo:AIGovernance"],
    "regulation": ["vair:Regulation"],
    "audit": ["aipo:AssessmentProcedure"],
}

ENTITY_TYPE_TO_ONTOLOGY: Dict[str, List[str]] = {
    "Dataset": ["dcat:Dataset"],
    "AISystem": ["aipo:AISystem"],
    # Model previously mapped identically to AISystem, contradicting
    # this schema's own declared class definition ("...as distinct from
    # AISystem, which is the deployed system that embeds and operates one or
    # more models" -- ontology/add_schema.py). Grounding every Model instance
    # to aipo:AISystem asserted at the RDF level that a model component IS a
    # full deployed AI system, which is not semantically defensible. No
    # external ontology class was found that precisely represents "the
    # underlying learned/computational artefact distinct from the system
    # that embeds it" -- rather than force an imprecise external alignment,
    # Model grounds to its own already-declared local class (etd:Model,
    # rdfs:subClassOf etd:System, sibling of etd:AISystem).
    "Model": ["etd:Model"],
    # Process/AuditProcess/Evaluation/AiGovernanceFramework are
    # already declared, correctly hierarchised local ETD classes with no
    # external ontology alignment declared for any of them -- ground each to
    # its own specific local class rather than the generic prov:Entity
    # fallback.
    "Process": ["etd:Process"],
    "AuditProcess": ["etd:AuditProcess"],
    "Evaluation": ["etd:Evaluation"],
    "AiGovernanceFramework": ["etd:AiGovernanceFramework"],
    "Person": ["foaf:Person"],
    "Expert": ["foaf:Person"],
    "Organization": ["org:Organization"],
    "EthicalConcept": ["aipo:EthicalConcept"],
    "BiasType": ["fmo:Bias"],
    "Risk": ["airo:Risk"],
    "Regulation": ["vair:Regulation"],
    "Theme": ["skos:Concept"],
    "Question": ["schema:Question"],
    "Answer": ["schema:Answer"],
    "Vote": ["prov:Activity"],
    "Document": ["foaf:Document"],
    # Resource is a heterogeneous information-hosting/-providing
    # class (repository, platform, catalogue, portal, collection host) that
    # is not itself a written work -- foaf:Document was semantically
    # inaccurate (it asserted every Resource instance IS a Document) and,
    # via canonical_linker's ontology_uri-driven coarse-type bucketing,
    # caused Resource entities to share an identity bucket with genuine
    # Document entities. dcat:Resource is DCAT's own broad superclass for
    # exactly this category (it is the superclass of dcat:Dataset,
    # dcat:Catalog, dcat:DataService in the real DCAT ontology) and DCAT is
    # already imported/used elsewhere in this schema (Dataset -> dcat:Dataset).
    "Resource": ["dcat:Resource"],
    "DiscourseUnit": ["prov:Entity"],
    "PolicyFramework": ["vair:Regulation"],
    "Law": ["vair:Regulation"],
    "Standard": ["vair:Regulation"],
    "Corpus": ["dcat:Dataset"],
    "DataSource": ["dcat:Dataset"],
    # Entity types the Dataset-family disambiguator can now
    # produce -- grounded to the same external alignment chosen elsewhere
    # in this schema (ResearchStudy/Analysis deliberately have no
    # forced external alignment there, so they are left unmapped here too
    # and fall back to the same generic grounding as any other class
    # without a specific external match).
    "Publication": ["bibo:Document"],
    "SoftwareProduct": ["schema:SoftwareApplication"],
    "Concept": ["skos:Concept"],
    # PopulationGroup/DemographicGroup are already declared,
    # correctly hierarchised local ETD classes (DemographicGroup subClassOf
    # PopulationGroup subClassOf Entity -- see ontology/add_schema.py) with
    # no external ontology alignment declared for either. Ground them to
    # their own specific local class rather than letting them fall through
    # to the generic prov:Entity fallback in _ensure_entity_ontology_grounding.
    "PopulationGroup": ["etd:PopulationGroup"],
    "DemographicGroup": ["etd:DemographicGroup"],
}

RELATION_TO_ONTOLOGY_PROPERTY: Dict[str, List[str]] = {
    "uses": ["aipo:uses"],
    "trainedOn": ["aipo:trainedOn"],
    "developedBy": ["prov:wasAttributedTo"],
    "evaluatedBy": ["aipo:evaluatedBy"],
    "affects": ["aipo:affects"],
    "exhibitsRisk": ["airo:hasRisk"],
    "violates": ["aipo:violates"],
    "compliesWith": ["aipo:compliesWith"],
    "requiresComplianceWith": ["vair:hasRequirement"],
    "requires": ["vair:hasRequirement"],
    "mitigates": ["aipo:mitigates"],
    "referencesSource": ["dcterms:source"],
    "cites": ["dcterms:references"],
    "contains": ["dcat:contains"],
}

DOMAIN_TO_VAIR: Dict[str, str] = {
    "health": "vair:HealthcareDomain",
    "medical": "vair:HealthcareDomain",
    "justice": "vair:JusticeDomain",
    "criminal": "vair:JusticeDomain",
    "finance": "vair:FinanceDomain",
    "education": "vair:EducationDomain",
    "employment": "vair:EmploymentDomain",
    "transport": "vair:TransportationDomain",
}

PROBLEM_CATEGORY_TO_ONTOLOGY: Dict[str, List[str]] = {
    "bias": ["airo:BiasRisk"],
    "privacy": ["dpv:Privacy"],
    "fairness": ["fmo:Fairness"],
    "transparency": ["aipo:Transparency"],
}

FMO_MAPPINGS: Dict[str, str] = {
    "equalized odds": "fmo:EqualizedOdds",
    "equal opportunity": "fmo:EqualOpportunity",
    "demographic parity": "fmo:DemographicParity",
    "predictive parity": "fmo:PredictiveParity",
    "calibration": "fmo:Calibration",
}

ETHICAL_CONCEPT_TO_AIRO_RISK: Dict[str, str] = {
    "bias": "airo:BiasRisk",
    "racial bias": "airo:RacialBiasRisk",
    "discrimination": "airo:DiscriminationRisk",
    "privacy breach": "dpv:DataBreach",
}

DEMOGRAPHIC_LEXICON: Dict[str, str] = {
    "black": "etd:Race_Black",
    "white": "etd:Race_White",
    "asian": "etd:Race_Asian",
    "women": "etd:Gender_Female",
    "men": "etd:Gender_Male",
}

RELATION_MAPPINGS: Dict[str, str] = {
    "domain": "vair:hasDomain",
    "ethicalConcept": "aipo:hasConcept",
    "demographic": "etd:hasPopulation",
    "fairnessMetric": "fmo:hasMetric",
    "privacyConcept": "dpv:hasConcept",
    "governanceOrRegulation": "vair:hasRequirement",
    "activityOrProcess": "aipo:involves",
    "stakeholder": "prov:wasAttributedTo",
}

# ============================================================
# NEW: ONTOLOGY_TERMS FOR LLM ENTITY EXTRACTOR
# ============================================================
ONTOLOGY_TERMS: Dict[str, str] = {
    "bias": "airo:BiasRisk",
    "racial bias": "airo:RacialBiasRisk",
    "algorithmic bias": "airo:BiasRisk",
    "fairness": "fmo:Fairness",
    "transparency": "aipo:Transparency",
    "privacy": "dpv:Privacy",
    "consent": "dpv:Consent",
    "safety": "airo:SafetyRisk",
    "governance": "aipo:AIGovernance",
    "audit": "aipo:AssessmentProcedure",
    "machine learning": "aipo:MachineLearning",
    "deep learning": "aipo:DeepLearning",
    "dataset": "dcat:Dataset",
    "model": "aipo:AISystem",
    "ai system": "aipo:AISystem",
    "equalized odds": "fmo:EqualizedOdds",
    "equal opportunity": "fmo:EqualOpportunity",
    "demographic parity": "fmo:DemographicParity",
    "high-risk": "vair:HighRiskAI",
    "gdpr": "dpv:GDPR",
    "personal data": "dpv:PersonalData",
    "criminal justice": "vair:JusticeDomain",
    "healthcare": "vair:HealthcareDomain",
    "employment": "vair:EmploymentDomain",
}
