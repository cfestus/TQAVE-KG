# semantic_typing.py
import re
from typing import Optional, Set
from rdflib import Literal
from rdflib.namespace import RDF, XSD

from utils.canonical_linker import _types_ontology_compatible
from ontology.property_registry import is_registered_etd_predicate


def _local_class_name(term) -> str:
    s = str(term)
    return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]


class SemanticTyper:
    def __init__(self, graph, etd_ns, allowed_classes: Optional[Set[str]] = None, domain_profile=None):
        self.graph = graph
        self.etd = etd_ns
        self.allowed_classes = allowed_classes  # None means "no filtering"
        # Active domain profile (see domain_profiles.py) -- same
        # resolution pattern as every other component in this pipeline
        # (EnhancedSemanticPipeline, RDFSerializer, CanonicalLinker,
        # EntityNormalizer, SemanticRelationPolicy): an explicit
        # domain_profile= overrides; otherwise resolved from
        # Config.DOMAIN_PROFILE, reproducing current AI-ethics behaviour
        # by default.
        if domain_profile is not None:
            self.domain_profile = domain_profile
        else:
            from domain_profiles import get_active_profile

            self.domain_profile = get_active_profile()

    def _add_type(self, uri, class_name: str) -> bool:
        """
        Only mint rdf:type etd:<class_name> if class_name is allowed/declared
        AND is ontology-compatible with every substantive type already
        asserted on this node. This is a general per-node gate,
        not a hard-coded pair list: a candidate is rejected only when it is
        neither an ancestor nor a descendant, in the real class hierarchy,
        of a type the node already carries -- e.g. a lexical proposal of
        "AISystem" is rejected once a node already carries the sibling type
        "Model", while a proposal that is a genuine superclass/subclass of an
        existing type (e.g. "PopulationGroup" alongside "DemographicGroup")
        is allowed through. "Entity" is the universal root of every branch
        and is excluded from the existing-type comparison set since it is
        trivially compatible with anything and asserting it is not itself a
        substantive typing decision.
        """
        if self.allowed_classes and class_name not in self.allowed_classes:
            print(f"⚠️ [SemanticTyper] Type blocked by allowed_classes: {class_name} for {uri}")
            return False

        existing_types = {
            _local_class_name(t) for t in self.graph.objects(uri, RDF.type)
        } - {"Entity"}
        for existing in existing_types:
            if not _types_ontology_compatible(class_name, existing):
                if is_registered_etd_predicate("supersededProposedType"):
                    self.graph.add((
                        uri,
                        self.etd.supersededProposedType,
                        Literal(class_name, datatype=XSD.string),
                    ))
                return False

        self.graph.add((uri, RDF.type, self.etd[class_name]))
        return True

    # --- Core public method -----------------------------
    def classify(self, uri, label_text):
        """
        Apply all semantic typing heuristics.
        label_text must be lowercase string.
        """

        # Normalize once
        label_text = label_text.lower().strip()

        # ------------------------------------------------------------
        # 1. Dataset / DataElement / DataIssue
        #
        # CORE, domain-independent: "is this a dataset-shaped resource"
        # applies identically regardless of subject-matter domain -- kept
        # here rather than moved to a domain profile.
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "dataset", "data set", "records", "csv",
            "311", "census", "demographics"
        ]):
            if any(kw in label_text for kw in ["missing", "skew", "drift", "imbalance"]):
                self._add_type(uri, "DataIssue")
            elif any(kw in label_text for kw in ["history", "attribute", "field"]):
                self._add_type(uri, "DataElement")
            else:
                self._add_type(uri, "Dataset")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 2. Standards
        #
        # CORE, domain-independent: standards bodies (NIST/ISO/IEEE/OECD/
        # UNESCO) issue standards across virtually every subject-matter
        # domain, not just AI -- kept here rather than moved.
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "nist", "iso", "ieee", "unesco", "standard", "oecd"
        ]):
            self._add_type(uri, "Standard")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 3. Processes / Evaluation / Action
        #
        # CORE, domain-independent: audits/assessments/reviews are a
        # near-universal governance concept, not AI-ethics-specific --
        # kept here rather than moved.
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "audit", "assessment", "evaluation", "review",
            "analysis", "process"
        ]):
            self._add_type(uri, "Process")
            self._add_type(uri, "Evaluation")
            self._add_type(uri, "Action")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 4. Domain-profile-driven positive keyword typing (formerly
        # separate "Bias/Harm/Impact", "System/Component",
        # "Ethical Principles", "Documentation artifacts", and
        # "Stakeholder" rules -- each hardcoded its own
        # AI-ethics-specific word list directly in this generic module,
        # some of them byte-for-byte duplicated in pipeline.py/
        # utils/semantic_relation_policy.py too).
        #
        # This mechanism itself (iterate rules, apply every triggered
        # class through the SAME ontology-compatibility gate as every
        # other rule in this file) is CORE and domain-independent. The
        # actual trigger vocabulary and classes-to-add live entirely in
        # the active domain profile's `type_assignment_rules` (see
        # domain_profiles.py::AI_ETHICS_PROFILE) -- this method does not
        # know "Bias", "AISystem", "EthicalPrinciple", or any AI-ethics
        # term by name. A domain profile with zero type_assignment_rules
        # (e.g. a synthetic non-AI domain that has not configured any)
        # simply adds nothing here, exactly like "no assumption" anywhere
        # else in this profile system -- it does not error.
        #
        # The former dedicated "Documentation artifacts" rule (typing
        # "Model Cards"/"datasheet" labels as Resource ahead of the
        # System/Component rule, to block the incompatible AISystem type
        # via the compatibility gate below) is now one of the profile's
        # ordinary type_assignment_rules, reusing the EXACT SAME compiled
        # pattern object as the AISystem ClassCorrectionRule's
        # disqualifying_signal in serializers/rdf_serializer.py -- not a
        # second, independently-maintained copy of that vocabulary.
        # ------------------------------------------------------------
        for rule in self.domain_profile.type_assignment_rules:
            matched = (
                rule.trigger_signal.fullmatch(label_text)
                if rule.exact
                else rule.trigger_signal.search(label_text)
            )
            if matched:
                for class_name in rule.classes_to_add:
                    self._add_type(uri, class_name)
            # no return/continue-out: allow every rule to apply, exactly
            # as the original hardcoded rules all fell through to one
            # another.

        # Case 3: label contains but does NOT equal → ignore (avoid false positives)
        return
