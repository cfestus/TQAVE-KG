import hashlib
import tempfile
import types
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from models import Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
if HAS_RDFLIB:
    import main


def _production_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "output"


def _hash_tree(directory: Path) -> dict:
    """sha256 of every regular file under `directory`, keyed by path
    relative to it -- used to prove a directory was not written to by a
    test run (targeted correction pass, item 3)."""
    digests = {}
    if not directory.exists():
        return digests
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for main orchestration tests")
class MainOrchestrationTests(unittest.TestCase):
    """Root-cause fix (targeted correction pass, item 3): every test here
    used to pass args.output="output" (a path relative to the repo root)
    straight through to main(), which sets Config.OUTPUT_DIR = Path(args
    .output) unconditionally -- so main.main() ran against the REAL
    production output/ directory, and the (unmocked) real
    write_semantic_quality_report() call inside it wrote a null/empty
    stub over the genuine report. setUp gives each test its OWN
    tempfile.TemporaryDirectory() and points args.output at it instead
    (dependency injection via the same args.output the CLI already
    supports, rather than a new seam) -- this redirects every output
    write main() makes (annotations/metrics/conflicts -- mocked via
    save_json regardless -- and, critically, the real, unmocked
    write_semantic_quality_report call) away from the repository.
    Config.OUTPUT_DIR is restored after each test so this isolation
    cannot leak into any other test module that runs afterward in the
    same session.
    """

    def setUp(self):
        self._tmp_output = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_output.cleanup)
        self._orig_output_dir = main.Config.OUTPUT_DIR
        self.addCleanup(lambda: setattr(main.Config, "OUTPUT_DIR", self._orig_output_dir))

    def _args(self):
        return types.SimpleNamespace(
            config=None,
            input="data/ai_ethics_datasets.json",
            output=self._tmp_output.name,
            debug=False,
            no_reasoning=False,
            no_wikidata=True,
            batch_size=1,
            use_llm=False,
            llm_model="gpt-4o-mini",
        )

    def test_stage_order_reasoning_before_rdf_build(self):
        order = []

        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                order.append("reasoning")
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                order.append("rdf_build")
                self.graph = object()

            def serialize_to_turtle(self, _):
                pass

            def serialize_to_rdf(self, _):
                pass

            def serialize_to_jsonld(self, _):
                pass

            def serialize_to_owl(self, _):
                pass

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": True, "report_path": "x"}
            main.Config.USE_GRAPHDB_UPLOAD = False
            main.main()

        self.assertLess(order.index("reasoning"), order.index("rdf_build"))
        self.assertTrue((Path(self._tmp_output.name) / "semantic_quality_report.json").exists())

    def test_shacl_fail_fast_stops_serialize_and_upload(self):
        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        serialize_calls = {"count": 0}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                self.graph = object()

            def serialize_to_turtle(self, _):
                serialize_calls["count"] += 1

            def serialize_to_rdf(self, _):
                serialize_calls["count"] += 1

            def serialize_to_jsonld(self, _):
                serialize_calls["count"] += 1

            def serialize_to_owl(self, _):
                serialize_calls["count"] += 1

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls, \
             patch("utils.graphdb_client.GraphDBClient") as gdb_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": False, "report_path": "x"}
            main.Config.FAIL_ON_SHACL_VIOLATIONS = True
            main.Config.USE_GRAPHDB_UPLOAD = True
            with self.assertRaises(SystemExit):
                main.main()

        self.assertEqual(serialize_calls["count"], 0)
        gdb_cls.assert_not_called()
        # Even on the SHACL-failing path, the report must have been
        # written (it runs before the SHACL check) -- and only to the
        # isolated temp directory.
        self.assertTrue((Path(self._tmp_output.name) / "semantic_quality_report.json").exists())

    def test_inferred_relations_are_integrated_before_serialization(self):
        captured = {}

        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return [
                    Relation(
                        source="https://alfie-project.eu/etd-hub#entity_a",
                        relation="relatedTo",
                        target="https://alfie-project.eu/etd-hub#entity_b",
                        confidence=0.8,
                        is_inferred=True,
                        metadata={"discourse_id": "x"},
                    )
                ]

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                captured["annotations"] = annotations_dict
                self.graph = object()

            def serialize_to_turtle(self, _):
                pass

            def serialize_to_rdf(self, _):
                pass

            def serialize_to_jsonld(self, _):
                pass

            def serialize_to_owl(self, _):
                pass

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": True, "report_path": "x"}
            main.Config.USE_GRAPHDB_UPLOAD = False
            main.main()

        anns = captured["annotations"]
        rels = anns.get("metadata", {}).get("inferred_relations", [])
        self.assertGreaterEqual(len(rels), 1)
        self.assertTrue(rels[0].get("is_inferred", False))

    def test_upload_failure_warns_and_continues_when_not_fail_fast(self):
        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        serialize_calls = {"count": 0}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                self.graph = object()

            def serialize_to_turtle(self, _):
                serialize_calls["count"] += 1

            def serialize_to_rdf(self, _):
                serialize_calls["count"] += 1

            def serialize_to_jsonld(self, _):
                serialize_calls["count"] += 1

            def serialize_to_owl(self, _):
                serialize_calls["count"] += 1

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls, \
             patch("utils.graphdb_client.GraphDBClient") as gdb_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": True, "report_path": "x"}
            gdb_cls.return_value.check_connection.return_value = {"success": True, "status_code": 200, "error": None}
            gdb_cls.return_value.upload_rdf.return_value = {"success": False, "status_code": 503, "error": "unavailable"}
            main.Config.USE_GRAPHDB_UPLOAD = True
            main.Config.FAIL_ON_UPLOAD_FAILURE = False
            main.main()

        self.assertEqual(serialize_calls["count"], 4)

    def test_upload_failure_raises_when_fail_fast_enabled(self):
        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        serialize_calls = {"count": 0}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                self.graph = object()

            def serialize_to_turtle(self, _):
                serialize_calls["count"] += 1

            def serialize_to_rdf(self, _):
                serialize_calls["count"] += 1

            def serialize_to_jsonld(self, _):
                serialize_calls["count"] += 1

            def serialize_to_owl(self, _):
                serialize_calls["count"] += 1

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls, \
             patch("utils.graphdb_client.GraphDBClient") as gdb_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": True, "report_path": "x"}
            gdb_cls.return_value.check_connection.return_value = {"success": True, "status_code": 200, "error": None}
            gdb_cls.return_value.upload_rdf.return_value = {"success": False, "status_code": 503, "error": "unavailable"}
            main.Config.USE_GRAPHDB_UPLOAD = True
            main.Config.FAIL_ON_UPLOAD_FAILURE = True
            with self.assertRaises(SystemExit):
                main.main()

        self.assertEqual(serialize_calls["count"], 0)

    def test_output_isolation_production_untouched_both_paths(self):
        """Targeted correction pass, item 3, explicit verification test:
        production output/ must remain byte-identical after running
        main() through BOTH the success path and the SHACL-failure
        path, and the real write_semantic_quality_report() must have
        written only into this test's own temp directory."""
        production_dir = _production_output_dir()
        before = _hash_tree(production_dir)

        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                self.quality_events = kwargs.get("quality_events") if isinstance(kwargs.get("quality_events"), list) else []
                self.graph = object()

            def serialize_to_turtle(self, _):
                pass

            def serialize_to_rdf(self, _):
                pass

            def serialize_to_jsonld(self, _):
                pass

            def serialize_to_owl(self, _):
                pass

        # --- Success path ---
        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": True, "report_path": "x"}
            main.Config.USE_GRAPHDB_UPLOAD = False
            main.main()

        success_report = Path(self._tmp_output.name) / "semantic_quality_report.json"
        self.assertTrue(success_report.exists(), "report must be written to the temp dir on the success path")
        self.assertEqual(_hash_tree(production_dir), before, "production output/ must be untouched after the success path")

        # --- Failure path, using a SECOND, independent temp directory ---
        with tempfile.TemporaryDirectory() as failure_tmp:
            def _failure_args():
                args = self._args()
                args.output = failure_tmp
                return args

            with patch("main.parse_arguments", return_value=_failure_args()), \
                 patch.object(main.Config, "validate", return_value=True), \
                 patch.object(main.Config, "print_config", return_value=None), \
                 patch("builtins.print", return_value=None), \
                 patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
                 patch("utils.helpers.validate_input_json", return_value=True), \
                 patch("utils.helpers.save_json", return_value=None), \
                 patch("main.Path.exists", return_value=True), \
                 patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
                 patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
                 patch("validation.shacl_validator.SHACLValidator") as shacl_cls:
                shacl_cls.return_value.validate_graph.return_value = {"conforms": False, "report_path": "x"}
                main.Config.FAIL_ON_SHACL_VIOLATIONS = True
                main.Config.USE_GRAPHDB_UPLOAD = False
                with self.assertRaises(SystemExit):
                    main.main()

            failure_report = Path(failure_tmp) / "semantic_quality_report.json"
            self.assertTrue(failure_report.exists(), "report must be written to the temp dir on the failure path too")

        self.assertEqual(_hash_tree(production_dir), before, "production output/ must be untouched after the failure path")
        # The two runs' reports must be genuinely separate files/dirs --
        # neither path is production, and they don't collide with each other.
        self.assertNotEqual(str(success_report.parent), failure_tmp)


if __name__ == "__main__":
    unittest.main()
