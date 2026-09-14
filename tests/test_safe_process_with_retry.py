import importlib.util
import unittest


HAS_REQUESTS = importlib.util.find_spec("requests") is not None


@unittest.skipUnless(HAS_REQUESTS, "requests is required for utils.helpers import")
class SafeProcessWithRetryTests(unittest.TestCase):
    def test_retry_failure_returns_processing_result_with_error_metadata(self):
        from models import ProcessingResult
        from utils.helpers import safe_process_with_retry

        def _always_fail(_item):
            raise ValueError("forced failure for retry test")

        result = safe_process_with_retry({"id": "x1"}, _always_fail, max_retries=2)

        self.assertIsInstance(result, ProcessingResult)
        self.assertEqual(result.entities, [])
        self.assertEqual(result.relations, [])
        self.assertTrue(result.metadata.get("processing_failed"))
        self.assertEqual(result.metadata.get("exception_type"), "ValueError")
        self.assertEqual(result.metadata.get("retries"), 2)
        self.assertIn("forced failure", result.metadata.get("error", ""))
        self.assertTrue(result.metadata.get("traceback"))

    def test_downstream_attribute_access_does_not_crash_on_failure_result(self):
        from utils.helpers import safe_process_with_retry

        def _always_fail(_item):
            raise RuntimeError("boom")

        result = safe_process_with_retry({"id": "x2"}, _always_fail, max_retries=1)

        # Mimics downstream pipeline access pattern.
        self.assertIsInstance(result.metadata, dict)
        self.assertIn("error", result.metadata)
        _ = result.metadata.get("error")
        _ = len(result.entities)
        _ = len(result.relations)


if __name__ == "__main__":
    unittest.main()

