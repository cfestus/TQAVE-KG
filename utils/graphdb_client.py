import requests
import time
import random
import logging
from rdflib import Graph

from config import Config

logger = logging.getLogger(__name__)


class GraphDBClient:
    """Production-safe GraphDB client (GraphDB endpoints only)."""

    def __init__(self, max_retries: int = 3, retry_base_delay: float = 0.5, retry_max_delay: float = 4.0):
        server_url = getattr(Config, "GRAPHDB_SERVER_URL", None)
        repo_name = getattr(Config, "GRAPHDB_REPO_NAME", None)
        use_auth = getattr(Config, "GRAPHDB_USE_AUTH", True)

        if not server_url or str(server_url).strip().lower() == "none":
            raise ValueError("GRAPHDB_SERVER_URL is required")
        if not repo_name or str(repo_name).strip().lower() == "none":
            raise ValueError("GRAPHDB_REPO_NAME is required")

        self.server_url = str(server_url).strip().rstrip("/")
        self.repo_name = str(repo_name).strip()
        self.use_auth = bool(use_auth)

        self.username = None
        self.password = None
        if self.use_auth:
            username = getattr(Config, "GRAPHDB_USERNAME", None)
            password = getattr(Config, "GRAPHDB_PASSWORD", None)
            if not username or str(username).strip().lower() == "none":
                raise ValueError("GRAPHDB_USERNAME is required when GRAPHDB_USE_AUTH=true")
            if not password or str(password).strip().lower() == "none":
                raise ValueError("GRAPHDB_PASSWORD is required when GRAPHDB_USE_AUTH=true")
            self.username = str(username).strip()
            self.password = str(password).strip()

        self.update_endpoint = f"{self.server_url}/repositories/{self.repo_name}/statements"
        self.query_endpoint = f"{self.server_url}/repositories/{self.repo_name}"
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)

    def _auth(self):
        if not self.use_auth:
            return None
        return (self.username, self.password)

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= int(status_code) <= 599

    def _retry_delay(self, attempt: int) -> float:
        base = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(0.25, base * 0.25))
        return base + jitter

    def check_connection(self) -> dict:
        """Check if GraphDB repository endpoint is reachable."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.query_endpoint, auth=self._auth(), timeout=10)
                if response.status_code == 200:
                    return {"success": True, "status_code": 200, "error": None}
                if self._is_transient_status(response.status_code) and attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient check_connection failure %s (attempt %d/%d); retrying in %.2fs",
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"Connection failed: {response.status_code} - {response.text}",
                }
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient check_connection exception (attempt %d/%d); retrying in %.2fs: %s",
                        attempt,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                break
        logger.error(
            "GraphDB final check_connection failure after %d attempts: %s",
            self.max_retries,
            last_error,
        )
        return {"success": False, "status_code": None, "error": last_error}

    def upload_rdf(self, rdf_graph: Graph) -> dict:
        """Upload an RDF graph to the configured GraphDB repository."""
        rdf_data = rdf_graph.serialize(format="turtle")
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.update_endpoint,
                    headers={"Content-Type": "text/turtle"},
                    auth=self._auth(),
                    data=rdf_data,
                    timeout=30,
                )
                if response.status_code in (200, 204):
                    return {"success": True, "status_code": response.status_code, "error": None}
                if self._is_transient_status(response.status_code) and attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient upload failure %s (attempt %d/%d); retrying in %.2fs",
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"Upload failed: {response.status_code} - {response.text}",
                }
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient upload exception (attempt %d/%d); retrying in %.2fs: %s",
                        attempt,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                break
        logger.error(
            "GraphDB final upload failure after %d attempts: %s",
            self.max_retries,
            last_error,
        )
        return {"success": False, "status_code": None, "error": last_error}

    def run_sparql_query(self, query: str) -> dict:
        """Execute a SPARQL query on the configured GraphDB repository."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.query_endpoint,
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth(),
                    params={"query": query},
                    timeout=30,
                )
                if response.status_code == 200:
                    return {
                        "success": True,
                        "status_code": 200,
                        "data": response.json(),
                        "error": None,
                    }
                if self._is_transient_status(response.status_code) and attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient query failure %s (attempt %d/%d); retrying in %.2fs",
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "data": None,
                    "error": f"SPARQL query failed: {response.status_code} - {response.text}",
                }
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "GraphDB transient query exception (attempt %d/%d); retrying in %.2fs: %s",
                        attempt,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                break
        logger.error(
            "GraphDB final query failure after %d attempts: %s",
            self.max_retries,
            last_error,
        )
        return {"success": False, "status_code": None, "data": None, "error": last_error}
