from pathlib import Path

from rdflib import Graph

from config import Config
from serializers.rdf_serializer import RDFSerializer


def main() -> None:
    annotations = {
        "themes": {
            "t1": {
                "entities": [],
                "relations": [
                    {
                        "source": "https://alfie-project.eu/etd-hub#entity_a",
                        "relation": "mentions",
                        "target": "https://alfie-project.eu/etd-hub#entity_b",
                        "confidence": 0.9,
                        "metadata": {"discourse_id": "7"},
                    }
                ],
            }
        }
    }

    s = RDFSerializer(annotations_dict=annotations, registry=None, namespaces=Config.NAMESPACES)
    out = Path("output/discourse_id_datatype_check.ttl")
    out.parent.mkdir(parents=True, exist_ok=True)
    s.serialize_to_turtle(out)

    ttl = out.read_text(encoding="utf-8")
    expected = 'etd:discourse_id "7"^^xsd:integer'
    if expected not in ttl:
        raise AssertionError(f"Expected '{expected}' in TTL output")

    g = Graph()
    g.parse(out.as_posix(), format="turtle")
    print("OK: discourse_id serialized as xsd:integer and TTL parses correctly")


if __name__ == "__main__":
    main()

