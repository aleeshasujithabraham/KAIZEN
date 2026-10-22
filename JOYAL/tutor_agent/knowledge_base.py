
import json
from pathlib import Path


class KnowledgeBase:
    def __init__(self):
        graph_path = (
            Path(__file__).parent
            / "data"
            / "learning_graph.json"
        )

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.concepts = {
            concept["id"]: concept
            for concept in self.graph["concepts"]
        }

    def get_concept(self, concept_id):
        return self.concepts.get(concept_id)


if __name__ == "__main__":
    kb = KnowledgeBase()

    print("Knowledge Base Loaded!")
    print("Total Concepts:", len(kb.concepts))

    print("\nTesting concept c01:")
    print(kb.get_concept("c01"))
