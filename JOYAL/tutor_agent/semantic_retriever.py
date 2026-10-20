
from vector_store import VectorStore


class SemanticRetriever:

    def __init__(self, document_id):
        self.store = VectorStore()
        self.document_id = document_id

    def retrieve(self, question, top_k=3):
        if not question.strip():
            return []

        return self.store.search(
            question=question,
            document_id=self.document_id,
            top_k=top_k
        )
