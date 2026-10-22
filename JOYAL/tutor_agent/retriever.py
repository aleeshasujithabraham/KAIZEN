
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:

    def __init__(self, concepts):
        self.concepts = list(concepts.values())

        self.documents = [
            concept["name"] + " " + concept["description"]
            for concept in self.concepts
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.vectors = self.vectorizer.fit_transform(
            self.documents
        )

    def retrieve(self, question, top_k=3):

        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            self.vectors
        )[0]

        indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in indices:
            results.append({
                "concept_id": self.concepts[index]["id"],
                "concept_name": self.concepts[index]["name"],
                "description": self.concepts[index]["description"],
                "score": float(scores[index])
            })

        return results
