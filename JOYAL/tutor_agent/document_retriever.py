
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentRetriever:

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No document chunks available")

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.vectors = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(self, question, top_k=3):
        if not question.strip():
            return []

        query_vector = self.vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            self.vectors
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(scores[index])
            })

        return results
