
from pathlib import Path
import chromadb

from embeddings import GeminiEmbeddings


class VectorStore:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(__file__).parent / "chroma_db"

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="kaizen_study_materials",
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = GeminiEmbeddings()

    def add_chunks(self, chunks, document_id):
        if not chunks:
            raise ValueError("No chunks to index")

        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(
                f"{document_id}_{chunk['chunk_id']}"
            )
            texts.append(chunk["text"])
            metadatas.append({
                "document_id": document_id,
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            })

            embeddings.append(
                self.embedder.embed_document(chunk["text"])
            )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(ids)

    def search(self, question, document_id, top_k=3):
        query_embedding = self.embedder.embed_query(
            question
        )

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        ):
            matches.append({
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "score": 1.0 - float(distance)
            })

        return matches
