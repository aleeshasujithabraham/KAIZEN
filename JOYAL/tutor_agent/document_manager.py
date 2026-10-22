
from pathlib import Path
from uuid import uuid4

from ingest import extract_pdf
from chunker import create_chunks
from vector_store import VectorStore


class DocumentManager:

    def __init__(self):
        self.store = VectorStore()

    def ingest_document(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")

        documents = extract_pdf(pdf_path)
        chunks = create_chunks(documents)

        if not chunks:
            raise ValueError(
                "The PDF contains no extractable text."
            )

        document_id = uuid4().hex

        count = self.store.add_chunks(
            chunks=chunks,
            document_id=document_id
        )

        return {
            "document_id": document_id,
            "filename": pdf_path.name,
            "pages_with_text": len(documents),
            "chunks_indexed": count
        }
