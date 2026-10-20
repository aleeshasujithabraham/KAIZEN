
from pathlib import Path

from ingest import extract_pdf
from chunker import create_chunks
from document_retriever import DocumentRetriever


pdf_path = (
    Path(__file__).parent
    / "data"
    / "learning_materials"
    / "python_notes.pdf"
)

documents = extract_pdf(pdf_path)
chunks = create_chunks(documents)

retriever = DocumentRetriever(chunks)

print("KAIZEN - PDF Retrieval Test")
print("=" * 40)

while True:
    question = input(
        "\nAsk a question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        break

    results = retriever.retrieve(question)

    if not results:
        print("No matching material found.")
        continue

    for result in results:
        print("\nSource:", result["source"])
        print("Page:", result["page"])
        print("Score:", round(result["score"], 3))
        print("Text:", result["text"][:500])
        print("-" * 40)
