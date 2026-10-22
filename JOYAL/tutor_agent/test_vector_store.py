
from pathlib import Path

from ingest import extract_pdf
from chunker import create_chunks
from vector_store import VectorStore


pdf_path = (
    Path(__file__).parent
    / "data"
    / "learning_materials"
    / "python_notes.pdf"
)

document_id = "python_notes_demo"

print("Loading PDF...")
documents = extract_pdf(pdf_path)
chunks = create_chunks(documents)

print("Chunks created:", len(chunks))

store = VectorStore()

print("\nGenerating embeddings and indexing chunks...")
print("This may take some time.")

count = store.add_chunks(chunks, document_id)

print("\nChunks indexed:", count)
print("Vector database ready!")

while True:
    question = input(
        "\nAsk a question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    results = store.search(
        question,
        document_id=document_id,
        top_k=3
    )

    if not results:
        print("No indexed material found.")
        continue

    print("\nSemantic Search Results:")

    for result in results:
        print("\nSource:", result["source"])
        print("Page:", result["page"])
        print("Similarity:", round(result["score"], 3))
        print("Text:", result["text"][:350])
        print("-" * 40)
