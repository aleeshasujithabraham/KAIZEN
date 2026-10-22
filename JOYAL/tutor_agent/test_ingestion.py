
from pathlib import Path
from ingest import extract_pdf
from chunker import create_chunks


pdf_path = (
    Path(__file__).parent
    / "data"
    / "learning_materials"
    / "python_notes.pdf"
)

print("Reading PDF:", pdf_path)

documents = extract_pdf(pdf_path)
chunks = create_chunks(documents)

print("\nPDF ingestion successful!")
print("Pages containing text:", len(documents))
print("Total chunks created:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print("Chunk ID:", chunks[0]["chunk_id"])
    print("Source:", chunks[0]["source"])
    print("Page:", chunks[0]["page"])
    print("Text:", chunks[0]["text"][:500])
else:
    print("No text could be extracted from the PDF.")
