
def create_chunks(documents, chunk_size=800, overlap=150):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    chunk_number = 0

    for document in documents:
        text = document["text"]
        step = chunk_size - overlap

        for start in range(0, len(text), step):
            content = text[start:start + chunk_size].strip()

            if not content:
                continue

            chunk_number += 1

            chunks.append({
                "chunk_id": f"chunk_{chunk_number:04d}",
                "source": document["source"],
                "page": document["page"],
                "text": content
            })

            if start + chunk_size >= len(text):
                break

    return chunks
