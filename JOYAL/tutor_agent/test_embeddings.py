
from embeddings import GeminiEmbeddings


embedding_model = GeminiEmbeddings()

document = (
    "A Python variable is a name that refers "
    "to a value. For example, age = 20."
)

question = "How do variables work in Python?"

print("Generating document embedding...")
document_vector = embedding_model.embed_document(document)

print("Generating question embedding...")
question_vector = embedding_model.embed_query(question)

print("\nDocument dimensions:", len(document_vector))
print("Question dimensions:", len(question_vector))
print("First five values:", document_vector[:5])

print("\nEmbedding test successful!")
