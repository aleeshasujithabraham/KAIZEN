
from llm_generator import LLMGenerator


generator = LLMGenerator()

chunks = [
    {
        "source": "test_notes.pdf",
        "page": 1,
        "text": (
            "A Python variable is a name that refers "
            "to a value. For example, age = 20."
        )
    }
]

answer = generator.generate(
    "What is a Python variable?",
    chunks
)

print("\nGemini response:")
print(answer)
