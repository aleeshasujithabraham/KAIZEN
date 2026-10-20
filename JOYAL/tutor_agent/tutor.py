
from semantic_retriever import SemanticRetriever
from llm_generator import LLMGenerator


class TutorAgent:

    def __init__(self, document_id="python_notes_demo"):

        self.retriever = SemanticRetriever(
            document_id=document_id
        )

        self.generator = LLMGenerator()

        # Experimental threshold; calibrate later.
        self.threshold = 0.55

    def ask(self, question):

        if not question.strip():
            return {
                "status": "invalid_question",
                "answer": "Please enter a question.",
                "sources": []
            }

        results = self.retriever.retrieve(
            question,
            top_k=3
        )

        print("\n--- SEMANTIC SEARCH DEBUG ---")

        for result in results:
            print(
                "Page:", result["page"],
                "| Similarity:",
                round(result["score"], 3)
            )

        if not results:
            return {
                "status": "insufficient_information",
                "answer": "No indexed learning material found.",
                "sources": []
            }

        relevant = [
            result
            for result in results
            if result["score"] >= self.threshold
        ]

        if not relevant:
            return {
                "status": "insufficient_information",
                "answer": (
                    "I don't have enough relevant "
                    "information in the uploaded notes "
                    "to answer your question."
                ),
                "sources": []
            }

        try:
            answer = self.generator.generate(
                question,
                relevant
            )

        except Exception as error:
            print(
                "Gemini error:",
                type(error).__name__
            )

            return {
                "status": "generation_error",
                "answer": "Unable to generate an answer right now.",
                "sources": []
            }

        if (
            not answer
            or "INSUFFICIENT_INFORMATION" in answer
        ):
            return {
                "status": "insufficient_information",
                "answer": (
                    "The retrieved notes don't contain "
                    "enough information to answer reliably."
                ),
                "sources": []
            }

        return {
            "status": "answered",
            "answer": answer,
            "sources": [
                {
                    "source": result["source"],
                    "page": result["page"],
                    "chunk_id": result["chunk_id"]
                }
                for result in relevant
            ]
        }
