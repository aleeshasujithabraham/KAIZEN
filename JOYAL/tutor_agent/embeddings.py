
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from httpx import ConnectError, ConnectTimeout


# Load Gemini API key from .env
load_dotenv(Path(__file__).parent / ".env")


class GeminiEmbeddings:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"

    def embed_with_retry(self, text, task_type):
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=768
                    )
                )

                return response.embeddings[0].values

            except (
                ConnectError,
                ConnectTimeout,
                errors.ServerError
            ) as error:

                if attempt == max_attempts - 1:
                    print(
                        "Embedding API failed:",
                        type(error).__name__
                    )
                    raise

                delay = 2 ** attempt

                print(
                    f"Temporary connection problem. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

    def embed_document(self, text):
        return self.embed_with_retry(
            text,
            "RETRIEVAL_DOCUMENT"
        )

    def embed_query(self, question):
        return self.embed_with_retry(
            question,
            "RETRIEVAL_QUERY"
        )
