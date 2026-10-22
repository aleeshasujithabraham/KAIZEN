
from pathlib import Path

from document_manager import DocumentManager
from tutor import TutorAgent


def main():
    print("=" * 45)
    print("KAIZEN - STUDENT DOCUMENT TUTOR")
    print("=" * 45)

    pdf_input = input(
        "\nEnter the full path of your PDF file: "
    ).strip().strip('"').strip("'")

    pdf_path = Path(pdf_input).expanduser()

    if not pdf_path.is_file():
        print("Error: PDF file not found.")
        return

    manager = DocumentManager()

    print("\nProcessing uploaded document...")

    try:
        result = manager.ingest_document(pdf_path)
    except Exception as error:
        print("Unable to process the PDF:", type(error).__name__)
        return

    print("\nDocument ready!")
    print("Filename:", result["filename"])
    print("Pages:", result["pages_with_text"])
    print("Chunks indexed:", result["chunks_indexed"])

    tutor = TutorAgent(
        document_id=result["document_id"]
    )

    while True:
        question = input(
            "\nAsk a question (or 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        response = tutor.ask(question)

        print("\nStatus:", response["status"])
        print("\nTutor Response:\n", response["answer"])

        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(
                    f"- {source['source']}, "
                    f"page {source['page']}, "
                    f"chunk {source['chunk_id']}"
                )


if __name__ == "__main__":
    main()
