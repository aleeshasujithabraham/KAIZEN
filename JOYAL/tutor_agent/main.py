
from tutor import TutorAgent


def main():
    tutor = TutorAgent()

    print("\nKAIZEN TUTOR AGENT")
    print("=" * 40)

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
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
