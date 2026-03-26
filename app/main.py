from app.rag import RAGAssistant


def main() -> None:
    assistant = RAGAssistant()

    print("NCPDP RAG Agent")
    print("Type your question or 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not question:
            continue

        try:
            answer = assistant.ask(question)
            print("\nAnswer:")
            print(answer)
            print("\n" + "-" * 60 + "\n")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()