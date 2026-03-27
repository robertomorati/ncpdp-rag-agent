from app.agent import NCPDPAgent


def main() -> None:
    agent = NCPDPAgent()

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
            result = agent.run(question)

            print("\n--- Judgment ---")
            print(result["judgment"])

            if result["rewritten_query"]:
                print("Rewritten query:", result["rewritten_query"])

            print("\n--- Reflection ---")
            print(result["reflection"])

            print("\n--- Final Answer ---")
            print(result["final_answer"])
            print("\n" + "-" * 60 + "\n")

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()