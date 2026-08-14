"""
CLI script to run the Google ADK agent interactively.

Usage:
    python -m scripts.run_agent
"""

import asyncio
import logging
import sys

from src.agent.rag_agent import run_agent_query


def main() -> None:
    """Interactive agent CLI."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n🤖 RAG Agent Ready (powered by Google ADK + Gemini)")
    print("   Session memory enabled — agent remembers previous questions")
    print("   Type 'quit' to exit\n")

    loop = asyncio.new_event_loop()

    while True:
        try:
            question = input("❓ Ask the agent: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\n🤖 Thinking...\n")
        try:
            answer = loop.run_until_complete(run_agent_query(question))
            print(f"📝 Agent:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

    loop.close()


if __name__ == "__main__":
    main()
