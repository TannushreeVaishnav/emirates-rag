"""
cli_ask.py — One-shot question answerer for quick terminal testing.

Useful during development to test a specific question without launching the
full Streamlit UI. Calls the baseline RAG chain (no LangGraph agent).

Usage:
    python -m src.cli_ask "What is the cabin baggage allowance for Economy?"
    python -m src.cli_ask   ← interactive: prompts for input if no argument given
"""

import sys

from src.rag_chain import ask


def main() -> None:
    # Accept the question as a command-line argument or prompt interactively
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Question: ").strip()

    if not question:
        print("No question provided.")
        return

    print("\nSearching...")
    res = ask(question)
    print(f"\n{res['answer']}\n")

    # Print each citation on its own line for readability
    for c in res["citations"]:
        page = f", p.{c['page']}" if c["page"] else ""
        print(f"  [{c['ref']}] {c['source']} | {c['section']}{page}")

    print(f"\n  ({res['latency']:.1f}s)")


if __name__ == "__main__":
    main()
