"""
cli_chat.py — Multi-turn conversational CLI for the LangGraph agent.

Lets me test the full agent (guardrails, router, retrieval, grading,
generation, output check, memory) from the terminal without launching Streamlit.

Commands during a session:
    /new   — start a fresh conversation (clears LangGraph memory)
    /quit  — exit

Each response shows the route the agent took, which LLM provider answered,
the number of rewrites used, and wall-clock latency.

Usage:
    python -m src.cli_chat
"""

import uuid

from src.graph import chat


def main() -> None:
    thread_id = str(uuid.uuid4())   # unique thread = separate memory in MemorySaver
    print("Emirates Policy Assistant (type /new to reset, /quit to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break
        if user_input.lower() == "/new":
            thread_id = str(uuid.uuid4())
            print("  [New conversation started]\n")
            continue

        res = chat(user_input, thread_id)
        print(f"\nAssistant: {res['answer']}\n")

        # Print citations compactly
        for c in res["citations"]:
            page = f", p.{c['page']}" if c["page"] else ""
            print(f"   [{c['ref']}] {c['source']} | {c['section']}{page}")

        # Metadata line: useful when debugging routing decisions
        print(
            f"   (route={res['route'] or 'blocked'}, "
            f"provider={res['provider_used'] or '-'}, "
            f"retries={res['retries']}, "
            f"{res['latency']:.1f}s)\n"
        )


if __name__ == "__main__":
    main()
