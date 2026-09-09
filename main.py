import os
import signal
import sys

from dotenv import load_dotenv

from gemini_agent_toolkit.agent import Agent

_shutdown_requested = False


def _handle_signal(signum: int, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    if signum == signal.SIGINT:
        print("\nInterrupt received. Exiting gracefully...", flush=True)
        sys.exit(130)
    elif signum == signal.SIGTERM:
        print("\nTermination signal received. Exiting gracefully...", flush=True)
        sys.exit(143)


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file or environment variables."
        )

    agent = Agent(api_key)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Gemini Agent started. Enter 'exit' to quit.")
    while True:
        try:
            task = input("> ")
        except EOFError:
            break
        if _shutdown_requested:
            break
        if task.lower() == 'exit':
            break
        try:
            result = agent.run_task(task)
            print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
