import argparse
import os
import signal
import sys

from dotenv import load_dotenv

from gemini_agent_toolkit.agent import Agent, AgentError

_shutdown_requested = False


def _handle_signal(signum: int, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    if signum == signal.SIGINT:
        print("\nInterrupt received. Exiting gracefully...", flush=True)
        sys.exit(130)
    elif signum == signal.SIGTERM:
        print("\nTermination signal received. Exiting gracefully.", flush=True)
        sys.exit(143)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemini-agent",
        description="Agentic workflow with Google Gemini (file ops + shell).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Working directory for tool execution (default: current dir).",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Single task to execute. If omitted, runs interactive REPL.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run a health check against the Gemini API and exit.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    os.chdir(args.directory)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    if args.health_check:
        result = Agent.health_check(api_key)
        print(f"Health: {result}")
        sys.exit(0 if result["healthy"] else 1)

    agent = Agent(api_key)

    if args.task:
        try:
            result = agent.run_task(args.task)
            print(result)
        except AgentError as e:
            print(f"Error [{e.code}]: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {type(e).__name__}")
            sys.exit(1)
        return

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
        except AgentError as e:
            print(f"Error [{e.code}]: {e}")
        except Exception as e:
            print(f"Error: {type(e).__name__}")


if __name__ == "__main__":
    main()
