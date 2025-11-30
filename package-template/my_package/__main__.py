import argparse
import os
from dotenv import load_dotenv
from my_package.agent import Agent

def main():
    load_dotenv() # Load environment variables from .env file

    parser = argparse.ArgumentParser(
        description="Run the AI agent on a specified directory with a given task."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="The path to the project directory for the agent to work in."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task for the AI agent to perform."
    )

    args = parser.parse_args()

    # Change to the specified directory
    try:
        os.chdir(args.directory)
        print(f"Changed current working directory to: {os.getcwd()}")
    except OSError as e:
        print(f"Error: Could not change to directory {args.directory}: {e}")
        return

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables. Please set it in a .env file or your system environment.")
        return

    agent = Agent(api_key=api_key)
    agent.run_task(args.task)

if __name__ == "__main__":
    main()
