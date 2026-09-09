import os

from dotenv import load_dotenv

from gemini_agent_toolkit.agent import Agent


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")

    agent = Agent(api_key)

    print("Gemini Agent started. Enter 'exit' to quit.")
    while True:
        task = input("> ")
        if task.lower() == 'exit':
            break
        try:
            result = agent.run_task(task)
            print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

