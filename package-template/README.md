# Python Package Template

This sample repo contains the recommended structure for a Python package.

## Setup Instructions 

This sample makes use of Dev Containers, in order to leverage this setup, make sure you have [Docker installed](https://www.docker.com/products/docker-desktop).

The code in this repo aims to follow Python style guidelines as outlined in [PEP 8](https://peps.python.org/pep-0008/).

To successfully run this example, we recommend the following VS Code extensions:

- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) 

In addition to these extension there a few settings that are also useful to enable. You can enable to following settings by opening the Settings editor (`Ctrl+,`) and searching for the following settings:

- Python > Analysis > **Type Checking Mode** : `basic`
- Python > Analysis > Inlay Hints: **Function Return Types** : `enable`
- Python > Analysis > Inlay Hints: **Variable Types** : `enable`

## Agent Overview

This package includes a `my_package.agent.Agent` class that leverages the Gemini API to perform tasks, interacting with the file system and executing shell commands through defined tools. It features robust error handling and an automatic retry mechanism with exponential backoff for API calls.

## API Key Setup

To use the agent, you need to provide your Google Gemini API key. It's recommended to set this as an environment variable, for example, in a `.env` file at the root of your project:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Then, ensure your application can access this environment variable.

## Configuration

The agent's behavior can be configured via `my_package/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_model: str = "gemini-2.5-flash"
    initial_backoff_seconds: int = 5
    max_backoff_seconds: int = 120

settings = Settings()
```

*   `gemini_model`: Specifies the Gemini model to use (e.g., "gemini-2.5-flash", "gemini-pro").
*   `initial_backoff_seconds`: The initial delay in seconds before retrying an API call after a `ResourceExhausted` error.
*   `max_backoff_seconds`: The maximum delay in seconds for exponential backoff during API retries.

## Running the Sample

- Open the template folder in VS Code (**File** > **Open Folder...**)
- Open the Command Palette in VS Code (**View > Command Palette...**) and run the **Dev Container: Reopen in Container** command
- Run the app using the Run and Debug view

**Example Usage (e.g., in `main.py` or a script):**

```python
import os
from my_package.agent import Agent

# Ensure your GEMINI_API_KEY is set in environment variables or .env file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
else:
    agent = Agent(api_key=api_key)
    # Example task for the agent
    agent.run_task("List all files in the current directory and save the output to 'file_list.txt'.")
```

- To test, navigate to the Test Panel to configure your Python test or by triggering the **Python: Configure Tests** command from the Command Palette
- Run tests in the Test Panel or by clicking the Play Button next to the individual tests in the `test_date_time.py` and `test_developer.py` file
