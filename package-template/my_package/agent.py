import time
import google.api_core.exceptions
import google.generativeai as genai
from my_package import tools
from my_package.config import settings


class Agent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            settings.gemini_model,
            tools=[
                tools.read_file,
                tools.write_file,
                tools.execute_command,
            ]
        )

        self.chat = self.model.start_chat(history=[])

    # ======================================================
    # SAFE MESSAGE SENDER WITH AUTO RETRY + BACKOFF
    # ======================================================
    def safe_send(self, payload):
        backoff = settings.initial_backoff_seconds  # Use configurable value
        max_backoff = settings.max_backoff_seconds  # Use configurable value

        while True:
            try:
                return self.chat.send_message(payload)

            except google.api_core.exceptions.ResourceExhausted as e:
                retry_delay = None

                # Try extracting API-provided retryDelay
                if hasattr(e, "errors") and e.errors:
                    err = e.errors[0]
                    if "retryDelay" in err:
                        retry_delay = int(err["retryDelay"].get("seconds", 0))

                wait_time = retry_delay if retry_delay else backoff

                print(f"\n⚠️ Quota exceeded — retrying in {wait_time}s...\n")
                time.sleep(wait_time)

                # Exponential backoff for next round
                backoff = min(int(backoff * 1.7), max_backoff)

            except Exception as e:
                print(f"Unexpected error, retrying in 10s: {e}")
                time.sleep(10)

    # ======================================================
    # RUN TASK
    # ======================================================
    def run_task(self, task: str):
        print(f"Executing task: {task}")

        # FIRST MESSAGE — SAFE VERSION
        response = self.safe_send(task)

        while True:
            candidate = response.candidates[0]
            has_tool_call = False
            tool_responses = []

            # Detect & execute tool calls
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_tool_call = True
                    tool_call = part.function_call
                    print(f"Executing tool: {tool_call.name}({tool_call.args})")

                    result = self._execute_tool(tool_call)

                    tool_responses.append({
                        "function_response": {
                            "name": tool_call.name,
                            "response": {"output": result}
                        }
                    })

            if has_tool_call:
                # Send tool response WITH SAFE RETRY
                response = self.safe_send(tool_responses)
                continue

            # Finished!
            print("Task finished.\n")

            # Print final text from model
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text)

            break

    # ======================================================
    # TOOL EXECUTION
    # ======================================================
    def _execute_tool(self, tool_call) -> str:
        name = tool_call.name
        args = dict(tool_call.args)

        try:
            if name == "read_file":
                return tools.read_file(**args)

            elif name == "write_file":
                return tools.write_file(**args)

            elif name == "execute_command":
                return tools.execute_command(**args)

            else:
                print(f"⚠️ Warning: Unknown tool called: {name}")
                return f"Error: Unknown tool: {name}"

        except Exception as e:
            print(f"❌ Error executing tool '{name}' with args {args}: {e}")
            return f"Error executing tool '{name}': {e}"