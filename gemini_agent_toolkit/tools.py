import os
import subprocess
import shlex

# Lock access to current working directory
BASE_DIR = os.path.abspath(os.getcwd())


# -------- SECURITY HELPERS -------- #

def is_safe_path(path: str) -> bool:
    """Ensure a path stays inside BASE_DIR."""
    real = os.path.abspath(os.path.join(BASE_DIR, path))
    return real.startswith(BASE_DIR)


def sanitize_path(path: str) -> str:
    """Reject unsafe paths such as ../ or absolute paths."""
    if ".." in path or path.startswith("/") or ":" in path:
        raise ValueError("Access outside the working directory is not allowed.")
    full = os.path.abspath(os.path.join(BASE_DIR, path))
    if not full.startswith(BASE_DIR):
        raise ValueError("Blocked: Path escapes the allowed directory.")
    return full


# -------- FILE FUNCTIONS -------- #

def read_file(file_path: str) -> str:
    """Reads the content of a file inside the allowed directory."""
    try:
        safe_path = sanitize_path(file_path)

        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return f"Error reading file: {e}"


def write_file(file_path: str, content: str) -> str:
    """Writes content to a file inside the allowed directory."""
    try:
        safe_path = sanitize_path(file_path)

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File '{file_path}' written successfully."

    except Exception as e:
        return f"Error writing file: {e}"


# -------- COMMAND EXECUTION -------- #

def execute_command(command: str) -> str:
    """Executes a shell command without allowing directory escape."""
    try:
        # Tokenize safely
        parts = shlex.split(command)

        # Check if any argument tries to use a path
        for p in parts:
            if os.path.sep in p:  # contains "/" or "\"
                if not is_safe_path(p):
                    return f"Command blocked: '{p}' is outside the allowed directory."

        # Prevent directory traversal or absolute commands
        if ".." in command or command.startswith("/") or ":" in parts[0]:
            return "Command blocked: Outside directory access is not allowed."

        # Execute the command in BASE_DIR
        result = subprocess.run(
            command,
            shell=True,
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if stdout and stderr:
            return f"Stdout:\n{stdout}\n\nStderr:\n{stderr}"

        if stdout:
            return f"Stdout:\n{stdout}"

        if stderr:
            return f"Stderr:\n{stderr}"

        return "Command executed successfully with no output."

    except Exception as e:
        return f"Error executing command: {e}"
