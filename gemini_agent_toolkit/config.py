from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_model: str = "gemini-2.5-flash"
    initial_backoff_seconds: int = 5
    max_backoff_seconds: int = 120
    max_retries: int = 5
    max_turns: int = 20
    max_prompt_length: int = 4000
    max_response_length: int = 8000
    command_timeout_seconds: int = 30
    log_level: str = "INFO"

    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    max_tool_iterations: int = 10

    allowed_commands: str = (
        "ls,cat,desc,dir,echo,find,where,grep,rg,head,tail,"
        "sort,uniq,wc,tree,diff,jq,xargs,sleep"
    )


settings = Settings()


def parse_allowed_commands(raw: str) -> frozenset[str]:
    """Parse a comma-separated string of allowed command names."""
    return frozenset(
        cmd.strip() for cmd in raw.split(",") if cmd.strip()
    )
