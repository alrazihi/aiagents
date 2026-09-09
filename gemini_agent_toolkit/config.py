from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_model: str = "gemini-2.5-flash"
    initial_backoff_seconds: int = 5
    max_backoff_seconds: int = 120
    max_retries: int = 5
    max_turns: int = 20
    max_prompt_length: int = 4000
    command_timeout_seconds: int = 30
    log_level: str = "INFO"


settings = Settings()
