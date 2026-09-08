from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_model: str = "gemini-2.5-flash"
    initial_backoff_seconds: int = 5
    max_backoff_seconds: int = 120

settings = Settings()
