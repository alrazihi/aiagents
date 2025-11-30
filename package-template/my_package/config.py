from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_model: str = "gemini-2.5-flash"

settings = Settings()
