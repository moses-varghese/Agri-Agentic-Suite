from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    """
    Loads all environment variables into a typed settings object.
    """
    # Project
    ENVIRONMENT: str = "development"
    SECRET_KEY: str

    # Database
    POSTGRES_SERVER: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Prototype 1: GroundTruth AI
    # TWILIO_ACCOUNT_SID: str
    # TWILIO_AUTH_TOKEN: str
    # TWILIO_PHONE_NUMBER: str

    # AI Models
    # OPENAI_API_KEY: str

    DATA_GOV_API_KEY: str

    # Local AI Model Settings
    LOCAL_LLM_MODEL: str
    OLLAMA_API_BASE_URL: str
    EMBEDDING_MODEL_NAME: str
    VISION_MODEL: str

    # env_file = "/app/.env" if Path("/app/.env").exists() else "/run/secrets/.env"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')


@lru_cache()
def get_settings() -> Settings:
    """
    Returns the settings object.
    The lru_cache decorator ensures this function is only run once.
    """
    return Settings()

# Instantiate once to be imported by other modules
settings = get_settings()