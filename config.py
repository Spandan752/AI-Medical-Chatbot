import os
import logging
import typing

try:
    from pydantic import BaseSettings, field_validator
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    ENVIRONMENT: typing.Literal["development", "production"] = os.getenv(
        "ENVIRONMENT", "development"
    )
    DEBUG: bool = ENVIRONMENT == "development"

    # API Configuration
    API_TITLE: str = "AI Medical Chatbot"
    API_VERSION: str = "0.1.0"
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Model Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", 0.7))

    # Vector Store Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 20))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", 3))

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG")

    # Security & Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

    @field_validator("GOOGLE_API_KEY")
    def validate_google_api_key(cls, v):
        """Validate that Google API key is set in production."""
        if not v and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("GOOGLE_API_KEY must be set in production")
        return v

    @field_validator("PINECONE_API_KEY")
    def validate_pinecone_api_key(cls, v):
        """Validate that Pinecone API key is set in production."""
        if not v and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("PINECONE_API_KEY must be set in production")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


# Load settings
settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(settings.LOG_LEVEL)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger