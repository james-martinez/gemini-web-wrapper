"""Configuration management for Gemini OpenAI-Compatible API."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini authentication cookies
    secure_1psid: str = os.getenv("SECURE_1PSID", "")
    secure_1psidts: str = os.getenv("SECURE_1PSIDTS", "")

    # Server settings
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    # Optional proxy
    proxy_url: Optional[str] = os.getenv("PROXY_URL")

    # Gemini client settings  (httpx timeout for gemini-webapi requests)
    timeout: int = 120
    watchdog_timeout: int = 120

    # Log level for gemini-webapi (DEBUG, INFO, WARNING, ERROR)
    gemini_log_level: str = os.getenv("GEMINI_LOG_LEVEL", "WARNING")

    # Debug mode - enables verbose logging
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()