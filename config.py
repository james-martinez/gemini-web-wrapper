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
    timeout: int = int(os.getenv("GEMINI_TIMEOUT", "60"))
    watchdog_timeout: int = int(os.getenv("GEMINI_WATCHDOG_TIMEOUT", "30"))

    # Outer timeout for the entire non-streaming generation call (seconds).
    # This caps the total wall-clock time including all library-internal retries.
    generate_timeout: int = int(os.getenv("GENERATE_TIMEOUT", "120"))

    # How many seconds to wait for the next streaming chunk before assuming
    # the stream is complete.  The gemini-webapi library often hangs for
    # 60-90s after the last real content before raising "Stream interrupted".
    stream_stall_timeout: int = int(os.getenv("STREAM_STALL_TIMEOUT", "10"))

    # Log level for gemini-webapi (DEBUG, INFO, WARNING, ERROR)
    gemini_log_level: str = os.getenv("GEMINI_LOG_LEVEL", "WARNING")

    # Debug mode - enables verbose logging
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()