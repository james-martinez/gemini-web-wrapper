"""Configuration management for Gemini OpenAI API."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gemini authentication
    secure_1psid: str = os.getenv("SECURE_1PSID", "")
    secure_1psidts: str = os.getenv("SECURE_1PSIDTS", "")
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Optional proxy
    proxy_url: Optional[str] = os.getenv("PROXY_URL")
    
    # Gemini client settings
    timeout: int = 600  # Request timeout in seconds (increased for long responses)
    watchdog_timeout: int = 300  # Watchdog timeout for thinking models (increased)
    
    # Model name to report in OpenAI-compatible responses
    model_name: str = "gemini-3.0-flash-thinking"
    
    # Log level for gemini-webapi (DEBUG, INFO, WARNING, ERROR)
    gemini_log_level: str = os.getenv("GEMINI_LOG_LEVEL", "WARNING")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()