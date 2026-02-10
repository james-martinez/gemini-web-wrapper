"""Configuration management for Gemini OpenAI API."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


def get_cookie_from_browser(cookie_name: str) -> Optional[str]:
    """
    Attempt to get a cookie value from the browser using browser-cookie3.
    
    Parameters
    ----------
    cookie_name : str
        The name of the cookie to retrieve (e.g., '__Secure-1PSID')
        
    Returns
    -------
    Optional[str]
        The cookie value if found, None otherwise
    """
    try:
        import browser_cookie3
        
        # Try Chrome first
        try:
            cookies = browser_cookie3.chrome(domain_name='.google.com')
            for cookie in cookies:
                if cookie.name == cookie_name:
                    return cookie.value
        except Exception:
            pass
        
        # Try Firefox
        try:
            cookies = browser_cookie3.firefox(domain_name='.google.com')
            for cookie in cookies:
                if cookie.name == cookie_name:
                    return cookie.value
        except Exception:
            pass
        
        # Try Edge
        try:
            cookies = browser_cookie3.edge(domain_name='.google.com')
            for cookie in cookies:
                if cookie.name == cookie_name:
                    return cookie.value
        except Exception:
            pass
        
        # Try Safari (macOS only)
        try:
            cookies = browser_cookie3.safari(domain_name='.google.com')
            for cookie in cookies:
                if cookie.name == cookie_name:
                    return cookie.value
        except Exception:
            pass
        
    except ImportError:
        print("[Config] browser-cookie3 not installed, skipping browser cookie extraction")
    except Exception as e:
        print(f"[Config] Error getting cookie from browser: {e}")
    
    return None


def get_secure_1psid() -> str:
    """
    Get SECURE_1PSID cookie value.
    First tries browser-cookie3, then falls back to environment variable.
    """
    # Try browser first
    value = get_cookie_from_browser('__Secure-1PSID')
    if value:
        print("[Config] Got SECURE_1PSID from browser cookies")
        return value
    
    # Fallback to environment variable
    env_value = os.getenv("SECURE_1PSID", "")
    if env_value:
        print("[Config] Got SECURE_1PSID from environment variable")
    return env_value


def get_secure_1psidts() -> str:
    """
    Get SECURE_1PSIDTS cookie value.
    First tries browser-cookie3, then falls back to environment variable.
    """
    # Try browser first
    value = get_cookie_from_browser('__Secure-1PSIDTS')
    if value:
        print("[Config] Got SECURE_1PSIDTS from browser cookies")
        return value
    
    # Fallback to environment variable
    env_value = os.getenv("SECURE_1PSIDTS", "")
    if env_value:
        print("[Config] Got SECURE_1PSIDTS from environment variable")
    return env_value


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gemini authentication - use browser cookies with env fallback
    secure_1psid: str = get_secure_1psid()
    secure_1psidts: str = get_secure_1psidts()
    
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
    
    # Debug mode - enables verbose logging of request/response content
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()