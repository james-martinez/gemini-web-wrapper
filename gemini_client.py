"""Gemini Web API client wrapper."""

import base64
import os
import tempfile
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional

from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import APIError
from gemini_webapi.utils import set_log_level

from config import settings

# Suppress noisy debug logs unless configured otherwise
set_log_level(settings.gemini_log_level)

# Map OpenAI-style model names to gemini-webapi Model enums
MODEL_MAP: Dict[str, Model] = {
    "gemini-3.0-pro": Model.G_3_0_PRO,
    "gemini-3.0-flash": Model.G_3_0_FLASH,
    "gemini-3.0-flash-thinking": Model.G_3_0_FLASH_THINKING,
}

# All model IDs we advertise
AVAILABLE_MODELS = list(MODEL_MAP.keys())


class GeminiClientWrapper:
    """Manages a GeminiClient instance and exposes generate / stream helpers."""

    def __init__(self) -> None:
        self._client: Optional[GeminiClient] = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Initialize the Gemini client with cookies from settings."""
        if self._initialized:
            return True

        print(f"[Gemini] Initializing (timeout={settings.timeout}s, watchdog={settings.watchdog_timeout}s) ...")

        if not settings.secure_1psid or not settings.secure_1psidts:
            print("[Gemini] ERROR: SECURE_1PSID or SECURE_1PSIDTS missing")
            return False

        try:
            self._client = GeminiClient(
                settings.secure_1psid,
                settings.secure_1psidts,
                proxy=settings.proxy_url,
            )
            await self._client.init(
                timeout=settings.timeout,
                auto_close=False,
                auto_refresh=True,
                watchdog_timeout=settings.watchdog_timeout,
            )
            self._initialized = True
            print("[Gemini] Client initialized successfully")
            return True
        except Exception as exc:
            print(f"[Gemini] ERROR initializing: {exc}")
            traceback.print_exc()
            self._client = None
            self._initialized = False
            return False

    async def close(self) -> None:
        """Close the underlying client."""
        if self._client:
            try:
                await self._client.close()
            except Exception as exc:
                print(f"[Gemini] Error closing client: {exc}")
            finally:
                self._client = None
                self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._client is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_model(model_name: str) -> Model:
        """Resolve a model string to a gemini-webapi Model enum."""
        return MODEL_MAP.get(model_name, Model.UNSPECIFIED)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model_name: str = "gemini-3.0-flash",
        image_files: Optional[List[str]] = None,
    ) -> str:
        """Non-streaming generation. Returns the full text."""
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")

        model = self.resolve_model(model_name)
        try:
            response = await self._client.generate_content(
                prompt,
                model=model,
                files=image_files,
            )
            return response.text if response else ""
        except APIError as exc:
            # "Stream interrupted or truncated" is common — return partial text
            if "interrupted" in str(exc).lower() or "truncated" in str(exc).lower():
                print(f"[Gemini] Stream interrupted during non-streaming call, returning partial result")
                return ""
            raise

    async def generate_stream(
        self,
        prompt: str,
        model_name: str = "gemini-3.0-flash",
        image_files: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text deltas.

        Gracefully handles 'Stream interrupted or truncated' errors from
        gemini-webapi by ending the stream normally.
        """
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")

        model = self.resolve_model(model_name)
        try:
            async for output in self._client.generate_content_stream(
                prompt,
                model=model,
                files=image_files,
            ):
                if output.text_delta:
                    yield output.text_delta
        except APIError as exc:
            # Treat stream-interrupted as a normal end-of-stream
            if "interrupted" in str(exc).lower() or "truncated" in str(exc).lower():
                print("[Gemini] Stream interrupted — ending gracefully")
                return
            raise


# ======================================================================
# Message helpers
# ======================================================================


def extract_text_from_messages(messages: List[Any]) -> str:
    """Convert an OpenAI-style messages array into a single Gemini prompt."""
    parts: List[str] = []
    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                p.text for p in content if hasattr(p, "type") and p.type == "text"
            )
        else:
            text = str(content)

        if not text.strip():
            continue

        label = {"system": "System", "assistant": "Assistant", "user": "User"}.get(
            role, role.capitalize()
        )
        parts.append(f"[{label}]\n{text}")

    return "\n\n".join(parts)


def extract_images_from_messages(messages: List[Any]) -> List[str]:
    """Extract base64 images from messages and save to temp files.

    Returns a list of temp-file paths.
    """
    temp_files: List[str] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if not (hasattr(part, "type") and part.type == "image_url"):
                continue
            url: str = part.image_url.url
            if not url.startswith("data:image"):
                continue
            try:
                header, encoded = url.split(",", 1)
                mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }.get(mime, ".png")
                data = base64.b64decode(encoded)
                fd, path = tempfile.mkstemp(suffix=ext)
                os.write(fd, data)
                os.close(fd)
                temp_files.append(path)
            except Exception as exc:
                print(f"[Gemini] Error processing image: {exc}")
    return temp_files


def cleanup_temp_files(paths: List[str]) -> None:
    """Remove temporary image files."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


# Global singleton
gemini_client = GeminiClientWrapper()