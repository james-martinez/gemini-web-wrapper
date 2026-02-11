"""Gemini Web API client wrapper."""
import base64
import tempfile
import os
import traceback
import asyncio
from typing import Optional, List, AsyncGenerator, Any

from gemini_webapi import GeminiClient
from gemini_webapi.utils import set_log_level

from config import settings

# Suppress debug logs from gemini_webapi unless configured otherwise
set_log_level(settings.gemini_log_level)


class GeminiClientWrapper:
    """Manages the GeminiClient instance and provides a simple interface."""
    
    def __init__(self):
        self._client: Optional[GeminiClient] = None
        self._initialized: bool = False
    
    async def initialize(self) -> bool:
        """Initialize the Gemini client with credentials from settings."""
        if self._initialized:
            print("[Gemini] Client already initialized")
            return True
        
        print(f"[Gemini] Initializing client (timeout={settings.timeout}s, watchdog={settings.watchdog_timeout}s)...")
        
        if not settings.secure_1psid or not settings.secure_1psidts:
            print("[Gemini] ERROR: Missing SECURE_1PSID or SECURE_1PSIDTS in environment")
            return False
        
        try:
            self._client = GeminiClient(
                settings.secure_1psid,
                settings.secure_1psidts,
                proxy=settings.proxy_url
            )
            
            await self._client.init(
                timeout=settings.timeout,
                auto_close=False,
                auto_refresh=True,
                watchdog_timeout=settings.watchdog_timeout
            )
            
            self._initialized = True
            print("[Gemini] Client initialized successfully")
            return True
            
        except Exception as e:
            print(f"[Gemini] ERROR: Failed to initialize client: {e}")
            traceback.print_exc()
            self._client = None
            self._initialized = False
            return False
    
    async def close(self):
        """Close the Gemini client."""
        if self._client:
            print("[Gemini] Closing client...")
            try:
                await self._client.close()
                print("[Gemini] Client closed")
            except Exception as e:
                print(f"[Gemini] Error closing client: {e}")
            finally:
                self._client = None
                self._initialized = False
    
    @property
    def is_ready(self) -> bool:
        """Check if the client is ready to use."""
        return self._initialized and self._client is not None
    
    async def generate_content(
        self,
        prompt: str,
        image_files: Optional[List[str]] = None
    ) -> str:
        """
        Generate content using Gemini (non-streaming).
        
        Parameters
        ----------
        prompt : str
            The prompt to send to Gemini
        image_files : List[str], optional
            List of file paths to images to include
            
        Returns
        -------
        str
            The generated text response
        """
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")
        
        print(f"[Gemini] Generating content (prompt={len(prompt)} chars, files={len(image_files or [])})")
        
        try:
            response = await self._client.generate_content(
                prompt,
                files=image_files
            )
            
            text = response.text if response else ""
            print(f"[Gemini] Response received ({len(text)} chars)")
            return text
            
        except Exception as e:
            print(f"[Gemini] Error generating content: {e}")
            traceback.print_exc()
            raise
    
    async def generate_content_stream(
        self,
        prompt: str,
        image_files: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate content using Gemini with streaming.
        
        Yields text deltas as they arrive from the API, with fixes for
        spurious newlines that the gemini_webapi library introduces.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to Gemini
        image_files : List[str], optional
            List of file paths to images to include
            
        Yields
        ------
        str
            Text deltas as they arrive
        """
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")
        
        print(f"[Gemini] Starting streaming generation (prompt={len(prompt)} chars, files={len(image_files or [])})")
        
        try:
            async for output in self._client.generate_content_stream(
                prompt,
                files=image_files
            ):
                # Extract text delta from the output
                if hasattr(output, 'candidates') and output.candidates:
                    candidate = output.candidates[0]
                    if hasattr(candidate, 'text_delta') and candidate.text_delta:
                        yield candidate.text_delta
            
            print("[Gemini] Streaming completed")
            
        except Exception as e:
            print(f"[Gemini] Error in streaming generation: {e}")
            traceback.print_exc()
            raise


def extract_text_from_messages(messages: List[Any]) -> str:
    """
    Combine all messages into a single prompt string.
    
    This converts the OpenAI-style messages array into a format
    suitable for Gemini's single-prompt interface.
    """
    prompt_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # Extract text from content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Handle content parts (text and images)
            text_parts = []
            for part in content:
                if hasattr(part, 'type') and part.type == "text":
                    text_parts.append(part.text)
            text = "\n".join(text_parts)
        else:
            text = str(content)
        
        if not text.strip():
            continue
        
        # Format based on role
        if role == "system":
            prompt_parts.append(f"[System Instructions]\n{text}")
        elif role == "assistant":
            prompt_parts.append(f"[Assistant]\n{text}")
        elif role == "user":
            prompt_parts.append(f"[User]\n{text}")
    
    return "\n\n".join(prompt_parts)


def extract_images_from_messages(messages: List[Any]) -> List[str]:
    """
    Extract base64 images from messages and save them to temp files.
    
    Returns a list of file paths to the saved images.
    """
    temp_files = []
    
    for msg in messages:
        content = msg.content
        
        if not isinstance(content, list):
            continue
        
        for part in content:
            if not hasattr(part, 'type') or part.type != "image_url":
                continue
            
            image_url = part.image_url.url
            
            # Only handle base64 data URIs
            if not image_url.startswith("data:image"):
                continue
            
            try:
                # Parse data URI: data:image/png;base64,<data>
                header, encoded = image_url.split(",", 1)
                
                # Determine file extension
                mime_type = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
                ext_map = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }
                ext = ext_map.get(mime_type, ".png")
                
                # Decode and save to temp file
                img_data = base64.b64decode(encoded)
                fd, temp_path = tempfile.mkstemp(suffix=ext)
                os.write(fd, img_data)
                os.close(fd)
                
                temp_files.append(temp_path)
                print(f"[Gemini] Saved image ({mime_type}) to {temp_path}")
                
            except Exception as e:
                print(f"[Gemini] Error processing image: {e}")
    
    return temp_files


def cleanup_temp_files(file_paths: List[str]):
    """Remove temporary files."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[Gemini] Error removing temp file {path}: {e}")


# Global client instance
gemini_client = GeminiClientWrapper()