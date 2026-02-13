"""Gemini Web API client wrapper."""

import asyncio
import base64
import os
import re
import tempfile
import time as _time
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
        """Non-streaming generation that collects streaming chunks.

        Uses generate_stream internally to avoid the library's non-streaming
        path which can hang for minutes due to internal retry logic on
        'Stream interrupted' errors.  An outer asyncio timeout caps total
        wall-clock time.
        """
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")

        timeout = settings.generate_timeout
        print(
            f"[Gemini] generate() called: model={model_name}  "
            f"prompt_len={len(prompt)}  images={len(image_files) if image_files else 0}  "
            f"timeout={timeout}s"
        )

        _start = _time.monotonic()
        collected: list[str] = []

        try:
            async def _collect():
                async for delta in self.generate_stream(
                    prompt, model_name=model_name, image_files=image_files
                ):
                    collected.append(delta)

            await asyncio.wait_for(_collect(), timeout=timeout)

        except asyncio.TimeoutError:
            _elapsed = _time.monotonic() - _start
            partial = "".join(collected)
            print(
                f"[Gemini] generate() TIMEOUT after {_elapsed:.1f}s  "
                f"(collected {len(partial)} chars so far)"
            )
            if partial.strip():
                print("[Gemini] Returning partial result from timeout")
                return partial
            raise RuntimeError(
                f"Gemini request timed out after {timeout}s with no content. "
                "Try increasing GENERATE_TIMEOUT or using streaming mode."
            )

        _elapsed = _time.monotonic() - _start
        result = "".join(collected)
        print(f"[Gemini] generate() complete in {_elapsed:.1f}s  ({len(result)} chars)")
        if settings.debug and result:
            preview = result[:300] + ("..." if len(result) > 300 else "")
            print(f"[Gemini] Response preview: {preview}")

        if not result.strip():
            raise RuntimeError(
                "Gemini returned an empty response. The model may be temporarily "
                "unavailable — try again or use a different model."
            )

        return unescape_gemini_markdown(result)

    async def generate_stream(
        self,
        prompt: str,
        model_name: str = "gemini-3.0-flash",
        image_files: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation using ChatSession.send_message_stream().

        Creates a new chat session per request and streams the response through it.
        Tracks cumulative text to deduplicate content across library-internal
        retries (the library's @running decorator can re-invoke _generate on
        'Stream interrupted' errors, which re-sends earlier content as deltas).
        """
        if not self.is_ready:
            raise RuntimeError("Gemini client not initialized")

        model = self.resolve_model(model_name)
        print(
            f"[Gemini] generate_stream() called: model={model_name} resolved={model}  "
            f"prompt_len={len(prompt)}  images={len(image_files) if image_files else 0}"
        )

        _start = _time.monotonic()
        _chunk_count = 0
        _total_chars = 0
        # Track full text built so far to deduplicate across retries.
        # The library may replay earlier content when it retries internally.
        _full_text = ""
        _unescaped_so_far = ""

        # Stall timeout: if no new content arrives within this many seconds,
        # assume the stream is done.  The gemini-webapi library often hangs
        # for 60-90s after the last real chunk before raising "Stream
        # interrupted", so this lets us finish much faster.
        STALL_TIMEOUT = settings.stream_stall_timeout

        try:
            print(f"[Gemini] Creating streaming chat session...")
            chat = self._client.start_chat(model=model)
            print(f"[Gemini] Starting stream ({len(prompt)} chars)...")

            stream_iter = chat.send_message_stream(
                prompt,
                files=image_files,
            ).__aiter__()

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(), timeout=STALL_TIMEOUT
                    )
                except StopAsyncIteration:
                    # Normal end of stream
                    break
                except asyncio.TimeoutError:
                    _elapsed = _time.monotonic() - _start
                    print(
                        f"[Gemini] Stream stalled for {STALL_TIMEOUT}s after "
                        f"{_chunk_count} chunks ({_total_chars} chars) — "
                        f"finishing at {_elapsed:.1f}s"
                    )
                    break

                if not chunk.text:
                    continue

                # The chunk.text is the FULL accumulated text so far from the
                # library.  chunk.text_delta is the new portion.  However, on
                # retries the library may replay content.  We use the full text
                # to detect overlap and only yield truly new characters.
                full_now = chunk.text
                if len(full_now) > len(_full_text):
                    # New content is the part beyond what we already sent.
                    # Apply unescape to the full accumulated text and yield
                    # only the truly new portion after unescaping.
                    _full_text = full_now
                    unescaped_full = unescape_gemini_markdown(_full_text)
                    new_part = unescaped_full[len(_unescaped_so_far):]
                    _unescaped_so_far = unescaped_full
                    _chunk_count += 1
                    _total_chars += len(new_part)
                    if _chunk_count == 1:
                        _first_chunk = _time.monotonic() - _start
                        print(f"[Gemini] First chunk received in {_first_chunk:.1f}s")
                    if new_part:
                        yield new_part
                # else: duplicate/overlapping content from a retry — skip

            _elapsed = _time.monotonic() - _start
            print(
                f"[Gemini] Stream complete in {_elapsed:.1f}s  "
                f"chunks={_chunk_count}  total_chars={_total_chars}"
            )
        except APIError as exc:
            _elapsed = _time.monotonic() - _start
            if "interrupted" in str(exc).lower() or "truncated" in str(exc).lower():
                print(
                    f"[Gemini] Stream interrupted after {_elapsed:.1f}s "
                    f"({_chunk_count} chunks, {_total_chars} chars) — ending gracefully"
                )
                return
            print(f"[Gemini] APIError during stream: {exc}")
            raise
        except Exception as exc:
            _elapsed = _time.monotonic() - _start
            print(
                f"[Gemini] Unexpected error in generate_stream() after {_elapsed:.1f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
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


# ======================================================================
# Markdown / escaping post-processor
# ======================================================================


def unescape_gemini_markdown(text: str) -> str:
    """Remove Gemini's spurious markdown escaping from a response.

    Gemini's web interface returns text with markdown formatting that
    breaks XML tool blocks used by coding agents (Roo Code, Cline,
    Kilo Code).  Specifically it:

      - Backslash-escapes special chars:  \\< \\> \\_ \\# \\! \\[ \\]
      - Sometimes inserts markdown code fences inside code content
      - Spaces out consecutive > characters (e.g. ``> > >`` instead of ``>>>``)

    This function reverses those transformations so the XML arrives
    clean and parseable.

    Rules:
      1. ``\\<``  → ``<``   (but ``\\\\`` i.e. real double-backslash is preserved)
      2. ``\\>``  → ``>``
      3. ``\\_``  → ``_``
      4. ``\\#``  → ``#``
      5. ``\\!``  → ``!``
      6. ``\\[``  → ``[``
      7. ``\\]``  → ``]``
      8. ``\\(``  → ``(``
      9. ``\\)``  → ``)``
     10. Runs of ``> > > ...`` collapsed to ``>>>...`` (for REPLACE markers)
     11. Stray markdown code fences removed (``` ``` ```)
    """
    if not text:
        return text

    # Step 1: Remove backslash escapes of markdown-special characters.
    # We must NOT touch \\n, \\t, \\\\, or other real escape sequences.
    # Gemini escapes:  < > _ # ! [ ] ( )
    # Pattern: a single backslash followed by one of these chars,
    # but NOT preceded by another backslash (to preserve \\).
    result = re.sub(r'(?<!\\)\\([<>_#!\[\]\(\)])', r'\1', text)

    # Step 2: Collapse spaced-out angle brackets for SEARCH/REPLACE markers.
    # Gemini turns  <<<<<<< SEARCH  into  < < < < < < < SEARCH  or
    #               >>>>>>> REPLACE into  > > > > > > > REPLACE
    # We collapse runs of ``< `` or ``> `` back to ``<<...`` / ``>>...``
    def collapse_angles(m: re.Match) -> str:
        chars = m.group(0).replace(' ', '')
        return chars
    result = re.sub(r'(?:<\s){2,}<', collapse_angles, result)
    result = re.sub(r'(?:>\s){2,}>', collapse_angles, result)

    # Step 3: Remove stray markdown code fences that appear inside code content.
    # These are triple-backtick lines optionally followed by a language tag.
    # Only remove them when they appear on their own line.
    result = re.sub(r'^```[a-zA-Z]*\s*$', '', result, flags=re.MULTILINE)

    return result


# Global singleton
gemini_client = GeminiClientWrapper()