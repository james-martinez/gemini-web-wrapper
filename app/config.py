# app/config.py
from typing import Literal

# --- Database ---
# Using aiosqlite for async access
DATABASE_URL = "sqlite+aiosqlite:///./chat_sessions.db" # Relative path

# --- Gemini Settings ---
# Available models in gemini-webapi v1.18.1 (from gemini_webapi.constants.Model):
#
# Gemini 3.0 Models:
#   - gemini-3.0-pro            : Most intelligent model, best for complex tasks (Model.G_3_0_PRO)
#   - gemini-3.0-flash          : Fast and balanced model (Model.G_3_0_FLASH)
#   - gemini-3.0-flash-thinking : Flash model with extended thinking (Model.G_3_0_FLASH_THINKING)
#
# You can also use Model.UNSPECIFIED to let the API choose the default model.
#
# Note: Use the string model name (e.g., "gemini-3.0-pro") or import from
# gemini_webapi.constants.Model for type-safe usage.
#
GEMINI_MODEL_NAME = "gemini-3.0-pro"  # Latest Gemini 3 Pro

# --- Chat Modes ---
# Defines the allowed mode names for validation purposes.
# The actual system prompt text associated with these modes is handled by the ChatService,
# likely by importing from the prompts module.
ALLOWED_MODES = Literal["Default", "Code", "Architect", "Debug", "Ask"]

# --- Roo Code / Kilo Code Compatibility ---
# When this app is used as an OpenAI-compatible API provider for Roo Code (Kilo Code),
# the /v1/chat/completions endpoint automatically detects system messages in the request
# and operates in "pass-through" mode:
#
# Pass-through mode (system messages present in request):
#   - All messages (system + user + assistant) are combined into a single prompt
#   - Roo Code's tool use instructions, mode prompts, and environment details
#     are passed through transparently to Gemini
#   - The hardcoded prompts from app/prompts/prompts.py are NOT used
#
# Legacy mode (no system messages in request - web UI):
#   - Only the last user message is extracted
#   - System prompts from app/prompts/prompts.py are injected during chat activation
#   - Mode switch detection via regex pattern is active
#
# To use with Roo Code, configure it as an OpenAI-compatible provider:
#   Base URL: http://<host>:8022/v1
#   Model: any (the app uses GEMINI_MODEL_NAME regardless)