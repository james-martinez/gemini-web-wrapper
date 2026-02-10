"""OpenAI-compatible request and response models."""
import time
import uuid
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Request Models (OpenAI Chat Completion API compatible)
# ============================================================================

class ContentPartText(BaseModel):
    """Text content part."""
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """Image URL details."""
    url: str
    detail: Optional[str] = "auto"


class ContentPartImage(BaseModel):
    """Image content part."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Union[ContentPartText, ContentPartImage]


class ChatMessage(BaseModel):
    """A single message in the chat."""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "gemini-3.0-flash-thinking"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


# ============================================================================
# Response Models (OpenAI Chat Completion API compatible)
# ============================================================================

class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """Message in a completion choice."""
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)
    # O1‑style reasoning content for thinking models
    reasoning_content: Optional[str] = None


# ============================================================================
# Streaming Response Models
# ============================================================================

class DeltaMessage(BaseModel):
    """Delta message for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """A single streaming choice."""
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# ============================================================================
# Models List Response
# ============================================================================

class ModelInfo(BaseModel):
    """Information about a model."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "google"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: List[ModelInfo]