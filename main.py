"""
Gemini OpenAI-Compatible API Server

Provides an OpenAI-compatible REST API that proxies requests to Google Gemini
via gemini-webapi.  Compatible with Roo Code, Kilo Code, Cline, and any other
client that speaks the OpenAI Chat Completions protocol.
"""

import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from gemini_client import (
    AVAILABLE_MODELS,
    cleanup_temp_files,
    extract_images_from_messages,
    extract_text_from_messages,
    gemini_client,
    unescape_gemini_markdown,
)
from models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelsResponse,
    StreamChoice,
    Usage,
)


# ============================================================================
# Application lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook."""
    print("=" * 60)
    print("  Gemini OpenAI-Compatible API Server")
    print("=" * 60)
    ok = await gemini_client.initialize()
    if not ok:
        print("WARNING: Gemini client failed to initialize!")
        print("  Check SECURE_1PSID / SECURE_1PSIDTS in your .env")
    yield
    print("Shutting down ...")
    await gemini_client.close()


app = FastAPI(
    title="Gemini OpenAI-Compatible API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health
# ============================================================================


@app.get("/health")
async def health():
    return {
        "status": "healthy" if gemini_client.is_ready else "degraded",
        "gemini_ready": gemini_client.is_ready,
    }


# ============================================================================
# Models
# ============================================================================


@app.get("/v1/models")
async def list_models():
    return ModelsResponse(data=[ModelInfo(id=mid) for mid in AVAILABLE_MODELS])


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return ModelInfo(id=model_id)


# ============================================================================
# Chat Completions
# ============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions (streaming and non-streaming)."""
    if not gemini_client.is_ready:
        raise HTTPException(503, "Gemini client not initialized. Check server logs.")

    prompt = extract_text_from_messages(request.messages)
    images = extract_images_from_messages(request.messages)

    # Always log incoming requests for diagnostics
    print(
        f"[API] >>> REQUEST: model={request.model}  stream={request.stream}"
        f"  messages={len(request.messages)}  prompt_len={len(prompt)}  images={len(images)}"
    )
    if settings.debug:
        # Show truncated prompt in debug mode
        preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
        print(f"[API] >>> PROMPT PREVIEW:\n{preview}")

    if not prompt.strip():
        raise HTTPException(400, "No content in messages")

    try:
        if request.stream:
            print(f"[API] Starting streaming response...")
            return StreamingResponse(
                _stream(prompt, images, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        print(f"[API] Starting non-streaming response...")
        result = await _complete(prompt, images, request.model)
        print(f"[API] <<< RESPONSE: status=200  model={result.model}  choices={len(result.choices)}  usage={result.usage.total_tokens} tokens")
        if settings.debug and result.choices:
            resp_preview = result.choices[0].message.content[:500] + ("..." if len(result.choices[0].message.content) > 500 else "")
            print(f"[API] <<< RESPONSE PREVIEW:\n{resp_preview}")
        return result
    except Exception as exc:
        cleanup_temp_files(images)
        print(f"[API] <<< ERROR: {type(exc).__name__}: {exc}")
        if settings.debug:
            traceback.print_exc()
        raise HTTPException(500, str(exc))


# ------------------------------------------------------------------
# Non-streaming helper
# ------------------------------------------------------------------


async def _complete(prompt: str, images: list, model: str) -> ChatCompletionResponse:
    """Generate a full (non-streaming) response."""
    try:
        text = await gemini_client.generate(
            prompt, model_name=model, image_files=images
        )

        return ChatCompletionResponse(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(text) // 4,
                total_tokens=(len(prompt) + len(text)) // 4,
            ),
        )
    finally:
        cleanup_temp_files(images)


# ------------------------------------------------------------------
# Streaming helper
# ------------------------------------------------------------------


async def _stream(
    prompt: str, images: list, model: str
) -> AsyncGenerator[str, None]:
    """Generate an SSE stream of chat completion chunks."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        # 1 --- role chunk
        yield _sse(
            ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                ],
            ).model_dump()
        )

        # 2 --- content deltas
        async for delta in gemini_client.generate_stream(
            prompt, model_name=model, image_files=images
        ):
            if delta:
                yield _sse(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(content=delta),
                                finish_reason=None,
                            )
                        ],
                    ).model_dump()
                )

        # 3 --- finish chunk
        yield _sse(
            ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            ).model_dump()
        )

        # 4 --- [DONE] sentinel
        yield "data: [DONE]\n\n"

    finally:
        cleanup_temp_files(images)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )