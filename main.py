"""
Gemini OpenAI-Compatible API Server

Provides an OpenAI-compatible REST API that proxies requests to Google Gemini.
Compatible with clients like Roo Code, Kilo Code, and Cline.
"""
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    StreamChoice,
    DeltaMessage,
    Usage,
    ModelsResponse,
    ModelInfo,
)
from gemini_client import (
    gemini_client,
    extract_text_from_messages,
    extract_images_from_messages,
    cleanup_temp_files,
    unescape_xml_content,
    clean_markdown_from_code,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("=" * 60)
    print("Starting Gemini OpenAI-Compatible API Server")
    print("=" * 60)
    
    # Initialize Gemini client
    success = await gemini_client.initialize()
    if not success:
        print("WARNING: Gemini client failed to initialize!")
        print("Check your SECURE_1PSID and SECURE_1PSIDTS environment variables")
    
    yield
    
    # Cleanup
    print("Shutting down server...")
    await gemini_client.close()
    print("Server shutdown complete")


app = FastAPI(
    title="Gemini OpenAI-Compatible API",
    description="OpenAI-compatible API that proxies to Google Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all origins (needed for web-based clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if gemini_client.is_ready else "degraded",
        "gemini_ready": gemini_client.is_ready,
    }


# ============================================================================
# Models Endpoint
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(id="gemini-3.0-flash-thinking"),
            ModelInfo(id="gemini-3.0-pro"),
            ModelInfo(id="gemini-3.0-flash"),
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model (OpenAI-compatible)."""
    return ModelInfo(id=model_id)


# ============================================================================
# Chat Completions Endpoint
# ============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming responses.
    """
    print(f"[API] POST /v1/chat/completions (model={request.model}, stream={request.stream})")
    
    if not gemini_client.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Gemini client not initialized. Check server logs."
        )
    
    # Extract prompt and images from messages
    prompt = extract_text_from_messages(request.messages)
    image_files = extract_images_from_messages(request.messages)
    
    print(f"[API] Prompt: {len(prompt)} chars, Images: {len(image_files)}")
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="No content in messages")
    
    try:
        if request.stream:
            return StreamingResponse(
                generate_stream(prompt, image_files, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            return await generate_response(prompt, image_files, request.model)
            
    except Exception as e:
        cleanup_temp_files(image_files)
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(
    prompt: str,
    image_files: list,
    model: str
) -> ChatCompletionResponse:
    """Generate a non-streaming response."""
    try:
        text = await gemini_client.generate_content(prompt, image_files)
        
        # Debug: Log the raw response content
        if settings.debug:
            print("=" * 60)
            print("[DEBUG] Non-streaming RAW response content:")
            print("=" * 60)
            print(text)
            print("=" * 60)
        
        # Unescape XML characters that Gemini escapes
        text = unescape_xml_content(text)
        
        # Clean up Markdown formatting artifacts from code content
        text = clean_markdown_from_code(text)
        
        # Debug: Log the processed response content
        if settings.debug:
            print("[DEBUG] Non-streaming PROCESSED response content:")
            print("=" * 60)
            print(text)
            print("=" * 60)
            print(f"[DEBUG] Response length: {len(text)} chars")
            print("=" * 60)
        
        # Include reasoning_content for thinking models (o1-style)
        reasoning = text if "thinking" in model.lower() else None
        
        return ChatCompletionResponse(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt) // 4,  # Rough estimate
                completion_tokens=len(text) // 4,
                total_tokens=(len(prompt) + len(text)) // 4
            ),
            reasoning_content=reasoning
        )
    finally:
        cleanup_temp_files(image_files)


async def generate_stream(
    prompt: str,
    image_files: list,
    model: str
) -> AsyncGenerator[str, None]:
    """Generate a streaming SSE response."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    
    # Debug: Accumulate full response for logging
    full_response = []
    
    try:
        # Send initial chunk with role
        initial_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None
                )
            ]
        )
        # Use json.dumps with ensure_ascii=False to preserve unicode and special chars
        yield f"data: {json.dumps(initial_chunk.model_dump(), ensure_ascii=False)}\n\n"
        
        # Stream content chunks
        async for text_delta in gemini_client.generate_content_stream(prompt, image_files):
            if text_delta:
                # Debug: Log raw chunk
                if settings.debug:
                    print(f"[DEBUG] Stream RAW chunk ({len(text_delta)} chars): {repr(text_delta[:100])}{'...' if len(text_delta) > 100 else ''}")
                
                # Unescape XML characters that Gemini escapes
                text_delta = unescape_xml_content(text_delta)
                
                # Clean up Markdown formatting artifacts from code content
                text_delta = clean_markdown_from_code(text_delta)
                
                # Debug: Accumulate processed content for logging
                if settings.debug:
                    full_response.append(text_delta)
                    print(f"[DEBUG] Stream PROCESSED chunk ({len(text_delta)} chars): {repr(text_delta[:100])}{'...' if len(text_delta) > 100 else ''}")
                
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=text_delta),
                            finish_reason=None
                        )
                    ]
                )
                # Use json.dumps with ensure_ascii=False to preserve unicode and special chars
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        
        # Debug: Log full accumulated response
        if settings.debug:
            complete_text = "".join(full_response)
            print("=" * 60)
            print("[DEBUG] Streaming complete - Full response content:")
            print("=" * 60)
            print(complete_text)
            print("=" * 60)
            print(f"[DEBUG] Total response length: {len(complete_text)} chars, chunks: {len(full_response)}")
            print("=" * 60)
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ]
        )
        # Use json.dumps with ensure_ascii=False to preserve unicode and special chars
        yield f"data: {json.dumps(final_chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    finally:
        cleanup_temp_files(image_files)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )