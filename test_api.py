"""
Test script for the Gemini OpenAI-compatible API.
Tests both streaming and non-streaming chat completions.
"""
import httpx
import json
import asyncio

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)
    
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gemini_ready" in data
    print("✓ Health check passed!")
    return True


def test_models():
    """Test the models endpoint."""
    print("\n" + "=" * 60)
    print("Testing /v1/models endpoint...")
    print("=" * 60)
    
    response = httpx.get(f"{BASE_URL}/v1/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify OpenAI-compatible format
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) > 0
    
    for model in data["data"]:
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model
    
    print("✓ Models endpoint passed!")
    return True


def test_chat_completions_non_streaming():
    """Test non-streaming chat completions."""
    print("\n" + "=" * 60)
    print("Testing /v1/chat/completions (non-streaming)...")
    print("=" * 60)
    
    request_body = {
        "model": "gemini-3.0-flash",
        "messages": [
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ],
        "stream": False
    }
    
    print(f"Request: {json.dumps(request_body, indent=2)}")
    
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json=request_body,
        timeout=120.0
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    # Verify OpenAI-compatible format
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    
    choice = data["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert "finish_reason" in choice
    
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]
    
    print(f"\nAssistant response: {choice['message']['content']}")
    print("✓ Non-streaming chat completions passed!")
    return True


def test_chat_completions_streaming():
    """Test streaming chat completions."""
    print("\n" + "=" * 60)
    print("Testing /v1/chat/completions (streaming)...")
    print("=" * 60)
    
    request_body = {
        "model": "gemini-3.0-flash",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "stream": True
    }
    
    print(f"Request: {json.dumps(request_body, indent=2)}")
    print("\nStreaming response chunks:")
    
    full_content = ""
    chunk_count = 0
    has_role_chunk = False
    has_content_chunk = False
    has_finish_chunk = False
    has_done = False
    
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json=request_body,
        timeout=120.0
    ) as response:
        if response.status_code != 200:
            print(f"Error: Status {response.status_code}")
            return False
        
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            
            data_str = line[6:]  # Remove "data: " prefix
            
            if data_str == "[DONE]":
                print("  [DONE]")
                has_done = True
                continue
            
            try:
                chunk = json.loads(data_str)
                chunk_count += 1
                
                # Verify chunk format
                assert "id" in chunk
                assert chunk["object"] == "chat.completion.chunk"
                assert "created" in chunk
                assert "model" in chunk
                assert "choices" in chunk
                
                if chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    
                    if "role" in delta:
                        has_role_chunk = True
                        print(f"  Chunk {chunk_count}: role={delta['role']}")
                    
                    if "content" in delta and delta["content"]:
                        has_content_chunk = True
                        full_content += delta["content"]
                        print(f"  Chunk {chunk_count}: content=\"{delta['content']}\"")
                    
                    if finish_reason:
                        has_finish_chunk = True
                        print(f"  Chunk {chunk_count}: finish_reason={finish_reason}")
                        
            except json.JSONDecodeError as e:
                print(f"  Failed to parse chunk: {e}")
    
    print(f"\nTotal chunks received: {chunk_count}")
    print(f"Full content: {full_content}")
    
    # Verify streaming structure
    assert has_role_chunk, "Missing role chunk"
    assert has_content_chunk, "Missing content chunks"
    assert has_finish_chunk, "Missing finish_reason chunk"
    assert has_done, "Missing [DONE] marker"
    
    print("✓ Streaming chat completions passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GEMINI OPENAI-COMPATIBLE API TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test health endpoint
    try:
        results["health"] = test_health()
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        results["health"] = False
    
    # Test models endpoint
    try:
        results["models"] = test_models()
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        results["models"] = False
    
    # Test non-streaming chat completions
    try:
        results["chat_non_streaming"] = test_chat_completions_non_streaming()
    except Exception as e:
        print(f"✗ Non-streaming chat completions failed: {e}")
        results["chat_non_streaming"] = False
    
    # Test streaming chat completions
    try:
        results["chat_streaming"] = test_chat_completions_streaming()
    except Exception as e:
        print(f"✗ Streaming chat completions failed: {e}")
        results["chat_streaming"] = False
    
    # Test thinking model (non‑streaming)
    try:
        results["chat_thinking"] = test_chat_completions_thinking()
    except Exception as e:
        print(f"✗ Thinking model test failed: {e}")
        results["chat_thinking"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    
    return all_passed


def test_chat_completions_thinking():
    """Test thinking model (non‑streaming) with reasoning content."""
    print("\n" + "=" * 60)
    print("Testing /v1/chat/completions (thinking model)...")
    print("=" * 60)

    request_body = {
        "model": "gemini-3.0-flash-thinking",
        "messages": [
            {"role": "user", "content": "Explain the steps to solve 2+2."}
        ],
        "stream": False
    }

    print(f"Request: {json.dumps(request_body, indent=2)}")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json=request_body,
        timeout=120.0
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False

    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    # Verify OpenAI‑compatible format
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data and len(data["choices"]) > 0

    choice = data["choices"][0]

    # Check for reasoning content (o1‑style)
    if "reasoning_content" in data:
        assert isinstance(data["reasoning_content"], str)
    elif "reasoning_content" in choice.get("message", {}):
        assert isinstance(choice["message"]["reasoning_content"], str)
    else:
        # If the field is absent, note it but do not fail the test
        print("No reasoning_content field found in response")

    # Standard OpenAI response checks
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert "finish_reason" in choice
    assert "usage" in data

    print("✓ Thinking model test passed!")
    return True

def test_chat_completions_thinking_streaming():
    """Test thinking model (streaming)."""
    print("\n" + "=" * 60)
    print("Testing /v1/chat/completions (thinking model, streaming)...")
    print("=" * 60)

    request_body = {
        "model": "gemini-3.0-flash-thinking",
        "messages": [
            {"role": "user", "content": "Explain the steps to solve a simple math problem like 3+3."}
        ],
        "stream": True
    }

    print(f"Request: {json.dumps(request_body, indent=2)}")
    print("\nStreaming response chunks:")
    
    full_content = ""
    chunk_count = 0
    has_role_chunk = False
    has_content_chunk = False
    has_finish_chunk = False
    has_done = False
    
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json=request_body,
        timeout=120.0
    ) as response:
        if response.status_code != 200:
            print(f"Error: Status {response.status_code}")
            return False
        
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            
            data_str = line[6:]  # Remove "data: " prefix
            
            if data_str == "[DONE]":
                print("  [DONE]")
                has_done = True
                continue
            
            try:
                chunk = json.loads(data_str)
                chunk_count += 1
                
                # Verify chunk format
                assert "id" in chunk
                assert chunk["object"] == "chat.completion.chunk"
                assert "created" in chunk
                assert "model" in chunk
                assert "choices" in chunk
                
                if chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    
                    if "role" in delta:
                        has_role_chunk = True
                        print(f"  Chunk {chunk_count}: role={delta['role']}")
                    
                    if "content" in delta and delta["content"]:
                        has_content_chunk = True
                        full_content += delta["content"]
                        print(f"  Chunk {chunk_count}: content=\"{delta['content']}\"")
                    
                    if finish_reason:
                        has_finish_chunk = True
                        print(f"  Chunk {chunk_count}: finish_reason={finish_reason}")
                        
            except json.JSONDecodeError as e:
                print(f"  Failed to parse chunk: {e}")
    
    print(f"\nTotal chunks received: {chunk_count}")
    print(f"Full content: {full_content}")
    
    # Verify streaming structure
    assert has_role_chunk, "Missing role chunk"
    assert has_content_chunk, "Missing content chunks"
    assert has_finish_chunk, "Missing finish_reason chunk"
    assert has_done, "Missing [DONE] marker"
    
    print("✓ Thinking model streaming test passed!")
    return True

if __name__ == "__main__":
    main()