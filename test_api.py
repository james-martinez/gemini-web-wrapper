"""
Test script for the Gemini OpenAI-compatible API.
Run the server first with: python main.py
Then run this script: python test_api.py
"""

import json
import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    print("=" * 60)
    print("GET /health")
    print("=" * 60)
    r = httpx.get(f"{BASE_URL}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Body:   {r.json()}")
    assert r.status_code == 200
    assert "gemini_ready" in r.json()
    print("  PASSED")


def test_models():
    print("\n" + "=" * 60)
    print("GET /v1/models")
    print("=" * 60)
    r = httpx.get(f"{BASE_URL}/v1/models")
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Models: {[m['id'] for m in data['data']]}")
    assert r.status_code == 200
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    print("  PASSED")


def test_chat_non_streaming():
    print("\n" + "=" * 60)
    print("POST /v1/chat/completions  (non-streaming)")
    print("=" * 60)
    body = {
        "model": "gemini-3.0-flash",
        "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
        "stream": False,
    }
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=120)
    print(f"  Status: {r.status_code}")
    if r.status_code != 200:
        print(f"  Error: {r.text}")
        return
    data = r.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    print(f"  Reply:  {content}")
    print("  PASSED")


def test_chat_streaming():
    print("\n" + "=" * 60)
    print("POST /v1/chat/completions  (streaming)")
    print("=" * 60)
    body = {
        "model": "gemini-3.0-flash",
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
    }
    full = ""
    with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=body, timeout=120) as r:
        if r.status_code != 200:
            print(f"  Error: status {r.status_code}")
            return
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                print("  [DONE]")
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                full += delta["content"]
    print(f"  Full reply: {full}")
    print("  PASSED")


if __name__ == "__main__":
    print("\nGemini OpenAI-Compatible API — Test Suite\n")
    for fn in [test_health, test_models, test_chat_non_streaming, test_chat_streaming]:
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {exc}")
    print("\nDone.")