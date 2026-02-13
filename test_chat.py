"""
Simple standalone test: send a chat message to the OpenAI-compatible API
and print the reply.  Tests both non-streaming and streaming modes with
proper timeouts so nothing hangs.
"""

import json
import httpx
import sys

BASE = "http://localhost:8000"
TIMEOUT = 120  # seconds


def chat_non_streaming(prompt: str) -> str:
    """Send a non-streaming chat request and return the reply text."""
    print(f"\n{'='*60}")
    print(f"NON-STREAMING: {prompt!r}")
    print(f"{'='*60}")

    body = {
        "model": "gemini-3.0-flash",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    r = httpx.post(f"{BASE}/v1/chat/completions", json=body, timeout=TIMEOUT)
    print(f"  Status: {r.status_code}")

    if r.status_code != 200:
        print(f"  Error: {r.text}")
        return ""

    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    print(f"  Reply: {reply}")
    print(f"  Tokens: {data.get('usage', {})}")
    return reply


def chat_streaming(prompt: str) -> str:
    """Send a streaming chat request and collect the full reply."""
    print(f"\n{'='*60}")
    print(f"STREAMING: {prompt!r}")
    print(f"{'='*60}")

    body = {
        "model": "gemini-3.0-flash",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    full = ""
    # Use a read timeout so we don't hang forever if the stream stalls
    timeout = httpx.Timeout(connect=10, read=30, write=10, pool=10)

    try:
        with httpx.stream(
            "POST",
            f"{BASE}/v1/chat/completions",
            json=body,
            timeout=timeout,
        ) as r:
            print(f"  Status: {r.status_code}")
            if r.status_code != 200:
                print(f"  Error: {r.text}")
                return ""

            for line in r.iter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                payload = line[6:]
                if payload == "[DONE]":
                    print("\n  [DONE]")
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    print(f"  [bad json: {payload!r}]")
                    continue

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full += content
                    # Print each chunk inline
                    sys.stdout.write(content)
                    sys.stdout.flush()

    except httpx.ReadTimeout:
        print(f"\n  [READ TIMEOUT — stream stalled, returning what we have]")
    except httpx.TimeoutException as exc:
        print(f"\n  [TIMEOUT: {exc}]")

    print(f"\n  Full reply ({len(full)} chars): {full}")
    return full


if __name__ == "__main__":
    # Quick health check first
    try:
        h = httpx.get(f"{BASE}/health", timeout=5)
        print(f"Health: {h.json()}")
    except Exception as e:
        print(f"Server not reachable at {BASE}: {e}")
        print("Start the server first:  python main.py")
        sys.exit(1)

    # Test non-streaming
    chat_non_streaming("Say hello in exactly 5 words.")

    # Test streaming
    chat_streaming("Count from 1 to 5.")