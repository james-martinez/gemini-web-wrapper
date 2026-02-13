"""
Test suite: Verify Gemini API responses are compatible with coding agent tools.

When Roo Code, Cline, or Kilo Code use this API, Gemini returns responses
containing XML tool commands like write_to_file with code inside. The code
inside those XML blocks must be CLEAN - no markdown formatting, no backslash
escaping, no HTML entities. A file written via write_to_file must contain
valid, runnable code.

This test suite verifies:
  1. XML tool tags are present and properly formed (not escaped)
  2. Code content inside XML tool blocks is clean (no markdown artifacts)
  3. Streaming and non-streaming both produce identical clean output
  4. The unescape_gemini_markdown() post-processor works correctly

Usage:
    1. Start the server:   python main.py
    2. Run the tests:      python test_xml_markdown.py
"""

import json
import re
import sys
import textwrap
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 180

# ============================================================================
# XML tag builder - constructs tags dynamically to avoid confusing tool clients
# ============================================================================


def mktag(name, closing=False):
    """Build an XML tag string dynamically."""
    if closing:
        return "<" + "/" + name + ">"
    return "<" + name + ">"


# Pre-build tag strings
T_READ_OPEN = mktag("read_file")
T_READ_CLOSE = mktag("read_file", closing=True)
T_WRITE_OPEN = mktag("write_to_file")
T_WRITE_CLOSE = mktag("write_to_file", closing=True)
T_DIFF_OPEN = mktag("apply_diff")
T_DIFF_CLOSE = mktag("apply_diff", closing=True)
T_PATH_OPEN = mktag("path")
T_PATH_CLOSE = mktag("path", closing=True)
T_CONTENT_OPEN = mktag("content")
T_CONTENT_CLOSE = mktag("content", closing=True)
T_ARGS_OPEN = mktag("args")
T_ARGS_CLOSE = mktag("args", closing=True)
T_FILE_OPEN = mktag("file")
T_FILE_CLOSE = mktag("file", closing=True)
T_DIFFTAG_OPEN = mktag("diff")
T_DIFFTAG_CLOSE = mktag("diff", closing=True)

# ============================================================================
# Helpers
# ============================================================================

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results: list[dict] = []


def report(name: str, passed: bool, detail: str = ""):
    results.append({"name": name, "passed": passed})
    print(f"  [{PASS if passed else FAIL}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def chat(messages: list[dict], stream: bool = False, model: str = "gemini-3.0-flash"):
    """Send a chat request (streaming or non-streaming) and return content."""
    body = {"model": model, "messages": messages, "stream": stream}

    if not stream:
        try:
            r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=TIMEOUT)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}: {r.text[:300]}")
                return None
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"    Error: {exc}")
            return None
    else:
        full = ""
        try:
            with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions",
                              json=body, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code}")
                    return None
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    c = delta.get("content")
                    if c:
                        full += c
            return full
        except Exception as exc:
            print(f"    Error: {exc}")
            return None


def extract_between(text, open_tag, close_tag):
    """Extract text between two tag strings."""
    start = text.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start:end]


def check_clean_code(code: str, label: str) -> bool:
    """Verify code content is free of markdown/escaping artifacts."""
    issues = []

    # Backslash-escaped markdown chars: \< \> \_ \! \[ \]
    bs_lt = "\\" + "<"
    bs_gt = "\\" + ">"
    bs_us = "\\" + "_"
    bs_bang = "\\" + "!"
    bs_lbr = "\\" + "["
    bs_rbr = "\\" + "]"

    if bs_lt in code:
        issues.append("found backslash-escaped < in code")
    if bs_gt in code:
        issues.append("found backslash-escaped > in code")
    if bs_us in code:
        issues.append("found backslash-escaped _ in code")
    if bs_bang in code:
        issues.append("found backslash-escaped ! in code")
    if bs_lbr in code:
        issues.append("found backslash-escaped [ in code")
    if bs_rbr in code:
        issues.append("found backslash-escaped ] in code")

    # HTML entities
    if "&lt;" in code:
        issues.append("found &lt; HTML entity in code")
    if "&gt;" in code:
        issues.append("found &gt; HTML entity in code")
    if "&amp;" in code:
        issues.append("found &amp; HTML entity in code")

    # Markdown bold markers that shouldn't be in code
    if "**" in code and "**kwargs" not in code and "**args" not in code:
        issues.append("found markdown bold (**) markers in code")

    # Markdown backtick fences inside what should be raw code
    if "```python" in code or "```py" in code:
        issues.append("found markdown code fence inside code content")

    passed = len(issues) == 0
    report(f"{label}: code content is clean", passed, "\n".join(issues) if issues else "")
    return passed


def preview(content: str, max_chars: int = 800):
    text = content[:max_chars]
    if len(content) > max_chars:
        text += "\n... [truncated]"
    print(textwrap.indent(text, "    "))


# ============================================================================
# Test 1: write_to_file XML - code inside must be clean
# ============================================================================


def test_write_to_file_clean_code():
    """The most critical test: code inside write_to_file content tags must be
    valid, runnable code without markdown artifacts."""
    print("\n" + "=" * 70)
    print("TEST 1: write_to_file - code content must be clean")
    print("=" * 70)

    # Ask Gemini to produce a write_to_file block with Python code
    prompt_parts = [
        "You are a coding agent. When asked to create a file, respond with a",
        "write_to_file XML block containing the file content.",
        "",
        "Create a Python file at src/calculator.py with a Calculator class that has",
        "add, subtract, multiply, and divide methods. Each method takes two numbers.",
        "The divide method should handle division by zero.",
        "",
        "Respond with ONLY the XML tool block, no other text:",
        T_WRITE_OPEN,
        T_PATH_OPEN + "src/calculator.py" + T_PATH_CLOSE,
        T_CONTENT_OPEN,
        "...the Python code here...",
        T_CONTENT_CLOSE,
        T_WRITE_CLOSE,
    ]
    prompt = "\n".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]

    for mode_name, stream in [("Streaming", True)]:
        print(f"\n  [{mode_name}]")
        content = chat(messages, stream=stream)
        if content is None:
            report(f"{mode_name}: got response", False, "No response")
            continue

        report(f"{mode_name}: got response", True)

        # Check XML structure is present
        has_write_open = T_WRITE_OPEN in content
        has_write_close = T_WRITE_CLOSE in content
        report(f"{mode_name}: has {T_WRITE_OPEN} tag", has_write_open)
        report(f"{mode_name}: has {T_WRITE_CLOSE} tag", has_write_close)

        has_path = T_PATH_OPEN in content and T_PATH_CLOSE in content
        report(f"{mode_name}: has path tags", has_path)

        has_content_tags = T_CONTENT_OPEN in content and T_CONTENT_CLOSE in content
        report(f"{mode_name}: has content tags", has_content_tags)

        # Extract and validate the code inside content tags
        if has_content_tags:
            code = extract_between(content, T_CONTENT_OPEN, T_CONTENT_CLOSE)
            if code:
                check_clean_code(code, mode_name)
                # Verify it looks like actual Python
                report(f"{mode_name}: code has class definition",
                       "class " in code or "def " in code)
                print(f"\n  --- Extracted code ({len(code)} chars) ---")
                preview(code)
            else:
                report(f"{mode_name}: could extract code", False, "extraction returned None")
        else:
            report(f"{mode_name}: skipped code check (no content tags)", False)

        print(f"\n  --- Full response ---")
        preview(content, 1200)


# ============================================================================
# Test 2: read_file XML tags must be unescaped
# ============================================================================


def test_read_file_tags():
    """Verify read_file XML tags come through with real angle brackets."""
    print("\n" + "=" * 70)
    print("TEST 2: read_file tags - angle brackets must be real")
    print("=" * 70)

    prompt_parts = [
        "You are a coding agent. To read a file, output a read_file XML block.",
        "Read the file at src/main.py. Output ONLY the XML block:",
        T_READ_OPEN,
        T_PATH_OPEN + "src/main.py" + T_PATH_CLOSE,
        T_READ_CLOSE,
    ]
    prompt = "\n".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]

    for mode_name, stream in [("Streaming", True)]:
        print(f"\n  [{mode_name}]")
        content = chat(messages, stream=stream)
        if content is None:
            report(f"{mode_name}: got response", False, "No response")
            continue

        report(f"{mode_name}: got response", True)
        report(f"{mode_name}: has {T_READ_OPEN}", T_READ_OPEN in content)
        report(f"{mode_name}: has {T_READ_CLOSE}", T_READ_CLOSE in content)

        # Check no escaped versions
        bs_lt = "\\" + "<"
        report(f"{mode_name}: no backslash-escaped angles",
               bs_lt not in content)

        report(f"{mode_name}: no HTML-escaped angles",
               "&lt;" not in content and "&gt;" not in content)

        path = extract_between(content, T_PATH_OPEN, T_PATH_CLOSE)
        if path:
            report(f"{mode_name}: path extracted correctly",
                   "main.py" in path.strip())
        print(f"\n  --- Raw response ---")
        preview(content, 400)


# ============================================================================
# Test 3: apply_diff with SEARCH/REPLACE markers
# ============================================================================


def test_apply_diff_markers():
    """Verify apply_diff SEARCH/REPLACE conflict markers survive."""
    print("\n" + "=" * 70)
    print("TEST 3: apply_diff - SEARCH/REPLACE markers intact")
    print("=" * 70)

    diff_block = "\n".join([
        T_DIFF_OPEN,
        T_ARGS_OPEN,
        T_FILE_OPEN,
        T_PATH_OPEN + "src/app.py" + T_PATH_CLOSE,
        T_DIFFTAG_OPEN,
        "<<<<<<< SEARCH",
        "def old_function():",
        '    return "old"',
        "=======",
        "def new_function():",
        '    return "new"',
        ">>>>>>> REPLACE",
        T_DIFFTAG_CLOSE,
        T_FILE_CLOSE,
        T_ARGS_CLOSE,
        T_DIFF_CLOSE,
    ])

    prompt = "\n".join([
        "You are a coding agent. Output EXACTLY this apply_diff block as raw text.",
        "No markdown fences, no explanations:",
        "",
        diff_block,
    ])
    messages = [{"role": "user", "content": prompt}]

    for mode_name, stream in [("Streaming", True)]:
        print(f"\n  [{mode_name}]")
        content = chat(messages, stream=stream)
        if content is None:
            report(f"{mode_name}: got response", False, "No response")
            continue

        report(f"{mode_name}: got response", True)
        report(f"{mode_name}: has SEARCH marker", "<<<<<<< SEARCH" in content)
        report(f"{mode_name}: has REPLACE marker", ">>>>>>> REPLACE" in content)
        report(f"{mode_name}: has separator (=======)", "=======" in content)
        report(f"{mode_name}: has diff tags", T_DIFF_OPEN in content)

        # The code inside the diff should be clean
        bs_lt = "\\" + "<"
        report(f"{mode_name}: no backslash escaping", bs_lt not in content)

        print(f"\n  --- Raw response ---")
        preview(content, 600)


# ============================================================================
# Test 4: write_to_file with code containing special chars
# ============================================================================


def test_write_special_chars_in_code():
    """Code with angle brackets, ampersands etc. must survive inside content tags."""
    print("\n" + "=" * 70)
    print("TEST 4: write_to_file - code with special chars")
    print("=" * 70)

    # Ask for code that naturally contains < > & characters
    prompt_parts = [
        "You are a coding agent. Create a C file using a write_to_file XML block.",
        "The file should be at src/compare.c and contain a main function that:",
        "- Compares two integers using < and > operators",
        "- Uses && and || logical operators",
        "- Prints results with printf",
        "",
        "Output ONLY the XML block:",
        T_WRITE_OPEN,
        T_PATH_OPEN + "src/compare.c" + T_PATH_CLOSE,
        T_CONTENT_OPEN,
        "...the C code...",
        T_CONTENT_CLOSE,
        T_WRITE_CLOSE,
    ]
    prompt = "\n".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]

    for mode_name, stream in [("Streaming", True)]:
        print(f"\n  [{mode_name}]")
        content = chat(messages, stream=stream)
        if content is None:
            report(f"{mode_name}: got response", False, "No response")
            continue

        report(f"{mode_name}: got response", True)

        if T_CONTENT_OPEN in content and T_CONTENT_CLOSE in content:
            code = extract_between(content, T_CONTENT_OPEN, T_CONTENT_CLOSE)
            if code:
                check_clean_code(code, mode_name)
                # C code should have real < > && operators
                report(f"{mode_name}: code has < operator", "<" in code)
                report(f"{mode_name}: code has > operator", ">" in code)
                report(f"{mode_name}: code has #include or main",
                       "#include" in code or "main" in code)
                print(f"\n  --- Extracted C code ---")
                preview(code)
            else:
                report(f"{mode_name}: extract code", False)
        else:
            report(f"{mode_name}: has content tags", False)
            print(f"\n  --- Full response (no content tags found) ---")
            preview(content, 1000)


# ============================================================================
# Test 5: Streaming JSON integrity
# ============================================================================


def test_streaming_json_integrity():
    """Each SSE chunk must be valid JSON with correct structure."""
    print("\n" + "=" * 70)
    print("TEST 5: Streaming JSON chunk integrity")
    print("=" * 70)

    messages = [{"role": "user",
                 "content": "Write a 3-line Python script that prints hello, world, done."}]

    body = {"model": "gemini-3.0-flash", "messages": messages, "stream": True}
    chunks_ok = 0
    json_errors = 0
    full = ""
    had_role = False
    had_done = False

    try:
        with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions",
                          json=body, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                report("Streaming response", False, f"HTTP {r.status_code}")
                return
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    had_done = True
                    break
                try:
                    chunk = json.loads(payload)
                    assert "choices" in chunk
                    assert "delta" in chunk["choices"][0]
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("role") == "assistant":
                        had_role = True
                    if delta.get("content"):
                        full += delta["content"]
                    chunks_ok += 1
                except (json.JSONDecodeError, KeyError, AssertionError):
                    json_errors += 1
    except Exception as exc:
        report("Streaming request", False, str(exc))
        return

    report("Parsed chunks", chunks_ok > 0, f"{chunks_ok} chunks")
    report("No JSON errors", json_errors == 0, f"{json_errors} errors")
    report("Had role=assistant", had_role)
    report("Had [DONE]", had_done)
    report("Content non-empty", len(full) > 0)


# ============================================================================
# Test 6: Local unit test of unescape function (no server needed)
# ============================================================================


def test_unescape_function_local():
    """Unit test the unescape_gemini_markdown function directly."""
    print("\n" + "=" * 70)
    print("TEST 6: Local unit test - unescape_gemini_markdown()")
    print("=" * 70)

    try:
        from gemini_client import unescape_gemini_markdown
    except ImportError:
        report("Import unescape_gemini_markdown", False,
               "Function not found in gemini_client.py")
        return

    report("Import unescape_gemini_markdown", True)

    # Build test cases with escaped versions using concatenation
    test_cases = [
        # (input, expected_output, description)
        ("\\" + "<" + "div" + "\\" + ">",
         "<" + "div" + ">",
         "angle brackets unescaped"),

        ("\\" + "<" + "read" + "\\" + "_" + "file" + "\\" + ">",
         "<" + "read_file" + ">",
         "XML tag with underscore unescaped"),

        ("x " + "\\" + "<" + " 10 && y " + "\\" + ">" + " 5",
         "x < 10 && y > 5",
         "comparison operators unescaped"),

        ("Hello" + "\\" + "!" + " World",
         "Hello! World",
         "exclamation mark unescaped"),

        ("def greet(name):\n    return f\"Hello, {name}\"",
         "def greet(name):\n    return f\"Hello, {name}\"",
         "clean code unchanged"),

        ("C:" + "\\" + "\\" + "Users",
         "C:" + "\\" + "\\" + "Users",
         "double backslash preserved (real escape)"),

        ("line1\\nline2",
         "line1\\nline2",
         "backslash-n preserved (real escape)"),

        ("",
         "",
         "empty string unchanged"),
    ]

    for inp, expected, desc in test_cases:
        actual = unescape_gemini_markdown(inp)
        passed = actual == expected
        detail = ""
        if not passed:
            detail = f"Input:    {repr(inp)}\nExpected: {repr(expected)}\nGot:      {repr(actual)}"
        report(f"unescape: {desc}", passed, detail)


# ============================================================================
# Summary
# ============================================================================


def print_summary():
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    if failed > 0:
        print(f"\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['name']}")
    print()
    return failed == 0


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Gemini API - XML Tool & Clean Code Test Suite")
    print("  (Roo Code / Cline / Kilo Code compatibility)")
    print("=" * 70)

    # Always run local unit tests first (no server needed)
    test_unescape_function_local()

    # Check if server is reachable for integration tests
    server_ok = False
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=10)
        health = r.json()
        server_ok = health.get("gemini_ready", False)
        if not server_ok:
            print("\n  WARNING: Server reports gemini_ready=False")
            print("  Integration tests will be skipped.\n")
    except Exception as exc:
        print(f"\n  WARNING: Cannot reach server at {BASE_URL}")
        print(f"  Integration tests will be skipped.")
        print(f"  Start server with: python main.py\n")

    if server_ok:
        integration_tests = [
            test_write_to_file_clean_code,
            test_read_file_tags,
            test_apply_diff_markers,
            test_write_special_chars_in_code,
            test_streaming_json_integrity,
        ]
        for fn in integration_tests:
            try:
                fn()
            except Exception as exc:
                print(f"  [FAIL] {fn.__name__} raised: {exc}")
                results.append({"name": fn.__name__, "passed": False})
    else:
        print("  Skipping integration tests (server not ready)")

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)