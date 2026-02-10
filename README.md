# Gemini OpenAI-Compatible API

An OpenAI-compatible REST API server that proxies requests to Google Gemini.

Compatible with AI coding assistants like:
- **Roo Code**
- **Kilo Code**  
- **Cline**
- Any other OpenAI API-compatible client

## Features

- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Streaming (SSE) and non-streaming responses
- ✅ Image/vision support via base64 data URIs
- ✅ Multiple message roles (system, user, assistant)
- ✅ Model listing endpoint `/v1/models`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your Gemini cookies:

```bash
cp .env.example .env
```

Edit `.env` and add your authentication cookies:

```
SECURE_1PSID=your_cookie_value_here
SECURE_1PSIDTS=your_cookie_value_here
```

**How to get cookies:**
1. Go to [gemini.google.com](https://gemini.google.com) and sign in
2. Open browser Developer Tools (F12)
3. Go to Application > Cookies > gemini.google.com
4. Copy the values for `__Secure-1PSID` and `__Secure-1PSIDTS`

### 3. Run the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```

OpenAI-compatible chat completion endpoint. Supports streaming.

**Example Request:**
```json
{
  "model": "gemini-3.0-flash-thinking",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

### List Models
```
GET /v1/models
```

Returns available models.

### Health Check
```
GET /health
```

Returns server health status.

## Configuring AI Assistants

### Roo Code / Kilo Code

1. Open settings
2. Set API Base URL to: `http://localhost:8000/v1`
3. Set API Key to any value (e.g., `dummy-key`)
4. Select model: `gemini-3.0-flash-thinking`

### Cline

1. Open Cline settings
2. Set provider to "OpenAI Compatible"
3. Set Base URL: `http://localhost:8000/v1`
4. Set API Key: `dummy-key`
5. Set Model: `gemini-3.0-flash-thinking`

## Available Models

- **gemini-3.0-flash-thinking** - Flash model with thinking/reasoning capabilities
- **gemini-3.0-pro** - Pro model for complex tasks
- **gemini-3.0-flash** - Fast flash model

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECURE_1PSID` | Gemini authentication cookie | Required |
| `SECURE_1PSIDTS` | Gemini authentication cookie | Required |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `PROXY_URL` | Optional proxy URL | None |

## Limitations

- This uses the web interface to Gemini, not the official API
- Cookies may expire and need to be refreshed periodically
- Rate limits depend on your Google account

## License

MIT