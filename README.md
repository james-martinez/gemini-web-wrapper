# Gemini OpenAI-Compatible API Server

An OpenAI-compatible REST API server that proxies requests to **Google Gemini** using [`gemini-webapi`](https://github.com/HanaokaYuzu/Gemini-API).  
Works with **Roo Code**, **Kilo Code**, **Cline**, and any client that speaks the OpenAI Chat Completions protocol.

## Available Models

| Model ID | Description |
|---|---|
| `gemini-3.0-pro` | Gemini 3.0 Pro |
| `gemini-3.0-flash` | Gemini 3.0 Flash |
| `gemini-3.0-flash-thinking` | Gemini 3.0 Flash Thinking |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure authentication

Copy `.env.example` to `.env` and fill in your Gemini cookies:

```bash
cp .env.example .env
```

Get the cookies from your browser when logged into [gemini.google.com](https://gemini.google.com):

- **Chrome**: F12 → Application → Cookies → `https://gemini.google.com`
- Copy `__Secure-1PSID` → `SECURE_1PSID`
- Copy `__Secure-1PSIDTS` → `SECURE_1PSIDTS`

### 3. Start the server

```bash
python main.py
```

The server starts on `http://0.0.0.0:8000` by default.

### 4. Configure your client

In Roo Code / Kilo Code / Cline, add a custom OpenAI-compatible provider:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:8000/v1` |
| API Key | `anything` (not validated) |
| Model | `gemini-3.0-flash` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{id}` | Get model info |
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |

## Testing

Start the server, then run:

```bash
python test_api.py
```

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `SECURE_1PSID` | — | Required. Gemini auth cookie |
| `SECURE_1PSIDTS` | — | Required. Gemini auth cookie |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `PROXY_URL` | — | Optional HTTP proxy |
| `GEMINI_LOG_LEVEL` | `WARNING` | Log level for gemini-webapi |
| `DEBUG` | `false` | Verbose request/response logging |

## Project Structure

```
├── main.py            # FastAPI application & routes
├── config.py          # Settings from .env
├── models.py          # OpenAI-compatible Pydantic models
├── gemini_client.py   # gemini-webapi wrapper
├── test_api.py        # Integration tests
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
└── .env               # Your local config (git-ignored)