# Poe Proxy (OpenAI-compatible) Server

A lightweight FastAPI server that proxies OpenAI-style Chat Completions requests to Poe’s OpenAI-compatible API. It adds a few quality-of-life features on top of Poe:

-   Model registry via models.yaml (grouped by modality)
-   Optional per-model and global context clipping by tokens
-   Robust streaming with layered 400 retries (remove max\_tokens, lowercase model name, etc.)
-   Footer that displays token usage and current Poe point balance using Poe’s Usage API
-   Optional usage guards:
    -   Pause usage if your balance is below a configurable cutoff (e.g., 100k points)
    -   Warn if a single message costs too many points (configurable threshold)
-   Simple per-IP rate limiting
-   OpenAI-compatible endpoints:
    -   POST /v1/chat/completions
    -   GET /v1/models

This is intended for local/self-hosted use, or as a proxy layer between OpenAI-compatible clients and Poe.

Website reference: Poe - Fast, Helpful AI Chat ([https://poe.com/login](https://poe.com/login))  
“The best AI, all in one place. Explore GPT-5, Claude Opus 4.1, DeepSeek-R1, Veo 3, ElevenLabs, and millions of others.”

## Directory structure

-   Dockerfile
-   requirements.txt
-   models.yaml
-   tests/
-   list\_models.py
-   pytest.ini
-   old-proxy.py
-   docker-compose.yaml
-   proxy.py

## Prerequisites

-   Docker and Docker Compose
-   A Poe API key
    -   You can generate one at: [https://poe.com/api\_key](https://poe.com/api_key)
-   Optional: Python 3.10+ if running locally without Docker

## Quick start (Docker Compose)

1.  Set your Poe API key securely

Do not hardcode your API key in docker-compose.yaml. Instead, create a local .env file (ignored by Git) in the repo root:

.env  
POE\_API\_KEY=your\_real\_poe\_api\_key\_here  
MODELS\_FILE=/app/models.yaml

# Optional configuration:

# POE\_BASE\_URL=[https://api.poe.com](https://api.poe.com/)

# POINTS\_CUTOFF=100000

# POINTS\_PER\_MESSAGE\_WARN=5000

# MAX\_POINTS\_PER\_MESSAGE=0

1.  Update docker-compose.yaml to use environment from .env

Replace any inlined keys in docker-compose.yaml with:  
services:  
poe-proxy:  
build: .  
container\_name: poe-proxy  
env\_file:  
\- ./.env  
ports:  
\- "0.0.0.0:9876:8080"  
restart: unless-stopped

poe-proxy-tests:  
build: .  
container\_name: poe-proxy-test-runner  
env\_file:  
\- ./.env  
volumes:  
\- ./:/app  
command: \["pytest", "-v", "tests/"\]

1.  Build and run

docker compose up --build -d

1.  Verify

-   Server runs at: [http://localhost:9876](http://localhost:9876/)
    
-   List models:  
    curl [http://localhost:9876/v1/models](http://localhost:9876/v1/models)
    
-   Example chat completion (non-streaming):  
    curl -X POST [http://localhost:9876/v1/chat/completions](http://localhost:9876/v1/chat/completions)  
    \-H "Content-Type: application/json"  
    \-d '{  
    "model": "GPT-4.1",  
    "messages": \[{"role":"user","content":"Hello!"}\]  
    }'
    

For streaming, set "stream": true in the payload and use an SSE-capable client.

## Environment variables

-   POE\_API\_KEY or POE\_TOKEN (required): Your Poe API key. Prefer POE\_API\_KEY.
-   MODELS\_FILE (optional): Path to models.yaml in the container. Default: /app/models.yaml
-   POE\_BASE\_URL (optional): Poe API base. Default: [https://api.poe.com](https://api.poe.com/)
-   POINTS\_CUTOFF (optional): If current balance is below this, proxy returns a friendly “paused” message. Default: 100000
-   POINTS\_PER\_MESSAGE\_WARN (optional): If a single response costs more than this, append a warning with cheaper model suggestions. Default: 5000
-   MAX\_POINTS\_PER\_MESSAGE (optional): If > 0, used to signal that extremely costly single messages may be blocked in future requests. Default: 0 (disabled)

Tip: Keep these in your local .env and never commit them.

## Models configuration (models.yaml)

-   Grouped by modality: text, image, video, audio.
-   Each entry may include:
    -   poe\_name: The exact Poe model/bot ID to request
    -   base and per\_1k\_tokens: Legacy local point estimate fields (now used only as a fallback if the Usage API entry isn’t available yet)
    -   max\_tokens (optional): A proxy-side context cap for that model; the proxy also enforces a global default via defaults.text\_max\_tokens.

Example snippet:  
models:  
text:  
\- poe\_name: GPT-4.1  
base: 226  
per\_1k\_tokens: 60  
max\_tokens: 200000

To see which model IDs your proxy exposes (from models.yaml), call GET /v1/models.

## Endpoints

-   POST /v1/chat/completions
    
    -   Accepts OpenAI-style payloads:
        -   model: must match a poe\_name from models.yaml
        -   messages: OpenAI format
        -   stream: boolean (default true for text models)
        -   Optional OpenAI-compatible params: max\_tokens, top\_p, temperature, stop, tools, tool\_choice, parallel\_tool\_calls, n (forced to 1), stream\_options
    -   Behavior:
        -   Clips context if needed
        -   Streams or returns once, depending on stream flag
        -   Appends a footer showing:
            -   Tokens: total/input/output (from Poe’s response usage)
            -   Points: fetched from Poe Usage API points\_history (first entry), otherwise estimated from models.yaml
            -   Current balance: fetched from Poe Usage API current\_balance
        -   If balance is below POINTS\_CUTOFF: returns a friendly “paused” message (no model call).
-   GET /v1/models
    
    -   Returns an OpenAI-compatible list of model IDs sourced from models.yaml.

## Usage API integration

The proxy uses Poe’s Usage API to:

-   Fetch current balance:  
    GET [https://api.poe.com/usage/current\_balance](https://api.poe.com/usage/current_balance)
    
-   Fetch the latest usage entry:  
    GET [https://api.poe.com/usage/points\_history?limit=1](https://api.poe.com/usage/points_history?limit=1)
    

This allows the footer to show actual cost\_points per message and the latest balance without relying on local pricing. If the most recent entry isn’t available (eventual consistency), it falls back to models.yaml estimates.

Authentication:  
Authorization: Bearer YOUR\_POE\_API\_KEY

## Local development (without Docker)

1.  Create and activate a virtualenv, then install deps:  
    python -m venv .venv  
    source .venv/bin/activate  
    pip install -r requirements.txt
    
2.  Set environment variables:  
    export POE\_API\_KEY=your\_real\_poe\_api\_key  
    export MODELS\_FILE=./models.yaml
    
3.  Run:  
    python proxy.py
    
    # Or: uvicorn proxy:app --host 0.0.0.0 --port 8080
    

## Testing (in development)

-   With Docker Compose:  
    docker compose run --rm poe-proxy-tests
    
-   Locally:  
    pytest -v
    

Note: tests/ is provided for your suite; adjust as needed.

## list\_models.py (utility)

A small helper for listing bots using fastapi-poe client. If you use it, ensure it reads the API key from an environment variable instead of a hardcoded string. Example:

import os, asyncio, fastapi\_poe as fp

async def list\_available\_bots():  
token = os.getenv("POE\_API\_KEY") or os.getenv("POE\_TOKEN") or ""  
if not token:  
print("Error: POE\_API\_KEY/POE\_TOKEN not set")  
return  
client = await fp.get\_client(token)  
bots = await client.get\_bot\_names()  
for bot in bots:  
print(f"- {bot}")

if **name** == "**main**":  
asyncio.run(list\_available\_bots())

## Security and publishing notes

-   Remove any hardcoded API keys from docker-compose.yaml, list\_models.py, commit history, or logs.
-   Rotate your key if it was ever committed.
-   Add .env to .gitignore and use env\_file in docker-compose to inject secrets at runtime.
-   Consider using Docker/Compose secrets or your orchestrator’s secret manager in production.

## Troubleshooting

-   401 Unauthorized: Ensure POE\_API\_KEY is set and valid.
-   404 Model not found: The “model” must exactly match a poe\_name in models.yaml. Check GET /v1/models.
-   429 Too many requests: Simple per-IP rate limiter may be triggering; slow down or adjust code.
-   Streaming errors: The proxy uses per-attempt httpx AsyncClient instances to avoid closed-client errors. If you see issues, check container logs.
-   Points show as “est.”: The points\_history endpoint may lag briefly; the proxy falls back to local estimates until a real entry appears.

## License
GNU AGPLv3

