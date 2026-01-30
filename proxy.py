import os
import json
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import uvicorn
import logging
import time
import asyncio
from typing import Optional, Dict, Any, Union, AsyncGenerator, List, Tuple
from uuid import uuid4
import traceback
import yaml
import httpx
import tiktoken

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_FILE = os.getenv("MODELS_FILE", "models.yaml")
POE_API_BASE = os.getenv("POE_BASE_URL", "https://api.poe.com")

# Global usage/balance controls
POINTS_CUTOFF = int(os.getenv("POINTS_CUTOFF", "10000"))  # pause usage below this balance
POINTS_PER_MESSAGE_WARN = int(os.getenv("POINTS_PER_MESSAGE_WARN", "5000"))  # warn if a single response costs more than this
MAX_POINTS_PER_MESSAGE = int(os.getenv("MAX_POINTS_PER_MESSAGE", "0"))  # hard block if > 0 and cost exceeds this; 0 disables hard block

# Error “when it happens” map per Poe docs
ERROR_WHEN_MAP = {
    400: ("invalid_request_error", "Malformed JSON or missing fields"),
    401: ("authentication_error", "Bad or expired API key"),
    402: ("insufficient_credits", "Balance ≤ 0"),
    403: ("moderation_error", "Permission denied or authorization issues"),
    404: ("not_found_error", "Wrong endpoint or model"),
    408: ("timeout_error", "Model didn't start in a reasonable time"),
    413: ("request_too_large", "Tokens > context window"),
    429: ("rate_limit_error", "RPM/TPM cap hit"),
    502: ("upstream_error", "Model backend not working"),
    529: ("overloaded_error", "Transient traffic spike"),
}

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

# Cheap model suggestions (adjust as you like)
CHEAP_ALTERNATIVES = {
    # expensive -> [cheaper, ...]
    "GPT-5": ["GPT-4.1-mini", "o4-mini", "Mistral-Small-3.1", "GLM-4.5-FW"],
    "GPT-4.1": ["GPT-4.1-mini", "o4-mini", "Mistral-Small-3.1"],
    "Claude-3.5-Sonnet": ["Mistral-Large-2", "Mistral-Small-3.1", "GLM-4.5-FW"],
    "Gemini-2.5-Pro": ["Gemini-2.5-Flash", "Mistral-Small-3.1"],
    "DeepSeek-R1": ["DeepSeek-V3.1", "Mistral-Small-3.1"],
    # default fallback suggestions used if model not mapped:
    "__default__": ["Mistral-Small-3.1", "GLM-4.5-FW", "GPT-4.1-mini"]
}

class ModelRegistry:
    """
    models.yaml example:

    defaults:
      text_max_tokens: 200000
    models:
      text:
        - poe_name: Claude-3.5-Sonnet
          max_tokens: 200000
        - GPT-4.1
      image:
        - GPT-Image-1
      video:
        - Veo-3
      audio:
        - ElevenLabs
    """
    def __init__(self, path: str):
        self.path = path
        self.defaults: Dict[str, Any] = {
            "text_max_tokens": 200000
        }
        self.models: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model config file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        defaults = data.get("defaults") or {}
        if "text_max_tokens" in defaults:
            self.defaults["text_max_tokens"] = int(defaults["text_max_tokens"])

        self.models.clear()
        grouped = (data.get("models") or {})
        for modality in ("text", "image", "video", "audio"):
            for item in grouped.get(modality, []) or []:
                if isinstance(item, str):
                    poe_name = item
                    model_cfg = {"poe_name": poe_name, "modality": modality}
                elif isinstance(item, dict):
                    poe_name = item.get("poe_name")
                    if not poe_name:
                        raise ValueError(f"Missing poe_name in models.{modality} entry: {item}")
                    model_cfg = {
                        "poe_name": poe_name,
                        "modality": modality,
                    }
                    if "base" in item:
                        model_cfg["base"] = float(item["base"])
                    if "per_1k_tokens" in item:
                        model_cfg["per_1k_tokens"] = float(item["per_1k_tokens"])
                    if "max_tokens" in item:
                        try:
                            model_cfg["max_tokens"] = int(item["max_tokens"])
                        except Exception:
                            logger.warning(f"Ignoring non-int max_tokens for {poe_name}")
                    if "context_limit" in item:
                        try:
                            model_cfg["context_limit"] = int(item["context_limit"])
                        except Exception:
                            logger.warning(f"Ignoring non-int context_limit for {poe_name}")
                else:
                    raise ValueError(f"Invalid entry in models.{modality}: {item}")
                self.models[poe_name] = model_cfg

        logger.info(f"Loaded {len(self.models)} models from {self.path}")

    def get(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found")
        return self.models[model_id]

    def list_ids(self) -> List[str]:
        return list(self.models.keys())

    def get_default_text_max_tokens(self) -> int:
        return int(self.defaults["text_max_tokens"])

try:
    REGISTRY = ModelRegistry(MODELS_FILE)
except Exception as e:
    logger.error(f"Failed to load model registry: {e}")
    raise

@lru_cache()
def get_poe_api_key() -> str:
    api_key = os.getenv("POE_API_KEY") or os.getenv("POE_TOKEN")
    if not api_key:
        raise ValueError("POE_API_KEY or POE_TOKEN environment variable not set")
    return api_key

def format_sse_chunk(data_dict: Dict, request_id: str, model_name: str, finish_reason: Optional[str] = None) -> str:
    sse_payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": data_dict,
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(sse_payload)}\n\n"

def make_clear_error_message(status_code: int, api_error: Optional[Dict[str, Any]]) -> str:
    mapped_type, when = ERROR_WHEN_MAP.get(status_code, ("unknown_error", "Unknown error"))
    e_type = api_error.get("type") if api_error and api_error.get("type") else mapped_type
    e_code = api_error.get("code") if api_error and api_error.get("code") else status_code
    e_msg = api_error.get("message") if api_error else "No message"
    return f"[Poe API Error] code={e_code}, type={e_type}, when='{when}', message='{e_msg}'"

def sanitize_messages(messages):
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append({"role": "user", "content": str(msg)})
            continue

        role = msg.get("role", "user").lower()
        if role not in ALLOWED_ROLES:
            role = "user"

        content = msg.get("content", "")

        # KEEP content lists (including images)
        if isinstance(content, list):
            cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({"role": role, "content": str(content)})

    if not cleaned:
        cleaned = [{"role": "user", "content": ""}]
    return cleaned

# Token counting and context management
@lru_cache(maxsize=1)
def get_tokenizer():
    """Get a tiktoken encoder for token counting. Uses cl100k_base (GPT-4/Claude compatible)."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        return None

def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Count tokens in a message list.
    Approximates OpenAI's token counting for chat messages.
    """
    tokenizer = get_tokenizer()
    if not tokenizer:
        # Fallback: rough estimate of 4 chars per token
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4

    num_tokens = 0
    for message in messages:
        # Every message follows <|im_start|>{role}\n{content}<|im_end|>\n
        num_tokens += 4  # message overhead

        role = message.get("role", "")
        content = message.get("content", "")

        num_tokens += len(tokenizer.encode(role))

        # Handle content that might be a list (for images, etc.)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        num_tokens += len(tokenizer.encode(text))
                    elif item.get("type") == "image_url":
                        # Image tokens: rough estimate, actual cost varies
                        num_tokens += 85  # Base image token cost
        else:
            num_tokens += len(tokenizer.encode(str(content)))

    num_tokens += 2  # Assistant reply priming
    return num_tokens

def truncate_message_content(content: Any, max_tokens: int) -> Any:
    """
    Truncate message content to fit within max_tokens.
    Handles both string content and list content (with images, etc.)
    """
    tokenizer = get_tokenizer()

    if isinstance(content, str):
        if not tokenizer:
            # Fallback: rough estimate of 4 chars per token
            max_chars = max_tokens * 4
            if len(content) > max_chars:
                return content[:max_chars] + "... [truncated]"
            return content

        tokens = tokenizer.encode(content)
        if len(tokens) <= max_tokens:
            return content

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens) + "... [truncated]"

    elif isinstance(content, list):
        # For list content, keep text items and truncate them
        result = []
        tokens_used = 0
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if tokenizer:
                    text_tokens = len(tokenizer.encode(text))
                else:
                    text_tokens = len(text) // 4

                if tokens_used + text_tokens > max_tokens:
                    remaining = max_tokens - tokens_used
                    if remaining > 0:
                        truncated_text = truncate_message_content(text, remaining)
                        result.append({"type": "text", "text": truncated_text})
                    break

                result.append(item)
                tokens_used += text_tokens
            else:
                # Keep non-text items (images, etc.) but count their tokens
                result.append(item)
                if item.get("type") == "image_url":
                    tokens_used += 85  # Rough image token estimate

        return result if result else [{"type": "text", "text": "[content truncated]"}]

    return content

def trim_messages_to_limit(messages: List[Dict[str, Any]], token_limit: int, aggressive: bool = False) -> List[Dict[str, Any]]:
    """
    Trim messages to fit within token limit.
    Preserves the first message (usually system prompt) and trims from the second message onwards.

    Strategy:
    1. Always keep the first message (position 0), but truncate if needed in aggressive mode
    2. Remove messages from position 1 onwards (oldest user/assistant exchanges)
    3. Keep removing until total tokens <= token_limit
    4. In aggressive mode, truncate individual messages if needed
    """
    if not messages:
        return messages

    current_tokens = count_message_tokens(messages)

    if current_tokens <= token_limit:
        return messages

    # If only one message and it's over the limit, truncate it
    if len(messages) == 1:
        logger.warning(f"Single message exceeds token limit ({current_tokens} > {token_limit}). Truncating content.")
        msg = dict(messages[0])
        msg["content"] = truncate_message_content(msg["content"], token_limit - 100)  # Leave some buffer
        return [msg]

    # Preserve first message, trim from position 1 onwards
    first_msg = dict(messages[0])
    remaining_msgs = messages[1:]

    # Check if first message alone is too big
    first_msg_tokens = count_message_tokens([first_msg])
    if first_msg_tokens > token_limit * 0.7:  # If first message uses more than 70% of limit
        if aggressive:
            logger.warning(f"First message uses {first_msg_tokens} tokens (>{token_limit*0.7}). Truncating to fit.")
            max_first_msg_tokens = int(token_limit * 0.3)  # Give first message max 30% of limit
            first_msg["content"] = truncate_message_content(first_msg["content"], max_first_msg_tokens)
            first_msg_tokens = count_message_tokens([first_msg])
        else:
            logger.warning(f"First message alone uses {first_msg_tokens} tokens (limit: {token_limit}). Consider using aggressive mode.")

    # Start with just the first message
    trimmed = [first_msg]

    # Add messages from the END (most recent) while staying under limit
    for i in range(len(remaining_msgs) - 1, -1, -1):
        test_msgs = [first_msg] + remaining_msgs[i:]
        test_tokens = count_message_tokens(test_msgs)

        if test_tokens <= token_limit:
            trimmed = test_msgs
        else:
            # Can't add this message, we're done
            break

    # If still too long in aggressive mode, truncate individual recent messages
    if aggressive and count_message_tokens(trimmed) > token_limit:
        logger.warning("Still over limit after trimming. Truncating recent messages.")
        available_tokens = token_limit - count_message_tokens([trimmed[0]])
        tokens_per_msg = available_tokens // max(1, len(trimmed) - 1)

        result = [trimmed[0]]
        for msg in trimmed[1:]:
            truncated_msg = dict(msg)
            truncated_msg["content"] = truncate_message_content(msg["content"], tokens_per_msg)
            result.append(truncated_msg)
        trimmed = result

    removed_count = len(messages) - len(trimmed)
    if removed_count > 0:
        logger.info(f"Trimmed {removed_count} messages to fit {token_limit} token limit ({current_tokens} -> {count_message_tokens(trimmed)} tokens)")

    return trimmed

# Usage API helpers
async def poe_get_current_balance(api_key: str) -> Optional[int]:
    url = f"{POE_API_BASE}/usage/current_balance"
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"Balance fetch failed: {resp.status_code} {resp.text}")
                return None
            data = resp.json()
            return int(data.get("current_point_balance")) if "current_point_balance" in data else None
        except Exception as e:
            logger.warning(f"Balance fetch error: {e}")
            return None

async def poe_get_latest_usage_entry(api_key: str) -> Optional[Dict[str, Any]]:
    url = f"{POE_API_BASE}/usage/points_history"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"limit": 1}
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                logger.warning(f"Usage history fetch failed: {resp.status_code} {resp.text}")
                return None
            data = resp.json()
            items = data.get("data") or []
            return items[0] if items else None
        except Exception as e:
            logger.warning(f"Usage history error: {e}")
            return None

def is_context_length_error(error_text: str) -> bool:
    """Check if error is a context length exceeded error."""
    if not error_text:
        return False
    return "context length" in error_text.lower() or "context window" in error_text.lower() or "too long" in error_text.lower()

async def call_poe_openai_compatible(payload: Dict[str, Any], stream: bool, model_context_limit: Optional[int] = None) -> Union[AsyncGenerator[Dict[str, Any], None], Dict[str, Any]]:
    """
    Retry ladder on 400:
      1) as-is
      2) remove max_tokens/max_completion_tokens
      3) lowercase model
      4) lowercase + remove max_tokens
      5) if context length error, aggressively trim messages and retry
    """
    api_key = get_poe_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = POE_API_BASE.rstrip("/")
    url = f"{base_url}/v1/chat/completions"

    def redact_payload_for_log(p: Dict[str, Any]) -> str:
        try:
            meta = {k: p[k] for k in p if k != "messages"}
            return json.dumps(meta, ensure_ascii=False)
        except Exception:
            return "{}"

    attempts: List[Dict[str, Any]] = []
    base_payload = dict(payload)
    attempts.append(base_payload)
    p2 = dict(base_payload); p2.pop("max_tokens", None); p2.pop("max_completion_tokens", None); attempts.append(p2)
    p3 = dict(base_payload);
    if isinstance(p3.get("model"), str):
        p3["model"] = p3["model"].lower()
    attempts.append(p3)
    p4 = dict(p3); p4.pop("max_tokens", None); p4.pop("max_completion_tokens", None); attempts.append(p4)

    if stream:
        async def stream_attempt(ap: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            timeout = httpx.Timeout(300.0, connect=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=ap) as resp:
                    if resp.status_code >= 400:
                        if resp.status_code == 400:
                            try:
                                logger.error(f"Poe 400 payload debug (model={ap.get('model')}): {redact_payload_for_log(ap)}; messages_count={len(ap.get('messages', []))}")
                            except Exception:
                                pass
                        try:
                            err = resp.json()
                        except Exception:
                            err = None
                        error_obj = err.get("error") if isinstance(err, dict) else None
                        error_msg = make_clear_error_message(resp.status_code, error_obj)
                        yield {"type": "error", "status_code": resp.status_code, "text": error_msg, "error_obj": error_obj}
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_line = line[len("data:"):].strip()
                        if data_line == "[DONE]":
                            yield {"type": "done"}
                            return
                        try:
                            evt = json.loads(data_line)
                            yield {"type": "chunk", "data": evt}
                        except Exception:
                            continue

        async def orchestrator():
            last_error = None
            for attempt_idx, ap in enumerate(attempts):
                first_event: Optional[Dict[str, Any]] = None
                first_yielded = False
                async for evt in stream_attempt(ap):
                    if not first_yielded:
                        first_event = evt
                        first_yielded = True
                        if evt.get("type") == "error":
                            last_error = evt
                            # try next attempt
                            break
                        else:
                            yield evt
                    else:
                        yield evt
                if first_event and first_event.get("type") != "error":
                    return

            # All regular attempts failed - check if it's a context length error
            if last_error and last_error.get("status_code") == 400:
                error_text = last_error.get("text", "")
                if is_context_length_error(error_text) and model_context_limit:
                    logger.warning("Context length error detected. Attempting aggressive trimming...")
                    # Try progressively more aggressive trimming
                    for trim_factor in [0.8, 0.6, 0.4, 0.2]:
                        reduced_limit = int(model_context_limit * trim_factor)
                        logger.info(f"Retrying with reduced context limit: {reduced_limit} tokens")
                        trimmed_messages = trim_messages_to_limit(base_payload.get("messages", []), reduced_limit, aggressive=True)
                        retry_payload = dict(base_payload)
                        retry_payload["messages"] = trimmed_messages
                        retry_payload.pop("max_tokens", None)
                        retry_payload.pop("max_completion_tokens", None)

                        first_event = None
                        first_yielded = False
                        async for evt in stream_attempt(retry_payload):
                            if not first_yielded:
                                first_event = evt
                                first_yielded = True
                                if evt.get("type") == "error":
                                    # This trimming level didn't work, try more aggressive
                                    break
                                else:
                                    yield evt
                            else:
                                yield evt

                        if first_event and first_event.get("type") != "error":
                            return

            yield {"type": "error", "text": "[All streaming attempts failed, including aggressive context trimming]"}

        return orchestrator()

    else:
        timeout = httpx.Timeout(300.0, connect=30.0)
        last_err = None
        last_status = 400

        for ap in attempts:
            # Create a fresh client for each attempt to avoid connection state issues
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=ap)
                if resp.status_code >= 400:
                    if resp.status_code == 400:
                        try:
                            logger.error(f"Poe 400 payload debug (model={ap.get('model')}): {redact_payload_for_log(ap)}; messages_count={len(ap.get('messages', []))}")
                        except Exception:
                            pass
                    # CRITICAL: Read full response body to avoid transfer encoding errors
                    await resp.aread()
                    try:
                        err = resp.json()
                    except Exception:
                        err = None
                    last_status = resp.status_code
                    last_err = make_clear_error_message(resp.status_code, err.get("error") if isinstance(err, dict) else None)
                    continue
                # Ensure full response is read before client closes
                await resp.aread()
                return resp.json()

        # All regular attempts failed - check if it's a context length error
        if last_status == 400 and is_context_length_error(last_err or "") and model_context_limit:
            logger.warning("Context length error detected. Attempting aggressive trimming...")
            # Try progressively more aggressive trimming
            for trim_factor in [0.8, 0.6, 0.4, 0.2]:
                reduced_limit = int(model_context_limit * trim_factor)
                logger.info(f"Retrying with reduced context limit: {reduced_limit} tokens")
                trimmed_messages = trim_messages_to_limit(base_payload.get("messages", []), reduced_limit, aggressive=True)
                retry_payload = dict(base_payload)
                retry_payload["messages"] = trimmed_messages
                retry_payload.pop("max_tokens", None)
                retry_payload.pop("max_completion_tokens", None)

                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, headers=headers, json=retry_payload)
                    if resp.status_code >= 400:
                        await resp.aread()
                        try:
                            err = resp.json()
                        except Exception:
                            err = None
                        last_err = make_clear_error_message(resp.status_code, err.get("error") if isinstance(err, dict) else None)
                        continue
                    await resp.aread()
                    return resp.json()

        raise HTTPException(status_code=last_status, detail=last_err or "Bad Request")

async def append_footer_with_usage_and_balance(model_id: str, usage_totals: Optional[Dict[str, Any]]) -> str:
    # Add a delay to prevent race condition with Poe's usage logging. ***
    await asyncio.sleep(1.5)

    # Fetch current balance and latest usage entry
    api_key = get_poe_api_key()
    balance = await poe_get_current_balance(api_key)
    latest = await poe_get_latest_usage_entry(api_key)

    lines = []
    if usage_totals:
        total = usage_totals.get('total_tokens')
        prompt = usage_totals.get('prompt_tokens')
        completion = usage_totals.get('completion_tokens')
        lines.append(f"📊 Tokens: {total if total is not None else 'N/A'} (Input: {prompt if prompt is not None else 'N/A'}, Output: {completion if completion is not None else 'N/A'})")

    points_used = None
    cost_usd = None
    if latest and isinstance(latest.get("cost_points"), (int, float)):
        points_used = int(latest["cost_points"])
        # Pricing: $30 per 1 million points (based on Poe add-on points pricing)
        cost_usd = (points_used / 1_000_000) * 30
        lines.append(f"💰 Points: {points_used} (${cost_usd:.6f} USD)")

    if balance is not None:
        balance_usd = (balance / 1_000_000) * 30
        lines.append(f"🏦 Balance: {balance:,} points (${balance_usd:.2f} USD)")

    # Warn if this single message used too many points
    warning = ""
    if points_used is not None and POINTS_PER_MESSAGE_WARN > 0 and points_used > POINTS_PER_MESSAGE_WARN:
        suggestions = CHEAP_ALTERNATIVES.get(model_id) or CHEAP_ALTERNATIVES.get("__default__", [])
        suggest_text = ", ".join(suggestions[:3]) if suggestions else "a cheaper model"
        warning = (
            "\n⚠️ High cost detected for this response.\n"
            "- Your current chat might have long context. Consider starting a new chat to reduce context length.\n"
            f"- Or try a cheaper model: {suggest_text}\n"
        )
        if MAX_POINTS_PER_MESSAGE > 0 and points_used > MAX_POINTS_PER_MESSAGE:
            warning += "- Further requests may be blocked due to exceeding per-message point limit.\n"

    footer = ""
    if lines or warning:
        footer = "\n\n---\n" + "\n".join(lines) + (("\n" + warning) if warning else "")

    return footer

async def stream_openai_sse_response_from_poe(
    payload: Dict[str, Any],
    request_id: str,
    model_context_limit: Optional[int] = None
) -> AsyncGenerator[str, None]:
    model_id = payload.get("model", "unknown")
    gen = await call_poe_openai_compatible(payload, stream=True, model_context_limit=model_context_limit)
    usage_totals: Optional[Dict[str, Any]] = None

    async for evt in gen:
        if evt["type"] == "error":
            yield format_sse_chunk(data_dict={"content": f"\n\n{evt['text']}"}, request_id=request_id, model_name=model_id, finish_reason="error")
            yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=model_id, finish_reason="error")
            yield "data: [DONE]\n\n"
            return

        if evt["type"] == "chunk":
            data = evt["data"]
            choices = data.get("choices", [{}])
            delta = choices[0].get("delta", {}) if choices else {}
            finish_reason = choices[0].get("finish_reason") if choices else None

            if "usage" in data and isinstance(data["usage"], dict):
                usage_totals = data["usage"]

            if delta.get("content"):
                yield format_sse_chunk(data_dict={"content": delta["content"]}, request_id=request_id, model_name=model_id)

            # If we got a finish_reason, the stream is ending - generate footer
            if finish_reason:
                # Generate and append the footer
                footer = await append_footer_with_usage_and_balance(model_id, usage_totals)

                # Send the footer content first if we have one
                if footer:
                    yield format_sse_chunk(
                        data_dict={"content": footer},
                        request_id=request_id,
                        model_name=model_id,
                        finish_reason=None
                    )

                # Send the final chunk with finish_reason
                yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=model_id, finish_reason=finish_reason)

                # Send [DONE] and return
                yield "data: [DONE]\n\n"
                return

        if evt["type"] == "done":
            # Fallback if Poe sends explicit done event
            footer = await append_footer_with_usage_and_balance(model_id, usage_totals)

            if footer:
                yield format_sse_chunk(
                    data_dict={"content": footer},
                    request_id=request_id,
                    model_name=model_id,
                    finish_reason=None
                )

            yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=model_id, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

async def nonstream_openai_response_from_poe(payload: Dict[str, Any], model_context_limit: Optional[int] = None) -> Dict[str, Any]:
    model_id = payload.get("model", "unknown")
    try:
        data = await call_poe_openai_compatible(payload, stream=False, model_context_limit=model_context_limit)
        if "choices" not in data or not data["choices"]:
            data["choices"] = [{"message": {"role": "assistant", "content": ""}}]

        usage = data.get("usage")
        footer = await append_footer_with_usage_and_balance(model_id, usage)

        if footer:
            content = data["choices"][0]["message"].get("content") or ""
            data["choices"][0]["message"]["content"] = f"{content}{footer}"
        return data
    except HTTPException as e:
        status = e.status_code
        msg = str(e.detail)
        raise HTTPException(status_code=status, detail={"error": {"code": status, "type": ERROR_WHEN_MAP.get(status, ('unknown_error',''))[0], "message": msg, "metadata": {}}})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request.state.start_time = time.time()

    try:
        logger.debug(f"New Request | Client: {request.client.host}")

        # Simple per-IP rate limit
        RATE_LIMIT = 10
        client_ip = request.client.host
        current_minute = int(time.time() // 60)
        if not hasattr(request.app.state, "rate_limits"):
            request.app.state.rate_limits = {}
        bucket = request.app.state.rate_limits.get(client_ip, {"count": 0, "minute": current_minute})
        if bucket["minute"] != current_minute:
            bucket = {"count": 0, "minute": current_minute}
        if bucket["count"] >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests")
        bucket["count"] += 1
        request.app.state.rate_limits[client_ip] = bucket

        # Balance gate: if balance < cutoff, refuse and inform user
        api_key = get_poe_api_key()
        balance = await poe_get_current_balance(api_key)
        if balance is not None and balance < POINTS_CUTOFF:
            message = (
                f"⚠️ API usage paused: Only {balance} points remaining. "
                f"Points replenish on the 11th of the month. Please try again after replenishment."
            )
            return {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "usage-guard",
                "choices": [{"message": {"role": "assistant", "content": message}}]
            }

        body = await request.body()
        data = json.loads(body.decode())

        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model'")
        try:
            model_cfg = REGISTRY.get(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        modality = model_cfg.get("modality", "text")
        is_stream = bool(data.get("stream", modality == "text"))  # stream all text bots by default

        # Sanitize messages
        sanitized_messages = sanitize_messages(data.get("messages", []))

        # Apply context limit if configured for this model
        context_limit = model_cfg.get("context_limit")
        if context_limit:
            sanitized_messages = trim_messages_to_limit(sanitized_messages, context_limit, aggressive=False)

        # Enforce n==1 if present
        if "n" in data and data["n"] != 1:
            logger.debug("Overriding n to 1 for Poe compatibility")
            data["n"] = 1

        # Build payload
        payload = {
            "model": model_id,
            "messages": sanitized_messages,
            "stream": is_stream,
        }
        passthrough_fields = [
            "max_tokens", "max_completion_tokens",
            "top_p", "tools", "tool_choice", "parallel_tool_calls",
            "stop", "temperature", "n", "stream_options"
        ]
        for f in passthrough_fields:
            if f in data:
                payload[f] = data[f]

        if is_stream:
            return StreamingResponse(
                stream_openai_sse_response_from_poe(payload, request_id=str(uuid4()), model_context_limit=context_limit),
                media_type="text/event-stream"
            )
        else:
            resp = await nonstream_openai_response_from_poe(payload, model_context_limit=context_limit)
            return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=502, detail={"error": {"code": 502, "type": "upstream_error", "message": str(e), "metadata": {}}})

@app.get("/v1/models")
async def list_models_endpoint():
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": int(time.time()), "owned_by": "poe"}
            for mid in REGISTRY.list_ids()
        ]
    }

if __name__ == "__main__":
    logger.info("Starting Poe Proxy Server")
    if not os.getenv("POE_TOKEN") and not os.getenv("POE_API_KEY"):
        logger.error("FATAL: POE_TOKEN or POE_API_KEY environment variable not set. Exiting.")
        exit(1)
    else:
        logger.info(f"Available model ids: {REGISTRY.list_ids()}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=None)
