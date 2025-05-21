import os
import json
import fastapi_poe as fp
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse # Import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import uvicorn
import logging
import time
from typing import Optional, Dict, Any, Union, AsyncGenerator, List # Add List
from uuid import uuid4
import traceback
import tiktoken
from contextlib import asynccontextmanager
import httpx
from datetime import datetime
import asyncio

# Enhanced logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable detailed logging for external libraries
# logging.getLogger("fastapi_poe").setLevel(logging.DEBUG) # Can be noisy
# logging.getLogger("httpx").setLevel(logging.DEBUG) # Can be noisy

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POE_TOKEN = os.getenv("POE_TOKEN")

# Define cost structures
POINT_COSTS = {
    "gemini-1.5-pro": {"base": 175, "per_1k_tokens": 0},
    "gemini-2.0-flash": {"base": 20, "per_1k_tokens": 2},
    "gemini-2.0-pro": {"base": 1500, "per_1k_tokens": 2},
    "gemini-2.5-pro-exp": {"base": 261, "per_1k_tokens": 13},
    "claude-3.5-sonnet": {"base": 297, "per_1k_tokens": 115},
    "claude-3.7-sonnet": {"base": 807, "per_1k_tokens": 115},
    "gpt-4.1": {"base": 213, "per_1k_tokens": 67},
    "gpt-4o-mini": {"base": 15, "per_1k_tokens": 0},
    "o3-mini": {"base": 255, "per_1k_tokens": 37},
    "o3-mini-high": {"base": 628, "per_1k_tokens": 37},
    "web-search": {"base": 15, "per_1k_tokens": 0},
    "DeepSeek-R1": {"base": 600, "per_1k_tokens": 0},
    "DeepSeek-V3-FW": {"base": 300, "per_1k_tokens": 0},
    "llama-3-70b-groq": {"base": 75, "per_1k_tokens": 0},
    "llama-3.1-405b": {"base": 43, "per_1k_tokens": 100},
    "llama-4-maverick": {"base": 50, "per_1k_tokens": 0},
    "llama-4-scout": {"base": 30, "per_1k_tokens": 0},
    "mistral-small-3": {"base": 9, "per_1k_tokens": 4},
    "mistral-large-2": {"base": 204, "per_1k_tokens": 100},
    "perplexity-sonar": {"base": 317, "per_1k_tokens": 0},
    "QwQ-32B-T": {"base": 250, "per_1k_tokens": 0}
}

# Map of model names to Poe bot names
MODEL_MAP = {
    "claude-3.5-sonnet": "Claude-3.5-Sonnet", # Poe often uses capitalized names
    "claude-3.7-sonnet": "Claude-3-Opus", # Example, ensure correct Poe bot name
    "gpt-4.1": "GPT-4", # Example, ensure correct Poe bot name
    "gpt-4o-mini": "GPT-4o-mini",
    "o3-mini": "o3-mini",
    "o3-mini-high": "o3-mini-high",
    "gemini-1.5-pro": "Gemini-1.5-Pro",
    "gemini-2.0-flash": "Gemini-Flash",
    "gemini-2.0-pro": "Gemini-Pro",
    "gemini-2.5-pro-exp": "gemini-2.5-pro-exp",
    "web-search": "Web-Search",
    "DeepSeek-R1": "DeepSeek-R1",
    "DeepSeek-V3-FW": "DeepSeek-V3-FW",
    "llama-3-70b-groq": "Llama-3-70b-Groq",
    "llama-3.1-405b": "Meta-Llama-3.1-405B",
    "llama-4-maverick": "llama-4-maverick",
    "llama-4-scout": "llama-4-scout",
    "mistral-small-3": "Mistral-Small",
    "mistral-large-2": "Mistral-Large",
    "perplexity-sonar": "Perplexity-Sonar",
    "QwQ-32B-T": "QwQ-32B-T"
}

# Define context limits for different models (ideally in TOKENS, these are still chars)
# IMPORTANT: These should be actual TOKEN limits for the Poe bots, not character counts.
# The current values are placeholders and likely incorrect for token-based limits.
MODEL_CONTEXT_LIMITS = {
    "Claude-3.5-Sonnet": 200000, # This is likely chars, find token limit
    "Claude-3-Opus": 200000,    # This is likely chars, find token limit
    "GPT-4": 128000,             # This is likely chars, find token limit for Poe's version
    "GPT-4o-mini": 128000,
    "o3-mini": 128000,
    "o3-mini-high": 128000,
    "Gemini-1.5-Pro": 1000000,   # Example token limit
    "Gemini-Flash": 1000000,
    "Gemini-Pro": 1000000,
    "gemini-2.5-pro-exp": 1000000,
    "Web-Search": 8000,          # Likely chars
    "DeepSeek-R1": 128000,       # Example token limit
    "DeepSeek-V3-FW": 128000,    # Example token limit (Poe's might be 32k or 128k)
    "Llama-3-70b-Groq": 8000,
    "Meta-Llama-3.1-405B": 128000,
    "llama-4-maverick": 1000000,
    "llama-4-scout": 10000000,
    "Mistral-Small": 32000,
    "Mistral-Large": 32000,
    "Perplexity-Sonar": 16000,
    "QwQ-32B-T": 32000
}

@lru_cache()
def get_poe_token():
    token = os.getenv("POE_TOKEN")
    if not token:
        raise ValueError("POE_TOKEN environment variable not set")
    return token

def count_tokens(text: str, model: str) -> int:
    """Estimate token count based on model"""
    try:
        # Use tiktoken for OpenAI models or models known to use similar tokenization
        if "gpt" in model.lower() or "claude" in model.lower() or "gemini" in model.lower(): # Add other knowns
            encoder = tiktoken.encoding_for_model("gpt-4") # A common default
            return len(encoder.encode(text))
        # Fallback for other models (very rough estimate)
        return len(text) // 4
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model} with tiktoken, falling back: {e}")
        return len(text) // 4 # Fallback

def calculate_points(model_key: str, total_tokens: int, has_images: bool = False) -> Dict[str, Union[int, str]]:
    # Use the original OpenAI model key to look up costs, as POINT_COSTS uses these keys
    if model_key not in POINT_COSTS:
        return {"points": "Unknown", "breakdown": f"Model key '{model_key}' not found in cost structure"}
    
    cost = POINT_COSTS[model_key]
    base_points = cost["base"]
    token_points = (total_tokens / 1000) * cost["per_1k_tokens"]
    
    image_points = 0
    # Ensure 'claude' check is against the original OpenAI model key if that's how costs are defined
    if has_images and "claude" in model_key.lower(): # Or use Poe bot name if costs are tied to that
        image_points = (total_tokens / 1000) * 100 # Example image point cost
    
    total_points_val = base_points + token_points + image_points
    
    breakdown = f"Base: {base_points}"
    if token_points > 0:
        breakdown += f", Tokens: {token_points:.1f}"
    if image_points > 0:
        breakdown += f", Images: {image_points:.1f}"
    
    return {
        "points": round(total_points_val, 1),
        "breakdown": breakdown
    }

def clean_content(content) -> Union[str, list]:
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        if "\n\n---\n📊" in content: # Your custom footer marker
            return content.split("\n\n---\n📊")[0].strip()
        return content.strip()
    return str(content).strip()

def enforce_context_limit(messages: List[Dict[str, Any]], poe_bot_name: str, limit_override: Optional[int] = None) -> List[Dict[str, Any]]:
    # Ensure count_tokens uses poe_bot_name if its logic depends on the specific bot
    # For now, count_tokens takes a generic model string, which might map to tiktoken or fallback.
    
    def calculate_message_tokens(message_content) -> int:
        if isinstance(message_content, list):
            total = 0
            for part in message_content:
                if part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), poe_bot_name) # Pass poe_bot_name
                elif part.get("type") == "image":
                    total += 512 
            return total
        return count_tokens(str(message_content), poe_bot_name) # Pass poe_bot_name

    # Use poe_bot_name to get the limit
    model_token_limit = limit_override if limit_override is not None else MODEL_CONTEXT_LIMITS.get(poe_bot_name, 32000) # Default if not found
    logger.debug(f"Enforcing token limit of {model_token_limit} for Poe bot: {poe_bot_name}")

    cleaned_messages: List[Dict[str, Any]] = []
    total_tokens = 0
    for msg in messages:
        cleaned_msg = msg.copy()
        if "content" in cleaned_msg:
            # clean_content might not be necessary here if we're just counting tokens for truncation
            # cleaned_msg["content"] = clean_content(cleaned_msg["content"]) # clean_content might alter it before token counting
            msg_tokens = calculate_message_tokens(cleaned_msg["content"])
            total_tokens += msg_tokens
            logger.debug(f"{cleaned_msg.get('role')} message tokens: {msg_tokens} (for {poe_bot_name})")
        cleaned_messages.append(cleaned_msg)
    
    logger.debug(f"Total tokens across all messages for {poe_bot_name}: {total_tokens}")

    if total_tokens <= model_token_limit:
        logger.debug(f"Content for {poe_bot_name} within token limits, no trimming needed.")
        return cleaned_messages

    logger.warning(f"Content for {poe_bot_name} exceeds token limit ({total_tokens} > {model_token_limit}), trimming...")
    system_msg = None
    user_msgs = []
    for msg in cleaned_messages:
        if msg.get("role") == "system":
            system_msg = msg
        else:
            user_msgs.append(msg)

    available_tokens = model_token_limit
    if system_msg:
        system_tokens = calculate_message_tokens(system_msg.get("content", ""))
        if system_tokens > available_tokens: # System prompt alone is too large
            logger.warning(f"System prompt for {poe_bot_name} ({system_tokens} tokens) exceeds limit ({available_tokens}). Returning empty.")
            return [] 
        available_tokens -= system_tokens
    
    trimmed_user_msgs: List[Dict[str, Any]] = []
    current_tokens = 0
    for msg in reversed(user_msgs):
        msg_tokens = calculate_message_tokens(msg.get("content", ""))
        if current_tokens + msg_tokens <= available_tokens:
            trimmed_user_msgs.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break
            
    final_trimmed_messages: List[Dict[str, Any]] = []
    if system_msg:
        final_trimmed_messages.append(system_msg)
    final_trimmed_messages.extend(trimmed_user_msgs)
    
    final_tokens = sum(calculate_message_tokens(msg.get("content", "")) for msg in final_trimmed_messages)
    logger.debug(f"Final token count for {poe_bot_name} after trimming: {final_tokens}")
    return final_trimmed_messages

async def discover_and_retry_bot_response(
    original_messages: List[Dict[str, Any]], # These are OpenAI format messages
    poe_bot_name: str,
    poe_token: str,
    openai_model_key: str, # For cost calculation
    max_retries: int = 3
) -> AsyncGenerator[Dict[str, Any], None]: # Yields dicts for easier processing
    """
    Handles context trimming, calling Poe, and yielding response parts.
    The footer is now a special dictionary yielded at the end.
    """
    initial_limit = MODEL_CONTEXT_LIMITS.get(poe_bot_name, 16000) # Default limit if bot not in map
    current_limit_override = initial_limit
    
    # Calculate original tokens based on the original_messages for the warning message
    original_total_tokens = sum(count_tokens(str(msg.get("content", "")), poe_bot_name) for msg in original_messages)
    
    for attempt in range(max_retries):
        logger.debug(f"Attempt {attempt + 1}/{max_retries} for {poe_bot_name} with context limit override: {current_limit_override}")
        
        # Enforce context limit on the original messages for this attempt
        messages_for_this_attempt_openai_fmt = enforce_context_limit(original_messages, poe_bot_name, limit_override=current_limit_override)
        
        if not messages_for_this_attempt_openai_fmt:
            logger.warning(f"No messages left after context enforcement for {poe_bot_name} on attempt {attempt + 1}. Skipping Poe call.")
            # Yield a special message or break, then yield footer
            yield {"type": "content", "text": "[Error: No content after context trimming]"}
            break 

        was_trimmed_for_this_attempt = len(messages_for_this_attempt_openai_fmt) < len(original_messages) or \
                                       sum(count_tokens(str(m.get("content","")), poe_bot_name) for m in messages_for_this_attempt_openai_fmt) < original_total_tokens


        # Convert to Poe message format
        poe_protocol_messages = []
        for msg in messages_for_this_attempt_openai_fmt:
            content = msg.get("content", "")
            if isinstance(content, list): # Your image handling
                formatted_content_list = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content_list.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image" and "image_url" in item:
                        image_url = item["image_url"]
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            formatted_content_list.append({
                                "type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_data}
                            })
                current_content = formatted_content_list
            else:
                current_content = str(content)
            poe_protocol_messages.append({
                "role": "bot" if msg.get("role") == "assistant" else msg.get("role", "user"),
                "content": current_content
            })

        full_response_parts = []
        received_chunks_this_attempt = False
        api_call_start_time = time.time()

        try:
            async for partial in fp.get_bot_response(
                messages=poe_protocol_messages,
                bot_name=poe_bot_name,
                api_key=poe_token
            ):
                if time.time() - api_call_start_time > 60: # 60s timeout for the call itself
                    logger.warning(f"Poe API call timeout for {poe_bot_name} during attempt {attempt + 1}")
                    yield {"type": "error", "text": "[Poe API call timeout]"}
                    break # Break from inner async for, will go to next retry if any

                if hasattr(partial, 'text') and partial.text:
                    # Don't yield internal markers like footer markers from Poe itself (if any)
                    if not any(marker in partial.text for marker in ['📊 Tokens:', '💰 Points:']):
                        full_response_parts.append(partial.text)
                        yield {"type": "content", "text": partial.text}
                        received_chunks_this_attempt = True
            
            if received_chunks_this_attempt: # Successful attempt
                logger.info(f"Successfully received response from {poe_bot_name} on attempt {attempt + 1}")
                # Prepare and yield footer data
                complete_response_text = "".join(full_response_parts)
                input_tokens = sum(count_tokens(str(msg.get("content", "")), poe_bot_name) for msg in messages_for_this_attempt_openai_fmt)
                output_tokens = count_tokens(complete_response_text, poe_bot_name)
                total_tokens = input_tokens + output_tokens
                
                has_images = any(isinstance(msg.get("content"), list) and any(item.get("type") == "image" for item in msg.get("content",[])) for msg in messages_for_this_attempt_openai_fmt)
                points_info = calculate_points(openai_model_key, total_tokens, has_images=has_images) # Use openai_model_key for costs

                footer_data_dict = {
                    "tokens_input": input_tokens,
                    "tokens_output": output_tokens,
                    "tokens_total": total_tokens,
                    "points": points_info['points'],
                    "points_breakdown": points_info['breakdown'],
                    "was_trimmed": was_trimmed_for_this_attempt,
                    "original_total_tokens": original_total_tokens if was_trimmed_for_this_attempt else None
                }
                yield {"type": "footer", "data": footer_data_dict}
                return # Success, exit generator

            # If no chunks received and no exception, it means Poe stream ended cleanly but empty
            logger.warning(f"No chunks received from {poe_bot_name} on attempt {attempt + 1} with limit {current_limit_override}. Trying smaller context.")
            current_limit_override //= 2
            if current_limit_override < 2000: # Arbitrary minimum
                logger.error(f"Context limit for {poe_bot_name} too small after retries ({current_limit_override}). Giving up.")
                yield {"type": "error", "text": "[Context limit too small after retries]"}
                break # Break from outer for loop

        except Exception as e:
            logger.error(f"Error in bot response from {poe_bot_name} on attempt {attempt + 1}: {str(e)}")
            traceback.print_exc()
            if attempt == max_retries - 1:
                yield {"type": "error", "text": f"[Poe API Error: {str(e)}]"}
                raise # Re-raise on last attempt
            current_limit_override //= 2 # Reduce context for next retry
            if current_limit_override < 2000:
                logger.error(f"Context limit for {poe_bot_name} too small after error ({current_limit_override}). Giving up.")
                yield {"type": "error", "text": "[Context limit too small after error]"}
                break
            logger.info(f"Retrying for {poe_bot_name} with new context limit override: {current_limit_override}")
            # Continue to next attempt

    # If all retries failed to get chunks
    yield {"type": "footer", "data": {"error": "Failed to get response after all retries."}}


# Helper function to format SSE data
def format_sse_chunk(data_dict: Dict, request_id: str, model_name: str, finish_reason: Optional[str] = None) -> str:
    sse_payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": data_dict, # e.g., {"content": "text"} or {} for done
            "finish_reason": finish_reason
        }]
    }
    json_data = json.dumps(sse_payload)
    return f"data: {json_data}\n\n"

async def stream_openai_sse_response(
    original_messages: List[Dict[str, Any]], # OpenAI format
    poe_bot_name: str,
    poe_token: str,
    openai_model_key: str, # For SSE "model" field and cost calculation
    request_id: str
) -> AsyncGenerator[str, None]:
    """
    Streams response from discover_and_retry_bot_response, formats as OpenAI SSE.
    """
    async for item in discover_and_retry_bot_response(
        original_messages=original_messages,
        poe_bot_name=poe_bot_name,
        poe_token=poe_token,
        openai_model_key=openai_model_key # Pass this along
    ):
        if item["type"] == "content":
            yield format_sse_chunk(data_dict={"content": item["text"]}, request_id=request_id, model_name=openai_model_key)
        elif item["type"] == "footer":
            footer_text = f"\n\n---\n📊 Tokens: {item['data'].get('tokens_total', 'N/A'):,} (Input: {item['data'].get('tokens_input', 'N/A'):,}, Output: {item['data'].get('tokens_output', 'N/A'):,})\n💰 Points: {item['data'].get('points', 'N/A')} ({item['data'].get('points_breakdown', 'N/A')})"
            if item["data"].get("was_trimmed"):
                footer_text += f"\n⚠️ Input was trimmed from {item['data'].get('original_total_tokens', 'N/A'):,} to {item['data'].get('tokens_input', 'N/A'):,} tokens due to model context limitations"
            elif item["data"].get("error"):
                 footer_text = f"\n\n---\nPROXY INFO: {item['data']['error']}"

            yield format_sse_chunk(data_dict={"content": footer_text}, request_id=request_id, model_name=openai_model_key)
            # After footer, send the done signal
            yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=openai_model_key, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return # End stream after footer
        elif item["type"] == "error":
            yield format_sse_chunk(data_dict={"content": f"\n\n[PROXY ERROR: {item['text']}]"}, request_id=request_id, model_name=openai_model_key, finish_reason="error")
            # After error, send the done signal
            yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=openai_model_key, finish_reason="error")
            yield "data: [DONE]\n\n"
            return # End stream after error
    
    # Fallback DONE signal if the loop completes without footer/error (shouldn't happen with current logic)
    logger.debug("Stream ended without explicit footer or error yield, sending DONE.")
    yield format_sse_chunk(data_dict={}, request_id=request_id, model_name=openai_model_key, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, poe_token: str = Depends(get_poe_token)):
    DEBUG_TAGS = {
        "request_start": "🌍🗳️", "request_end": "🔚", "api_call": "📡",
        "chunk_receive": "📦", "error": "🔴", "warning": "⚠️",
        "cache": "💾", "rate_limit": "⏳"
    }
    request.state.api_calls = 0
    request.state.chunks_received = 0 # For non-streaming chunk count
    request.state.start_time = time.time()
    
    try:
        logger.debug(f"{DEBUG_TAGS['request_start']} New Request | Client: {request.client.host}")
        # ... (Rate Limiting logic - unchanged) ...
        RATE_LIMIT = 10; WINDOW_MINUTES = 1; client_ip = request.client.host; current_minute = int(time.time() // 60)
        if not hasattr(request.app.state, "rate_limits"): request.app.state.rate_limits = {}
        client_bucket = request.app.state.rate_limits.get(client_ip, {"count": 0, "minute": current_minute})
        if client_bucket["minute"] != current_minute: client_bucket = {"count": 0, "minute": current_minute}
        if client_bucket["count"] >= RATE_LIMIT:
            logger.warning(f"{DEBUG_TAGS['rate_limit']} Rate limited: {client_ip}")
            raise HTTPException(status_code=429, detail="Too many requests")
        client_bucket["count"] += 1; request.app.state.rate_limits[client_ip] = client_bucket
        logger.debug(f"{DEBUG_TAGS['rate_limit']} Requests: {client_bucket['count']}/{RATE_LIMIT}")

        request_body = await request.body()
        data = json.loads(request_body.decode())
        
        is_streaming_request = data.get("stream", False)
        openai_model_key = data.get("model", "claude-3.5-sonnet") # Original key from request
        
        request_hash_content = (
            f"{openai_model_key}_"
            f"{str(data.get('messages', []))}_" # Ensure messages are part of hash
            f"{data.get('temperature', 1.0)}_"
            f"{data.get('max_tokens', 0)}" # Or other relevant params
        )
        request_hash = hash(request_hash_content)

        if not is_streaming_request: # Only use cache for non-streaming
            if not hasattr(request.app.state, "request_cache"): request.app.state.request_cache = {}
            if cached := request.app.state.request_cache.get(request_hash):
                logger.debug(f"{DEBUG_TAGS['cache']} Serving cached non-streaming response")
                return cached

        original_openai_messages = data.get("messages", [])
        
        poe_bot_name = MODEL_MAP.get(openai_model_key)
        if not poe_bot_name:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {openai_model_key}")

        # Initial context enforcement is now inside discover_and_retry_bot_response
        # and stream_openai_sse_response will call it.
        # No, enforce_context_limit should be called before passing to these generators
        # as they expect OpenAI format messages.
        # However, discover_and_retry_bot_response now handles its own enforcement internally.

        request_id = str(uuid4())

        if is_streaming_request:
            logger.debug(f"{DEBUG_TAGS['api_call']} Starting STREAMING API call to Poe bot: {poe_bot_name} (for OpenAI model: {openai_model_key})")
            return StreamingResponse(
                stream_openai_sse_response(
                    original_messages=original_openai_messages, # Pass original OpenAI messages
                    poe_bot_name=poe_bot_name,
                    poe_token=poe_token,
                    openai_model_key=openai_model_key,
                    request_id=request_id
                ),
                media_type="text/event-stream"
            )
        else: # Non-streaming request
            logger.debug(f"{DEBUG_TAGS['api_call']} Starting NON-STREAMING API call to Poe bot: {poe_bot_name} (for OpenAI model: {openai_model_key})")
            request.state.api_calls = 1
            api_start_non_stream = time.time()
            
            collected_content_parts = []
            final_footer_data = None

            try:
                async for item in discover_and_retry_bot_response(
                    original_messages=original_openai_messages, # Pass original OpenAI messages
                    poe_bot_name=poe_bot_name,
                    poe_token=poe_token,
                    openai_model_key=openai_model_key
                ):
                    request.state.chunks_received +=1 # Count all items yielded by discover_and_retry
                    if item["type"] == "content":
                        collected_content_parts.append(item["text"])
                    elif item["type"] == "footer":
                        final_footer_data = item["data"]
                        break # Footer is the last thing for non-streaming
                    elif item["type"] == "error":
                        # For non-streaming, we might want to include this error in the response
                        # or raise an HTTPException. Let's include it in content.
                        collected_content_parts.append(f"\n\n[PROXY ERROR: {item['text']}]")
                        # final_footer_data might still be None or from a previous attempt if error on retry
                        if final_footer_data is None: # Ensure footer data exists
                             final_footer_data = {"error_message": item['text']}
                        break
                
                api_duration_non_stream = time.time() - api_start_non_stream
                logger.debug(f"{DEBUG_TAGS['api_call']} Non-streaming call processing completed in {api_duration_non_stream:.2f}s")

            except Exception as e:
                logger.error(f"{DEBUG_TAGS['error']} API failure (non-streaming): {str(e)}")
                raise HTTPException(status_code=502, detail=f"Model service unavailable: {str(e)}")

            if not collected_content_parts and not final_footer_data: # Should not happen if discover_and_retry yields error/footer
                logger.error(f"{DEBUG_TAGS['error']} Empty response (non-streaming) from discover_and_retry")
                raise HTTPException(status_code=504, detail="Model timeout or empty response from internal processing")

            final_response_text = "".join(collected_content_parts)
            
            # Append footer to text for non-streaming
            if final_footer_data:
                if final_footer_data.get("error"): # Error from discover_and_retry_bot_response itself
                     final_response_text += f"\n\n---\nPROXY INFO: {final_footer_data['error']}"
                elif "tokens_total" in final_footer_data: # Normal footer
                    footer_text = f"\n\n---\n📊 Tokens: {final_footer_data.get('tokens_total', 'N/A'):,} (Input: {final_footer_data.get('tokens_input', 'N/A'):,}, Output: {final_footer_data.get('tokens_output', 'N/A'):,})\n💰 Points: {final_footer_data.get('points', 'N/A')} ({final_footer_data.get('points_breakdown', 'N/A')})"
                    if final_footer_data.get("was_trimmed"):
                        footer_text += f"\n⚠️ Input was trimmed from {final_footer_data.get('original_total_tokens', 'N/A'):,} to {final_footer_data.get('tokens_input', 'N/A'):,} tokens due to model context limitations"
                    final_response_text += footer_text
                elif final_footer_data.get("error_message"): # Error from Poe API call
                    # Already appended to collected_content_parts
                    pass


            response_data = {
                "id": f"chatcmpl-{request_id}", "object": "chat.completion", "created": int(time.time()),
                "model": openai_model_key,
                "choices": [{"message": {"role": "bot", "content": final_response_text}}]
            }
            
            if len(final_response_text) > 10: # Only cache substantial responses
                if not hasattr(request.app.state, "request_cache"): request.app.state.request_cache = {} # Ensure init
                request.app.state.request_cache[request_hash] = response_data
                logger.debug(f"{DEBUG_TAGS['cache']} Cached non-streaming response")
            
            duration = time.time() - request.state.start_time
            logger.debug(f"{DEBUG_TAGS['request_end']} Non-streaming completed in {duration:.2f}s | Items from discover: {request.state.chunks_received} | API Calls: {request.state.api_calls}")
            return response_data

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"{DEBUG_TAGS['error']} Unhandled error in chat_completions: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/models")
async def list_models_endpoint(): # Renamed to avoid conflict if you import list_models from elsewhere
    return {
        "object": "list",
        "data": [
            {
                "id": model_id, # This is the OpenAI model key
                "object": "model",
                "created": int(time.time()),
                "owned_by": "poe"
            }
            for model_id in MODEL_MAP.keys() # Iterate over OpenAI model keys
        ]
    }

if __name__ == "__main__":
    logger.info("Starting Poe Proxy Server")
    # Ensure POE_TOKEN is set before starting
    if not os.getenv("POE_TOKEN"):
        logger.error("FATAL: POE_TOKEN environment variable not set. Exiting.")
        exit(1)
    else:
        logger.info(f"POE_TOKEN is set. Available OpenAI model keys: {list(MODEL_MAP.keys())}")
    
    # For Uvicorn, set log_config=None to use the logging configured by logging.basicConfig
    # Or pass a Uvicorn log config dict. Default Uvicorn logging can be quite different.
    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=None)
