import os
import json
import fastapi_poe as fp
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import uvicorn
import logging
import time
from typing import Optional, Dict, Any, Union
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
logging.getLogger("fastapi_poe").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

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
    "claude-3.5-sonnet": "claude-3.5-sonnet",
    "claude-3.7-sonnet": "claude-3.7-sonnet",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o-mini": "gpt-4o-mini",
    "o3-mini": "o3-mini",
    "o3-mini-high": "o3-mini-high",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-pro": "gemini-2.0-pro",
    "gemini-2.5-pro-exp": "gemini-2.5-pro-exp",
    "web-search": "web-search",
    "DeepSeek-R1": "DeepSeek-R1",
    "DeepSeek-V3-FW": "DeepSeek-V3-FW",
    "llama-3-70b-groq": "llama-3-70b-groq",
    "llama-3.1-405b": "llama-3.1-405b",
    "llama-4-maverick": "llama-4-maverick",
    "llama-4-scout": "llama-4-scout",
    "mistral-small-3": "mistral-small-3",
    "mistral-large-2": "mistral-large-2",
    "perplexity-sonar": "perplexity-sonar",
    "QwQ-32B-T": "QwQ-32B-T"
}

# Define context limits for different models (in characters)
MODEL_CONTEXT_LIMITS = {
    "claude-3.5-sonnet": 200000,
    "claude-3.7-sonnet": 200000,
    "gpt-4.1": 1000000,
    "gpt-4-mini": 128000,
    "o3-mini": 128000,
    "o3-mini-high": 128000,
    "gemini-1.5-pro": 10000,
    "gemini-2.0-flash": 1000000,
    "gemini-2.0-pro": 2000000,
    "gemini-2.5-pro-exp": 1000000,
    "web-search": 8000,
    "DeepSeek-R1": 164000,
    "DeepSeek-V3-FW": 131000,
    "llama-3-70b-groq": 2000,
    "llama-3.1-405b": 32000,
    "llama-4-scout": 1000000,
    "llama-4-scout": 10000000,
    "mistral-small-3": 32000,
    "mistral-large-2": 32000,
    "perplexity-sonar": 127000,
    "QwQ-32B-T": 131000
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
        if "gpt" in model.lower():
            encoder = tiktoken.encoding_for_model("gpt-4")
            return len(encoder.encode(text))
        return len(text) // 4
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return len(text) // 4

def calculate_points(model: str, total_tokens: int, has_images: bool = False) -> Dict[str, Union[int, str]]:
    if model not in POINT_COSTS:
        return {"points": "Unknown", "breakdown": "Model not found in cost structure"}
    
    cost = POINT_COSTS[model]
    base_points = cost["base"]
    token_points = (total_tokens / 1000) * cost["per_1k_tokens"]
    
    image_points = 0
    if has_images and "claude" in model:
        image_points = (total_tokens / 1000) * 100
    
    total_points = base_points + token_points + image_points
    
    breakdown = f"Base: {base_points}"
    if token_points > 0:
        breakdown += f", Tokens: {token_points:.1f}"
    if image_points > 0:
        breakdown += f", Images: {image_points:.1f}"
    
    return {
        "points": round(total_points, 1),
        "breakdown": breakdown
    }

def clean_content(content) -> str | list:
    # If content is a list (image format), return as is
    if isinstance(content, list):
        return content
        
    # Otherwise handle text content as before
    if isinstance(content, str):
        if "\n\n---\n📊" in content:
            return content.split("\n\n---\n📊")[0].strip()
        return content.strip()
    
    return str(content).strip()

async def discover_and_retry_bot_response(messages, bot_name, poe_token, max_retries=3):
    initial_limit = MODEL_CONTEXT_LIMITS.get(bot_name, 16000)
    current_limit = initial_limit
    original_tokens = sum(count_tokens(msg.get("content", ""), bot_name) for msg in messages)
    was_trimmed = False

    for attempt in range(max_retries):
        try:
            trimmed_messages = enforce_context_limit(messages, bot_name, limit_override=current_limit)
            was_trimmed = len(trimmed_messages) < len(messages)
            
            full_response = []
            received_chunks = False
            
            # Convert messages to Poe format while preserving image content
            poe_messages = []
            for msg in trimmed_messages:
                content = msg["content"]
                
                # Convert list content (images) to Poe's format for Claude
                if isinstance(content, list):
                    # Format specifically for Claude
                    formatted_content = []
                    for item in content:
                        if item.get("type") == "text":
                            formatted_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image":
                            formatted_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",  # or the actual image type
                                    "data": item.get("image_url", "").split("base64,")[1]
                                }
                            })
                    poe_msg = {
                        "role": "bot" if msg["role"] == "assistant" else msg["role"],
                        "content": formatted_content
                    }
                else:
                    poe_msg = {
                        "role": "bot" if msg["role"] == "assistant" else msg["role"],
                        "content": str(content)
                    }
                poe_messages.append(poe_msg)


            async for partial in fp.get_bot_response(
                messages=poe_messages,
                bot_name=bot_name,
                api_key=poe_token
            ):
                if hasattr(partial, 'text'):
                    if not any(marker in partial.text for marker in ['📊 Tokens:', '💰 Points:']):
                        full_response.append(partial.text)
                        received_chunks = True
                    yield partial 
            if received_chunks:
                complete_response = "".join(full_response)
                
                input_tokens = sum(count_tokens(msg.get("content", ""), bot_name) for msg in trimmed_messages)
                output_tokens = count_tokens(complete_response, bot_name)
                total_tokens = input_tokens + output_tokens
                
                # Check if any messages contain images
                has_images = any(
                    isinstance(msg.get("content"), list) and 
                    any(item.get("type") == "image" for item in msg.get("content", []))
                    for msg in trimmed_messages
                )
                
                points_info = calculate_points(bot_name, total_tokens, has_images=has_images)
                
                footer = f"\n\n---\n📊 Tokens: {total_tokens:,} (Input: {input_tokens:,}, Output: {output_tokens:,})\n💰 Points: {points_info['points']} ({points_info['breakdown']})"
                
                if was_trimmed:
                    footer += f"\n⚠️ Input was trimmed from {original_tokens:,} to {input_tokens:,} tokens due to model context limitations"
                
                yield type('PartialResponse', (), {
                    'text': footer,
                    'done': True
                })
                return

            logger.warning(f"Token length {current_limit} failed, trying shorter context")
            current_limit = current_limit // 2
            
            if current_limit < 2000:
                raise Exception(f"Token length discovery failed: could not find working length below {initial_limit:,} tokens")
            
            continue
        except Exception as e:
            logger.error(f"Error in bot response: {str(e)}")
            print("fail")

def enforce_context_limit(messages: list, model: str, limit_override: int = None) -> list:
    def calculate_message_tokens(message_content) -> int:
        if isinstance(message_content, list):
            # For image content, sum up tokens from text parts
            total = 0
            for part in message_content:
                if part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), model)
                # You might want to add a fixed token count for images
                elif part.get("type") == "image":
                    total += 512  # Approximate token cost for images
            return total
        return count_tokens(message_content, model)

    model_token_limit = limit_override if limit_override is not None else MODEL_CONTEXT_LIMITS.get(model, 32000)
    logger.debug(f"Enforcing token limit of {model_token_limit} for {model}")

    # Clean and calculate total tokens for all messages including system
    cleaned_messages = []
    total_tokens = 0
    for msg in messages:
        cleaned_msg = msg.copy()
        if "content" in cleaned_msg:
            cleaned_msg["content"] = clean_content(cleaned_msg["content"])
            msg_tokens = calculate_message_tokens(cleaned_msg["content"])
            total_tokens += msg_tokens
            logger.debug(f"{cleaned_msg['role']} message tokens: {msg_tokens}")
        cleaned_messages.append(cleaned_msg)
    
    logger.debug(f"Total tokens across all messages: {total_tokens}")

    if total_tokens <= model_token_limit:
        logger.debug("Content within token limits, no trimming needed")
        return cleaned_messages

    # Log warning about trimming
    logger.warning(f"Content exceeds token limit ({total_tokens} > {model_token_limit}), trimming...")

    # Separate system message and user messages
    system_msg = None
    user_msgs = []
    for msg in cleaned_messages:
        if msg.get("role") == "system":
            system_msg = msg
        else:
            user_msgs.append(msg)

    # Calculate available tokens after system message
    available_tokens = model_token_limit
    if system_msg:
        system_tokens = calculate_message_tokens(system_msg.get("content", ""))
        available_tokens -= system_tokens
        logger.debug(f"System message uses {system_tokens} tokens, {available_tokens} remaining")

    # Trim from oldest messages first until we fit
    trimmed_msgs = []
    current_tokens = 0
    
    # Process messages from newest to oldest
    for msg in reversed(user_msgs):
        msg_tokens = calculate_message_tokens(msg.get("content", ""))
        if current_tokens + msg_tokens <= available_tokens:
            trimmed_msgs.insert(0, msg)
            current_tokens += msg_tokens
            logger.debug(f"Keeping message with {msg_tokens} tokens, total now {current_tokens}")
        else:
            logger.debug(f"Skipping message with {msg_tokens} tokens as it would exceed limit")
            break

    # Add back system message if it exists
    if system_msg:
        trimmed_msgs.insert(0, system_msg)

    final_tokens = sum(calculate_message_tokens(msg.get("content", "")) for msg in trimmed_msgs)
    logger.debug(f"Final token count after trimming: {final_tokens}")
    
    return trimmed_msgs


async def retry_bot_response(messages, bot_name, poe_token, max_retries=3):
    for attempt in range(max_retries):
        try:
            async for partial in fp.get_bot_response(
                messages=messages,
                bot_name=bot_name,
                api_key=poe_token
            ):
                yield partial
            return  # Success, exit the function
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise the last error
            
            if "allow_retry" in str(e) and "true" in str(e).lower():
                logger.warning(f"Retrying API call (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(1)  # Wait before retry
            else:
                raise  # Not a retryable error

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, poe_token: str = Depends(get_poe_token)):
    DEBUG_TAGS = {
        "request_start": "🌍🗳️",
        "request_end": "🔚",
        "api_call": "📡",
        "chunk_receive": "📦",
        "error": "🔴",
        "warning": "⚠️",
        "cache": "💾",
        "rate_limit": "⏳"
    }

    # Initialize request state
    request.state.api_calls = 0
    request.state.chunks_received = 0
    request.state.start_time = time.time()
    
    try:
        logger.debug(f"{DEBUG_TAGS['request_start']} New Request | Client: {request.client.host}")

        # Rate Limiting logic...
        RATE_LIMIT = 10
        WINDOW_MINUTES = 1
        client_ip = request.client.host
        current_minute = int(time.time() // 60)

        if not hasattr(request.app.state, "rate_limits"):
            request.app.state.rate_limits = {}

        client_bucket = request.app.state.rate_limits.get(client_ip, {"count": 0, "minute": current_minute})
        
        if client_bucket["minute"] != current_minute:
            client_bucket = {"count": 0, "minute": current_minute}
            
        if client_bucket["count"] >= RATE_LIMIT:
            logger.warning(f"{DEBUG_TAGS['rate_limit']} Rate limited: {client_ip}")
            raise HTTPException(status_code=429, detail="Too many requests")
            
        client_bucket["count"] += 1
        request.app.state.rate_limits[client_ip] = client_bucket
        logger.debug(f"{DEBUG_TAGS['rate_limit']} Requests: {client_bucket['count']}/{RATE_LIMIT}")

        # Request processing
        request_body = await request.body()
        data = json.loads(request_body.decode())
        
        # Create a request hash based on relevant fields
        request_hash = hash(
            f"{data.get('model', '')}_"
            f"{str(data.get('messages', []))}_"
            f"{data.get('temperature', 1.0)}_"
            f"{data.get('max_tokens', 0)}"
        )

        # Cache checking
        CACHE_TTL = 60  # 1 minute
        if not hasattr(request.app.state, "request_cache"):
            request.app.state.request_cache = {}
            
        if cached := request.app.state.request_cache.get(request_hash):
            logger.debug(f"{DEBUG_TAGS['cache']} Serving cached response")
            return cached

        # Request Validation
        messages = data.get("messages", [])
        model = data.get("model", "claude-3-sonnet")
        
        messages = enforce_context_limit(messages, model)

        if not messages:
            logger.debug(f"{DEBUG_TAGS['warning']} Empty messages")
            raise HTTPException(status_code=400, detail="No messages provided")

        # Convert messages to Poe format, handling images
        poe_messages = []
        for msg in messages:
            content = msg.get("content", "")
            
            # Handle image content for Claude
            if isinstance(content, list):
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image":
                        # Get the base64 data from the image URL
                        image_url = item.get("image_url", "")
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            formatted_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",  # or the actual image type
                                    "data": base64_data
                                }
                            })
                
                poe_msg = {
                    "role": "bot" if msg.get("role") == "assistant" else msg.get("role", "user"),
                    "content": formatted_content
                }
            else:
                poe_msg = {
                    "role": "bot" if msg.get("role") == "assistant" else msg.get("role", "user"),
                    "content": str(content)
                }
            poe_messages.append(poe_msg)

        # 4. Model Handling
        bot_name = MODEL_MAP.get(model)
        if not bot_name:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

        # 5. API Call Execution
        try:
            logger.debug(f"{DEBUG_TAGS['api_call']} Starting API call to {bot_name}")
            request.state.api_calls = 1
            api_start = time.time()
            
            response_chunks = []
            async for partial in discover_and_retry_bot_response(
                messages=poe_messages,
                bot_name=bot_name,
                poe_token=poe_token
            ):
                if hasattr(partial, 'text') and partial.text:
                    request.state.chunks_received += 1
                    #logger.debug(f"{DEBUG_TAGS['chunk_receive']} Chunk {request.state.chunks_received}")
                    response_chunks.append(partial.text)    
                # Safety: Prevent runaways
                if time.time() - api_start > 60:
                    logger.error(f"{DEBUG_TAGS['warning']} API timeout")
                    break

            logger.debug(f"{DEBUG_TAGS['api_call']} Completed in {time.time()-api_start:.2f}s")
            
        except Exception as e:
            logger.error(f"{DEBUG_TAGS['error']} API failure: {str(e)}")
            raise HTTPException(status_code=502, detail="Model service unavailable")
        # 6. Response Processing
        if not response_chunks:
            logger.error(f"{DEBUG_TAGS['error']} Empty response")
            raise HTTPException(status_code=504, detail="Model timeout")

        #response_text = fix_duplicates("".join(response_chunks))
        response_text = "".join(response_chunks)

        # 7. Caching & Final Response
        response_data = {
            "id": f"chatcmpl-{uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "message": {"role": "bot", "content": response_text}
            }]
        }

        # Cache valid responses for 1 minute
        if len(response_text) > 10:  # Only cache substantial responses
            request.app.state.request_cache[request_hash] = response_data
            logger.debug(f"{DEBUG_TAGS['cache']} Cached response")

        return response_data

    except Exception as e:
        logger.error(f"{DEBUG_TAGS['error']} Request failed: {str(e)}")
        raise
    finally:
        duration = time.time() - request.state.start_time
        logger.debug(f"{DEBUG_TAGS['request_end']} Completed in {duration:.2f}s | "
                     f"Chunks: {request.state.chunks_received} | "
                     f"API Calls: {request.state.api_calls}")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "poe"
            }
            for model_id in MODEL_MAP.keys()
        ]
    }

if __name__ == "__main__":
    logger.info("Starting Poe Proxy Server")
    logger.info(f"Available models: {list(MODEL_MAP.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
