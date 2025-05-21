# tests/test_context_handling.py
import pytest
from proxy import enforce_context_limit, MODEL_CONTEXT_LIMITS, count_tokens

# Simplified MODEL_CONTEXT_LIMITS for testing, ensure 'test-model' is present
TEST_MODEL_CONTEXT_LIMITS = {
    "test-model": 100, # Tokens
    "test-model-large": 1000,
    "gpt-4": 100 # Override for testing
}
REAL_MODEL_CONTEXT_LIMITS = MODEL_CONTEXT_LIMITS.copy() # Keep original for other tests

@pytest.fixture(autouse=True)
def patch_model_context_limits(monkeypatch):
    # Patch the global MODEL_CONTEXT_LIMITS in your proxy module for these tests
    monkeypatch.setattr('proxy.MODEL_CONTEXT_LIMITS', TEST_MODEL_CONTEXT_LIMITS)
    yield
    monkeypatch.setattr('proxy.MODEL_CONTEXT_LIMITS', REAL_MODEL_CONTEXT_LIMITS) # Restore


def _calculate_total_tokens(messages, model):
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), model)
                elif part.get("type") == "image":
                    total += 512 # As per your code's approximation
        else:
            total += count_tokens(content, model)
    return total


def test_enforce_context_limit_no_trimming_needed():
    messages = [
        {"role": "system", "content": "System prompt"}, # ~3 tokens for gpt-4
        {"role": "user", "content": "User query"}      # ~2 tokens for gpt-4
    ]
    # MODEL_CONTEXT_LIMITS["test-model"] is 100 tokens
    trimmed = enforce_context_limit(messages, "test-model")
    assert len(trimmed) == 2
    assert trimmed == messages # No changes

def test_enforce_context_limit_trimming_user_messages():
    messages = [
        {"role": "system", "content": "System prompt " * 5}, # len 65 -> 16 tokens (fallback)
        {"role": "user", "content": "Old message " * 20},    # len 240 -> 60 tokens (fallback)
        {"role": "user", "content": "Newer message " * 20}   # len 300 -> 75 tokens (fallback)
    ]
    TEST_MODEL_CONTEXT_LIMITS['test-model'] = 60 # Limit
    
    trimmed = enforce_context_limit(messages, "test-model")
    assert len(trimmed) == 1 # Only system message should remain
    assert trimmed[0]["role"] == "system"
    assert _calculate_total_tokens(trimmed, "test-model") <= 60
    
    TEST_MODEL_CONTEXT_LIMITS['test-model'] = 100 # Reset

def test_enforce_context_limit_system_message_preserved():
    messages = [
        {"role": "system", "content": "Important System Instructions " * 10}, # ~30 tokens
        {"role": "user", "content": "User message that is very long and needs to be cut. " * 30} # ~90 tokens
    ]
    # Total ~120 tokens. Limit for 'test-model' is 100.
    # System should be kept, user message should be trimmed (or dropped if system alone is too big)
    trimmed = enforce_context_limit(messages, "test-model")
    assert len(trimmed) >= 1
    assert trimmed[0]["role"] == "system"
    assert _calculate_total_tokens(trimmed, "test-model") <= 100

def test_enforce_context_limit_with_image_tokens():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Short text"}, # ~2 tokens
            {"type": "image", "image_url": "data:..."} # Approx 512 tokens
        ]}
    ]
    # Total ~514 tokens.
    TEST_MODEL_CONTEXT_LIMITS['test-model'] = 500 # Set limit below image cost
    trimmed = enforce_context_limit(messages, "test-model")
    assert len(trimmed) == 0 # Image message alone exceeds limit

    TEST_MODEL_CONTEXT_LIMITS['test-model'] = 600 # Set limit above image cost
    trimmed = enforce_context_limit(messages, "test-model")
    assert len(trimmed) == 1
    assert _calculate_total_tokens(trimmed, "test-model") <= 600
    TEST_MODEL_CONTEXT_LIMITS['test-model'] = 100 # Reset
def test_enforce_context_limit_limit_override():
    messages = [{"role": "user", "content": "A very long message indeed " * 50}] # len 1350 -> 337 tokens
    # Model 'test-model' has limit 100 in TEST_MODEL_CONTEXT_LIMITS
    trimmed_default = enforce_context_limit(messages, "test-model")
    assert len(trimmed_default) == 0 # Message (337 tokens) exceeds limit (100)

    # Override limit to 50
    trimmed_override = enforce_context_limit(messages, "test-model", limit_override=50)
    assert len(trimmed_override) == 0 # Message (337 tokens) exceeds override limit (50)
    assert _calculate_total_tokens(trimmed_override, "test-model") <= 50
