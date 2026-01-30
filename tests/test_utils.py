# tests/test_utils.py
import pytest
from proxy import (
    get_poe_api_key,
    sanitize_messages,
    make_clear_error_message,
    format_sse_chunk,
    REGISTRY
)

def test_get_poe_api_key_env_set(mock_poe_token):
    """Test that POE_API_KEY can be retrieved."""
    get_poe_api_key.cache_clear()
    assert get_poe_api_key() == "test_poe_token_123"

def test_get_poe_api_key_env_not_set(monkeypatch):
    """Test that missing POE_API_KEY raises an error."""
    get_poe_api_key.cache_clear()
    monkeypatch.delenv("POE_API_KEY", raising=False)
    monkeypatch.delenv("POE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="POE_API_KEY or POE_TOKEN"):
        get_poe_api_key()

def test_sanitize_messages_handles_various_inputs():
    """Test sanitize_messages handles different input formats."""
    # Valid messages
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
        {"role": "assistant", "content": "Assistant"}
    ]
    result = sanitize_messages(messages)
    assert len(result) == 3
    assert all(msg["role"] in ["system", "user", "assistant"] for msg in result)

def test_sanitize_messages_handles_invalid_roles():
    """Test that invalid roles are converted to 'user'."""
    messages = [{"role": "invalid", "content": "Test"}]
    result = sanitize_messages(messages)
    assert result[0]["role"] == "user"

def test_sanitize_messages_handles_image_content():
    """Test that image content arrays are preserved."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]
    }]
    result = sanitize_messages(messages)
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "text"

def test_make_clear_error_message():
    """Test error message formatting."""
    # Test with known error code
    msg = make_clear_error_message(401, {"type": "auth_error", "code": 401, "message": "Invalid key"})
    assert "401" in msg
    assert "Invalid key" in msg
    assert "auth" in msg.lower()

    # Test with unknown error code
    msg = make_clear_error_message(999, None)
    assert "999" in msg

def test_format_sse_chunk():
    """Test SSE chunk formatting."""
    chunk = format_sse_chunk(
        data_dict={"content": "Hello"},
        request_id="test-123",
        model_name="test-model",
        finish_reason=None
    )
    assert "data: " in chunk
    assert "chatcmpl-test-123" in chunk
    assert "Hello" in chunk
    assert "test-model" in chunk

def test_format_sse_chunk_with_finish_reason():
    """Test SSE chunk with finish_reason."""
    chunk = format_sse_chunk(
        data_dict={},
        request_id="test-123",
        model_name="test-model",
        finish_reason="stop"
    )
    assert "data: " in chunk
    assert "stop" in chunk

def test_model_registry_loads_models():
    """Test that ModelRegistry loads models from YAML."""
    assert len(REGISTRY.list_ids()) > 0
    # Check some expected models exist
    models = REGISTRY.list_ids()
    assert any("Claude" in m or "GPT" in m or "Gemini" in m for m in models)

def test_model_registry_get_model():
    """Test getting a model from registry."""
    models = REGISTRY.list_ids()
    if models:
        model = REGISTRY.get(models[0])
        assert "poe_name" in model
        assert "modality" in model

def test_model_registry_get_nonexistent_model():
    """Test that getting a nonexistent model raises KeyError."""
    with pytest.raises(KeyError):
        REGISTRY.get("nonexistent-model-xyz-123")

def test_model_registry_context_limit():
    """Test that context_limit is loaded for models that have it."""
    # Check if Claude-3.5-Sonnet has context_limit set
    try:
        model = REGISTRY.get("Claude-3.5-Sonnet")
        # Should have context_limit set in models.yaml
        assert "context_limit" in model
        assert isinstance(model["context_limit"], int)
    except KeyError:
        # Model not in registry, skip this test
        pytest.skip("Claude-3.5-Sonnet not in registry")

def test_model_registry_default_max_tokens():
    """Test that default text_max_tokens is set."""
    default = REGISTRY.get_default_text_max_tokens()
    assert isinstance(default, int)
    assert default > 0
