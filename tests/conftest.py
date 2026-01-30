# tests/conftest.py
import pytest
import pytest_asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure POE_API_KEY is set for tests
os.environ.setdefault("POE_API_KEY", "test_poe_token_123")
os.environ.setdefault("MODELS_FILE", "models.yaml")

@pytest.fixture
def mock_poe_token(monkeypatch):
    """Mocks the POE_API_KEY environment variable."""
    token = "test_poe_token_123"
    monkeypatch.setenv("POE_API_KEY", token)
    return token

@pytest.fixture
def sample_openai_messages_basic():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ]

@pytest.fixture
def sample_openai_messages_long():
    """Messages that will exceed typical context limits for testing trimming."""
    return [
        {"role": "system", "content": "You are a helpful assistant designed to summarize."},
        {"role": "user", "content": "This is a very long message that will definitely exceed the context limit of many models. " * 500},
        {"role": "assistant", "content": "I understand. " * 100},
        {"role": "user", "content": "More context. " * 300}
    ]

@pytest.fixture
def sample_openai_messages_very_long():
    """Very long conversation for testing aggressive trimming."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(50):
        messages.append({"role": "user", "content": f"User message {i}. " * 50})
        messages.append({"role": "assistant", "content": f"Assistant response {i}. " * 50})
    return messages

@pytest.fixture
def sample_openai_messages_with_images():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs="}}
        ]}
    ]

@pytest.fixture
def mock_poe_balance():
    """Mock for poe_get_current_balance."""
    async def _mock_balance(api_key: str):
        return 500000  # 500k points
    return _mock_balance

@pytest.fixture
def mock_poe_usage():
    """Mock for poe_get_latest_usage_entry."""
    async def _mock_usage(api_key: str):
        return {
            "cost_points": 1234,
            "model": "Claude-3.5-Sonnet",
            "timestamp": "2025-01-01T00:00:00Z"
        }
    return _mock_usage

@pytest_asyncio.fixture
async def mock_poe_api_responses(mocker):
    """
    Mocks httpx calls to Poe's OpenAI-compatible API.
    Returns a mock that can be configured per-test.
    """
    # Mock streaming response
    async def mock_stream_response(*args, **kwargs):
        """Default mock streaming response."""
        class MockStreamResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def aiter_lines(self):
                yield "data: " + '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}],"usage":null}'
                yield "data: " + '{"choices":[{"delta":{"content":" from"},"finish_reason":null}],"usage":null}'
                yield "data: " + '{"choices":[{"delta":{"content":" Poe!"},"finish_reason":"stop"}],"usage":{"total_tokens":20,"prompt_tokens":10,"completion_tokens":10}}'
                yield "data: [DONE]"

        return MockStreamResponse()

    # Mock non-streaming response
    async def mock_post_response(*args, **kwargs):
        """Default mock non-streaming response."""
        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": kwargs.get("json", {}).get("model", "test-model"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Poe!"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "total_tokens": 20,
                        "prompt_tokens": 10,
                        "completion_tokens": 10
                    }
                }

        return MockResponse()

    # Create mock client with configurable methods
    mock_client = MagicMock()
    mock_client.stream = AsyncMock(side_effect=mock_stream_response)
    mock_client.post = AsyncMock(side_effect=mock_post_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Patch httpx.AsyncClient
    with patch("httpx.AsyncClient", return_value=mock_client):
        yield mock_client
