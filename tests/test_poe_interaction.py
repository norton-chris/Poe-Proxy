# tests/test_poe_interaction.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from proxy import app as fastapi_app, REGISTRY

@pytest.fixture(scope="module")
def client():
    with TestClient(fastapi_app) as c:
        yield c

@pytest.mark.asyncio
async def test_poe_get_current_balance_success():
    """Test fetching current balance from Poe Usage API."""
    from proxy import poe_get_current_balance

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current_point_balance": 500000}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        balance = await poe_get_current_balance("test_api_key")
        assert balance == 500000

@pytest.mark.asyncio
async def test_poe_get_current_balance_failure():
    """Test handling of balance fetch failure."""
    from proxy import poe_get_current_balance

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        balance = await poe_get_current_balance("invalid_key")
        assert balance is None

@pytest.mark.asyncio
async def test_poe_get_latest_usage_entry_success():
    """Test fetching latest usage entry from Poe Usage API."""
    from proxy import poe_get_latest_usage_entry

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"cost_points": 1234, "model": "Claude-3.5-Sonnet", "timestamp": "2025-01-01T00:00:00Z"}
            ]
        }
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        entry = await poe_get_latest_usage_entry("test_api_key")
        assert entry is not None
        assert entry["cost_points"] == 1234
        assert entry["model"] == "Claude-3.5-Sonnet"

@pytest.mark.asyncio
async def test_poe_get_latest_usage_entry_empty():
    """Test handling of empty usage history."""
    from proxy import poe_get_latest_usage_entry

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        entry = await poe_get_latest_usage_entry("test_api_key")
        assert entry is None

@pytest.mark.asyncio
async def test_call_poe_openai_compatible_streaming():
    """Test streaming call to Poe's OpenAI-compatible API."""
    from proxy import call_poe_openai_compatible

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":10,"prompt_tokens":5,"completion_tokens":5}}'
            yield "data: [DONE]"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        # stream() should return the context manager directly
        mock_client.stream.return_value = MockStreamResponse()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        gen = await call_poe_openai_compatible(payload, stream=True)

        events = []
        async for event in gen:
            events.append(event)

        assert len(events) > 0
        assert any(event.get("type") == "chunk" for event in events)
        assert any(event.get("type") == "done" for event in events)

@pytest.mark.asyncio
async def test_call_poe_openai_compatible_non_streaming():
    """Test non-streaming call to Poe's OpenAI-compatible API."""
    from proxy import call_poe_openai_compatible

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }

    async def mock_post_response(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Poe!"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "total_tokens": 10,
                        "prompt_tokens": 5,
                        "completion_tokens": 5
                    }
                }
        return MockResponse()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=mock_post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await call_poe_openai_compatible(payload, stream=False)

        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Hello from Poe!"

@pytest.mark.asyncio
async def test_call_poe_openai_compatible_retry_on_400():
    """Test retry logic for 400 errors."""
    from proxy import call_poe_openai_compatible

    payload = {
        "model": "Test-Model",  # Capital letters
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1000,
        "stream": False
    }

    call_count = 0

    async def mock_post_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        class MockResponse:
            status_code = 400 if call_count < 3 else 200
            def json(self):
                if self.status_code == 400:
                    return {"error": {"code": 400, "message": "Invalid request"}}
                return {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "message": {"role": "assistant", "content": "Success!"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
                }
        return MockResponse()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=mock_post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await call_poe_openai_compatible(payload, stream=False)

        # Should have retried and eventually succeeded
        assert call_count >= 2  # At least one retry
        assert result["choices"][0]["message"]["content"] == "Success!"

@pytest.mark.asyncio
async def test_append_footer_with_usage_and_balance():
    """Test footer generation with usage and balance."""
    from proxy import append_footer_with_usage_and_balance

    usage_totals = {
        "total_tokens": 1000,
        "prompt_tokens": 800,
        "completion_tokens": 200
    }

    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage, \
         patch("asyncio.sleep", new_callable=AsyncMock):  # Skip the delay

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 1234}

        footer = await append_footer_with_usage_and_balance("test-model", usage_totals)

        assert "📊 Tokens:" in footer
        assert "1000" in footer  # Total tokens
        assert "800" in footer   # Input tokens
        assert "200" in footer   # Output tokens
        assert "💰 Points: 1234" in footer
        assert "🏦 Current balance:" in footer
        assert "500000" in footer  # Balance

@pytest.mark.asyncio
async def test_append_footer_with_high_cost_warning():
    """Test that footer includes warning for high-cost messages."""
    from proxy import append_footer_with_usage_and_balance

    usage_totals = {
        "total_tokens": 10000,
        "prompt_tokens": 9000,
        "completion_tokens": 1000
    }

    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage, \
         patch("asyncio.sleep", new_callable=AsyncMock):

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 10000}  # High cost

        footer = await append_footer_with_usage_and_balance("GPT-4.1", usage_totals)

        assert "⚠️ High cost detected" in footer
        assert "cheaper model" in footer.lower()
