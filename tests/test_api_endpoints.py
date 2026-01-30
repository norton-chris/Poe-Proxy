# tests/test_api_endpoints.py
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app
from proxy import app as fastapi_app, REGISTRY

# Fixture to provide a TestClient instance
@pytest.fixture(scope="module")
def client():
    with TestClient(fastapi_app) as c:
        yield c

def test_list_models_endpoint(client):
    """Test GET /v1/models returns list of models from registry."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == len(REGISTRY.list_ids())
    assert data["data"][0]["id"] in REGISTRY.list_ids()
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "poe"

def test_chat_completions_endpoint_success_streaming(client, mock_poe_token):
    """Test POST /v1/chat/completions with streaming enabled."""
    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage:

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 1234}

        # Mock the Poe API streaming response
        class MockStreamResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
                yield 'data: {"choices":[{"delta":{"content":" from Poe!"},"finish_reason":"stop"}],"usage":{"total_tokens":20,"prompt_tokens":10,"completion_tokens":10}}'
                yield "data: [DONE]"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            # stream() should return the context manager directly, not a coroutine
            mock_client.stream.return_value = MockStreamResponse()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            payload = {
                "model": "Gemini-2.5-Flash",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True
            }
            response = client.post("/v1/chat/completions", json=payload)

            assert response.status_code == 200
            # For streaming, check that we get SSE data
            content = response.text
            assert "data:" in content

def test_chat_completions_endpoint_success_non_streaming(client, mock_poe_token):
    """Test POST /v1/chat/completions with streaming disabled."""
    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage:

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 1234}

        # Mock the Poe API non-streaming response
        async def mock_post_response(*args, **kwargs):
            class MockResponse:
                status_code = 200
                def json(self):
                    return {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "Gemini-2.5-Flash",
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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=mock_post_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            payload = {
                "model": "Gemini-2.5-Flash",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": False
            }
            response = client.post("/v1/chat/completions", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "Hello from Poe!" in data["choices"][0]["message"]["content"]
            # Footer should be appended
            assert "📊 Tokens:" in data["choices"][0]["message"]["content"]

def test_chat_completions_unsupported_model(client, mock_poe_token):
    """Test that requesting an unsupported model returns 404."""
    payload = {
        "model": "super-advanced-model-9000",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()

def test_chat_completions_missing_model(client, mock_poe_token):
    """Test that missing model field returns 400."""
    payload = {
        "messages": [{"role": "user", "content": "Hello!"}]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()

def test_chat_completions_balance_too_low(client, mock_poe_token):
    """Test that low balance returns a paused message."""
    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance:
        mock_balance.return_value = 5000  # Below default POINTS_CUTOFF of 10000

        payload = {
            "model": "Gemini-2.5-Flash",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "paused" in data["choices"][0]["message"]["content"].lower()
        assert "5000 points remaining" in data["choices"][0]["message"]["content"]

def test_chat_completions_rate_limit(client, mock_poe_token):
    """Test that rate limiting works."""
    # Make 11 requests quickly (limit is 10 per minute)
    payload = {
        "model": "Gemini-2.5-Flash",
        "messages": [{"role": "user", "content": "Hello!"}]
    }

    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance:
        mock_balance.return_value = 500000

        # Clear any existing rate limit state by accessing the app state
        # Reset rate limits for a fresh test
        if hasattr(client.app.state, "rate_limits"):
            client.app.state.rate_limits = {}

        # First 10 should work
        for i in range(10):
            response = client.post("/v1/chat/completions", json=payload)
            # May get errors from mocking but shouldn't be rate limited yet
            assert response.status_code != 429, f"Request {i+1} was rate limited unexpectedly"

        # 11th should be rate limited
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 429, f"Request 11 should have been rate limited but got {response.status_code}"

def test_chat_completions_sanitizes_messages(client, mock_poe_token):
    """Test that messages with invalid roles are sanitized."""
    # We'll test this by checking that the proxy doesn't crash when receiving invalid roles
    # The actual sanitization is tested in test_utils.py
    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage:

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 1234}

        # Reset rate limits for this test
        if hasattr(client.app.state, "rate_limits"):
            client.app.state.rate_limits = {}

        async def mock_post_response(*args, **kwargs):
            # Verify messages were sanitized
            sent_payload = kwargs.get("json", {})
            if "messages" in sent_payload:
                # All roles should be valid by the time they reach Poe
                for msg in sent_payload["messages"]:
                    assert msg["role"] in ["system", "user", "assistant", "tool"]

            class MockResponse:
                status_code = 200
                def json(self):
                    return {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "Gemini-2.5-Flash",
                        "choices": [{
                            "message": {"role": "assistant", "content": "OK"},
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

            payload = {
                "model": "Gemini-2.5-Flash",
                "messages": [
                    {"role": "invalid_role", "content": "This should become user role"},
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": False
            }
            response = client.post("/v1/chat/completions", json=payload)

            assert response.status_code == 200
            # The request succeeded, which means sanitization worked

def test_chat_completions_preserves_image_content(client, mock_poe_token, sample_openai_messages_with_images):
    """Test that image content in messages is preserved."""
    with patch("proxy.poe_get_current_balance", new_callable=AsyncMock) as mock_balance, \
         patch("proxy.poe_get_latest_usage_entry", new_callable=AsyncMock) as mock_usage:

        mock_balance.return_value = 500000
        mock_usage.return_value = {"cost_points": 1234}

        # Reset rate limits for this test
        if hasattr(client.app.state, "rate_limits"):
            client.app.state.rate_limits = {}

        async def mock_post_response(*args, **kwargs):
            # Verify image content was preserved
            sent_payload = kwargs.get("json", {})
            if "messages" in sent_payload and len(sent_payload["messages"]) > 0:
                msg = sent_payload["messages"][0]
                assert isinstance(msg["content"], list), "Image content should be a list"
                assert msg["content"][0]["type"] == "text"
                assert msg["content"][1]["type"] == "image_url"

            class MockResponse:
                status_code = 200
                def json(self):
                    return {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "Gemini-2.5-Flash",
                        "choices": [{
                            "message": {"role": "assistant", "content": "I see an image"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"total_tokens": 100, "prompt_tokens": 85, "completion_tokens": 15}
                    }
            return MockResponse()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=mock_post_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            payload = {
                "model": "Gemini-2.5-Flash",
                "messages": sample_openai_messages_with_images,
                "stream": False
            }
            response = client.post("/v1/chat/completions", json=payload)

            assert response.status_code == 200
            # The assertions in mock_post_response verify image content was preserved
