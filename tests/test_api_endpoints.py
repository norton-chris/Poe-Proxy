# tests/test_api_endpoints.py
import pytest
import httpx # For making async HTTP requests
import json
from unittest.mock import patch, AsyncMock, MagicMock

# Assuming your FastAPI app instance is named 'app' in 'proxy.py'
# and can be imported. If not, these tests would need to run against
# a live server started separately, which is more complex for automated testing.
from proxy import app as fastapi_app
from proxy import MODEL_MAP, POINT_COSTS # For assertions
from fastapi.testclient import TestClient # Use TestClient for synchronous tests of an ASGI app

# Fixture to provide a TestClient instance
@pytest.fixture(scope="module")
def client():
    with TestClient(fastapi_app) as c:
        yield c

# Mock fp.get_bot_response for all tests in this file that hit /chat/completions
@pytest.fixture(autouse=True) # Apply to all tests in this module
def auto_mock_fp_get_bot_response(mocker):
    # This mock will be active for endpoint tests calling the chat completion logic
    mock_async_gen = AsyncMock() # This is the async generator itself

    # We need to make the mock_async_gen itself an async iterable
    # and control what it yields.
    # Default behavior: yield one response part.
    async def default_stream_behavior(*args, **kwargs):
        part = MagicMock(spec=fp.PartialResponse) # Assuming fp is imported in proxy
        part.text = "Mocked AI response. "
        yield part
        # Simulate the footer being added by discover_and_retry_bot_response
        # by having it yield another part that looks like a footer.
        # This is a simplification; the actual footer is built by the calling function.
        # For a more accurate test, the mock should only yield content parts,
        # and the test should verify the full response including the generated footer.
        # Let's assume discover_and_retry_bot_response will add its own footer.
        
    mock_async_gen.return_value = default_stream_behavior() # Return an actual async generator
    
    # Patch 'proxy.fp.get_bot_response' because that's where it's used by your endpoint logic
    # (via discover_and_retry_bot_response)
    active_patch = mocker.patch('proxy.fp.get_bot_response', new=mock_async_gen)
    return active_patch


def test_list_models_endpoint(client): # Use the TestClient
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == len(MODEL_MAP)
    assert data["data"][0]["id"] in MODEL_MAP

# For async endpoint testing with httpx, you'd typically run the server.
# But with TestClient, it handles running the app in a test mode.

def test_chat_completions_endpoint_success(client, mock_poe_token, auto_mock_fp_get_bot_response, sample_openai_messages_basic):
    # Configure the mock specifically for this test if needed
    # The autouse fixture already set up a default mock.
    # If you need it to yield specific things:
    async def specific_stream(*args, **kwargs):
        yield MagicMock(text="Specific test response. ", spec=fp.PartialResponse)
    auto_mock_fp_get_bot_response.return_value = specific_stream() # Reconfigure the mock

    payload = {
        "model": "claude-3.5-sonnet", # A model in your MODEL_MAP
        "messages": sample_openai_messages_basic
    }
    response = client.post("/v1/chat/completions", json=payload) # TestClient handles json

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "claude-3.5-sonnet"
    assert "choices" in data
    assert len(data["choices"]) == 1
    # The content will be "Specific test response. " + the footer from discover_and_retry
    assert "Specific test response. " in data["choices"][0]["message"]["content"]
    assert "📊 Tokens:" in data["choices"][0]["message"]["content"] # Footer check

def test_chat_completions_unsupported_model(client, mock_poe_token, sample_openai_messages_basic):
    payload = {
        "model": "super-advanced-model-9000",
        "messages": sample_openai_messages_basic
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400 # Or whatever your proxy returns
    data = response.json()
    assert "Unsupported model" in data["detail"]

def test_chat_completions_empty_messages(client, mock_poe_token):
    payload = {
        "model": "claude-3.5-sonnet",
        "messages": []
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400 # As per your proxy logic
    assert "No messages provided" in response.json()["detail"]


@pytest.mark.asyncio # Keep this if you were to use httpx with a live server
async def test_chat_completions_response_cutoff_simulation(
    client, mock_poe_token, auto_mock_fp_get_bot_response, sample_openai_messages_basic
):
    """
    Test that if Poe (mocked) sends an incomplete stream, the proxy
    still assembles what it received and includes its footer.
    The "cutoff" here means the mocked stream ends prematurely.
    """
    async def incomplete_stream(*args, **kwargs):
        yield MagicMock(text="This is the first part. ", spec=fp.PartialResponse)
        # No more parts, simulating a cutoff from Poe's stream
        # The discover_and_retry_bot_response should still add its footer.

    auto_mock_fp_get_bot_response.return_value = incomplete_stream()

    payload = {
        "model": "claude-3.5-sonnet",
        "messages": sample_openai_messages_basic
    }
    # Use TestClient for synchronous call from async test (if TestClient is sync)
    # Or use httpx.AsyncClient if testing a live server.
    # Since client is TestClient, the call is synchronous.
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    
    assert "This is the first part. " in content
    assert "📊 Tokens:" in content # Footer should still be there
    assert "💰 Points:" in content

    # To test if the "thinking part" is cut off:
    # Your mock would need to yield something like "Thinking..." and then stop.
    # Then assert that "Thinking..." is present but a "final" part is missing,
    # yet the footer is still there.

@pytest.mark.asyncio
async def test_chat_completions_thinking_then_cutoff(
    client, mock_poe_token, auto_mock_fp_get_bot_response, sample_openai_messages_basic
):
    async def thinking_stream(*args, **kwargs):
        yield MagicMock(text="Thinking... ", spec=fp.PartialResponse)
        # Stream ends here, no actual answer
    
    auto_mock_fp_get_bot_response.return_value = thinking_stream()

    payload = {
        "model": "claude-3.5-sonnet",
        "messages": sample_openai_messages_basic
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200 # It didn't error, just incomplete from Poe
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    assert "Thinking... " in content
    # Assert that a typical "answer" part is NOT in content if you expect one
    # e.g. if "Mocked AI response." was the usual, assert it's not there.
    assert "Mocked AI response." not in content # Assuming default mock yields this
    assert "📊 Tokens:" in content # Footer should still be there
    assert "💰 Points:" in content

@pytest.mark.asyncio
async def test_chat_completions_large_input_full_output(
    client, mock_poe_token, auto_mock_fp_get_bot_response_for_api, sample_openai_messages_very_long
):
    model_id = "claude-3.5-sonnet"
    expected_poe_response_parts_api = [f"API chunk {i}. " for i in range(30)]
    complete_expected_poe_text_api = "".join(expected_poe_response_parts_api)
    
    captured_messages_to_poe_holder = {} # Use a dict to pass by reference effectively

    async def mock_poe_stream_for_api_side_effect(*args, **kwargs):
        # Capture the messages that discover_and_retry_bot_response passes to fp.get_bot_response
        captured_messages_to_poe_holder['messages'] = kwargs.get('messages', args[0] if args else [])
        
        async def _gen(): # The actual async generator
            for part_text in expected_poe_response_parts_api:
                yield MagicMock(text=part_text, spec=fp.PartialResponse)
        return _gen() # Return the async generator object
            
    auto_mock_fp_get_bot_response_for_api.side_effect = mock_poe_stream_for_api_side_effect

    payload = { "model": model_id, "messages": sample_openai_messages_very_long }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    final_content = data["choices"][0]["message"]["content"]

    # 1. Verify input to Poe (captured by the mock's side_effect)
    assert 'messages' in captured_messages_to_poe_holder
    # Further assertions on captured_messages_to_poe_holder['messages'] can be added here,
    # comparing them to what enforce_context_limit would produce.

    # 2. Verify complete Poe response is in the final output
    assert final_content.startswith(complete_expected_poe_text_api)
    assert "📊 Tokens:" in final_content
    assert "💰 Points:" in final_content
