# tests/conftest.py
import pytest
import pytest_asyncio # Ensure this is imported if using older pytest-asyncio
import os
from unittest.mock import AsyncMock, MagicMock

# If your proxy.py is not directly importable due to how it's structured
# for uvicorn.run, you might need to adjust imports or refactor proxy.py
# slightly to separate the FastAPI app instance.
# For now, we'll assume functions from proxy.py can be imported.
# If not, you'd test the running app via HTTP requests primarily.

# To make functions from proxy.py available for testing:
# One way is to ensure proxy.py can be imported as a module.
# If proxy.py has an `if __name__ == "__main__": uvicorn.run(...)`,
# the parts above it (definitions) are importable.

@pytest.fixture
def mock_poe_token(monkeypatch):
    """Mocks the POE_TOKEN environment variable."""
    token = "test_poe_token_123"
    monkeypatch.setenv("POE_TOKEN", token)
    return token

@pytest_asyncio.fixture
async def mock_fp_get_bot_response(mocker):
    """
    Mocks proxy.fp.get_bot_response.
    The mock itself is a callable (like the original function).
    When called, its side_effect (which is an async function) is executed.
    This side_effect function must RETURN an async generator object.
    """
    
    # This is the function that will be executed when the mock is called.
    # It's an async function because fp.get_bot_response might be,
    # and more importantly, it needs to return an async generator.
    async def default_side_effect_func(*args, **kwargs):
        # This is the async generator that the mocked function will return
        async def _async_generator():
            part = MagicMock(spec=fp.PartialResponse)
            part.text = "Default mock response part from conftest. "
            yield part
        return _async_generator() # Return the async generator *object*

    # mock_function is what replaces proxy.fp.get_bot_response
    # When mock_function is called, its side_effect is invoked.
    mock_function = mocker.patch('proxy.fp.get_bot_response', side_effect=default_side_effect_func)
    return mock_function

@pytest.fixture
def sample_openai_messages_basic():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ]

@pytest.fixture
def sample_openai_messages_long():
    return [
        {"role": "system", "content": "You are a helpful assistant designed to summarize."},
        {"role": "user", "content": "This is a very long message that will definitely exceed the context limit of many models. " * 500}
    ]

@pytest.fixture
def sample_openai_messages_with_images():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image", "image_url": "data:image/jpeg;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs="} # 1x1 black pixel
        ]}
    ]

# If you want to test the FastAPI app directly using TestClient (good for unit testing endpoints)
# from fastapi.testclient import TestClient
# from proxy import app as fastapi_app # Assuming your FastAPI app instance is named 'app' in proxy.py

# @pytest.fixture(scope="module")
# def test_client():
#     client = TestClient(fastapi_app)
#     yield client
