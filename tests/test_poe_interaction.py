# tests/test_poe_interaction.py
import pytest
import fastapi_poe as fp # For fp.PartialResponse if needed for type hints or constructing mocks
from unittest.mock import MagicMock, AsyncMock
from proxy import discover_and_retry_bot_response, MODEL_CONTEXT_LIMITS, calculate_points, POINT_COSTS

# Use a test-specific model context limit
TEST_MODEL_CONTEXT_LIMITS_INTERACTION = {
    "test-bot": 200, # tokens
    "claude-3.5-sonnet": 200000 # Keep some real ones if used in tests
}
REAL_MODEL_CONTEXT_LIMITS_PROXY = MODEL_CONTEXT_LIMITS.copy()

@pytest.fixture(autouse=True)
def patch_model_context_limits_interaction(monkeypatch):
    monkeypatch.setattr('proxy.MODEL_CONTEXT_LIMITS', TEST_MODEL_CONTEXT_LIMITS_INTERACTION)
    yield
    monkeypatch.setattr('proxy.MODEL_CONTEXT_LIMITS', REAL_MODEL_CONTEXT_LIMITS_PROXY)


async def generate_mock_responses(texts, done_at_end=True):
    for i, text_content in enumerate(texts):
        mock_partial = MagicMock(spec=fp.PartialResponse)
        mock_partial.text = text_content
        # Simulate 'done' attribute if your code relies on it from PartialResponse
        # For fp.get_bot_response, there isn't an explicit 'done' in PartialResponse itself.
        # The stream ending signifies completion.
        yield mock_partial

@pytest.mark.asyncio
async def test_discover_successful_response(mock_fp_get_bot_response, sample_openai_messages_basic):
    bot_name = "test-bot"
    poe_token = "fake_token"
    
    # Configure the mock for fp.get_bot_response
    # It needs to return an async iterable (like an async generator)
    mock_fp_get_bot_response.return_value = generate_mock_responses(["Full response."])

    responses = []
    async for partial in discover_and_retry_bot_response(sample_openai_messages_basic, bot_name, poe_token):
        responses.append(partial.text)

    assert "".join(responses).startswith("Full response.")
    assert "📊 Tokens:" in "".join(responses) # Check for footer
    assert "💰 Points:" in "".join(responses)
    mock_fp_get_bot_response.assert_called_once()
    # You can add more assertions on the arguments passed to mock_fp_get_bot_response

@pytest.mark.asyncio
async def test_discover_retry_on_context_length(mock_fp_get_bot_response, sample_openai_messages_long):
    bot_name = "test-bot" # Limit is 200 tokens
    poe_token = "fake_token"

    # Simulate failure on first attempt (e.g., by raising an error or yielding nothing)
    # and success on second attempt with halved context.
    
    # This mock setup is a bit more complex as it needs to change behavior across calls
    # or simulate the "no chunks received" scenario.
    # For simplicity, let's assume the first call yields nothing (simulating timeout/error from Poe)
    # and the second call (after context halving) yields response.

    attempt_count = 0
    async def mock_stream_with_retry(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        # messages_arg = kwargs.get('messages', args[0] if args else [])
        # current_tokens = sum(count_tokens(m['content'], bot_name) for m in messages_arg if isinstance(m['content'], str))
        # print(f"Mock attempt {attempt_count}, tokens: {current_tokens}")

        if attempt_count == 1: # First attempt (long context) - simulate failure (no chunks)
            # Check if context was indeed long
            # This requires inspecting kwargs['messages'] passed to the mock
            # For now, we assume the test setup (sample_openai_messages_long) ensures this
            if False: # Hack to make it an async generator
                yield
            return # No response chunks
        else: # Second attempt (shorter context) - simulate success
            yield MagicMock(text="Shortened response.", spec=fp.PartialResponse)

    mock_fp_get_bot_response.side_effect = mock_stream_with_retry # Use side_effect for dynamic mock

    responses = []
    async for partial in discover_and_retry_bot_response(sample_openai_messages_long, bot_name, poe_token):
        responses.append(partial.text)
    
    assert "".join(responses).startswith("Shortened response.")
    assert "📊 Tokens:" in "".join(responses)
    assert "💰 Points:" in "".join(responses)
    assert "⚠️ Input was trimmed" in "".join(responses) # Check for trimming warning
    assert attempt_count >= 2 # Should have tried at least twice

@pytest.mark.asyncio
async def test_discover_footer_calculation(mock_fp_get_bot_response, sample_openai_messages_basic):
    bot_name = "gemini-1.5-pro" # Use a model with known costs
    poe_token = "fake_token"
    
    response_text = "This is the AI response."
    mock_fp_get_bot_response.return_value = generate_mock_responses([response_text])

    # Manually calculate expected tokens and points
    # This is a bit of a self-fulfilling test if calculate_points is also under test here,
    # but it checks the integration.
    from proxy import count_tokens # Ensure it's the one from your proxy
    input_tokens = sum(count_tokens(msg.get("content", ""), bot_name) for msg in sample_openai_messages_basic)
    output_tokens = count_tokens(response_text, bot_name)
    total_tokens = input_tokens + output_tokens
    expected_points_info = calculate_points(bot_name, total_tokens, has_images=False)

    full_response_text = ""
    async for partial in discover_and_retry_bot_response(sample_openai_messages_basic, bot_name, poe_token):
        full_response_text += partial.text
    
    assert response_text in full_response_text
    assert f"📊 Tokens: {total_tokens:,} (Input: {input_tokens:,}, Output: {output_tokens:,})" in full_response_text
    assert f"💰 Points: {expected_points_info['points']}" in full_response_text
    assert expected_points_info['breakdown'] in full_response_text

@pytest.mark.asyncio
async def test_discover_large_input_and_full_output(
    mock_fp_get_bot_response, sample_openai_messages_very_long # Use the large input
):
    bot_name = "claude-3.5-sonnet"
    poe_token = "fake_token"
    
    expected_poe_response_parts = [f"Response chunk {i}. " for i in range(50)]
    complete_expected_poe_text = "".join(expected_poe_response_parts)

    actual_messages_sent_to_poe_container = [] # Use a container to capture

    async def side_effect_func(*args, **kwargs):
        # Capture messages passed to the actual fp.get_bot_response call
        actual_messages_sent_to_poe_container.append(kwargs.get('messages', args[0] if args else []))
        
        async def _gen(): # The actual async generator
            for part_text in expected_poe_response_parts:
                mock_partial = MagicMock(spec=fp.PartialResponse)
                mock_partial.text = part_text
                yield mock_partial
        return _gen() # Return the async generator object

    mock_fp_get_bot_response.side_effect = side_effect_func

    all_yielded_parts_text = []
    async for partial in discover_and_retry_bot_response(
        sample_openai_messages_very_long, bot_name, poe_token
    ):
        if hasattr(partial, 'text'):
            all_yielded_parts_text.append(partial.text)

    # Assertions
    assert actual_messages_sent_to_poe_container # Ensure it was captured
    actual_messages_sent_to_poe = actual_messages_sent_to_poe_container[0]

    # Use the original MODEL_CONTEXT_LIMITS from proxy.py for this check,
    # as discover_and_retry_bot_response uses it for its initial calculation.
    from proxy import MODEL_CONTEXT_LIMITS as PROXY_MODEL_CONTEXT_LIMITS
    initial_limit_for_bot = PROXY_MODEL_CONTEXT_LIMITS.get(bot_name, 16000)
    
    expected_trimmed_input = enforce_context_limit(
        sample_openai_messages_very_long, bot_name, limit_override=initial_limit_for_bot
    )
    assert len(actual_messages_sent_to_poe) == len(expected_trimmed_input)
    for sent_msg, expected_msg in zip(actual_messages_sent_to_poe, expected_trimmed_input):
        assert sent_msg['role'] == expected_msg['role']
        assert sent_msg['content'] == expected_msg['content']

    assert len(all_yielded_parts_text) == len(expected_poe_response_parts) + 1
    assembled_poe_content_from_proxy = "".join(all_yielded_parts_text[:-1])
    assert assembled_poe_content_from_proxy == complete_expected_poe_text
    
    footer_part = all_yielded_parts_text[-1]
    assert "📊 Tokens:" in footer_part
    mock_fp_get_bot_response.assert_called_once() 

# TODO: Add a test for image handling if you get that working.
# This would involve checking that `poe_messages` passed to the mocked `fp.get_bot_response`
# are formatted correctly (e.g., with attachments or specific content structure for images).
