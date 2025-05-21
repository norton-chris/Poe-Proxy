# tests/test_utils.py
import pytest
from proxy import count_tokens, calculate_points, clean_content, get_poe_token, POINT_COSTS, MODEL_CONTEXT_LIMITS

def test_get_poe_token_env_set(mock_poe_token): # mock_poe_token fixture sets the env var
    get_poe_token.cache_clear() # Clear cache before this test too, for isolation
    assert get_poe_token() == "test_poe_token_123"

def test_get_poe_token_env_not_set(monkeypatch):
    get_poe_token.cache_clear() # Crucial: clear the cache
    monkeypatch.delenv("POE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="POE_TOKEN environment variable not set"):
        get_poe_token()

@pytest.mark.parametrize("text, model, expected_tokens", [
    ("Hello world", "gpt-4", 2),
    ("你好世界", "gpt-4", 4), # Assuming more tokens for non-ASCII with gpt-4 tokenizer
    ("Test", "some-other-model", 1), # Fallback: len // 4
    ("", "gpt-4", 0),
    ("", "some-other-model", 0),
])
def test_count_tokens(text, model, expected_tokens):
    assert count_tokens(text, model) == expected_tokens

@pytest.mark.parametrize("model, total_tokens, has_images, expected_points_range", [
    ("gemini-1.5-pro", 1000, False, (POINT_COSTS["gemini-1.5-pro"]["base"], POINT_COSTS["gemini-1.5-pro"]["base"] + 1)),
    ("claude-3.5-sonnet", 2000, False, (POINT_COSTS["claude-3.5-sonnet"]["base"] + (2 * POINT_COSTS["claude-3.5-sonnet"]["per_1k_tokens"]),
                                       POINT_COSTS["claude-3.5-sonnet"]["base"] + (2 * POINT_COSTS["claude-3.5-sonnet"]["per_1k_tokens"]) + 1)),
    ("claude-3.5-sonnet", 1000, True, (POINT_COSTS["claude-3.5-sonnet"]["base"] + POINT_COSTS["claude-3.5-sonnet"]["per_1k_tokens"] + 100,
                                      POINT_COSTS["claude-3.5-sonnet"]["base"] + POINT_COSTS["claude-3.5-sonnet"]["per_1k_tokens"] + 101)), # 100 for image
    ("unknown-model", 1000, False, ("Unknown", "Unknown")),
])
def test_calculate_points(model, total_tokens, has_images, expected_points_range):
    result = calculate_points(model, total_tokens, has_images)
    if expected_points_range[0] == "Unknown":
        assert result["points"] == "Unknown"
    else:
        # Check if points fall within a small range due to float precision
        assert expected_points_range[0] <= result["points"] < expected_points_range[1]

@pytest.mark.parametrize("content, expected_cleaned_content", [
    ("Hello\n\n---\n📊 Stats", "Hello"),
    ("Just normal text", "Just normal text"),
    (["img_part1", "img_part2"], ["img_part1", "img_part2"]), # List should pass through
    ("", ""),
])
def test_clean_content(content, expected_cleaned_content):
    assert clean_content(content) == expected_cleaned_content
