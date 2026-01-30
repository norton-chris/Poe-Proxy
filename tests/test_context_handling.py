# tests/test_context_handling.py
import pytest
from proxy import count_message_tokens, trim_messages_to_limit, sanitize_messages

def test_count_message_tokens_basic():
    """Test basic token counting."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    tokens = count_message_tokens(messages)
    assert tokens > 0
    assert isinstance(tokens, int)

def test_count_message_tokens_with_images():
    """Test token counting with image content."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]}
    ]
    tokens = count_message_tokens(messages)
    assert tokens > 0
    # Should include base image tokens (85)
    assert tokens >= 85

def test_count_message_tokens_long_content():
    """Test token counting with very long content."""
    messages = [
        {"role": "user", "content": "This is a test. " * 1000}
    ]
    tokens = count_message_tokens(messages)
    # Should be roughly 4000+ tokens (1000 * 4 words * ~1 token per word)
    assert tokens > 3000

def test_trim_messages_to_limit_no_trimming_needed():
    """Test that messages under the limit are not trimmed."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    original_count = len(messages)
    trimmed = trim_messages_to_limit(messages, token_limit=100000)
    assert len(trimmed) == original_count
    assert trimmed == messages

def test_trim_messages_preserves_first_message():
    """Test that trimming always preserves the first message."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Old message " * 1000},
        {"role": "assistant", "content": "Old response " * 1000},
        {"role": "user", "content": "Newer message " * 1000},
        {"role": "assistant", "content": "Newer response " * 1000},
        {"role": "user", "content": "Latest message"}
    ]

    # Set a limit that forces trimming
    trimmed = trim_messages_to_limit(messages, token_limit=500)

    # First message should always be preserved
    assert trimmed[0] == messages[0]
    assert trimmed[0]["role"] == "system"
    # Should have fewer messages than original
    assert len(trimmed) < len(messages)

def test_trim_messages_removes_oldest_first():
    """Test that trimming removes oldest messages (from position 1 onwards)."""
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Message 1 " * 500},
        {"role": "assistant", "content": "Response 1 " * 500},
        {"role": "user", "content": "Message 2 " * 500},
        {"role": "assistant", "content": "Response 2 " * 500},
        {"role": "user", "content": "Message 3"}
    ]

    # Trim to a small limit
    trimmed = trim_messages_to_limit(messages, token_limit=1000)

    # Should preserve first message (system)
    assert trimmed[0]["role"] == "system"
    # Should preserve most recent messages
    assert trimmed[-1]["content"] == "Message 3"
    # Should not include the oldest user/assistant exchanges
    assert len(trimmed) < len(messages)

def test_trim_messages_single_message_over_limit():
    """Test behavior when a single message exceeds the limit."""
    messages = [
        {"role": "user", "content": "Very long message " * 10000}
    ]

    original_tokens = count_message_tokens(messages)
    # Trim to a limit smaller than the message
    trimmed = trim_messages_to_limit(messages, token_limit=100)

    # Should return the message as-is (can't trim a single message)
    assert len(trimmed) == 1
    assert trimmed[0] == messages[0]

def test_trim_messages_first_message_over_limit():
    """Test behavior when first message alone exceeds the limit."""
    messages = [
        {"role": "system", "content": "Very long system prompt " * 10000},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]

    # Trim to a limit smaller than the first message
    trimmed = trim_messages_to_limit(messages, token_limit=100)

    # Should return all messages as-is since we can't trim when first message is too large
    assert len(trimmed) == len(messages)
    assert trimmed == messages

def test_trim_messages_empty_list():
    """Test trimming an empty message list."""
    messages = []
    trimmed = trim_messages_to_limit(messages, token_limit=1000)
    assert trimmed == []

def test_trim_messages_preserves_recent_context():
    """Test that trimming keeps the most recent conversation context."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Message 0 " * 100},
        {"role": "assistant", "content": "Response 0 " * 100},
        {"role": "user", "content": "Message 1 " * 100},
        {"role": "assistant", "content": "Response 1 " * 100},
        {"role": "user", "content": "Message 2 " * 100},
        {"role": "assistant", "content": "Response 2 " * 100},
        {"role": "user", "content": "Latest message"}
    ]

    trimmed = trim_messages_to_limit(messages, token_limit=500)

    # Should have system message
    assert trimmed[0]["role"] == "system"
    # Should have the latest message
    assert trimmed[-1]["content"] == "Latest message"
    # Should have recent messages but not all
    assert len(trimmed) < len(messages)
    # Verify Message 0 is NOT in trimmed (oldest conversation)
    message_0_present = any("Message 0" in msg.get("content", "") for msg in trimmed)
    assert not message_0_present

def test_sanitize_messages_basic():
    """Test basic message sanitization."""
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    sanitized = sanitize_messages(messages)
    assert len(sanitized) == 1
    assert sanitized[0]["role"] == "user"
    assert sanitized[0]["content"] == "Hello"

def test_sanitize_messages_invalid_role():
    """Test that invalid roles are converted to 'user'."""
    messages = [
        {"role": "invalid", "content": "Test"}
    ]
    sanitized = sanitize_messages(messages)
    assert sanitized[0]["role"] == "user"

def test_sanitize_messages_preserves_valid_roles():
    """Test that all valid roles are preserved."""
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "User"},
        {"role": "assistant", "content": "Assistant"},
        {"role": "tool", "content": "Tool"}
    ]
    sanitized = sanitize_messages(messages)
    assert [msg["role"] for msg in sanitized] == ["system", "user", "assistant", "tool"]

def test_sanitize_messages_preserves_image_content():
    """Test that image content arrays are preserved."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]}
    ]
    sanitized = sanitize_messages(messages)
    assert isinstance(sanitized[0]["content"], list)
    assert sanitized[0]["content"][0]["type"] == "text"
    assert sanitized[0]["content"][1]["type"] == "image_url"

def test_sanitize_messages_empty_list():
    """Test that empty message list gets a default user message."""
    messages = []
    sanitized = sanitize_messages(messages)
    assert len(sanitized) == 1
    assert sanitized[0]["role"] == "user"
    assert sanitized[0]["content"] == ""

def test_sanitize_messages_non_dict():
    """Test that non-dict messages are converted to user messages."""
    messages = ["Just a string", 123]
    sanitized = sanitize_messages(messages)
    assert len(sanitized) == 2
    assert all(msg["role"] == "user" for msg in sanitized)
    assert sanitized[0]["content"] == "Just a string"
    assert sanitized[1]["content"] == "123"
