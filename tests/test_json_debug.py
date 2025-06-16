#!/usr/bin/env python3
"""Test script to demonstrate JSON parsing debug output."""

from pathlib import Path

import pytest

from lucan.core import LucanChat


@pytest.fixture
def _chat() -> LucanChat:
    """Create a LucanChat instance with debug enabled for testing."""
    persona_path = Path("memory/personas/lucan")
    return LucanChat(persona_path, debug=True)


def test_json_debug(_chat: LucanChat) -> None:
    """Test that malformed JSON is properly handled and debugged."""

    # Test case: Malformed JSON (missing closing brace)
    test_response_malformed = """I'll try to be more direct.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": -1,
    "reason": "User wants more directness"

```

Let me get straight to the point then."""

    # Test case: Valid JSON for comparison
    test_response_valid = """I'll try to be more direct.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": -1,
    "reason": "User wants more directness"
}
```

Let me get straight to the point then."""

    # Test malformed JSON
    warmth_before_malformed = _chat.lucan.modifiers.get("warmth", 0)
    processed_malformed = _chat._process_modifier_adjustment(test_response_malformed)
    warmth_after_malformed = _chat.lucan.modifiers.get("warmth", 0)

    # Malformed JSON should not be processed, so warmth should be unchanged
    assert warmth_after_malformed == warmth_before_malformed, (
        "Malformed JSON should not change modifiers"
    )
    # Malformed JSON blocks are left as-is when they can't be parsed
    assert "```json" in processed_malformed, (
        "Malformed JSON block should be left in place"
    )
    assert "Let me get straight to the point then." in processed_malformed, (
        "Main content should be preserved"
    )

    # Test valid JSON for comparison
    warmth_before_valid = _chat.lucan.modifiers.get("warmth", 0)
    processed_valid = _chat._process_modifier_adjustment(test_response_valid)
    warmth_after_valid = _chat.lucan.modifiers.get("warmth", 0)

    # Valid JSON should be processed correctly
    assert warmth_after_valid == warmth_before_valid - 1, (
        "Valid JSON should decrease warmth by 1"
    )
    assert "```json" not in processed_valid, "Valid JSON block should be removed"
    assert "Let me get straight to the point then." in processed_valid, (
        "Main content should be preserved"
    )
