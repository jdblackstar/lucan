#!/usr/bin/env python3
"""Test script to demonstrate JSON parsing debug output."""

from lucan.core import LucanChat

from .utils import (
    assert_content_preserved,
    assert_json_removed,
    process_modifier_adjustment_for_test,
)


def test_json_debug(chat: LucanChat) -> None:
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
    warmth_before_malformed = chat.lucan.modifiers.get("warmth", 0)
    processed_malformed = process_modifier_adjustment_for_test(
        chat, test_response_malformed
    )
    warmth_after_malformed = chat.lucan.modifiers.get("warmth", 0)

    # Malformed JSON should not be processed, so warmth should be unchanged
    assert warmth_after_malformed == warmth_before_malformed, (
        "Malformed JSON should not change modifiers"
    )
    # Malformed JSON blocks are left as-is when they can't be parsed
    assert "```json" in processed_malformed, (
        "Malformed JSON block should be left in place"
    )
    assert_content_preserved(
        processed_malformed, "Let me get straight to the point then."
    )

    # Test valid JSON for comparison
    warmth_before_valid = chat.lucan.modifiers.get("warmth", 0)
    processed_valid = process_modifier_adjustment_for_test(chat, test_response_valid)
    warmth_after_valid = chat.lucan.modifiers.get("warmth", 0)

    # Valid JSON should be processed correctly
    assert warmth_after_valid == warmth_before_valid - 1, (
        "Valid JSON should decrease warmth by 1"
    )

    # Valid JSON should be removed
    assert_json_removed(processed_valid, "Valid JSON")
    assert_content_preserved(processed_valid, "Let me get straight to the point then.")
