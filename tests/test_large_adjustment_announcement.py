#!/usr/bin/env python3
"""Test script to demonstrate the new large adjustment announcement behavior."""

from lucan.core import LucanChat
from .utils import assert_content_preserved, assert_json_removed


def test_large_adjustment_announcement(chat: LucanChat) -> None:
    """Test that large adjustments are announced naturally in conversation."""

    # Test case: User requests gentler approach (warmth +2)
    test_response = """I hear you. Let me shift how I'm approaching this and try a gentler touch.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": 2,
    "reason": "User explicitly requested gentler approach and is feeling hopeless"
}
```

Right now, can we just sit with where you are instead of where you think you should be? You're feeling overwhelmed, and that's okay. You don't have to solve your whole life today.

What would help you feel even slightly more at peace in this moment?"""

    warmth_before = chat.lucan.modifiers.get("warmth", 0)

    # Process the adjustment (this should show debug output and apply the change)
    processed = chat.process_modifier_adjustment(test_response)

    warmth_after = chat.lucan.modifiers.get("warmth", 0)

    # Verify the large adjustment was applied
    assert warmth_after == warmth_before + 2, (
        f"Expected warmth to increase by 2, went from {warmth_before} to {warmth_after}"
    )

    # Verify the announcement is preserved and JSON is removed
    assert_content_preserved(
        processed, "Let me shift how I'm approaching this and try a gentler touch"
    )
    assert_json_removed(processed, "JSON")
    assert_content_preserved(
        processed, "What would help you feel even slightly more at peace"
    )

    # Test case 2: User wants less verbosity (verbosity -2)
    test_response_2 = """You're right - I'm going to dial back and be more concise.

```json
{
    "action": "adjust_modifier", 
    "modifier": "verbosity",
    "adjustment": -2,
    "reason": "User indicated responses are too long and overwhelming"
}
```

What's one small step you could take today?"""

    verbosity_before = chat.lucan.modifiers.get("verbosity", 0)

    processed_2 = chat.process_modifier_adjustment(test_response_2)

    verbosity_after = chat.lucan.modifiers.get("verbosity", 0)

    # Verify the verbosity adjustment was applied
    assert verbosity_after == verbosity_before - 2, (
        f"Expected verbosity to decrease by 2, went from {verbosity_before} to {verbosity_after}"
    )

    # Verify announcement and content preservation
    assert_content_preserved(processed_2, "I'm going to dial back and be more concise")
    assert_json_removed(processed_2, "JSON")
    assert_content_preserved(processed_2, "What's one small step you could take today?")
