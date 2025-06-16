#!/usr/bin/env python3
"""Test script to demonstrate modifier adjustment debug output."""

from .utils import assert_content_preserved, assert_json_removed, chat


def test_debug_output() -> None:
    """Test debug output for modifier adjustments."""

    # Test small adjustment (should be applied silently)
    test_response = """I'll try to be more concise going forward.

```json
{
    "action": "adjust_modifier",
    "modifier": "verbosity",
    "adjustment": -1,
    "reason": "User requested shorter responses"
}
```

Is there anything specific you'd like to focus on?"""

    # Show current modifiers
    verbosity_before = chat.lucan.modifiers.get("verbosity", 0)

    # Process the adjustment (this should show debug output)
    processed = chat.process_modifier_adjustment(test_response)

    verbosity_after = chat.lucan.modifiers.get("verbosity", 0)

    # Assert that the adjustment was applied correctly
    assert verbosity_after == verbosity_before - 1, (
        f"Expected verbosity to decrease by 1, but went from {verbosity_before} to {verbosity_after}"
    )

    # Assert that the JSON was removed from the response
    assert_json_removed(processed, "JSON")
    assert_content_preserved(
        processed, "Is there anything specific you'd like to focus on?"
    )
