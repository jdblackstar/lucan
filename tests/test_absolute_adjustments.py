#!/usr/bin/env python3
"""Test script to verify absolute modifier adjustments work correctly."""

from .utils import assert_json_removed, chat


def test_absolute_adjustments() -> None:
    """Test that absolute adjustments are calculated correctly."""

    # Start with some baseline modifiers by making relative adjustments first
    warmth_adjustment_response = """I'll be more supportive.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": 1,
    "reason": "User needs more support"
}
```

Here to help."""

    chat._process_modifier_adjustment(warmth_adjustment_response)

    # Test case: Make an absolute adjustment to warmth using set_modifier
    test_response = """I'm going to be much more formal and professional with you.

```json
{
    "action": "set_modifier",
    "modifier": "warmth",
    "value": -3,
    "reason": "User requested professional, formal interaction style"
}
```

How may I assist you today?"""

    processed = chat.process_modifier_adjustment(test_response)

    warmth_after = chat.lucan.modifiers.get("warmth", 0)

    # With set_modifier, the value should be set to exactly -3, regardless of current value
    assert warmth_after == -3, (
        f"Expected warmth to be -3 (absolute value), but got {warmth_after}"
    )

    # Test case 2: Another absolute adjustment to verbosity
    test_response_2 = """I'll be extremely brief.

```json
{
    "action": "set_modifier",
    "modifier": "verbosity", 
    "value": -2,
    "reason": "User wants very short responses"
}
```

Ok."""

    processed_2 = chat.process_modifier_adjustment(test_response_2)

    verbosity_after = chat.lucan.modifiers.get("verbosity", 0)

    # Verbosity should be exactly -2 (absolute value)
    assert verbosity_after == -2, (
        f"Expected verbosity to be -2 (absolute value), but got {verbosity_after}"
    )

    # Verify JSON was removed from responses
    assert_json_removed(processed, "First response JSON")
    assert_json_removed(processed_2, "Second response JSON")
