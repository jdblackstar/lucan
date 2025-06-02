#!/usr/bin/env python3
"""Test script to verify absolute value adjustments work correctly."""

from pathlib import Path

from lucan.core import LucanChat


def test_absolute_adjustments() -> None:
    """Test that absolute value requests calculate the correct adjustment."""

    # Create chat instance with debug enabled
    persona_path = Path("memory/personas/lucan")
    chat = LucanChat(persona_path, debug=True)

    # Test case: User requests "set verbosity to 0"
    # Should calculate: current (-2) → target (0) = adjustment +2
    test_response = """Got it! Let me go back to my normal verbosity level.

```json
{
    "action": "adjust_modifier",
    "modifier": "verbosity",
    "adjustment": 2,
    "reason": "User requested to set verbosity back to 0 (current: -2, target: 0)"
}
```

There we go - back to my usual chatty self!"""

    verbosity_before = chat.lucan.modifiers.get("verbosity", 0)

    processed = chat._process_modifier_adjustment(test_response)

    verbosity_after = chat.lucan.modifiers.get("verbosity", 0)

    # Verify the adjustment was applied correctly
    assert verbosity_after == verbosity_before + 2, (
        f"Expected verbosity to increase by 2, went from {verbosity_before} to {verbosity_after}"
    )
    assert "```json" not in processed, "JSON should be removed from processed response"
    assert "There we go - back to my usual chatty self!" in processed, (
        "Main response content should remain"
    )

    # Test case 2: Set warmth adjustment
    warmth_before = chat.lucan.modifiers.get("warmth", 0)

    test_response_2 = """I'll dial back the warmth a bit.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": -2,
    "reason": "User requested to set warmth to 1 (current: 3, target: 1)"
}
```

A bit more balanced now."""

    processed_2 = chat._process_modifier_adjustment(test_response_2)

    warmth_after = chat.lucan.modifiers.get("warmth", 0)

    # Verify the warmth adjustment was applied correctly
    assert warmth_after == warmth_before - 2, (
        f"Expected warmth to decrease by 2, went from {warmth_before} to {warmth_after}"
    )
    assert "```json" not in processed_2, (
        "JSON should be removed from second processed response"
    )
    assert "A bit more balanced now." in processed_2, (
        "Main response content should remain"
    )


if __name__ == "__main__":
    test_absolute_adjustments()
