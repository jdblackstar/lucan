#!/usr/bin/env python3
"""Test script to verify set_modifier action works correctly."""

from pathlib import Path

from lucan.core import LucanChat


def test_set_modifier() -> None:
    """Test that set_modifier directly sets values without calculation."""

    # Create chat instance with debug enabled
    persona_path = Path("memory/personas/lucan")
    chat = LucanChat(persona_path, debug=True)

    # Test case: User requests "set verbosity to 0"
    test_response = """Got it! Let me go back to my normal verbosity level.

```json
{
    "action": "set_modifier",
    "modifier": "verbosity",
    "value": 0,
    "reason": "User requested to set verbosity back to 0"
}
```

There we go - back to my usual chatty self!"""

    processed = chat._process_modifier_adjustment(test_response)

    assert chat.lucan.modifiers.get("verbosity", 0) == 0, "Verbosity should be set to 0"

    # Test case 2: Set warmth to 2
    test_response_2 = """I'll warm up my approach significantly.

```json
{
    "action": "set_modifier",
    "modifier": "warmth",
    "value": 2,
    "reason": "User needs more emotional support"
}
```

Let me be more supportive and caring with you."""

    processed_2 = chat._process_modifier_adjustment(test_response_2)

    assert chat.lucan.modifiers.get("warmth", 0) == 2, "Warmth should be set to 2"

    # Test case 3: Test bounds checking (set to value > 3)
    test_response_3 = """I'll max out the warmth.

```json
{
    "action": "set_modifier",
    "modifier": "warmth",
    "value": 5,
    "reason": "User needs maximum warmth (testing bounds)"
}
```

As warm as I can be."""

    processed_3 = chat._process_modifier_adjustment(test_response_3)

    assert chat.lucan.modifiers.get("warmth", 0) == 3, (
        "Warmth should be capped at 3, not set to 5"
    )

    # Test that JSON was removed from all responses
    assert "```json" not in processed, "JSON should be removed from first response"
    assert "```json" not in processed_2, "JSON should be removed from second response"
    assert "```json" not in processed_3, "JSON should be removed from third response"


if __name__ == "__main__":
    test_set_modifier()
