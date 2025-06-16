#!/usr/bin/env python3
"""Test script to verify set_modifier action works correctly."""

from lucan.core import LucanChat
from .utils import assert_json_removed


def test_set_modifier(chat: LucanChat) -> None:
    """Test that set_modifier directly sets values without calculation."""

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

    processed = chat.process_modifier_adjustment(test_response)

    assert chat.lucan.modifiers.get("verbosity", 0) == 0, (
        "Verbosity should be set to 0"
    )

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

    processed_2 = chat.process_modifier_adjustment(test_response_2)

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

    processed_3 = chat.process_modifier_adjustment(test_response_3)

    assert chat.lucan.modifiers.get("warmth", 0) == 3, (
        "Warmth should be capped at 3, not set to 5"
    )

    # Test that JSON was removed from all responses
    assert_json_removed(processed, "First response JSON")
    assert_json_removed(processed_2, "Second response JSON")
    assert_json_removed(processed_3, "Third response JSON")
