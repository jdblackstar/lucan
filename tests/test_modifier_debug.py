#!/usr/bin/env python3
"""Test script to demonstrate modifier adjustment debug output."""

from pathlib import Path

import pytest

from lucan.core import LucanChat


@pytest.fixture
def _chat() -> LucanChat:
    """Create a LucanChat instance with debug enabled for testing."""
    persona_path = Path("memory/personas/lucan")
    return LucanChat(persona_path, debug=True)


def test_debug_output(_chat: LucanChat) -> None:
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
    verbosity_before = _chat.lucan.modifiers.get("verbosity", 0)

    # Process the adjustment (this should show debug output)
    processed = _chat._process_modifier_adjustment(test_response)

    verbosity_after = _chat.lucan.modifiers.get("verbosity", 0)

    # Assert that the adjustment was applied correctly
    assert verbosity_after == verbosity_before - 1, (
        f"Expected verbosity to decrease by 1, but went from {verbosity_before} to {verbosity_after}"
    )

    # Assert that the JSON was removed from the response
    assert "```json" not in processed, (
        "JSON block should be removed from processed response"
    )
    assert "Is there anything specific you'd like to focus on?" in processed, (
        "Main response content should remain"
    )
