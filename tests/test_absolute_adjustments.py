#!/usr/bin/env python3
"""Test script to verify absolute modifier adjustments work correctly."""

from pathlib import Path

import pytest

from lucan.core import LucanChat


@pytest.fixture
def _chat() -> LucanChat:
    """Create a LucanChat instance with debug enabled for testing."""
    persona_path = Path("memory/personas/lucan")
    return LucanChat(persona_path, debug=True)


def test_absolute_adjustments(_chat: LucanChat) -> None:
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

    _chat._process_modifier_adjustment(warmth_adjustment_response)

    # Test case: Make an absolute adjustment to warmth
    test_response = """I'm going to be much more formal and professional with you.

```json
{
    "action": "adjust_modifier",
    "modifier": "warmth",
    "adjustment": -3,
    "absolute": true,
    "reason": "User requested professional, formal interaction style"
}
```

How may I assist you today?"""

    processed = _chat._process_modifier_adjustment(test_response)

    warmth_after = _chat.lucan.modifiers.get("warmth", 0)

    # With absolute=true, the adjustment should be from baseline 0, not from current value
    # So warmth should be 0 + (-3) = -3, not initial_warmth + (-3)
    assert warmth_after == -3, (
        f"Expected warmth to be -3 (absolute adjustment), but got {warmth_after}"
    )

    # Test case 2: Another absolute adjustment to verbosity
    test_response_2 = """I'll be extremely brief.

```json
{
    "action": "adjust_modifier",
    "modifier": "verbosity", 
    "adjustment": -2,
    "absolute": true,
    "reason": "User wants very short responses"
}
```

Ok."""

    processed_2 = _chat._process_modifier_adjustment(test_response_2)

    verbosity_after = _chat.lucan.modifiers.get("verbosity", 0)

    # Verbosity should be exactly -2 (from baseline 0)
    assert verbosity_after == -2, (
        f"Expected verbosity to be -2 (absolute adjustment), but got {verbosity_after}"
    )

    # Verify JSON was removed from responses
    assert "```json" not in processed, "JSON should be removed from first response"
    assert "```json" not in processed_2, "JSON should be removed from second response"
