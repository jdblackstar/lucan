"""Shared pytest fixtures for all Lucan tests."""

from pathlib import Path

import pytest

from lucan.core import LucanChat


@pytest.fixture
def chat() -> LucanChat:
    """Create a LucanChat instance with debug enabled for testing.
    
    This fixture provides an isolated test environment by:
    - Loading the lucan persona from memory/personas/lucan
    - Enabling debug mode for test visibility
    - Resetting all modifiers to 0 for consistent test state
    
    Returns:
        LucanChat instance ready for testing
    """
    persona_path = Path("memory/personas/lucan")
    chat = LucanChat(persona_path, debug=True)
    
    # Reset all modifiers to 0 for consistent test state
    for modifier in chat.lucan.modifiers:
        chat.lucan.modifiers[modifier] = 0
    chat.lucan.save_modifiers()
    
    return chat 