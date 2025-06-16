"""Shared test utilities for Lucan tests."""

from lucan.core import LucanChat


def create_test_response(action: str, modifier: str, **kwargs) -> str:
    """Helper function to create test responses with JSON blocks.
    
    Args:
        action: Either "adjust_modifier" or "set_modifier"
        modifier: The modifier to adjust (e.g., "warmth", "verbosity")
        **kwargs: Additional JSON fields like "adjustment", "value", "reason"
        
    Returns:
        Formatted test response string with JSON block
    """
    # Build the JSON content
    json_content = {
        "action": action,
        "modifier": modifier,
        **kwargs
    }
    
    # Convert to JSON string manually for consistent formatting
    json_lines = ["{"]
    for key, value in json_content.items():
        if isinstance(value, str):
            json_lines.append(f'    "{key}": "{value}",')
        else:
            json_lines.append(f'    "{key}": {value},')
    
    # Remove trailing comma from last line
    if json_lines[-1].endswith(','):
        json_lines[-1] = json_lines[-1][:-1]
    
    json_lines.append("}")
    json_block = "\n".join(json_lines)
    
    return f"""Test response with modifier adjustment.

```json
{json_block}
```

Response continues after JSON block."""


def assert_modifier_change(
    chat: LucanChat, 
    modifier: str, 
    expected_value: int, 
    operation_desc: str
) -> None:
    """Helper to assert modifier changes with descriptive error messages.
    
    Args:
        chat: The LucanChat instance to check
        modifier: The modifier name to check
        expected_value: The expected value after change
        operation_desc: Description of the operation for error messages
    """
    actual_value = chat.lucan.modifiers.get(modifier, 0)
    assert actual_value == expected_value, (
        f"{operation_desc}: expected {modifier} to be {expected_value}, "
        f"but got {actual_value}"
    )


def assert_json_removed(response: str, operation_desc: str = "JSON") -> None:
    """Helper to assert that JSON blocks were removed from responses.
    
    Args:
        response: The processed response to check
        operation_desc: Description for error messages
    """
    assert "```json" not in response, f"{operation_desc} block should be removed"


def assert_content_preserved(response: str, expected_content: str) -> None:
    """Helper to assert that main content was preserved in responses.
    
    Args:
        response: The processed response to check
        expected_content: Content that should be present
    """
    assert expected_content in response, (
        f"Main content should be preserved: '{expected_content}'"
    )
