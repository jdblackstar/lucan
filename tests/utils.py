"""Shared test utilities for Lucan tests."""

import json
import re

from lucan.core import LucanChat
from lucan.tools import ModifierAdjustmentTool


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
    json_content = {"action": action, "modifier": modifier, **kwargs}

    # Convert to JSON string manually for consistent formatting
    json_lines = ["{"]
    for key, value in json_content.items():
        if isinstance(value, str):
            json_lines.append(f'    "{key}": "{value}",')
        else:
            json_lines.append(f'    "{key}": {value},')

    # Remove trailing comma from last line
    if json_lines[-1].endswith(","):
        json_lines[-1] = json_lines[-1][:-1]

    json_lines.append("}")
    json_block = "\n".join(json_lines)

    return f"""Test response with modifier adjustment.

```json
{json_block}
```

Response continues after JSON block."""


def process_modifier_adjustment_for_test(chat: LucanChat, response: str) -> str:
    """Process modifier adjustments in test responses using the tool system.

    This replaces the old process_modifier_adjustment method for testing.

    Args:
        chat: The LucanChat instance
        response: The response containing JSON modifier adjustments

    Returns:
        The response with JSON blocks removed
    """
    # Create modifier tool
    modifier_tool = ModifierAdjustmentTool(chat.lucan, debug=True)

    # Pattern to find JSON blocks
    json_pattern = r"```json\s*\n(.*?)\n```"

    def process_json_block(match):
        json_content = match.group(1)
        try:
            # Parse the JSON
            data = json.loads(json_content)

            # Only process modifier adjustments
            if data.get("action") in ["adjust_modifier", "set_modifier"]:
                action = data["action"]
                modifier = data["modifier"]
                reason = data.get("reason", "")

                # Convert set_modifier to tool's "set" action
                if action == "set_modifier":
                    tool_action = "set"
                    value = data.get("value")
                    result = modifier_tool.execute(
                        action=tool_action,
                        modifier=modifier,
                        value=value,
                        reason=reason,
                    )
                elif action == "adjust_modifier":
                    tool_action = "adjust"
                    adjustment = data.get("adjustment")
                    result = modifier_tool.execute(
                        action=tool_action,
                        modifier=modifier,
                        adjustment=adjustment,
                        reason=reason,
                    )

                if not result.success:
                    print(f"[TEST] Tool execution failed: {result.error}")

            # Return empty string to remove the JSON block
            return ""

        except json.JSONDecodeError:
            # If JSON is malformed, leave it as-is
            return match.group(0)

    # Process all JSON blocks and remove them
    processed_response = re.sub(
        json_pattern, process_json_block, response, flags=re.DOTALL
    )

    return processed_response


def assert_modifier_change(
    chat: LucanChat, modifier: str, expected_value: int, operation_desc: str
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


def assert_json_removed(processed_response: str, description: str) -> None:
    """Assert that JSON blocks have been removed from the response.

    Args:
        processed_response: The processed response to check
        description: Description for error messages
    """
    assert "```json" not in processed_response, f"{description} should be removed"


def assert_content_preserved(processed_response: str, expected_content: str) -> None:
    """Assert that specific content is preserved in the response.

    Args:
        processed_response: The processed response to check
        expected_content: Content that should be present
    """
    assert expected_content in processed_response, (
        f"Content '{expected_content}' should be preserved in response"
    )
