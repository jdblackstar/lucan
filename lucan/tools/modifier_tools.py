"""Tools for adjusting personality modifiers."""

from typing import Optional

from .base import BaseTool, ToolResult


class ModifierAdjustmentTool(BaseTool):
    """Tool for adjusting personality modifiers."""

    def __init__(self, lucan_instance, debug: bool = False):
        self.lucan = lucan_instance
        self.debug = debug

    @property
    def name(self) -> str:
        return "adjust_modifier"

    @property
    def description(self) -> str:
        return (
            "Adjust your own personality modifiers based on user feedback or "
            "your own perception of misalignment. Use this tool when: "
            "1. The user explicitly asks for behavior changes (e.g., 'be less verbose', 'be warmer') "
            "2. You perceive your current behavior isn't working well for the user. "
            "For small adjustments (±1): Just apply the change and continue naturally. "
            "For larger changes (±2 or more, or any set_modifier): Always announce that you're "
            "shifting your approach."
        )

    def execute(
        self,
        action: str,
        modifier: str,
        value: Optional[int] = None,
        adjustment: Optional[int] = None,
        reason: str = "",
    ) -> ToolResult:
        """Adjust a personality modifier.

        Args:
            action: Either "adjust" (relative change) or "set" (absolute value)
            modifier: Which modifier to change (warmth, verbosity, challenge, emotional_depth, structure)
            value: Absolute value for "set" action (range: -3 to +3)
            adjustment: Relative change for "adjust" action (e.g., -1, +2)
            reason: Why this change is being made
        """
        try:
            # Validate action
            if action not in ["adjust", "set"]:
                return ToolResult(
                    success=False, error="Action must be either 'adjust' or 'set'"
                )

            # Validate modifier exists
            available_modifiers = [
                "warmth",
                "challenge",
                "verbosity",
                "emotional_depth",
                "structure",
            ]
            if modifier not in available_modifiers:
                return ToolResult(
                    success=False,
                    error=f"Unknown modifier '{modifier}'. Available: {available_modifiers}",
                )

            # Validate required parameters based on action
            if action == "adjust" and adjustment is None:
                return ToolResult(
                    success=False,
                    error="'adjustment' parameter required for 'adjust' action",
                )

            if action == "set" and value is None:
                return ToolResult(
                    success=False, error="'value' parameter required for 'set' action"
                )

            # Get current value
            current_value = self.lucan.modifiers.get(modifier, 0)

            # Calculate new value
            if action == "adjust":
                new_value = current_value + adjustment
            else:  # action == "set"
                new_value = value

            # Clamp to valid range
            new_value = max(-3, min(3, new_value))

            # Apply the change
            old_value = self.lucan.modifiers[modifier]
            self.lucan.modifiers[modifier] = new_value
            self.lucan.save_modifiers()

            if self.debug:
                print(
                    f"[DEBUG] Modified {modifier}: {old_value} -> {new_value} (reason: {reason})"
                )

            # Determine if this is a large change that should be announced
            is_large_change = False
            if action == "set":
                is_large_change = True  # All set operations are considered significant
            elif abs(adjustment) >= 2:
                is_large_change = True

            return ToolResult(
                success=True,
                data={
                    "modifier": modifier,
                    "old_value": old_value,
                    "new_value": new_value,
                    "action": action,
                    "is_large_change": is_large_change,
                    "reason": reason,
                },
                message=f"Modified {modifier} from {old_value} to {new_value}",
            )

        except Exception as e:
            return ToolResult(
                success=False, error=f"Error adjusting modifier: {str(e)}"
            )
