"""Tools for goal tracking and management."""

from typing import Optional

from .base import BaseTool, ToolResult


class TrackUserGoalTool(BaseTool):
    """Tool for tracking user goals."""

    def __init__(self, goal_manager, debug: bool = False):
        self.goal_manager = goal_manager
        self.debug = debug

    @property
    def name(self) -> str:
        return "track_user_goal"

    @property
    def description(self) -> str:
        return (
            "Track or update user goals when they mention wanting to work on something, "
            "achieve something, or change their focus. Use this when the user expresses "
            "goals like: 'I want to reduce my anxiety', 'My goal is to get promoted', "
            "'I'm working on improving my relationships', 'I want to stop procrastinating', etc. "
            "This helps maintain goal consistency tracking. Don't announce when you're using this tool."
        )

    def execute(
        self, goal: str, action: str, timeframe: Optional[str] = None
    ) -> ToolResult:
        """Track or update a user goal.

        Args:
            goal: The user's goal in clear, specific terms
            action: Whether to add a new goal, replace all current goals with this one, or remove a specific goal (add, replace, remove)
            timeframe: The timeframe for this goal (short-term, medium-term, long-term, ongoing)
        """
        try:
            # Validate action
            valid_actions = ["add", "replace", "remove"]
            if action not in valid_actions:
                return ToolResult(
                    success=False, error=f"Action must be one of: {valid_actions}"
                )

            # Validate timeframe if provided
            if timeframe is not None:
                valid_timeframes = ["short-term", "medium-term", "long-term", "ongoing"]
                if timeframe not in valid_timeframes:
                    return ToolResult(
                        success=False,
                        error=f"Timeframe must be one of: {valid_timeframes}",
                    )

            if not goal.strip():
                return ToolResult(success=False, error="Goal cannot be empty")

            # Execute the goal tracking
            result = self.goal_manager.handle_goal_tracking(goal, action, timeframe)

            if self.debug:
                print(
                    f"[DEBUG] Goal tracking: {action} '{goal}' (timeframe: {timeframe})"
                )

            # Convert the goal manager result to our ToolResult format
            if result.get("success", False):
                return ToolResult(
                    success=True,
                    data={
                        "goal": goal,
                        "action": action,
                        "timeframe": timeframe,
                        "total_goals": len(self.goal_manager.get_active_goals()),
                    },
                    message=result.get("message", "Goal updated successfully"),
                )
            else:
                return ToolResult(
                    success=False, error=result.get("message", "Failed to update goal")
                )

        except Exception as e:
            return ToolResult(success=False, error=f"Error tracking goal: {str(e)}")
