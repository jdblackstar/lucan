"""Goal tracking and management for Lucan conversations.

This module handles tracking user goals, managing goal changes,
and providing goal consistency data for sidecar metrics.
"""

from typing import Dict, List, Optional


class GoalManager:
    """Manages user goals within a conversation."""

    def __init__(self, debug: bool = False):
        """Initialize the goal manager.

        Args:
            debug: Whether to enable debug output
        """
        self.debug = debug
        self._active_goals: List[str] = []
        self._goal_cache: Dict[
            str, Optional[object]
        ] = {}  # Goal text → embeddings (future)

    def add_goal(self, goal: str, timeframe: Optional[str] = None) -> Dict:
        """Add a new goal to the active goals list.

        Args:
            goal: The goal text
            timeframe: Optional timeframe (short-term, medium-term, long-term, ongoing)

        Returns:
            Result dictionary with success status
        """
        if goal not in self._active_goals:
            self._active_goals.append(goal)
            self._update_cache(goal)
            if self.debug:
                print(f"[DEBUG] Added goal: '{goal}' (timeframe: {timeframe})")
            return {
                "success": True,
                "action": "added",
                "goal": goal,
                "total_goals": len(self._active_goals),
            }
        else:
            return {
                "success": True,
                "action": "already_exists",
                "goal": goal,
                "total_goals": len(self._active_goals),
            }

    def replace_all_goals(self, goal: str, timeframe: Optional[str] = None) -> Dict:
        """Replace all existing goals with a single new goal.

        Args:
            goal: The new goal text
            timeframe: Optional timeframe

        Returns:
            Result dictionary with success status
        """
        self._active_goals.clear()
        self._goal_cache.clear()
        self._active_goals.append(goal)
        self._update_cache(goal)
        if self.debug:
            print(f"[DEBUG] Replaced all goals with: '{goal}' (timeframe: {timeframe})")
        return {"success": True, "action": "replaced", "goal": goal, "total_goals": 1}

    def remove_goal(self, goal: str) -> Dict:
        """Remove a specific goal.

        Args:
            goal: The goal text to remove

        Returns:
            Result dictionary with success status
        """
        if goal in self._active_goals:
            self._active_goals.remove(goal)
            if goal in self._goal_cache:
                del self._goal_cache[goal]
            if self.debug:
                print(f"[DEBUG] Removed goal: '{goal}'")
            return {
                "success": True,
                "action": "removed",
                "goal": goal,
                "total_goals": len(self._active_goals),
            }
        else:
            return {
                "success": False,
                "action": "not_found",
                "goal": goal,
                "total_goals": len(self._active_goals),
            }

    def handle_goal_tracking(
        self, goal: str, action: str, timeframe: Optional[str] = None
    ) -> Dict:
        """Handle goal tracking based on the specified action.

        Args:
            goal: The goal text
            action: One of "add", "replace", or "remove"
            timeframe: Optional timeframe for the goal

        Returns:
            Result dictionary with success status
        """
        try:
            if action == "add":
                return self.add_goal(goal, timeframe)
            elif action == "replace":
                return self.replace_all_goals(goal, timeframe)
            elif action == "remove":
                return self.remove_goal(goal)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error in goal tracking: {e}")
            return {"success": False, "error": str(e)}

    def get_active_goals(self) -> List[str]:
        """Get the list of currently active goals."""
        return self._active_goals.copy()

    def get_goal_cache(self) -> Dict[str, Optional[object]]:
        """Get the goal cache for metrics evaluation."""
        return self._goal_cache

    def get_goals_summary(self) -> str:
        """Get a formatted summary of current goals for display.

        Returns:
            Formatted string describing current goals
        """
        if not self._active_goals:
            return "Goals: none set"

        goal_count = len(self._active_goals)
        goal_preview = (
            self._active_goals[0][:30] + "..."
            if len(self._active_goals[0]) > 30
            else self._active_goals[0]
        )

        if goal_count == 1:
            return f"Goals: '{goal_preview}'"
        else:
            return f"Goals: '{goal_preview}' +{goal_count - 1} more"

    def _update_cache(self, goal: str) -> None:
        """Update the goal cache with a new goal.

        Args:
            goal: The goal text to cache
        """
        if goal not in self._goal_cache:
            self._goal_cache[goal] = None  # Placeholder for future embedding
            if self.debug:
                print(f"[DEBUG] Added goal to cache: '{goal}'")
