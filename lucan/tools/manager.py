"""Tool manager for handling tool registration and execution."""

from typing import Dict, List

from ..goals import GoalManager
from ..relationships import RelationshipManager
from .base import ToolResult
from .goal_tools import TrackUserGoalTool
from .modifier_tools import ModifierAdjustmentTool
from .registry import ToolRegistry
from .relationship_tools import AddRelationshipNoteTool, GetRelationshipNotesTool


class ToolManager:
    """Manages tool definitions and handles tool execution."""

    def __init__(
        self,
        relationship_manager: RelationshipManager,
        goal_manager: GoalManager,
        lucan_instance=None,
        debug: bool = False,
    ):
        """Initialize the tool manager with the new registry system.

        Args:
            relationship_manager: Instance for managing relationship notes
            goal_manager: Instance for managing user goals
            lucan_instance: The Lucan persona instance for modifier tools
            debug: Whether to enable debug output
        """
        self.relationship_manager = relationship_manager
        self.goal_manager = goal_manager
        self.lucan_instance = lucan_instance
        self.debug = debug

        # Initialize the registry
        self.registry = ToolRegistry(debug=debug)

        # Register all available tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools with the registry."""
        # Relationship tools
        self.registry.register_tool(
            AddRelationshipNoteTool(self.relationship_manager, self.debug)
        )
        self.registry.register_tool(
            GetRelationshipNotesTool(self.relationship_manager, self.debug)
        )

        # Goal tracking tool
        self.registry.register_tool(TrackUserGoalTool(self.goal_manager, self.debug))

        # Modifier tool (only if lucan_instance is provided)
        if self.lucan_instance:
            self.registry.register_tool(
                ModifierAdjustmentTool(self.lucan_instance, self.debug)
            )

    def get_tool_definitions(self) -> List[Dict]:
        """Get the list of available tool definitions.

        Returns:
            List of tool definitions for the Anthropic API
        """
        return self.registry.get_tool_definitions()

    def handle_tool_call(self, tool_name: str, tool_input: Dict) -> Dict:
        """Handle a tool call and return the result.

        Args:
            tool_name: The name of the tool being called
            tool_input: The input parameters for the tool

        Returns:
            Dict containing the tool result
        """
        # Execute the tool using the registry
        result = self.registry.execute_tool(tool_name, **tool_input)

        # Convert ToolResult to the old Dict format for backward compatibility
        return self._convert_tool_result_to_dict(result)

    def _convert_tool_result_to_dict(self, result: ToolResult) -> Dict:
        """Convert a ToolResult to the old dictionary format for backward compatibility.

        Args:
            result: The ToolResult to convert

        Returns:
            Dictionary in the old format
        """
        response = {"success": result.success}

        if result.message:
            response["message"] = result.message

        if result.error:
            response["message"] = result.error

        if result.data:
            # Merge data into the response for backward compatibility
            response.update(result.data)

        return response

    def list_available_tools(self) -> List[str]:
        """Get a list of all available tool names.

        Returns:
            List of tool names
        """
        return self.registry.list_tools()

    def get_tool_registry(self) -> ToolRegistry:
        """Get direct access to the tool registry.

        Returns:
            The ToolRegistry instance
        """
        return self.registry

    # Legacy methods for backward compatibility
    def _handle_add_relationship_note(self, tool_input: Dict) -> Dict:
        """Handle adding a relationship note (legacy method)."""
        return self.handle_tool_call("add_relationship_note", tool_input)

    def _handle_get_relationship_notes(self, tool_input: Dict) -> Dict:
        """Handle retrieving relationship notes (legacy method)."""
        return self.handle_tool_call("get_relationship_notes", tool_input)

    def _handle_track_user_goal(self, tool_input: Dict) -> Dict:
        """Handle tracking user goals (legacy method)."""
        return self.handle_tool_call("track_user_goal", tool_input)

    def _infer_relationship_type(self, name: str) -> str:
        """Infer a relationship type for a given name (legacy method).

        This is a simplified version - in the full implementation,
        this would look at conversation context.

        Args:
            name: The name of the person

        Returns:
            Inferred relationship type
        """
        # For now, return a generic type
        # The full implementation in core.py looks at conversation history
        return "person"
