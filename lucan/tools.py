"""Tool definitions and handlers for Lucan conversations.

This module defines the tools available to Claude and handles
their execution when called during conversations.
"""

from typing import Dict, List

from .goals import GoalManager
from .relationships import RelationshipManager


class ToolManager:
    """Manages tool definitions and handles tool execution."""

    def __init__(
        self,
        relationship_manager: RelationshipManager,
        goal_manager: GoalManager,
        debug: bool = False,
    ):
        """Initialize the tool manager.

        Args:
            relationship_manager: Instance for managing relationship notes
            goal_manager: Instance for managing user goals
            debug: Whether to enable debug output
        """
        self.relationship_manager = relationship_manager
        self.goal_manager = goal_manager
        self.debug = debug

    def get_tool_definitions(self) -> List[Dict]:
        """Get the list of available tool definitions.

        Returns:
            List of tool definitions for the Anthropic API
        """
        return [
            {
                "name": "add_relationship_note",
                "description": "Add or update information about someone the user mentions. Use this tool when the user shares important information about people in their life, such as: relationship changes (breakups, marriages), life updates (new jobs, moves, health issues), new people they mention, or any significant details worth remembering. Examples:'My girlfriend and I broke up', 'My mom got a new job', 'I have a new therapist named Dr. Smith', 'My friend Sarah is getting married'. Don't announce when you're using this tool - just naturally remember the information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person's name"},
                        "relationship_type": {
                            "type": "string",
                            "description": "Their relationship to the user (e.g., friend, family, colleague, therapist, pet, partner, etc.)",
                        },
                        "note": {
                            "type": "string",
                            "description": "What to remember about this person (updates, context, interests, concerns, relationship changes, etc.)",
                        },
                    },
                    "required": ["name", "relationship_type", "note"],
                },
            },
            {
                "name": "get_relationship_notes",
                "description": "Look up information about someone the user asks about. Use this tool when the user asks questions like 'Do you know my mom?', 'Tell me about Sarah', 'Do you remember my therapist?', 'What do you know about my friend John?', etc. You can search by either a person's name (like 'Sarah') or by relationship type (like 'mom', 'therapist', 'friend'). Don't announce when you're using this tool - just naturally recall the information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The person's name OR their relationship type (e.g., 'Sarah', 'mom', 'therapist', 'friend', 'dog')",
                        }
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "track_user_goal",
                "description": "Track or update user goals when they mention wanting to work on something, achieve something, or change their focus. Use this when the user expresses goals like: 'I want to reduce my anxiety', 'My goal is to get promoted', 'I'm working on improving my relationships', 'I want to stop procrastinating', etc. This helps maintain goal consistency tracking. Don't announce when you're using this tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "The user's goal in clear, specific terms",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["add", "replace", "remove"],
                            "description": "Whether to add a new goal, replace all current goals with this one, or remove a specific goal",
                        },
                        "timeframe": {
                            "type": "string",
                            "enum": [
                                "short-term",
                                "medium-term",
                                "long-term",
                                "ongoing",
                            ],
                            "description": "The timeframe for this goal",
                        },
                    },
                    "required": ["goal", "action"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, tool_input: Dict) -> Dict:
        """Handle a tool call and return the result.

        Args:
            tool_name: The name of the tool being called
            tool_input: The input parameters for the tool

        Returns:
            Dict containing the tool result
        """
        if tool_name == "add_relationship_note":
            return self._handle_add_relationship_note(tool_input)
        elif tool_name == "get_relationship_notes":
            return self._handle_get_relationship_notes(tool_input)
        elif tool_name == "track_user_goal":
            return self._handle_track_user_goal(tool_input)
        else:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}

    def _handle_add_relationship_note(self, tool_input: Dict) -> Dict:
        """Handle adding a relationship note."""
        name = tool_input.get("name", "")
        relationship_type = tool_input.get("relationship_type", "")
        note = tool_input.get("note", "")

        if name and note:
            success = self.relationship_manager.add_note(name, relationship_type, note)
            if self.debug:
                print(f"[DEBUG] Added note for {name} ({relationship_type}): {note}")
            return {"success": success, "message": f"Added note for {name}"}
        else:
            return {"success": False, "message": "Missing required fields"}

    def _handle_get_relationship_notes(self, tool_input: Dict) -> Dict:
        """Handle retrieving relationship notes."""
        name = tool_input.get("name", "")
        if not name:
            return {"success": False, "message": "Name is required"}

        # First try direct name lookup
        notes = self.relationship_manager.get_notes(name)
        if self.debug:
            if notes:
                print(f"[DEBUG] Retrieved {len(notes['notes'])} notes for {name}")
            else:
                print(f"[DEBUG] No notes found for {name}")

        if notes:
            return {
                "success": True,
                "name": notes["name"],
                "relationship": notes["relationship"],
                "notes": notes["notes"],
            }

        # If no direct name match, try searching by relationship type
        if self.debug:
            print(f"[DEBUG] Trying relationship type search for '{name}'")

        relationship_results = self.relationship_manager.find_by_relationship_type(name)

        if relationship_results:
            if self.debug:
                names = [r["name"] for r in relationship_results]
                print(
                    f"[DEBUG] Found {len(relationship_results)} people with relationship '{name}': {names}"
                )

            # Return the first match (could be enhanced to return all matches)
            first_match = relationship_results[0]
            return {
                "success": True,
                "name": first_match["name"],
                "relationship": first_match["relationship"],
                "notes": first_match["notes"],
                "found_by": "relationship_type",  # Indicate how it was found
            }
        else:
            if self.debug:
                print(f"[DEBUG] No relationship type matches found for '{name}'")

            # If no notes found anywhere, create a basic note for this person
            relationship_type = self._infer_relationship_type(name)
            success = self.relationship_manager.add_note(
                name, relationship_type, "First mentioned in conversation"
            )
            if self.debug and success:
                print(f"[DEBUG] Created initial note for {name} as {relationship_type}")
            return {
                "success": True,
                "name": name,
                "relationship": relationship_type,
                "notes": ["First mentioned in conversation"],
            }

    def _handle_track_user_goal(self, tool_input: Dict) -> Dict:
        """Handle tracking user goals."""
        goal = tool_input.get("goal", "")
        action = tool_input.get("action", "add")
        timeframe = tool_input.get("timeframe")

        if goal:
            return self.goal_manager.handle_goal_tracking(goal, action, timeframe)
        else:
            return {"success": False, "message": "Goal is required"}

    def _infer_relationship_type(self, name: str) -> str:
        """Infer a relationship type for a given name.

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
