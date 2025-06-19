"""Tools for managing relationship information."""

from ..relationships import RelationshipManager
from .base import BaseTool, ToolResult


class AddRelationshipNoteTool(BaseTool):
    """Tool for adding relationship notes."""

    def __init__(self, relationship_manager: RelationshipManager, debug: bool = False):
        self.relationship_manager = relationship_manager
        self.debug = debug

    @property
    def name(self) -> str:
        return "add_relationship_note"

    @property
    def description(self) -> str:
        return (
            "Add or update information about someone the user mentions. "
            "Use this tool when the user shares important information about "
            "people in their life, such as: relationship changes (breakups, marriages), "
            "life updates (new jobs, moves, health issues), new people they mention, "
            "or any significant details worth remembering. Examples: 'My girlfriend and I broke up', "
            "'My mom got a new job', 'I have a new therapist named Dr. Smith', "
            "'My friend Sarah is getting married'. Don't announce when you're using this tool - "
            "just naturally remember the information."
        )

    def execute(self, name: str, relationship_type: str, note: str) -> ToolResult:
        """Add a relationship note.

        Args:
            name: The person's name
            relationship_type: Their relationship to the user (e.g., friend, family, colleague, therapist, pet, partner, etc.)
            note: What to remember about this person (updates, context, interests, concerns, relationship changes, etc.)
        """
        try:
            if not name.strip():
                return False

            success = self.relationship_manager.add_note(name, relationship_type, note)

            if self.debug:
                print(f"[DEBUG] Added note for {name} ({relationship_type}): {note}")

            if success:
                return ToolResult(
                    success=True,
                    data={
                        "name": name,
                        "relationship_type": relationship_type,
                        "note": note,
                    },
                    message=f"Added note for {name}",
                )
            else:
                return ToolResult(
                    success=False, error="Failed to save relationship note"
                )

        except Exception as e:
            return ToolResult(
                success=False, error=f"Error adding relationship note: {str(e)}"
            )


class GetRelationshipNotesTool(BaseTool):
    """Tool for retrieving relationship notes."""

    def __init__(self, relationship_manager: RelationshipManager, debug: bool = False):
        self.relationship_manager = relationship_manager
        self.debug = debug

    @property
    def name(self) -> str:
        return "get_relationship_notes"

    @property
    def description(self) -> str:
        return (
            "Look up information about someone the user asks about. "
            "Use this tool when the user asks questions like 'Do you know my mom?', "
            "'Tell me about Sarah', 'Do you remember my therapist?', "
            "'What do you know about my friend John?', etc. You can search by either "
            "a person's name (like 'Sarah') or by relationship type (like 'mom', 'therapist', 'friend'). "
            "Don't announce when you're using this tool - just naturally recall the information."
        )

    def execute(self, name: str) -> ToolResult:
        """Get relationship notes for a person.

        Args:
            name: The person's name OR their relationship type (e.g., 'Sarah', 'mom', 'therapist', 'friend', 'dog')
        """
        try:
            if not name.strip():
                return ToolResult(success=False, error="Name cannot be empty")

            # First try direct name lookup
            notes = self.relationship_manager.get_notes(name)

            if self.debug:
                if notes:
                    print(f"[DEBUG] Retrieved {len(notes['notes'])} notes for {name}")
                else:
                    print(f"[DEBUG] No notes found for {name}")

            if notes:
                return ToolResult(
                    success=True,
                    data={
                        "name": notes["name"],
                        "relationship": notes["relationship"],
                        "notes": notes["notes"],
                        "found_by": "name",
                    },
                )

            # If no direct name match, try searching by relationship type
            if self.debug:
                print(f"[DEBUG] Trying relationship type search for '{name}'")

            relationship_results = self.relationship_manager.find_by_relationship_type(
                name
            )

            if relationship_results:
                if self.debug:
                    names = [r["name"] for r in relationship_results]
                    print(
                        f"[DEBUG] Found {len(relationship_results)} people with relationship '{name}': {names}"
                    )

                # Return the first match (could be enhanced to return all matches)
                first_match = relationship_results[0]
                return ToolResult(
                    success=True,
                    data={
                        "name": first_match["name"],
                        "relationship": first_match["relationship"],
                        "notes": first_match["notes"],
                        "found_by": "relationship_type",
                    },
                )
            else:
                if self.debug:
                    print(f"[DEBUG] No relationship type matches found for '{name}'")

                # First: Return clean "not found" result
                result = ToolResult(
                    success=True,
                    data={
                        "name": name,
                        "relationship": None,
                        "notes": [],
                        "found_by": "not_found",
                    },
                    message=f"No information found about {name}",
                )

                # Then: Try to create empty record in background (can fail silently)
                try:
                    relationship_type = self._infer_relationship_type(name)
                    success = self.relationship_manager.add_note(
                        name, relationship_type, ""
                    )
                    if self.debug and success:
                        print(f"[DEBUG] Created empty relationship record for {name}")
                    elif self.debug:
                        print(f"[DEBUG] Failed to create empty record for {name}")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Exception creating empty record: {e}")

                return result

        except Exception as e:
            return ToolResult(
                success=False, error=f"Error retrieving relationship notes: {str(e)}"
            )

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
