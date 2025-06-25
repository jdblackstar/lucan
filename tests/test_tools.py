"""Tests for tool functionality."""

import pytest

from lucan.tools import (
    AddRelationshipNoteTool,
    GetRelationshipNotesTool,
    ModifierAdjustmentTool,
    ToolRegistry,
    TrackUserGoalTool,
)


class MockRelationshipManager:
    """Mock relationship manager for testing."""

    def __init__(self):
        self.notes = {}

    def add_note(self, name: str, relationship_type: str, note: str) -> bool:
        if name not in self.notes:
            self.notes[name] = {"relationship": relationship_type, "notes": []}
        self.notes[name]["notes"].append(note)
        return True

    def get_notes(self, name: str):
        if name in self.notes:
            return {
                "name": name,
                "relationship": self.notes[name]["relationship"],
                "notes": self.notes[name]["notes"],
            }
        return None

    def find_by_relationship_type(self, relationship_type: str):
        results = []
        for name, data in self.notes.items():
            if data["relationship"].lower() == relationship_type.lower():
                results.append(
                    {
                        "name": name,
                        "relationship": data["relationship"],
                        "notes": data["notes"],
                    }
                )
        return results


class MockGoalManager:
    """Mock goal manager for testing."""

    def __init__(self):
        self.goals = []

    def handle_goal_tracking(self, goal: str, action: str, timeframe: str = None):
        if action == "add":
            self.goals.append({"goal": goal, "timeframe": timeframe})
            return {"success": True, "message": f"Added goal: {goal}"}
        elif action == "remove":
            self.goals = [g for g in self.goals if g["goal"] != goal]
            return {"success": True, "message": f"Removed goal: {goal}"}
        elif action == "replace":
            self.goals = [{"goal": goal, "timeframe": timeframe}]
            return {"success": True, "message": f"Replaced goals with: {goal}"}
        else:
            return {"success": False, "message": "Invalid action"}

    def get_active_goals(self):
        return [g["goal"] for g in self.goals]


class MockLucanInstance:
    """Mock Lucan instance for testing modifier tools."""

    def __init__(self):
        self.modifiers = {
            "warmth": 0,
            "challenge": 0,
            "verbosity": 0,
            "emotional_depth": 0,
            "structure": 0,
        }

    def save_modifiers(self):
        pass  # Mock implementation


@pytest.fixture
def tool_registry():
    """Create a tool registry for testing."""
    return ToolRegistry(debug=True)


@pytest.fixture
def mock_relationship_manager():
    """Create a mock relationship manager."""
    return MockRelationshipManager()


@pytest.fixture
def mock_goal_manager():
    """Create a mock goal manager."""
    return MockGoalManager()


@pytest.fixture
def mock_lucan_instance():
    """Create a mock Lucan instance."""
    return MockLucanInstance()


@pytest.fixture
def add_note_tool(mock_relationship_manager):
    """Create an add relationship note tool."""
    return AddRelationshipNoteTool(mock_relationship_manager, debug=True)


@pytest.fixture
def get_notes_tool(mock_relationship_manager):
    """Create a get relationship notes tool."""
    return GetRelationshipNotesTool(mock_relationship_manager, debug=True)


@pytest.fixture
def modifier_tool(mock_lucan_instance):
    """Create a modifier adjustment tool."""
    return ModifierAdjustmentTool(mock_lucan_instance, debug=True)


@pytest.fixture
def goal_tool(mock_goal_manager):
    """Create a goal tracking tool."""
    return TrackUserGoalTool(mock_goal_manager, debug=True)


def test_tool_registration(tool_registry, add_note_tool):
    """Test tool registration."""
    tool_registry.register_tool(add_note_tool)

    assert "add_relationship_note" in tool_registry.list_tools()
    definitions = tool_registry.get_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == "add_relationship_note"


def test_add_relationship_note_execution(tool_registry, add_note_tool):
    """Test successful relationship note addition."""
    tool_registry.register_tool(add_note_tool)

    result = tool_registry.execute_tool(
        "add_relationship_note",
        name="Alice",
        relationship_type="friend",
        note="Met at work",
    )

    assert result.success
    assert result.data["name"] == "Alice"
    assert result.data["relationship_type"] == "friend"
    assert result.data["note"] == "Met at work"


def test_get_relationship_notes_execution(
    tool_registry, get_notes_tool, mock_relationship_manager
):
    """Test retrieving relationship notes."""
    # Add some test data
    mock_relationship_manager.add_note("Alice", "friend", "Met at work")

    tool_registry.register_tool(get_notes_tool)

    result = tool_registry.execute_tool("get_relationship_notes", name="Alice")

    assert result.success
    assert result.data["name"] == "Alice"
    assert result.data["relationship"] == "friend"
    assert "Met at work" in result.data["notes"]


def test_modifier_adjustment_tool(tool_registry, modifier_tool):
    """Test modifier adjustment functionality."""
    tool_registry.register_tool(modifier_tool)

    # Test adjust action
    result = tool_registry.execute_tool(
        "adjust_modifier",
        action="adjust",
        modifier="warmth",
        adjustment=2,
        reason="User wants more warmth",
    )

    assert result.success
    assert result.data["modifier"] == "warmth"
    assert result.data["old_value"] == 0
    assert result.data["new_value"] == 2
    assert result.data["is_large_change"]


def test_modifier_set_tool(tool_registry, modifier_tool):
    """Test modifier set functionality."""
    tool_registry.register_tool(modifier_tool)

    # Test set action
    result = tool_registry.execute_tool(
        "adjust_modifier",
        action="set",
        modifier="verbosity",
        value=-1,
        reason="User wants less verbosity",
    )

    assert result.success
    assert result.data["modifier"] == "verbosity"
    assert result.data["new_value"] == -1
    assert result.data["is_large_change"]


def test_goal_tracking_tool(tool_registry, goal_tool):
    """Test goal tracking functionality."""
    tool_registry.register_tool(goal_tool)

    # Test adding a goal
    result = tool_registry.execute_tool(
        "track_user_goal",
        goal="Get promoted at work",
        action="add",
        timeframe="medium-term",
    )

    assert result.success
    assert result.data["goal"] == "Get promoted at work"
    assert result.data["action"] == "add"
    assert result.data["timeframe"] == "medium-term"


def test_tool_validation_error(tool_registry, add_note_tool):
    """Test tool validation errors."""
    tool_registry.register_tool(add_note_tool)

    # Missing required parameter
    result = tool_registry.execute_tool(
        "add_relationship_note",
        name="Alice",
        # Missing relationship_type and note
    )

    assert not result.success
    assert "Validation error" in result.error


def test_unknown_tool(tool_registry):
    """Test execution of unknown tool."""
    result = tool_registry.execute_tool("unknown_tool")

    assert not result.success
    assert "Unknown tool" in result.error


def test_invalid_modifier(tool_registry, modifier_tool):
    """Test invalid modifier name."""
    tool_registry.register_tool(modifier_tool)

    result = tool_registry.execute_tool(
        "adjust_modifier", action="adjust", modifier="invalid_modifier", adjustment=1
    )

    assert not result.success
    assert "Unknown modifier" in result.error


def test_modifier_clamping(tool_registry, modifier_tool, mock_lucan_instance):
    """Test that modifier values are clamped to valid range."""
    tool_registry.register_tool(modifier_tool)

    # Try to set value beyond valid range
    result = tool_registry.execute_tool(
        "adjust_modifier",
        action="set",
        modifier="warmth",
        value=10,  # Should be clamped to 3
    )

    assert result.success
    assert result.data["new_value"] == 3  # Clamped to max


def test_tool_schema_generation(add_note_tool):
    """Test that tools generate proper JSON schemas."""
    schema = add_note_tool.get_schema()

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema

    # Check required parameters
    assert "name" in schema["required"]
    assert "relationship_type" in schema["required"]
    assert "note" in schema["required"]

    # Check parameter types
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["relationship_type"]["type"] == "string"
    assert schema["properties"]["note"]["type"] == "string"


def test_multiple_tools_registration(
    tool_registry, add_note_tool, get_notes_tool, modifier_tool
):
    """Test registering multiple tools."""
    tool_registry.register_tool(add_note_tool)
    tool_registry.register_tool(get_notes_tool)
    tool_registry.register_tool(modifier_tool)

    tools = tool_registry.list_tools()
    assert len(tools) == 3
    assert "add_relationship_note" in tools
    assert "get_relationship_notes" in tools
    assert "adjust_modifier" in tools


def test_relationship_type_search(
    tool_registry, get_notes_tool, mock_relationship_manager
):
    """Test searching by relationship type."""
    # Add some test data
    mock_relationship_manager.add_note("Dr. Smith", "therapist", "Very helpful")
    mock_relationship_manager.add_note("Dr. Jones", "doctor", "Primary care")

    tool_registry.register_tool(get_notes_tool)

    # Search by relationship type
    result = tool_registry.execute_tool("get_relationship_notes", name="therapist")

    assert result.success
    assert result.data["name"] == "Dr. Smith"
    assert result.data["found_by"] == "relationship_type"
