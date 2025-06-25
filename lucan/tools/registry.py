"""Tool registry for automatic tool discovery and management."""

from typing import Any, Dict, List

from .base import BaseTool, ToolResult, ToolValidationError


class ToolRegistry:
    """Registry for managing tool instances and execution."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool_instance: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool_instance.name] = tool_instance

        if self.debug:
            print(f"[DEBUG] Registered tool: {tool_instance.name}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for the OpenAI API."""
        definitions = []

        for tool in self._tools.values():
            definition = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_schema(),
                },
            }
            definitions.append(definition)

        return definitions

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with validation."""
        if tool_name not in self._tools:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]

        try:
            # Validate input
            tool.validate_input(**kwargs)

            # Execute tool
            result = tool.execute(**kwargs)

            if self.debug:
                print(f"[DEBUG] Tool {tool_name} executed: {result.success}")

            return result

        except ToolValidationError as e:
            return ToolResult(success=False, error=f"Validation error: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Execution error: {str(e)}")

    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a specific tool instance."""
        return self._tools.get(tool_name)
