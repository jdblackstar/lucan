"""Base classes and decorators for tool implementation."""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Optional, Type, get_type_hints


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ToolValidationError(Exception):
    """Raised when tool input validation fails."""

    pass


class BaseTool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for registration."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the AI."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from execute method signature."""
        sig = inspect.signature(self.execute)
        type_hints = get_type_hints(self.execute)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "kwargs":
                continue

            param_type = type_hints.get(param_name, str)
            property_def = self._type_to_schema(param_type)

            # Extract description from docstring parameter documentation
            docstring = self.execute.__doc__ or ""
            param_description = self._extract_param_description(docstring, param_name)
            if param_description:
                property_def["description"] = param_description

            properties[param_name] = property_def

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def _type_to_schema(self, python_type: Type) -> Dict[str, Any]:
        """
        Convert a Python type to a JSON schema type.

        Args:
            python_type: The Python type to convert.

        Returns:
            A dictionary representing the JSON schema type.
        """
        # Mapping of Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            bool: "boolean",
            float: "number",
            list: "array",
        }

        # Handle typing generics (e.g., List[str])
        origin = getattr(python_type, "__origin__", None)
        if origin is list:
            return {"type": "array"}
        elif python_type in type_map:
            return {"type": type_map[python_type]}
        else:
            return {"type": "string"}  # Default fallback

    def _extract_param_description(
        self, docstring: str, param_name: str
    ) -> Optional[str]:
        """Extract parameter description from docstring."""
        if not docstring:
            return None

        lines = docstring.split("\n")
        in_args_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("Args:"):
                in_args_section = True
                continue
            elif in_args_section and line.startswith(f"{param_name}:"):
                return line.split(":", 1)[1].strip()
            elif (
                in_args_section
                and line
                and not line.startswith(" ")
                and ":" not in line
            ):
                # End of args section
                break

        return None

    def validate_input(self, **kwargs) -> None:
        """Validate input parameters against method signature."""
        sig = inspect.signature(self.execute)

        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param_name == "kwargs":
                continue

            if param.default is inspect.Parameter.empty and param_name not in kwargs:
                raise ToolValidationError(f"Missing required parameter: {param_name}")

        # Check for unexpected parameters
        valid_params = set(sig.parameters.keys()) - {"kwargs"}
        provided_params = set(kwargs.keys())
        unexpected = provided_params - valid_params

        if unexpected:
            raise ToolValidationError(f"Unexpected parameters: {unexpected}")


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to register a function as a tool."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_name = name or func.__name__
        wrapper._tool_description = description or func.__doc__ or ""
        return wrapper

    return decorator
