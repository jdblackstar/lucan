"""
Global configuration settings for the Lucan project.

This module centralizes all configuration values, file paths, styling options,
and constants used throughout the project.
"""

from pathlib import Path
from typing import Dict

# === Project Structure ===
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
PERSONAS_DIR = MEMORY_DIR / "personas"
RELATIONSHIPS_DIR = MEMORY_DIR / "relationships"


# === Persona Configuration ===
DEFAULT_PERSONA_NAME = "lucan"
DEFAULT_PERSONA_PATH = PERSONAS_DIR / DEFAULT_PERSONA_NAME

# Required files for a valid persona
REQUIRED_PERSONA_FILES = {
    "personality": "personality.txt",
    "modifiers": "modifiers.txt",
}

# Persona template directory (to exclude from listings)
PERSONA_TEMPLATE_DIR = "template"


# === File Extensions and Names ===
PERSONALITY_FILE = "personality.txt"
MODIFIERS_FILE = "modifiers.txt"
RELATIONSHIPS_FILE = "relationships.json"


# === CLI Configuration ===
# Exit commands
EXIT_COMMANDS = ["quit", "exit", "bye"]

# Special commands
CLEAR_COMMAND = "/clear"
HELP_COMMANDS = ["/help", "help"]

# Command prefixes
COMMAND_PREFIX = "/"


# === Console Styling ===
class ConsoleStyles:
    """Console styling configuration for Rich components."""

    # Border styles
    WELCOME_BORDER = "cyan"
    LUCAN_RESPONSE_BORDER = "green"
    DEBUG_BORDER = "blue"
    ERROR_BORDER = "red"
    WARNING_BORDER = "yellow"
    HELP_BORDER = "yellow"
    MODIFIER_DEBUG_BORDER = "yellow"
    SYSTEM_PROMPT_DEBUG_BORDER = "magenta"

    # Text styles
    USER_PROMPT_STYLE = "[bold blue]You[/bold blue]"
    PERSONA_NAME_STYLE = "bold cyan"
    DIM_STYLE = "dim"
    ERROR_STYLE = "red"
    WARNING_STYLE = "yellow"
    SUCCESS_STYLE = "green"


# === Messages and Text Templates ===
class Messages:
    """Centralized message templates and text content."""

    # Welcome messages
    WELCOME_PREFIX = "Welcome to "
    WELCOME_SUFFIX = " - your loyal AI friend"

    # Status messages
    THINKING_STATUS = "{persona_name} is thinking..."
    GOODBYE_MESSAGE = "Goodbye! Take care."
    CONVERSATION_CLEARED = "Conversation history cleared."
    CHAT_INTERRUPTED = "Chat interrupted. Goodbye!"

    # Error messages
    GENERIC_ERROR = "An error occurred: {error}"
    PERSONA_NOT_FOUND = "Persona '{persona}' not found. Available personas: {available}"
    PERSONA_NOT_DIRECTORY = "Persona path '{path}' is not a directory"
    MISSING_PERSONALITY_FILE = "Persona '{persona}' is missing personality.txt file"
    MISSING_MODIFIERS_FILE = "Persona '{persona}' is missing modifiers.txt file"
    NO_PERSONAS_FOUND = "No personas found in memory/personas/"

    # Help content
    HELP_TEXT = """
**Available commands:**
- `/clear` - Clear conversation history
- `/help` - Show this help message  
- `quit`, `exit`, or `bye` - Exit the chat

**Persona selection:**
- Use `--persona <name>` when starting to choose a different persona
- Use `--list-personas` to see all available personas

**Tips:**
- Lucan responds best to direct, honest communication
- Try asking about goals, challenges, or decisions you're facing
- Lucan will challenge you constructively to help you grow
- Say things like "be more supportive" or "be less verbose" to adjust communication style
"""

    # CLI usage examples
    CLI_EXAMPLES = """
Examples:
  python main.py                    # Use default Lucan persona
  python main.py --persona coach    # Use the coach persona
  python main.py --persona memory/personas/therapist  # Use full path
  python main.py --list-personas    # Show available personas
  python main.py --debug            # Enable debug mode
"""


# === Panel Titles and Labels ===
class PanelTitles:
    """Titles and labels for Rich panels."""

    WELCOME_TITLE = "🤖 {persona_name} Chat"
    WELCOME_SUBTITLE = (
        "Type 'quit', 'exit', or 'bye' to leave • '/clear' to reset conversation"
    )

    LUCAN_RESPONSE_TITLE = "💭 {persona_name}"
    DEBUG_TITLE = "Debug"
    HELP_TITLE = "Help"

    # Debug panels
    MODIFIER_DEBUG_TITLE = "🔧 Debug - Modifier Values"
    SYSTEM_PROMPT_DEBUG_TITLE = "🔧 Debug - Generated System Prompt"

    # Persona listing
    AVAILABLE_PERSONAS_TITLE = "Available Personas:"


# === Debug Configuration ===
class DebugConfig:
    """Debug mode configuration."""

    SHOW_MODIFIERS = True
    SHOW_SYSTEM_PROMPT = True
    PANEL_PADDING = (1, 2)

    # Debug text templates
    MODIFIERS_HEADER = "**Loaded Modifiers:**\n\n"
    MODIFIER_ITEM = "- **{key}**: `{value}`\n"
    NO_MODIFIERS_TEXT = "*No modifiers loaded*\n"
    MODIFIERS_FILE_PATH = "\n*Modifiers file: {path}*"


# === API and Model Configuration ===
# Note: Add these when you integrate with AI models
class ModelConfig:
    """Configuration for AI model integration."""

    # Placeholder for future model settings
    DEFAULT_MODEL = "gpt-4"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7

    # System prompt templates
    SYSTEM_PROMPT_TEMPLATE = "{personality}\n\nModifiers: {modifiers}"


# === Validation Settings ===
class ValidationConfig:
    """Settings for input validation and constraints."""

    MIN_PERSONA_NAME_LENGTH = 1
    MAX_PERSONA_NAME_LENGTH = 50
    ALLOWED_PERSONA_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_-"

    # File size limits (in bytes)
    MAX_PERSONALITY_FILE_SIZE = 10 * 1024  # 10KB
    MAX_MODIFIERS_FILE_SIZE = 5 * 1024  # 5KB


# === Utility Functions ===
def get_default_persona_path() -> Path:
    """Get the default persona path."""
    return DEFAULT_PERSONA_PATH


def get_persona_files(persona_path: Path) -> Dict[str, Path]:
    """Get the expected file paths for a persona directory."""
    return {
        name: persona_path / filename
        for name, filename in REQUIRED_PERSONA_FILES.items()
    }


def is_command(user_input: str) -> bool:
    """Check if user input is a command."""
    stripped = user_input.strip().lower()
    return (
        stripped in EXIT_COMMANDS
        or stripped in HELP_COMMANDS
        or stripped.startswith(COMMAND_PREFIX)
    )


# === Export commonly used values ===
__all__ = [
    "PROJECT_ROOT",
    "MEMORY_DIR",
    "PERSONAS_DIR",
    "RELATIONSHIPS_DIR",
    "DEFAULT_PERSONA_NAME",
    "DEFAULT_PERSONA_PATH",
    "REQUIRED_PERSONA_FILES",
    "PERSONALITY_FILE",
    "MODIFIERS_FILE",
    "EXIT_COMMANDS",
    "CLEAR_COMMAND",
    "HELP_COMMANDS",
    "ConsoleStyles",
    "Messages",
    "PanelTitles",
    "DebugConfig",
    "ModelConfig",
    "ValidationConfig",
    "get_default_persona_path",
    "get_persona_files",
    "is_command",
]
