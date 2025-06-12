import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .config import (
    CLEAR_COMMAND,
    EXIT_COMMANDS,
    HELP_COMMANDS,
    PERSONA_TEMPLATE_DIR,
    PERSONAS_DIR,
    ConsoleStyles,
    DebugConfig,
    Messages,
    PanelTitles,
    get_default_persona_path,
    get_persona_files,
)
from .core import LucanChat


def _get_personas_directory() -> Path:
    """Get the path to the personas directory."""
    return PERSONAS_DIR


def _list_available_personas() -> list[str]:
    """
    Get a list of available persona names.

    Returns:
        List of persona directory names that contain both personality.txt and modifiers.txt
    """
    personas_dir = _get_personas_directory()
    available_personas = []

    if not personas_dir.exists():
        return available_personas

    for persona_dir in personas_dir.iterdir():
        # Skip template directory
        if persona_dir.is_dir() and persona_dir.name != PERSONA_TEMPLATE_DIR:
            persona_files = get_persona_files(persona_dir)

            if all(file_path.exists() for file_path in persona_files.values()):
                available_personas.append(persona_dir.name)

    return sorted(available_personas)


def _resolve_persona_path(persona_input: str) -> Path:
    """
    Resolve persona input to a full path.

    Args:
        persona_input: Either a persona name (like "coach") or a full/relative path

    Returns:
        Path to the persona directory

    Raises:
        FileNotFoundError: If the persona doesn't exist or is invalid
    """
    # If it's already a path (contains slashes), use it directly
    if "/" in persona_input or "\\" in persona_input:
        persona_path = Path(persona_input)
    else:
        # Treat it as a persona name
        persona_path = PERSONAS_DIR / persona_input

    # Verify the persona exists and is valid
    if not persona_path.exists():
        available = _list_available_personas()
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            Messages.PERSONA_NOT_FOUND.format(
                persona=persona_input, available=available_text
            )
        )

    if not persona_path.is_dir():
        raise FileNotFoundError(
            Messages.PERSONA_NOT_DIRECTORY.format(path=persona_path)
        )

    persona_files = get_persona_files(persona_path)

    if not persona_files["personality"].exists():
        raise FileNotFoundError(
            Messages.MISSING_PERSONALITY_FILE.format(persona=persona_input)
        )

    if not persona_files["modifiers"].exists():
        raise FileNotFoundError(
            Messages.MISSING_MODIFIERS_FILE.format(persona=persona_input)
        )

    return persona_path


class LucanCLI:
    """
    Command-line interface for chatting with Lucan.
    """

    def __init__(self, persona_path: Optional[str] = None, debug: bool = False):
        """
        Initialize the CLI with an optional persona path and debug mode.

        Args:
            persona_path: Path to persona directory. Defaults to memory/personas/lucan/
            debug: Whether to display debug information on startup
        """
        self.console = Console()
        self.debug = debug

        if persona_path is None:
            persona_path = get_default_persona_path()
        else:
            persona_path = Path(persona_path)

        self.chat = LucanChat(persona_path, debug=debug)

        if self.debug:
            # Debug: Display loaded modifiers
            self._display_debug_modifiers()

            # Debug: Display generated system prompt
            self._display_debug_system_prompt()

    def _display_debug_modifiers(self) -> None:
        """
        Display the currently loaded modifier values for debugging.
        """
        modifiers = self.chat.lucan.modifiers
        debug_text = DebugConfig.MODIFIERS_HEADER

        for key, value in modifiers.items():
            debug_text += DebugConfig.MODIFIER_ITEM.format(key=key, value=value)

        if not modifiers:
            debug_text += DebugConfig.NO_MODIFIERS_TEXT

        debug_text += DebugConfig.MODIFIERS_FILE_PATH.format(
            path=self.chat.lucan.modifiers_file
        )

        self.console.print(
            Panel(
                Markdown(debug_text),
                title=PanelTitles.MODIFIER_DEBUG_TITLE,
                border_style=ConsoleStyles.MODIFIER_DEBUG_BORDER,
                padding=DebugConfig.PANEL_PADDING,
            )
        )

    def _display_debug_system_prompt(self) -> None:
        """
        Display the generated system prompt for debugging.
        """
        system_prompt = self.chat.system_prompt

        self.console.print(
            Panel(
                system_prompt,
                title=PanelTitles.SYSTEM_PROMPT_DEBUG_TITLE,
                border_style=ConsoleStyles.SYSTEM_PROMPT_DEBUG_BORDER,
                padding=DebugConfig.PANEL_PADDING,
            )
        )

    def _display_welcome(self) -> None:
        """
        Display the welcome message.
        """
        persona_name = self.chat.lucan.personality.get("name", "Lucan")

        welcome_text = Text()
        welcome_text.append(Messages.WELCOME_PREFIX, style="white")
        welcome_text.append(persona_name, style=ConsoleStyles.PERSONA_NAME_STYLE)
        welcome_text.append(Messages.WELCOME_SUFFIX, style="white")

        panel = Panel(
            welcome_text,
            title=PanelTitles.WELCOME_TITLE.format(persona_name=persona_name),
            subtitle=PanelTitles.WELCOME_SUBTITLE,
            border_style=ConsoleStyles.WELCOME_BORDER,
        )
        self.console.print(panel)

    def _display_message(self, message: str, sender: str = "lucan") -> None:
        """
        Display a message from Lucan or system.

        Args:
            message: The message to display
            sender: Either 'lucan' or 'system'
        """
        if sender == "lucan":
            persona_name = self.chat.lucan.personality.get("name", "Lucan")
            # Display Lucan's response as markdown for better formatting
            self.console.print(
                Panel(
                    Markdown(message),
                    title=PanelTitles.LUCAN_RESPONSE_TITLE.format(
                        persona_name=persona_name
                    ),
                    border_style=ConsoleStyles.LUCAN_RESPONSE_BORDER,
                    padding=DebugConfig.PANEL_PADDING,
                )
            )
        else:
            self.console.print(
                Panel(
                    Markdown(message),
                    title=PanelTitles.DEBUG_TITLE,
                    border_style=ConsoleStyles.DEBUG_BORDER,
                    padding=DebugConfig.PANEL_PADDING,
                )
            )
            # self.console.print(f"[dim]{message}[/dim]") # previous version

    def _get_user_input(self) -> str:
        """
        Get input from the user with a nice prompt.
        """
        return Prompt.ask(ConsoleStyles.USER_PROMPT_STYLE, console=self.console)

    def _handle_command(self, user_input: str) -> bool:
        """
        Handle special commands. Returns True if it was a command, False otherwise.

        Args:
            user_input: The user's input

        Returns:
            True if a command was handled, False if it's a regular message
        """
        command = user_input.strip().lower()

        if command in EXIT_COMMANDS:
            self.console.print(
                f"[{ConsoleStyles.DIM_STYLE}]{Messages.GOODBYE_MESSAGE}[/{ConsoleStyles.DIM_STYLE}]"
            )
            return True
        elif command == CLEAR_COMMAND:
            self.chat.clear_history()
            self.console.clear()
            self._display_welcome()
            self.console.print(
                f"[{ConsoleStyles.DIM_STYLE}]{Messages.CONVERSATION_CLEARED}[/{ConsoleStyles.DIM_STYLE}]"
            )
            return False
        elif command in HELP_COMMANDS:
            self.console.print(
                Panel(
                    Markdown(Messages.HELP_TEXT),
                    title=PanelTitles.HELP_TITLE,
                    border_style=ConsoleStyles.HELP_BORDER,
                )
            )
            return False

        return False

    def run(self) -> None:
        """
        Run the main chat loop.
        """
        self._display_welcome()

        try:
            while True:
                user_input = self._get_user_input()

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle commands
                if self._handle_command(user_input):
                    break

                # Get response from Lucan
                persona_name = self.chat.lucan.personality.get("name", "Lucan")
                with self.console.status(
                    f"[{ConsoleStyles.DIM_STYLE}]{Messages.THINKING_STATUS.format(persona_name=persona_name)}[/{ConsoleStyles.DIM_STYLE}]"
                ):
                    response = self.chat.send_message(user_input)

                # Display Lucan's response
                self._display_message(response)

        except KeyboardInterrupt:
            self.console.print(
                f"\n[{ConsoleStyles.DIM_STYLE}]{Messages.CHAT_INTERRUPTED}[/{ConsoleStyles.DIM_STYLE}]"
            )
        except Exception as e:
            self.console.print(
                f"[{ConsoleStyles.ERROR_STYLE}]{Messages.GENERIC_ERROR.format(error=str(e))}[/{ConsoleStyles.ERROR_STYLE}]"
            )
            sys.exit(1)


def _run_cli() -> None:
    """
    Entry point for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Lucan CLI - Your adaptive AI friend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=Messages.CLI_EXAMPLES.strip(),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--persona", type=str, help="Specify a persona by name (e.g., 'coach') or path"
    )
    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List all available personas and exit",
    )

    args = parser.parse_args()

    # Handle --list-personas
    if args.list_personas:
        console = Console()
        available_personas = _list_available_personas()

        if not available_personas:
            console.print(
                f"[{ConsoleStyles.WARNING_STYLE}]{Messages.NO_PERSONAS_FOUND}[/{ConsoleStyles.WARNING_STYLE}]"
            )
            console.print(
                f"[{ConsoleStyles.DIM_STYLE}]Create personas using the template in memory/personas/template/[/{ConsoleStyles.DIM_STYLE}]"
            )
        else:
            console.print(f"[bold]{PanelTitles.AVAILABLE_PERSONAS_TITLE}[/bold]")
            for persona in available_personas:
                console.print(f"  • {persona}")
            console.print(
                f"\n[{ConsoleStyles.DIM_STYLE}]Use with: python main.py --persona <name>[/{ConsoleStyles.DIM_STYLE}]"
            )

        return

    # Resolve persona path
    persona_path = None
    if args.persona:
        try:
            persona_path = _resolve_persona_path(args.persona)
        except FileNotFoundError as e:
            console = Console()
            console.print(
                f"[{ConsoleStyles.ERROR_STYLE}]Error: {e}[/{ConsoleStyles.ERROR_STYLE}]"
            )
            console.print(
                f"[{ConsoleStyles.DIM_STYLE}]Use --list-personas to see available options[/{ConsoleStyles.DIM_STYLE}]"
            )
            sys.exit(1)

    # Create and run CLI
    cli = LucanCLI(persona_path=persona_path, debug=args.debug)
    cli.run()


if __name__ == "__main__":
    _run_cli()
