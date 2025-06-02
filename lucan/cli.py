import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .core import LucanChat


def _get_personas_directory() -> Path:
    """Get the path to the personas directory."""
    return Path(__file__).parent.parent / "memory" / "personas"


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
        if persona_dir.is_dir() and persona_dir.name != "template":
            personality_file = persona_dir / "personality.txt"
            modifiers_file = persona_dir / "modifiers.txt"

            if personality_file.exists() and modifiers_file.exists():
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
        persona_path = _get_personas_directory() / persona_input

    # Verify the persona exists and is valid
    if not persona_path.exists():
        available = _list_available_personas()
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            f"Persona '{persona_input}' not found. Available personas: {available_text}"
        )

    if not persona_path.is_dir():
        raise FileNotFoundError(f"Persona path '{persona_path}' is not a directory")

    personality_file = persona_path / "personality.txt"
    modifiers_file = persona_path / "modifiers.txt"

    if not personality_file.exists():
        raise FileNotFoundError(
            f"Persona '{persona_input}' is missing personality.txt file"
        )

    if not modifiers_file.exists():
        raise FileNotFoundError(
            f"Persona '{persona_input}' is missing modifiers.txt file"
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
            persona_path = (
                Path(__file__).parent.parent / "memory" / "personas" / "lucan"
            )
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
        debug_text = "**Loaded Modifiers:**\n\n"

        for key, value in modifiers.items():
            debug_text += f"- **{key}**: `{value}`\n"

        if not modifiers:
            debug_text += "*No modifiers loaded*\n"

        debug_text += f"\n*Modifiers file: {self.chat.lucan.modifiers_file}*"

        self.console.print(
            Panel(
                Markdown(debug_text),
                title="🔧 Debug - Modifier Values",
                border_style="yellow",
                padding=(1, 2),
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
                title="🔧 Debug - Generated System Prompt",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    def _display_welcome(self) -> None:
        """
        Display the welcome message.
        """
        persona_name = self.chat.lucan.personality.get("name", "Lucan")

        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append(persona_name, style="bold cyan")
        welcome_text.append(" - your loyal AI friend", style="white")

        panel = Panel(
            welcome_text,
            title=f"🤖 {persona_name} Chat",
            subtitle="Type 'quit', 'exit', or 'bye' to leave • '/clear' to reset conversation",
            border_style="cyan",
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
                    title=f"💭 {persona_name}",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            self.console.print(f"[dim]{message}[/dim]")

    def _get_user_input(self) -> str:
        """
        Get input from the user with a nice prompt.
        """
        return Prompt.ask("[bold blue]You[/bold blue]", console=self.console)

    def _handle_command(self, user_input: str) -> bool:
        """
        Handle special commands. Returns True if it was a command, False otherwise.

        Args:
            user_input: The user's input

        Returns:
            True if a command was handled, False if it's a regular message
        """
        command = user_input.strip().lower()

        if command in ["quit", "exit", "bye"]:
            self.console.print("[dim]Goodbye! Take care.[/dim]")
            return True
        elif command == "/clear":
            self.chat.clear_history()
            self.console.clear()
            self._display_welcome()
            self.console.print("[dim]Conversation history cleared.[/dim]")
            return False
        elif command in ["/help", "help"]:
            help_text = """
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
            self.console.print(
                Panel(Markdown(help_text), title="Help", border_style="yellow")
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
                with self.console.status(f"[dim]{persona_name} is thinking...[/dim]"):
                    response = self.chat.send_message(user_input)

                # Display Lucan's response
                self._display_message(response)

        except KeyboardInterrupt:
            self.console.print("\n[dim]Chat interrupted. Goodbye![/dim]")
        except Exception as e:
            self.console.print(f"[red]An error occurred: {str(e)}[/red]")
            sys.exit(1)


def _run_cli() -> None:
    """
    Entry point for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Lucan CLI - Your adaptive AI friend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default Lucan persona
  python main.py --persona coach    # Use the coach persona
  python main.py --persona memory/personas/therapist  # Use full path
  python main.py --list-personas    # Show available personas
  python main.py --debug            # Enable debug mode
        """.strip(),
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
            console.print("[yellow]No personas found in memory/personas/[/yellow]")
            console.print(
                "[dim]Create personas using the template in memory/personas/template/[/dim]"
            )
        else:
            console.print("[bold]Available Personas:[/bold]")
            for persona in available_personas:
                console.print(f"  • {persona}")
            console.print("\n[dim]Use with: python main.py --persona <name>[/dim]")

        return

    # Resolve persona path
    persona_path = None
    if args.persona:
        try:
            persona_path = _resolve_persona_path(args.persona)
        except FileNotFoundError as e:
            console = Console()
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Use --list-personas to see available options[/dim]")
            sys.exit(1)

    # Create and run CLI
    cli = LucanCLI(persona_path=persona_path, debug=args.debug)
    cli.run()


if __name__ == "__main__":
    _run_cli()
