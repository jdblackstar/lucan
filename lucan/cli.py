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
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append("Lucan", style="bold cyan")
        welcome_text.append(" - your loyal AI friend", style="white")

        panel = Panel(
            welcome_text,
            title="🤖 Lucan Chat",
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
            # Display Lucan's response as markdown for better formatting
            self.console.print(
                Panel(
                    Markdown(message),
                    title="💭 Lucan",
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

            **Tips:**
            - Lucan responds best to direct, honest communication
            - Try asking about goals, challenges, or decisions you're facing
            - Lucan will challenge you constructively to help you grow
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
                with self.console.status("[dim]Lucan is thinking...[/dim]"):
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
    parser = argparse.ArgumentParser(description="Lucan CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    cli = LucanCLI(debug=args.debug)
    cli.run()


if __name__ == "__main__":
    _run_cli()
