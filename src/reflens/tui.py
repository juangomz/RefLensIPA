"""
An interactive terminal UI that maintains conversation history and allows
the user to send multiple messages to the agent in a loop.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.reflens.agent import Agent


class TUI:
    """
    Interactive TUI for continuous agent operation.

    Maintains agent conversation history and allows multi-turn interaction
    with slash-command support for reset, help, and exit.
    """

    def __init__(self, agent: Agent | None = None, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        if agent is not None:
            self.agent = agent
        else:
            # Default: agent sin tools (para que arranque ya)
            self.agent = Agent(tools=[])

    def _print_header(self) -> None:
        title = Text("🤖 Agent TUI", style="bold cyan")
        self.console.print(Panel(title, expand=False, border_style="cyan"))
        self.console.print(
            Text(
                "Type your question or command. Type '/help' for available commands.",
                style="dim italic",
            )
        )
        self.console.print()

    def _print_help(self) -> None:
        self.console.print(Rule("Available Commands", style="cyan"))
        help_text = """
[cyan]/help[/cyan]    Show this help message
[cyan]/reset[/cyan]   Clear conversation history (keeps system prompt)
[cyan]/verbose[/cyan] Toggle verbose mode (shows tool calls)
[cyan]/quit[/cyan]    Exit the TUI
[cyan]/exit[/cyan]    Same as /quit
        """.strip()
        self.console.print(help_text)
        self.console.print()

    def _handle_command(self, line: str) -> bool:
        line_lower = line.strip().lower()

        if line_lower in ("/quit", "/exit"):
            self.console.print(Text("Goodbye!", style="cyan"))
            return True

        if line_lower == "/reset":
            self.agent.reset()
            self.console.print(Text("✓ Conversation history cleared.", style="green"))
            return False

        if line_lower == "/help":
            self._print_help()
            return False

        if line_lower == "/verbose":
            self.verbose = not self.verbose
            status = "enabled" if self.verbose else "disabled"
            self.console.print(Text(f"✓ Verbose mode {status}.", style="green"))
            return False

        self.console.print(Text(f"Unknown command: {line}", style="yellow"))
        self.console.print(Text("Type '/help' for available commands.", style="dim"))
        return False

    def _format_tool_calls(self, tool_calls: list[dict]) -> str:
        if not tool_calls:
            return ""

        lines = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call["tool"]
            arguments = call["arguments"]
            result = call.get("result", {})

            lines.append(f"[yellow]Tool {i}:[/yellow] [cyan]{tool_name}[/cyan]")
            lines.append("  [dim]Arguments:[/dim]")
            for key, value in arguments.items():
                lines.append(f"    {key}: {value}")

            if result:
                lines.append("  [dim]Result:[/dim]")
                for key, value in result.items():
                    lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def _send(self, user_input: str) -> None:
        try:
            self.console.print(Panel(user_input, title="You", border_style="blue", expand=False))

            messages_before = len(self.agent.messages)
            response = self.agent.run(user_input)

            if self.verbose:
                new_tool_calls = []
                for message in self.agent.messages[messages_before:]:
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            arguments = json.loads(tool_call["function"]["arguments"])
                            new_tool_calls.append({"tool": tool_name, "arguments": arguments})
                    elif message.get("role") == "tool" and new_tool_calls:
                        result = json.loads(message.get("content", "{}"))
                        new_tool_calls[-1]["result"] = result

                if new_tool_calls:
                    tool_calls_text = self._format_tool_calls(new_tool_calls)
                    self.console.print(Panel(tool_calls_text, title="Tool Calls", border_style="yellow", expand=False))

            self.console.print(Panel(response, title="Agent", border_style="green", expand=False))

        except Exception as e:
            self.console.print(
                Panel(f"Error: {str(e)}", title="Error", border_style="red", style="red")
            )

        self.console.print()

    def run(self) -> None:
        self._print_header()

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                    continue

                self._send(user_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print(Text("\nGoodbye!", style="cyan"))
                break


if __name__ == "__main__":
    TUI().run()