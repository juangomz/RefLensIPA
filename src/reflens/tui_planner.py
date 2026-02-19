"""
Interactive TUI specialized for PlannerAgent.

Displays the planning, execution, and re-planning process in a structured way.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from src.reflens.subagents.planner_agent import PlannerAgent


class PlannerTUI:
    """
    Interactive TUI for PlannerAgent operation.

    Displays plan, execution steps, and re-planning process in organized panels.
    """

    def __init__(self, agent: PlannerAgent | None = None, verbose: bool = False):
        """
        Initialize the PlannerTUI and agent.

        Args:
            agent: Optional pre-configured PlannerAgent instance
            verbose: If True, display tool calls and their arguments/results
        """
        self.console = Console()
        self.verbose = verbose
        if agent is not None:
            self.agent = agent
        else:
            # Default: Create a planner agent with calculator tools
            self.agent = PlannerAgent(
                executor_tools=[calculator.calculator_tool],
            )

    def _print_header(self) -> None:
        """Print welcome banner."""
        title = Text("🤖 PlannerAgent TUI", style="bold cyan")
        self.console.print(Panel(title, expand=False, border_style="cyan"))
        self.console.print(
            Text(
                "Type your question or command. Type '/help' for available commands.",
                style="dim italic",
            )
        )
        self.console.print()

    def _print_help(self) -> None:
        """Print available commands."""
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
        """
        Handle slash commands.

        Args:
            line: The user input line

        Returns:
            True if should quit, False otherwise
        """
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

        # Unknown command
        self.console.print(
            Text(f"Unknown command: {line}", style="yellow"),
        )
        self.console.print(Text("Type '/help' for available commands.", style="dim"))
        return False

    def _display_plan(self) -> None:
        """Display the generated plan."""
        if not self.agent.last_plan:
            return

        plan_lines = []
        for i, step in enumerate(self.agent.last_plan, 1):
            plan_lines.append(f"[yellow]{i}.[/yellow] {step}")

        plan_text = "\n".join(plan_lines)
        plan_panel = Panel(
            plan_text,
            title="📋 Plan",
            border_style="blue",
            expand=False,
        )
        self.console.print(plan_panel)

    def _display_execution(self) -> None:
        """Display the executed steps and their results."""
        if not self.agent.last_executed_steps:
            return

        for i, step_result in enumerate(self.agent.last_executed_steps, 1):
            status = step_result.get("status", "unknown")
            input_text = step_result.get("input", "")
            output = step_result.get("output", "")
            error = step_result.get("error", "")

            execution_lines = []

            # Display input
            if input_text:
                execution_lines.append("[dim]Input:[/dim]")
                execution_lines.append(input_text)
                execution_lines.append("")

            # Display result or error
            if status == "success":
                execution_lines.append("[dim]Result:[/dim]")
                execution_lines.append(f"[green]✓ {output}[/green]")
            else:
                execution_lines.append("[dim]Error:[/dim]")
                execution_lines.append(f"[red]✗ {error}[/red]")

            execution_text = "\n".join(execution_lines)
            execution_panel = Panel(
                execution_text,
                title=f"Step {i}",
                border_style="yellow",
                expand=False,
            )
            self.console.print(execution_panel)

    def _display_replans(self) -> None:
        """Display any re-planning iterations."""
        if not self.agent.last_replans or not self.agent.last_replans:
            return

        for attempt, replan in enumerate(self.agent.last_replans, 1):
            replan_lines = []
            for i, step in enumerate(replan, 1):
                replan_lines.append(f"[yellow]{i}.[/yellow] {step}")

            replan_text = "\n".join(replan_lines)
            replan_panel = Panel(
                replan_text,
                title=f"📋 Replan (Attempt {attempt})",
                border_style="magenta",
                expand=False,
            )
            self.console.print(replan_panel)

    def _send(self, user_input: str) -> None:
        """
        Send user input to agent and display response with planning details.

        Args:
            user_input: The user's message
        """
        try:
            # Show user input
            user_panel = Panel(
                user_input,
                title="You",
                border_style="blue",
                expand=False,
            )
            self.console.print(user_panel)

            # Get agent response
            response = self.agent.run(user_input)

            # Display plan
            self._display_plan()

            # Display execution
            self._display_execution()

            # Display replans if any
            self._display_replans()

            # Show agent response
            agent_panel = Panel(
                response,
                title="Agent",
                border_style="green",
                expand=False,
            )
            self.console.print(agent_panel)

        except Exception as e:
            # Show error
            error_text = f"Error: {str(e)}"
            error_panel = Panel(
                error_text,
                title="Error",
                border_style="red",
                style="red",
            )
            self.console.print(error_panel)

        self.console.print()  # Spacing

    def run(self) -> None:
        """Main TUI loop."""
        self._print_header()

        while True:
            try:
                # Get user input with styled prompt
                user_input = input("You: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                    continue

                # Send regular message to agent
                self._send(user_input)

            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl-C and Ctrl-D gracefully
                self.console.print(Text("\nGoodbye!", style="cyan"))
                break
