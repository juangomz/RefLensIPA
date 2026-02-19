"""
Interactive TUI for PlannerAgent continuous interaction.

The PlannerAgent breaks down complex tasks into steps and executes them sequentially.

Do: uv run main-planner-tui.py --verbose to see tool calls and their results
if you don't want to see tool calls and their results, do: uv run main-planner-tui.py
"""

from lab4_agents.subagents.planner_agent import PlannerAgent
from lab4_agents.tools import calculator
from lab4_agents import PlannerTUI
import fire


def main(verbose: bool = False):
    # Define the planner agent with calculator tools
    planner = PlannerAgent(
        system_prompt="You are a helpful planning assistant that breaks down complex tasks into clear, sequential steps. Use the available tools to execute each step precisely.",
        executor_tools=[calculator.calculator_tool],
    )

    # Create and run the specialized PlannerTUI with the planner agent
    tui = PlannerTUI(agent=planner, verbose=verbose)
    tui.run()


if __name__ == "__main__":
    fire.Fire(main)
