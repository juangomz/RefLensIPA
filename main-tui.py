"""
Interactive TUI for continuous agent interaction.

Do: uv run main-tui.py --verbose to see tool calls and their results
if you dont want to see tool calls and their results, do: uv run main-tui.py

"""

from lab4_agents.agent import Agent
from lab4_agents.tools import calculator
from lab4_agents import TUI
import fire


def main(verbose: bool = False):
    # Define the agent with calculator tools
    agent = Agent(
        
        system_prompt="You are a helpful assistant that can answer questions and help with tasks related to math.",
        tools=[calculator.calculator_tool],
    )

    # Create and run the TUI with the agent
    tui = TUI(agent=agent, verbose=verbose)
    tui.run()


if __name__ == "__main__":
    fire.Fire(main)
