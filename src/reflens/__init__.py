"""
lab4_agents - A minimal AI agents framework

A didactic implementation showing how AI agents work with tools.
"""
from .agent import Agent
from .tool import Tool

__all__ = ["Agent", "Tool"]

# from .agent import Agent
# from .tool import Tool
# from .tui import TUI
# # from .tui_planner import PlannerTUI

# # Make submodules accessible
# from . import tools
# from . import subagents

# __all__ = [
#     "Agent",
#     "Tool",
#     "tools",
#     "subagents",
#     "TUI",
#     "PlannerTUI",
# ]
