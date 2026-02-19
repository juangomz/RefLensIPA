"""
Tool definition for AI agents.

A Tool represents a function that an AI agent can invoke to complete tasks.
"""

from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel


@dataclass
class Tool:
    """
    Represents a tool that the agent can use.

    Attributes:
        name: Unique identifier for the tool
        description: What the tool does (sent to the LLM)
        parameters: Pydantic BaseModel class describing the input parameters
        func: Python function to execute when the tool is called
    """

    name: str
    description: str
    parameters: Type[BaseModel]
    func: Callable[[dict[str, Any] | None], dict[str, Any]]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI's tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with the given arguments."""
        return self.func(arguments)
