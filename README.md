# Lab4 Agents Framework

A minimal, educational AI agents framework demonstrating how AI agents interact with tools. Built with async-first design, this framework shows the core concepts of agentic workflows including tool integration, message history management, and iterative problem solving.

**Key Features:**

- 🤖 Extensible agent framework
- 🛠️ Easy tool definition and integration
- 💬 Interactive TUI for conversation-based workflows
- 🔄 Context-aware multi-turn conversations
- 📊 Langfuse observability integration
- ⚡ Retry logic with exponential backoff

## Quick Start

### Installation

```bash
uv sync --frozen
```

### Run HTTP app

```bash
uv run env PYTHONPATH=src uvicorn main_http.app:app --reload --port 8000
```

Then open http://127.0.0.1:8000

## Architecture

### Core Components

- **Agent** (`src/lab4_agents/agent.py`): Core agent loop with tool integration
- **Tool** (`src/lab4_agents/tool.py`): Tool definition and execution
- **TUI** (`src/lab4_agents/tui.py`): Interactive terminal UI
- **Tools** (`src/lab4_agents/tools/`): Available tool implementations

### Tools

- **Calculator**: Basic arithmetic operations (add, subtract, multiply, divide)

## Examples

### Simple Usage

```python
from lab4_agents.agent import Agent
from lab4_agents.tools import calculator

agent = Agent(tools=[calculator.calculator_tool])
response = agent.run("What is 10 + 20?")
print(response)  # Output: 10 + 20 = 30.
```

### Context-Aware Conversations

```python
agent = Agent(tools=[calculator.calculator_tool])

# First query
response1 = agent.run("What is 15 * 4?")
print(response1)  # Output: 15 × 4 = 60.

# Follow-up uses context from previous message
response2 = agent.run("Double that result")
print(response2)  # Output: Doubling 60 gives 120.
```

## Configuration

### Prerequisites

- Python 3.11+
- An LLM API key (GROQ, OpenAI, etc.)

### Environment Variables

Create a `.env` file with your LLM credentials. For GROQ:

```env
# GROQ API Configuration
GROQ_API_ENDPOINT=https://api.groq.com/openai/v1
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# Optional: Langfuse Observability
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

See `.env.example` for all available configuration options.

## Development

### Project Structure

```
lab4_agents/
├── main_http/                   # FastAPI app + web UI
├── src/lab4_agents/
│   ├── agent.py                 # Core Agent class
│   ├── tool.py                  # Tool definition
│   ├── tui.py                   # Interactive TUI class
│   ├── tools/
│   │   └── calculator.py        # Calculator tool
│   └── subagents/
│       ├── calculator_agent.py  # Specialized calculator agent
│       └── planner_agent.py     # Plan-and-execute agent
└── pyproject.toml               # Project metadata
```
