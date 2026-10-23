# Customer Support Agent

A stateful, observable, end-to-end AI customer support agent built with **LangGraph**. This project implements a multi-stage pipeline that ingests customer messages, retrieves relevant order and policy information, makes decisions using an LLM, and generates appropriate responses, all while tracking detailed execution traces.

## Features

- **Stateful Conversation Tracking**: Preserves context across multiple turns via `AgentState`.
- **Modular Node Architecture**: Composed of distinct graph nodes for routing, retrieval, reasoning, and response generation.
- **Policy-Aware Responses**: Uses Retrieval-Augmented Generation (RAG) to ground decisions in company policies.
- **Observability**: Built-in logging and tracing to track every step of the agent's execution.
- **Testing Framework**: Unit tests for each node and full graph integration tests.

## Table of Contents

- [Features](#features)
- [Structure](#structure)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Structure

```
src/
├── agent/
│   ├── __init__.py
│   ├── llm_client.py      # LLM Abstraction and Mock Implementation
│   ├── router.py          # Intent Classification Node
│   ├── retrieval.py       # RAG and State Enrichment Node
│   ├── reasoner.py        # Decision Making Node
│   ├── response_builder.py # Final Response Generation Node
│   ├── graph.py           # LangGraph Workflow Definition
│   ├── state.py           # AgentState Schema (TypedDict)
│   └── observability/
│       ├── __init__.py
│       └── logger.py      # With_logging Decorator
├── cli.py                 # Command-line Interface
└── data/
    ├── sample_orders.json   # Mock Order Data
    └── policy_index.pkl     # Mock Policy Index (Vector Store)

tests/
├── __init__.py
├── test_nodes.py          # Unit Tests for Individual Nodes
├── test_full_workflow.py  # End-to-End Integration Tests
├── test_cli.py            # CLI Interaction Tests
└── test_observability.py    # Tracing and Logging Tests

pyproject.toml
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd support-agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface

The agent includes a simple CLI for interactive testing:

```bash
python src/cli.py
```

**Example Interaction:**
```
Customer: Hi, where is my order?
> Router: intent=order_status
> Retrieval: Found order 123e4567-e89b-12d3-a456-426614174000 (status: SHIPPED)
> Reasoner: Action=ANSWER, Rationale=Order context available, answering directly
> Response Builder: Final Response: Your order is currently 'SHIPPED'. ...
Agent: Your order is currently 'SHIPPED'. Let me know if you'd like more detail.

Customer: What is your refund policy?
...
Agent: Your refund policy allows returns within 30 days of delivery for a full refund.
```

### Python API

You can also use the agent programmatically:

```python
from src.agent.graph import build_graph

# Initialize the graph with LLM client
llm_client = MockLLMClient()
app = build_graph(llm_client)

# Invoke with initial state
initial_state = {
    "conversation_id": "test-123",
    "customer_id": "customer-456",
    "messages": [{"role": "user", "content": "Where is my order?"}]
}

result = app.invoke(initial_state)

print("Final Response:", result["final_response"])
print("Conversation Status:", result["status"])
```

## Testing

The project uses `pytest` for comprehensive testing.

### Run All Tests
```bash
pytest
```

### Test Individual Components
```bash
# Test router node
pytest tests/test_nodes.py::test_router_node_classifies_intent

# Test full workflow
pytest tests/test_full_workflow.py::test_simple_resolution_path
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
