# 🤖 AI Customer Support Agent

A stateful, observable, end-to-end customer support automation platform built with **LangGraph**, **PostgreSQL**, and **Groq LLMs**. 

The agent combines **Retrieval-Augmented Generation (RAG)**, **multimodal computer vision**, **deterministic guardrails**, and **Human-in-the-Loop (HITL)** supervisor workflows to resolve e-commerce inquiries autonomously while enforcing company policies.

---

## 📑 Table of Contents

- [Key Architecture & Capabilities](#-key-architecture--capabilities)
- [Workflow Architecture](#-workflow-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Environment Setup](#installation--environment-setup)
  - [Starting PostgreSQL](#starting-postgresql)
  - [Policy RAG Ingestion](#policy-rag-ingestion)
- [Running the Interactive CLI](#-running-the-interactive-cli)
  - [Attaching Photos & Vision Inspection](#attaching-photos--vision-inspection)
  - [Simulating Human Supervisor Approval](#simulating-human-supervisor-approval)
- [Programmatic API Usage](#-programmatic-api-usage)
- [Guardrails & Policy Rules](#-guardrails--policy-rules)
- [Testing](#-testing)
- [License](#-license)

---

## 🚀 Key Architecture & Capabilities

1. **Stateful LangGraph Workflow**:
   - Tracks complete conversation history and context across multiple turns using `AgentState`.
   - Thread persistence via PostgreSQL (`PostgresSaver`) with automatic pool recovery and fallback to `MemorySaver`.
   - Multi-turn context preservation (e.g., remembers order context across turns when a customer follows up with a photo).

2. **Multimodal Vision Pipeline**:
   - Supports customer-submitted photos for damaged or defective item claims via `inspect_damage_photo`.
   - Uses Groq Vision (`qwen/qwen3.8-27b`) to verify physical breakage, dents, or defects against company return policies.
   - Verified damage within 14 days of delivery auto-authorizes replacements/refunds; unverified or late claims are escalated to human supervisors.

3. **Deterministic Financial & Operational Guardrails**:
   - **Refund Protection**: Auto-approves refunds $\le \$75.00$ and $\le 80\%$ of order total; amounts exceeding either threshold are paused for human supervisor review.
   - **Address Changes**: Allowed strictly when the order is in `pending` status; rejected once `shipped` or `delivered`.
   - **Lost Packages**: Flags carrier claims and escalates for investigation.
   - **Repeat Contact Detection**: Automatically flags and escalates inquiries when an order reaches $\ge 3$ customer support tickets.

4. **Human-in-the-Loop (HITL) Workflow**:
   - Graph interrupts via conditional routing when high-risk actions or policy exceptions occur (`PENDING_APPROVAL`).
   - Supervisors can inspect reasons and approve or reject actions in real time (`resume_approval`), dynamically driving the final response.

5. **Policy Grounding (RAG)**:
   - Semantic retrieval over structured policy documents (`policies/shipping_policy.md`, `policies/refund_policy.md`) using chunked vector embeddings.

6. **Production Groq LLM Client**:
   - High-speed inference using `qwen/qwen3.8-27b` with structured JSON schema outputs and automatic fallback to `openai/gpt-oss-20b`.
   - Output token throttling (`max_tokens=350`) designed to maintain stability within free-tier rate limits.

---

## 🏛 Workflow Architecture

```mermaid
flowchart TD
    User([Customer Message / Image]) --> Router[Router Node<br/><i>Intent Classification</i>]
    
    subgraph Enrichment [Context Enrichment]
        Router --> Retrieval[Retrieval Node<br/><i>Order DB + Policy RAG + Contact History</i>]
    end

    Retrieval --> Reasoner[Reasoner Node<br/><i>LLM Structured Decision</i>]

    Reasoner --> Condition{Decision Type?}

    Condition -->|Answer| ResponseBuilder[Response Builder Node]
    Condition -->|Tool Call| Action[Action Node<br/><i>Tool Execution & Guardrails</i>]
    Condition -->|Escalate| Approval[Human Approval Node<br/><i>HITL Interrupt</i>]

    Action -->|Requires Supervisor Review| Approval
    Action -->|Success / Direct Outcome| ResponseBuilder

    Approval -->|Supervisor Resumes a/r| ResponseBuilder
    ResponseBuilder --> Output([Final Response & Execution Trace])
```

---

## 📂 Project Structure

```
support-agent/
├── docker/
│   ├── init.sql                     # PostgreSQL schema and initial seed data
│   └── postgres.Dockerfile          # Database container definition
├── data/
│   ├── schema.sql                   # Reference PostgreSQL database DDL
│   └── generate_synthetic_data.py   # Script to populate mock orders and tickets
├── policies/
│   ├── refund_policy.md             # E-commerce refund and damaged item policies
│   └── shipping_policy.md           # Shipping, delivery, and address change rules
├── src/
│   ├── cli.py                       # Interactive CLI with multimodal image support
│   └── agent/
│       ├── state.py                 # Pydantic models & TypedDict AgentState
│       ├── graph.py                 # LangGraph state machine & conditional edges
│       ├── llm_client.py            # Groq & Mock LLM abstraction layer
│       ├── persistence.py           # PostgreSQL connection pool & checkpointer
│       ├── router.py                # Intent classification node
│       ├── retrieval.py             # Multi-turn order & policy RAG retrieval node
│       ├── reasoner.py              # LLM decision synthesis node
│       ├── action.py                # Tool dispatch & execution node
│       ├── human_approval.py        # HITL interrupt pause/resume logic
│       ├── response_builder.py      # Final empathetic response generation node
│       ├── observability/
│       │   ├── logger.py            # Node execution tracing decorator
│       │   └── tracer.py            # Latency and step log collectors
│       └── tools/
│           ├── order_tools.py       # Refunds, address changes, damaged goods tools
│           ├── customer_tools.py    # Customer profile, ticket history & repeat contacts
│           ├── vision_tools.py      # Multimodal photo damage inspection tool
│           └── policy_rag/
│               ├── vector_store.py  # Local vector store implementation
│               └── ingest_policies.py # Policy chunking & index builder
├── tests/                           # 52 unit, integration, and E2E tests
│   ├── test_approval_flow.py        # HITL pause/resume & persistence tests
│   ├── test_cli.py                  # CLI interaction, parsing, & image input tests
│   ├── test_full_workflow.py        # End-to-end customer journeys
│   ├── test_guardrails.py           # Refund thresholds & policy constraint tests
│   ├── test_llm_client.py           # Intent & structured decision tests
│   ├── test_nodes.py                # Individual graph node tests
│   └── test_tools.py                # Order, customer, and vision tool tests
├── docker-compose.yml               # PostgreSQL 16 container service
├── pyproject.toml                   # PEP 621 package specification
└── .env.example                     # Environment template
```

---

## 🛠 Getting Started

### Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (for PostgreSQL)
- **Groq API Key** (for live LLM inference & vision verification)

### Installation & Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd support-agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .bot_env
   source .bot_env/bin/activate
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Configure your environment**:
   ```bash
   cp .env.example .env
   ```
   Add your Groq API key in `.env`:
   ```ini
   GROQ_API_KEY=gsk_your_groq_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/support_agent
   GROQ_MODEL=qwen/qwen3.8-27b
   PYTHONPATH=.
   ```

### Starting PostgreSQL

Start the PostgreSQL 16 Alpine container with the preconfigured schema:
```bash
docker compose up -d
```
Verify the container is healthy:
```bash
docker compose ps
```

### Policy RAG Ingestion

Build the vector index from markdown policy documents (`policies/`):
```bash
python src/agent/tools/policy_rag/ingest_policies.py
```
This produces `policy_index.pkl`, ready for semantic similarity search in `retrieval_node`.

---

## 💻 Running the Interactive CLI

Launch the interactive command-line interface:
```bash
python src/cli.py
```

### Attaching Photos & Vision Inspection

When reporting a damaged or defective item, you can attach photos directly through the CLI:

1. **Slash Command with Message**:
   ```text
   Customer: /image /Users/alex/Desktop/broken_screen.png The screen is completely cracked
   ```
2. **Interactive Prompt**:
   ```text
   Customer: /image
   📸 [Image Attachment Mode]
      Enter image file path or URL: /path/to/damage.jpg
      Message or notes (press Enter to skip): Arrived damaged in box
   ```
3. **Finder Drag & Drop / Direct Paste**:
   - Drag and drop an image file directly from macOS Finder into the terminal window.
   - Or paste a public image URL (`https://.../photo.jpg`).
   The CLI automatically normalizes terminal-escaped spaces (`\ `), removes surrounding quotes, and extracts the file path.

### Simulating Human Supervisor Approval

If an action violates policy limits (e.g. requesting a refund exceeding \$75.00 or reporting an unconfirmed damaged item), the agent pauses execution:

```text
[⚠️ ESCALATION] Request paused for supervisor review: amount $150.00 exceeds $75.00 threshold
Supervisor Action: (a)pprove / (r)eject? [a/r]: a
[✅ Resumed] Graph resumed with supervisor decision.

Agent: Your request has been approved and processed. Refund of $150.00 approved and processed (refund_id=12)
```

---

## 🐍 Programmatic API Usage

You can invoke the graph in Python workflows:

```python
import uuid
from src.agent.graph import build_graph
from src.agent.llm_client import get_llm_client
from src.agent.persistence import get_checkpointer, get_thread_config

# Initialize components
llm = get_llm_client()
checkpointer = get_checkpointer(use_postgres=True)
app = build_graph(llm, checkpointer=checkpointer)

thread_id = f"conv-{uuid.uuid4().hex[:8]}"
config = get_thread_config(thread_id)

# Execute turn with image attachment
state = {
    "conversation_id": thread_id,
    "customer_id": "cust-1001",
    "image_source": "/path/to/cracked_screen.jpg",
    "messages": [
        {
            "role": "user",
            "content": "[Image attached: /path/to/cracked_screen.jpg] Order 11111111-1111-1111-1111-111111111111 arrived damaged.",
            "image_source": "/path/to/cracked_screen.jpg",
        }
    ],
}

result = app.invoke(state, config=config)
print("Agent Response:", result["final_response"])
print("Status:", result["status"])
```

---

## 🛡 Guardrails & Policy Rules

| Operation | Condition / Policy | Automated Behavior | Escalation Behavior |
| :--- | :--- | :--- | :--- |
| **Refund** | Amount $\le \$75.00$ AND $\le 80\%$ order total | Auto-approved & recorded in database | Pauses graph for human supervisor review |
| **Address Change** | Order status is `pending` | Address updated in database | Prohibited if order is `shipped` or `delivered` |
| **Damaged Items** | Delivered within 14 days + Photo provided | Multimodal vision verifies damage $\rightarrow$ Auto-authorizes refund/replacement | Escalates if $>14$ days old or image damage unverified |
| **Lost Package** | Delivery delay reported | Creates carrier claim ticket | Escalates to human agent for carrier trace |
| **Repeat Contacts** | Customer reaches $\ge 3$ tickets on order | Flagged during retrieval | Reasoner routes directly to human escalation |

---

## 🧪 Testing

The repository includes a 52-test suite covering guardrails, persistence, multimodal tools, CLI parsing, and multi-turn workflows.

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Modules
```bash
# Test deterministic guardrails and refund thresholds
pytest tests/test_guardrails.py -v

# Test multimodal vision and order tools
pytest tests/test_tools.py -v

# Test CLI parsing and commands
pytest tests/test_cli.py -v

# Test multi-turn end-to-end conversation flows
pytest tests/test_full_workflow.py -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
