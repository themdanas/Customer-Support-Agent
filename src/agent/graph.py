from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

try:
    from src.agent.state import AgentState, ActionType
    from src.agent.llm_client import LLMClient
    from src.agent.router import make_router_node
    from src.agent.retrieval import retrieval_node
    from src.agent.reasoner import make_reasoner_node
    from src.agent.action import action_node
    from src.agent.human_approval import human_approval_node
    from src.agent.response_builder import response_builder_node
except ImportError:
    from state import AgentState, ActionType
    from llm_client import LLMClient
    from router import make_router_node
    from retrieval import retrieval_node
    from reasoner import make_reasoner_node
    from action import action_node
    from human_approval import human_approval_node
    from response_builder import response_builder_node


def route_after_reasoner(state: dict) -> str:
    decision = state.get("decision")
    if not decision:
        return "response_builder"
    action_type = getattr(decision, "action_type", None) or (
        decision.get("action_type") if isinstance(decision, dict) else None
    )
    if action_type == ActionType.TOOL_CALL:
        return "action"
    return "response_builder"


def route_after_action(state: dict) -> str:
    action_result = state.get("action_result")
    if action_result and getattr(action_result, "requires_escalation", False):
        return "human_approval"
    return "response_builder"


def build_graph(
    llm: LLMClient,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    graph = StateGraph(AgentState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoner", make_reasoner_node(llm))
    graph.add_node("action", action_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("response_builder", response_builder_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "reasoner")

    graph.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "action": "action",
            "response_builder": "response_builder",
        },
    )

    graph.add_conditional_edges(
        "action",
        route_after_action,
        {
            "human_approval": "human_approval",
            "response_builder": "response_builder",
        },
    )

    graph.add_edge("human_approval", "response_builder")
    graph.add_edge("response_builder", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
