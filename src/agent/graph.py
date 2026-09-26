from __future__ import annotations
from langgraph.graph import StateGraph, END

from state import AgentState
from llm_client import LLMClient
from router import make_router_node
from retrieval import retrieval_node
from reasoner import make_resaoner_node
from response_builder import response_builder_node


def build_graph(llm: LLMClient):
    graph = StateGraph(AgentState)


    graph.add_node("router", make_router_node(llm))
    graph.add_node("retrieval",retrieval_node)
    graph.add_node("reasoner", make_resaoner_node(llm))
    graph.add_node("response_builder", response_builder_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("reasoner", "response_builder")
    graph.add_edge("response_builder", END)

    return graph.compile()
