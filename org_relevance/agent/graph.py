import json
from typing import TypedDict, Literal, Optional, Dict, Any, List, Sequence, Tuple

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

from org_relevance.common.types import AgentState
from org_relevance.agent.nodes import router_node, make_search_query_node, web_search_node, augment_context_node, \
    classify_node

def route_after_router(state: AgentState) -> Literal["classify", "web_search"]:
    return "classify" if state.get("can_decide_now", False) else "web_search"

def build_graph(llm):
    g = StateGraph(AgentState)
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

    # nodes (оборачиваем, чтобы прокинуть llm)
    g.add_node("router", lambda s: router_node(s, llm))
    # g.add_node("make_search_query", lambda s: make_search_query_node(s, llm))
    g.add_node("web_search", web_search_node)
    g.add_node("augment_context", lambda s: augment_context_node(s, embedding_model))
    g.add_node("classify", lambda s: classify_node(s, llm))

    # edges
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {
        "classify": "classify",
        "web_search": "web_search",
    })

    # g.add_edge("make_search_query", "web_search")
    g.add_edge("web_search", "augment_context")
    g.add_edge("augment_context", "router")

    g.add_edge("classify", END)

    return g.compile()