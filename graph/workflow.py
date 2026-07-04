from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import (
    intent_node,
    greeting_node,
    task_node,
    general_llm_node,
    generic_cluster_node
)

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("greeting", greeting_node)
graph.add_node("task", task_node)
graph.add_node("general", general_llm_node)
graph.add_node("generic_cluster", generic_cluster_node)

graph.set_entry_point("intent")

def route_after_intent(state):
    if state.get("intent") == "GREETING":
        return END

    if state.get("intent") == "GENERIC_CLUSTER":
        return "generic_cluster"

    if state.get("intent") not in ("GENERAL", None):
        return "task"

    return "general"

graph.add_conditional_edges(
    "intent", route_after_intent
)

graph.add_edge("task", END)
graph.add_edge("general", END)
graph.add_edge("generic_cluster", END)

app = graph.compile()
