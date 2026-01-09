from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import (
    intent_node,
    greeting_node,
    task_node,
    general_llm_node
)

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("greeting", greeting_node)
graph.add_node("task", task_node)
graph.add_node("general", general_llm_node)

graph.set_entry_point("intent")

graph.add_conditional_edges(
    "intent",
    lambda s: END if s.get("intent") == "GREETING" else "task" if s.get("intent") not in ("GENERAL", None)
        else "general"
)

app = graph.compile()
