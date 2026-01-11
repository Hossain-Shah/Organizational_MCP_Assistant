from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import json

async def call_tool_json(session, name, args):
    result = await session.call_tool(name, args)
    print(f"\n[DEBUG] Tool: {name}")
    print("[DEBUG] Raw content:", result.content)

    if not result.content:
        raise RuntimeError(f"{name} returned no content")

    text = result.content[0].text
    print("[DEBUG] Raw text:", repr(text))

    return json.loads(text)

GREET_SERVER = StdioServerParameters(
    command="python",
    args=["mcp_servers/greeting_intent_server.py"]
)

TASK_SERVER = StdioServerParameters(
    command="python",
    args=["mcp_servers/task_intent_server.py"]
)

NER_SERVER = StdioServerParameters(
    command="python",
    args=["mcp_servers/task_ner_server.py"]
)

RESP_SERVER = StdioServerParameters(
    command="python",
    args=["mcp_servers/task_response_server.py"]
)

GENERAL_LLM_SERVER = StdioServerParameters(
    command="python",
    args=["mcp_servers/general_llm_server.py"]
)
CONF_THRESHOLD = 0.8

async def intent_node(state):
    text = state.get("user_input", "") or ""
    scores = {}
    # Greeting
    async with stdio_client(GREET_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            g = await call_tool_json(session, "detect_greeting", {"text": text})

    if g.get("is_greeting"):
        scores["GREETING"] = g["confidence"]
        state["model_response"] = g.get("response")

    # Task intent
    async with stdio_client(TASK_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            t = await call_tool_json(session, "classify_task_intent", {"text": text})

    if t["intent"] != "Irrelevant":
        scores[t["intent"]] = t["confidence"]

    state["task_intents"] = scores

    # Pick best intent
    if not scores:
        state["intent"] = "GENERAL"
        state["confidence"] = 0.0
        return state

    # Pick best intent
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    # Global rejection threshold
    if best_score < CONF_THRESHOLD:
        state["intent"] = "GENERAL"
        state["confidence"] = best_score
        return state

    state["intent"] = best_intent
    state["confidence"] = best_score

    if best_intent == "GREETING":
        state["response"] = state.get("model_response")
    return state

def greeting_node(state):
    state["response"] = state.get("model_response")
    return state

async def task_node(state):
    intent = state["intent"]
    text = state["user_input"]

    if intent not in {"Room Booking System", "HR", "VMS"}:
        return state

    # extract entities
    async with stdio_client(NER_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            entities = await call_tool_json(session, "extract_entities", {"text": text})

    state["entities"] = entities
    state["response"] = None
    return state

async def general_llm_node(state):
    text = state["user_input"]

    async with stdio_client(GENERAL_LLM_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await call_tool_json(
                session,
                "generate_answer",
                {"text": text}
            )

    state["response"] = res.get("response")
    return state
