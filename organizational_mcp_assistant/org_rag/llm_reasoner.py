from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from graph.nodes import GENERAL_LLM_SERVER, call_tool_json

async def reason(query, context):
    async with stdio_client(GENERAL_LLM_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await call_tool_json(
                session,
                "generate_answer",
                {
                    "text": query,
                    "context": "\n\n".join(context)
                }
            )
    return res["response"]
