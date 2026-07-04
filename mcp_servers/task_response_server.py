import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from models.ffn_response import ResponseFFN

model = ResponseFFN()

mcp = FastMCP("task_response_ffn")

@mcp.tool()
def generate_task_response(intent: str, entities: dict) -> dict:
    response = model.generate(intent, entities)
    return {"response": response}

if __name__ == "__main__":
    mcp.run()
