import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from ner.spacy_ner import CustomNER

ner = CustomNER()
mcp = FastMCP("task_ner_service")

@mcp.tool()
def extract_entities(text: str) -> dict:
    """
    Extract ROOM / VEHICLE / LEAVE entities
    """
    return ner.extract(text)

if __name__ == "__main__":
    mcp.run()
