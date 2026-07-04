from typing import TypedDict, Optional, Dict

class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    confidence: Optional[float]
    entities: Optional[Dict]
    response: Optional[str]
