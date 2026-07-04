from typing import TypedDict, Optional, Dict, List, Any

class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[str]
    confidence: Optional[float]
    entities: Optional[Dict]
    response: Optional[Any]

    generic_cluster_records: Optional[List[Dict]]
