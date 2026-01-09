import json
from pathlib import Path

FILE = Path("memory/memory.json")

def save(state):
    if FILE.exists():
        data = json.loads(FILE.read_text())
    else:
        data = []

    data.append(state)
    FILE.write_text(json.dumps(data, indent=2))
