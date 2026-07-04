import uuid

class DocGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, text, meta):
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = {"text": text, "meta": meta}
        return node_id

    def add_edge(self, src, dst, rel):
        self.edges.append((src, dst, rel))
