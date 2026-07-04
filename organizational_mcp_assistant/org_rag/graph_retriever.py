from sentence_transformers import SentenceTransformer, util
from graph_builder import build_graph
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BUILD GRAPH ONCE
graph = build_graph("C:/Users/shahnawaz.hossain/Tasks/Unsupervised_learning/Sprint_35/ResumeEvaluator-Web-APP/output/OCR")

class GraphRAGRetriever:
    def __init__(self, graph):
        self.graph = graph
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = {
            nid: self.encoder.encode(n["text"], convert_to_tensor=True)
            for nid, n in graph.nodes.items()
            if n["text"]
        }

    def retrieve(self, query, top_k=3):
        q_emb = self.encoder.encode(query, convert_to_tensor=True)

        scores = []
        for nid, emb in self.embeddings.items():
            score = util.cos_sim(q_emb, emb).item()
            scores.append((score, nid))

        scores.sort(reverse=True)
        return [
            self.graph.nodes[nid]["text"]
            for _, nid in scores[:top_k]
        ]
    
# SINGLETON RETRIEVER INSTANCE
retrieve = GraphRAGRetriever(graph)
