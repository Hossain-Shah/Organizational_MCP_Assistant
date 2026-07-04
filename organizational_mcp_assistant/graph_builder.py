from pathlib import Path
from docx_loader import load_docx
from graphs import DocGraph

def build_graph(folder="C:/Users/shahnawaz.hossain/Tasks/Unsupervised_learning/Sprint_35/ResumeEvaluator-Web-APP/output/OCR"):
    graph = DocGraph()

    for doc_path in Path(folder).glob("*.docx"):
        doc_node = graph.add_node(
            text="",
            meta={"type": "document", "name": doc_path.name}
        )

        sections = load_docx(doc_path)
        for sec in sections:
            sec_node = graph.add_node(
                text=sec["text"],
                meta={"type": "section", "title": sec["title"]}
            )
            graph.add_edge(doc_node, sec_node, "HAS_SECTION")

    return graph
