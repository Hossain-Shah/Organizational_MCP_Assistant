from docx import Document
from pathlib import Path

def load_docx(path: Path):
    doc = Document(path)
    sections = []
    current = {"title": "Introduction", "text": ""}

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            sections.append(current)
            current = {"title": para.text.strip(), "text": ""}
        else:
            current["text"] += para.text + "\n"

    sections.append(current)
    return sections
