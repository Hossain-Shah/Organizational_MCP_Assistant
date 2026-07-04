from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

mcp = FastMCP("general_llm")

@mcp.tool()
def generate_answer(text: str) -> dict:
    """
    Open-domain response for out-of-scope queries
    """

    prompt = f"""
    You are an organizational assistant for internal and general queries.

    If the question is general knowledge, answer factually.
    If the question is about your role, explain your capabilities clearly.

    Question: {text}
    Answer:
    """


    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": answer}

if __name__ == "__main__":
    mcp.run()
