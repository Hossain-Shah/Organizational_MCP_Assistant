from mcp.server.fastmcp import FastMCP
from transformers import pipeline

LABEL_MAPPING = {
    "LABEL_0": "Room Booking System",
    "LABEL_1": "HR",
    "LABEL_2": "VMS"
}

# ⚠️ Force TensorFlow loading
classifier = pipeline(
    "text-classification",
    model="models/artifacts/booking_topic_finetuned_model",
    framework="tf"
)

mcp = FastMCP("task_intent_topic")

@mcp.tool()
def classify_task_intent(text: str) -> dict:
    result = classifier(text)[0]

    label = result["label"]
    score = float(result["score"])

    if score < 0.75:
        return {
            "intent": "Irrelevant",
            "confidence": score
        }

    return {
        "intent": LABEL_MAPPING[result["label"]],
        "confidence": score
    }

if __name__ == "__main__":
    mcp.run()
