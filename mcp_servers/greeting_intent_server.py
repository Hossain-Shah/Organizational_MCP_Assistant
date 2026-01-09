from mcp.server.fastmcp import FastMCP
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import random

stemmer = PorterStemmer()

def tokenize(sentence: str):
    return nltk.word_tokenize(sentence)

def stem(word: str):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "artifacts" / "rf-ci-greeter.pth"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
responses = checkpoint.get("responses", {})
all_words = checkpoint["all_words"]
tags = checkpoint["tags"]
print("Loaded tags:", tags)

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]

class GreetingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

model = GreetingModel(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

GREETING_TAG = {'capable_of', 'funny', 'goodbye', 'greeting', 'items', 'payments', 'pr_request', 'sci_info', 'sysinfo', 'thanks'}

mcp = FastMCP("greeting_intent")

def encode(text: str):
    tokens = tokenize(text)
    bow = bag_of_words(tokens, all_words)
    return torch.tensor(bow, dtype=torch.float32).unsqueeze(0)

@mcp.tool()
def detect_greeting(text: str):
    x = encode(text)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

    confidence, label_id = torch.max(probs, dim=-1)
    predicted_tag = tags[label_id.item()]

    response = None
    if predicted_tag in responses:
        response = random.choice(responses[predicted_tag])

    return {
        "is_greeting": predicted_tag in GREETING_TAG,
        "predicted_tag": predicted_tag,
        "confidence": float(confidence.item()),
        "response": response
    }

if __name__ == "__main__":
    mcp.run()
