import sys
import os
import pandas as pd
from mcp.server.fastmcp import FastMCP
from managers.model_manager import ModelManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mcp = FastMCP("fraud")

MODEL_CACHE = {}

@mcp.tool()
def train(
    csv_path: str = "data/creditcard.csv",
    model_name: str = "xgboost"
) -> dict:
    df = pd.read_csv(csv_path)

    manager = ModelManager(model_name=model_name, df=df)
    model = manager.get_model()

    model.split()
    model.train()
    report = model.validate()

    MODEL_CACHE[model_name] = model

    return {
        "status": "trained",
        "model_name": model_name,
        "validation_report": report
    }


@mcp.tool()
def predict(
    transaction: dict,
    model_name: str = "xgboost"
) -> dict:
    if model_name not in MODEL_CACHE:
        train(model_name=model_name)

    model = MODEL_CACHE[model_name]

    df = pd.DataFrame([transaction])
    prediction = model.model.predict(df)[0]

    return {
        "model_name": model_name,
        "prediction": int(prediction),
        "label": "Fraud" if int(prediction) == 1 else "Legitimate"
    }