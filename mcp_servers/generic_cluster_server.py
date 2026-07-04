import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from mcp.server.fastmcp import FastMCP
import pandas as pd
import pickle, traceback

from trend_model.generic_cluster_model import GenericClusterModel

mcp = FastMCP("generic_cluster")

MODEL_PATH = os.getenv(
    "GENERIC_CLUSTER_MODEL_PATH",
    "models/artifacts/attendance_model.pkl"
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = GenericClusterModel()
        _model.load(MODEL_PATH)
    return _model


@mcp.tool()
def predict_generic_cluster(records: list[dict]) -> dict:
    """
    Prediction-only MCP tool for generic cluster trend model.
    Input: list of attendance-like records.
    """

    if not records:
        return {
            "success": False,
            "message": "No records provided.",
            "predictions": []
        }

    df = pd.DataFrame(records)

    model = get_model()
    result_df = model.predict(df)

    return {
        "success": True,
        "message": "Generic cluster prediction complete.",
        "predictions": result_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    mcp.run()