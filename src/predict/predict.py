from pathlib import Path

import pandas as pd

from ..models.model import load_model


def predict_text(model_path: str, text: str) -> tuple[float, float]:
    """Load a model and predict coordinates for a single text input."""
    model = load_model(model_path)
    pred = model.predict([text])
    return float(pred[0][0]), float(pred[0][1])


def predict_coordinates(model, text_series: pd.Series) -> pd.DataFrame:
    """Predict coordinates for a pandas Series of text values."""
    predicted = model.predict(text_series)
    return pd.DataFrame(
        predicted,
        columns=["pred_latitude", "pred_longitude"],
    )
