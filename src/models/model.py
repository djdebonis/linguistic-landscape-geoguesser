from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


def create_coord_model(alpha: float = 0.1) -> Pipeline:
    """Create a text-to-coordinate regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                lowercase=True,
                min_df=2,
            ),
        ),
        (
            "regressor",
            MultiOutputRegressor(Ridge(alpha=alpha)),
        ),
    ])


def train_coord_model(
    model: Pipeline,
    X: pd.Series,
    y: pd.DataFrame,
) -> Pipeline:
    """Fit the coordinate model on text and latitude/longitude labels."""
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: str) -> None:
    """Save a trained model to a file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str) -> Pipeline:
    """Load a trained model from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return joblib.load(path)


def evaluate_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate model performance and return MAE for lat/lon."""
    y_pred = model.predict(X_test)
    results = {
        "mae_latitude": mean_absolute_error(y_test["latitude"], y_pred[:, 0]),
        "mae_longitude": mean_absolute_error(y_test["longitude"], y_pred[:, 1]),
    }
    return results
