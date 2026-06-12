import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction_vs_actual_testing(
    prediction_df: pd.DataFrame,
    predicted_latitude_col: str = "pred_latitude",
    predicted_longitude_col: str = "pred_longitude",
    actual_latitude_col: str = "actual_latitude",
    actual_longitude_col: str = "actual_longitude",
    title: str = "Prediction vs Actual Coordinates",
    output_dir: str = "data/plots",
) -> str:
    """Create a scatter plot comparing predicted vs actual coordinates.

    The plot is saved as a PNG file in the provided output directory.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot some points for reference
    plt.text(-104.996583333333, 39.74852777777778, "Downtown Denver")
    plt.text(-104.875277777777, 39.74022222222222, "Aurora on Colfax")
    plt.text(-104.768055555555, 39.53711111111111, "Parker/Lincoln")

    plt.scatter(
        prediction_df[actual_longitude_col],
        prediction_df[actual_latitude_col],
        color="blue",
        label="True",
        s=80,
    )

    plt.scatter(
        prediction_df[predicted_longitude_col],
        prediction_df[predicted_latitude_col],
        color="red",
        label="Predicted",
        s=80,
    )

    for _, row in prediction_df.iterrows():
        plt.plot(
            [row[actual_longitude_col], row[predicted_longitude_col]],
            [row[actual_latitude_col], row[predicted_latitude_col]],
            color="gray",
            alpha=0.5,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mean_error_km = prediction_df["error_km"].mean()
    error_label = f"{mean_error_km:.2f}"

    plt.title(f"{title}\n{timestamp_str} | Mean Error: {error_label} km")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    filename = f"prediction_vs_actual_{timestamp_str}_mean_error_{error_label}.png"
    output_file = output_dir_path / filename
    plt.savefig(output_file)
    plt.close()

    return str(output_file)
