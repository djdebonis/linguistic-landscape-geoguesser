# src/visualize.py
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime

def plot_prediction_vs_actual_testing(
        prediction_df: pd.DataFrame, 
        predicted_latitude_col: str = "pred_latitude",
        predicted_longitude_col: str = "pred_longitude",
        actual_latitude_col: str = "actual_latitude",
        actual_longitude_col: str = "actual_longitude",
        title: str="Prediction vs Actual Coordinates"):
    
    """
    Create a scatter plot comparing predicted vs actual coordinates, with lines connecting them.
    
    """

    plt.figure(figsize=(12,8))

    # Plot some points for reference
    plt.text(-104.996583333333, 39.74852777777778,"Downtown Denver")
    plt.text(-104.875277777777, 39.74022222222222, "Aurora on Colfax")
    plt.text(-104.768055555555, 39.53711111111111, "Parker/Lincoln")


    # true locations
    plt.scatter(
        prediction_df[actual_longitude_col],
        prediction_df[actual_latitude_col],
        color="blue",
        label="True",
        s=80
    )

    # predicted locations
    plt.scatter(
        prediction_df[predicted_longitude_col],
        prediction_df[predicted_latitude_col],
        color="red",
        label="Predicted",
        s=80
    )

    # arrows / lines
    for _, row in prediction_df.iterrows():

        plt.plot(
            [row[actual_longitude_col], row[predicted_longitude_col]],
            [row[actual_latitude_col], row[predicted_latitude_col]],
            color="gray",
            alpha=0.5
        )



    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # get datetime data for logging
    now = now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str = timestamp_str.replace(" ", "_").replace(":", "-")

    # get error stats for logging
    mean_error_km = str(prediction_df["error_km"].mean())

    # log title with timestamp and error stats
    title = f"{title}\n{timestamp_str} | Mean Error: {mean_error_km} km"

    # create file string
    filename = f"prediction_vs_actual_{timestamp_str}_mean_error_{mean_error_km}.png"
    
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)