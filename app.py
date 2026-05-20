import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

MODEL_PATH = "models/coord_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

coord_model = load_model()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

st.title("Linguistic Landscape GeoGuesser")

st.subheader("Enter linguistic landscape text")

intersection_text = st.text_area(
    "Text from signs/intersection",
    placeholder="taqueria mercado wireless pho smoke shop..."
)

intersection_code = st.text_area(
    "Actual Intersection code, ideally with E-W street followed by N-S street, combined with a hyphen. E.g lincoln-chambers or colfax-yosemite",
    placeholder = "lincoln-parker"
)
    

actual_lat = st.number_input(
    "Actual latitude",
    value=39.7392,
    format="%.6f"
)

actual_lon = st.number_input(
    "Actual longitude",
    value=-104.9903,
    format="%.6f"
)

if st.button("Predict and Compare"):
    if intersection_text.strip() == "":
        st.warning("Enter some text first.")
    else:
        pred = coord_model.predict([intersection_text])

        pred_lat = pred[0][0]
        pred_lon = pred[0][1]

        error_km = haversine_distance(
            actual_lat,
            actual_lon,
            pred_lat,
            pred_lon
        )

        st.success(f"Model was {error_km:.2f} km away.")

        st.write("### Predicted Coordinates")
        st.write(f"Latitude: `{pred_lat:.6f}`")
        st.write(f"Longitude: `{pred_lon:.6f}`")

        st.write("### Actual Coordinates")
        st.write(f"Latitude: `{actual_lat:.6f}`")
        st.write(f"Longitude: `{actual_lon:.6f}`")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            actual_lon,
            actual_lat,
            color="blue",
            s=120,
            label="Actual"
        )

        ax.scatter(
            pred_lon,
            pred_lat,
            color="red",
            s=120,
            label="Model Prediction"
        )

        ax.plot(
            [actual_lon, pred_lon],
            [actual_lat, pred_lat],
            color="gray",
            alpha=0.7
        )

        mid_x = (actual_lon + pred_lon) / 2
        mid_y = (actual_lat + pred_lat) / 2

        ax.text(
            mid_x,
            mid_y,
            f"{error_km:.2f} km",
            fontsize=10
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Actual vs Predicted Location")
        ax.legend()

        st.pyplot(fig)

# export data 
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

table = {
    "intersection":	[intersection_code],
    "text_blob": [intersection_text],
    "latitude": [actual_lat],
    "longitude": [actual_lon],
    "code_type": ["mixed"]
}

df = pd.DataFrame(table)


csv = df.to_csv(index=False)

st.download_button(
    label="Download this entry as CSV",
    data=csv,
    file_name=f"{intersection_code}_{timestamp}.csv",
    mime="text/csv"
)