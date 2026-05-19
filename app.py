import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "models/coord_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

coord_model = load_model()

st.title("Linguistic Landscape GeoGuesser")

st.subheader("Enter linguistic landscape text")

user_text = st.text_area(
    "Text from signs/intersection",
    placeholder="taqueria mercado wireless pho smoke shop..."
)

if st.button("Predict Coordinates"):
    if user_text.strip() == "":
        st.warning("Enter some text first.")
    else:
        pred = coord_model.predict([user_text])

        pred_lat = pred[0][0]
        pred_lon = pred[0][1]

        st.success("Predicted coordinates:")

        st.write(f"Latitude: `{pred_lat:.6f}`")
        st.write(f"Longitude: `{pred_lon:.6f}`")

        DATA_PATH = "output/cleaned_concatenated.csv"

        df = pd.read_csv(DATA_PATH)

        # basic graph
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            df["longitude"],
            df["latitude"],
            alpha=0.4,
            s=50,
            label="Training Intersections"
        )
        
        ax.scatter(
            pred_lon,
            pred_lat,
            color="red",
            s=120,
            label="Model Prediction"
        )

        ax.text(
            pred_lon,
            pred_lat,
            " Predicted Point",
            fontsize=10
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Predicted Geographic Location")
        ax.legend()

        st.pyplot(fig)