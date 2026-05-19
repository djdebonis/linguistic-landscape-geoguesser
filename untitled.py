import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/intersections.csv")

st.title("Linguistic Landscape GeoGuessr")

df = pd.read_csv(DATA_PATH)

# pick a random intersection
if "current_idx" not in st.session_state:
    st.session_state.current_idx = np.random.choice(df.index)

row = df.loc[st.session_state.current_idx]

st.subheader("Guess the location from the linguistic landscape")

st.write("### Text blob")
st.write(row["text_blob"])

st.write("Enter your guess:")

guess_lat = st.number_input("Guessed latitude", value=39.7392, format="%.6f")
guess_lon = st.number_input("Guessed longitude", value=-104.9903, format="%.6f")

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

if st.button("Submit Guess"):
    error_km = haversine_distance(
        row["latitude"],
        row["longitude"],
        guess_lat,
        guess_lon
    )

    st.success(f"You were {error_km:.2f} km away.")

    st.write("### True location")
    st.write(f"Latitude: {row['latitude']}")
    st.write(f"Longitude: {row['longitude']}")
    st.write(f"City: {row['city']}")

if st.button("Next Round"):
    st.session_state.current_idx = np.random.choice(df.index)
    st.rerun()