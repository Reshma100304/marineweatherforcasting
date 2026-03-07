"""
utils/data_fetcher.py
Handles geocoding, fetching live data from the StormGlass API, and generating sample data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import streamlit as st


def geocode_city(city_name: str) -> tuple[float, float, str] | None:
    """
    Convert a city name to (latitude, longitude, display_name) using the
    free OpenStreetMap Nominatim API. Returns None if the city is not found.
    No API key required.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city_name, "format": "json", "limit": 1},
            headers={"User-Agent": "MarineWeatherPredictor/1.0"},
            timeout=5,
        )
        results = resp.json()
        if not results:
            return None
        r = results[0]
        return float(r["lat"]), float(r["lon"]), r.get("display_name", city_name)
    except Exception:
        return None


def fetch_live_data(lat: float, lng: float, api_key: str, hours: int) -> pd.DataFrame | None:
    """Fetch real-time marine data from the StormGlass API."""
    url = (
        f"https://api.stormglass.io/v2/weather/point"
        f"?lat={lat}&lng={lng}"
        f"&params=waveHeight,windSpeed,swellHeight,swellPeriod"
        f"&start={int(datetime.now().timestamp()) - (hours * 3600)}"
        f"&end={int(datetime.now().timestamp())}"
    )
    with st.spinner("Fetching live data from StormGlass..."):
        response_raw = requests.get(url, headers={"Authorization": api_key})

    if response_raw.status_code != 200:
        st.warning(f"⚠️ Could not fetch data: {response_raw.status_code} - {response_raw.text}")
        return None

    try:
        response = response_raw.json()
        df = pd.json_normalize(response.get("hours", []))
        if df.empty:
            st.info("No hourly data returned by the API. Falling back to sample data.")
            return None
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "waveHeight.sg", "windSpeed.sg", "swellHeight.sg", "swellPeriod.sg"]]
        df.columns = ["Time", "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)", "Swell Period (s)"]
        return df
    except Exception as e:
        st.warning(f"⚠️ Error processing API data: {e}")
        return None


def get_sample_data() -> pd.DataFrame:
    """Return a simple one-row sample DataFrame for offline/demo use."""
    return pd.DataFrame([
        {
            "Time": pd.Timestamp.now(),
            "Wave Height (m)": 1.2,
            "Wind Speed (m/s)": 6.5,
            "Swell Height (m)": 0.8,
            "Swell Period (s)": 10,
        }
    ])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived wind and wave energy features required by the model."""
    df = df.copy()
    df["wind_x"] = df["Wind Speed (m/s)"] * np.cos(np.radians(45))
    df["wind_y"] = df["Wind Speed (m/s)"] * np.sin(np.radians(45))
    df["wave_energy"] = 0.5 * 1025 * 9.81 * (df["Wave Height (m)"] ** 2)
    return df


FEATURE_COLS = [
    "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)",
    "Swell Period (s)", "wind_x", "wind_y", "wave_energy",
]
