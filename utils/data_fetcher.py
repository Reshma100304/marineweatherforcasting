"""
utils/data_fetcher.py
Handles geocoding, fetching live data from Open-Meteo (free) and StormGlass, and sample data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import streamlit as st


def geocode_city(city_name: str) -> tuple[float, float, str] | None:
    """
    Convert a city name to (latitude, longitude, display_name) using the
    free OpenStreetMap Nominatim API. No API key required.
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


def fetch_openmeteo_data(lat: float, lng: float, hours: int = 24) -> pd.DataFrame | None:
    """
    Fetch real marine + wind data using the free Open-Meteo APIs.
    No API key required. Works for any coastal coordinate worldwide.

    Combines:
      - Open-Meteo Marine API  -> wave height, swell height, swell period
      - Open-Meteo Weather API -> wind speed at 10m
    """
    try:
        with st.spinner("Fetching real marine data from Open-Meteo (free)..."):
            forecast_days = max(1, hours // 24 + 1)

            # Marine data
            marine_resp = requests.get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude": lat,
                    "longitude": lng,
                    "hourly": "wave_height,swell_wave_height,swell_wave_period",
                    "forecast_days": forecast_days,
                    "timezone": "auto",
                },
                timeout=10,
            )
            # Wind speed from standard weather API (m/s)
            wind_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lng,
                    "hourly": "windspeed_10m",
                    "forecast_days": forecast_days,
                    "timezone": "auto",
                    "windspeed_unit": "ms",
                },
                timeout=10,
            )

        if marine_resp.status_code != 200 or wind_resp.status_code != 200:
            st.warning(
                f"Open-Meteo error — marine: {marine_resp.status_code}, "
                f"wind: {wind_resp.status_code}"
            )
            return None

        marine = marine_resp.json().get("hourly", {})
        wind = wind_resp.json().get("hourly", {})

        if not marine.get("time") or not wind.get("time"):
            return None

        df = pd.DataFrame({
            "Time":             pd.to_datetime(marine["time"]),
            "Wave Height (m)":  marine["wave_height"],
            "Swell Height (m)": marine["swell_wave_height"],
            "Swell Period (s)": marine["swell_wave_period"],
            "Wind Speed (m/s)": wind["windspeed_10m"],
        }).dropna()

        # Keep only upcoming hours
        now = pd.Timestamp.now()
        df["Time"] = df["Time"].dt.tz_localize(None)
        df = df[df["Time"] >= now].head(hours).reset_index(drop=True)

        return df if not df.empty else None

    except Exception as e:
        st.warning(f"Could not fetch Open-Meteo data: {e}")
        return None


def fetch_stormglass_data(lat: float, lng: float, api_key: str, hours: int) -> pd.DataFrame | None:
    """Fetch real-time marine data from the StormGlass API (requires paid key)."""
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
        st.warning(f"StormGlass error: {response_raw.status_code} - {response_raw.text[:200]}")
        return None

    try:
        response = response_raw.json()
        df = pd.json_normalize(response.get("hours", []))
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "waveHeight.sg", "windSpeed.sg", "swellHeight.sg", "swellPeriod.sg"]]
        df.columns = ["Time", "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)", "Swell Period (s)"]
        return df
    except Exception as e:
        st.warning(f"Error processing StormGlass data: {e}")
        return None


def get_sample_data() -> pd.DataFrame:
    """Return a one-row fallback DataFrame (last resort only)."""
    return pd.DataFrame([{
        "Time": pd.Timestamp.now(),
        "Wave Height (m)": 1.2,
        "Wind Speed (m/s)": 6.5,
        "Swell Height (m)": 0.8,
        "Swell Period (s)": 10,
    }])


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
