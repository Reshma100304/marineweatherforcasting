"""
app.py — Marine Weather Predictor
Entry point for the Streamlit dashboard.
"""
import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load .env file if it exists (silently ignored if missing)
load_dotenv()

from utils.model_loader import get_model
from utils.data_fetcher import fetch_live_data, get_sample_data, engineer_features, FEATURE_COLS
from utils.charts import render_trend_chart, render_feature_importance

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌊 Marine Weather Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #021124 0%, #061A2B 100%); color: #e6f0ff; }
    .stButton>button { background: #0ea5a0; color: white; border-radius: 6px; }
    .streamlit-expanderHeader { color: #cfeffd; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 Marine Weather Predictor")
st.markdown(
    "Fetch live oceanic parameters via the StormGlass API or use built-in presets. "
    "The trained model predicts a sea-state label: **Calm**, **Moderate**, or **Rough**."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
PRESETS = {
    "San Francisco, USA": (37.7749, -122.4194),
    "Sydney, AU": (-33.8688, 151.2093),
    "Cape Town, ZA": (-33.9249, 18.4241),
    "Mumbai, IN": (19.0760, 72.8777),
    "Honolulu, US-HI": (21.3069, -157.8583),
}

with st.sidebar:
    st.header("⚙️ Controls")

    preset_name = st.selectbox("Location preset", list(PRESETS.keys()))
    default_lat, default_lng = PRESETS[preset_name]
    lat = st.number_input("Latitude", value=default_lat, format="%f")
    lng = st.number_input("Longitude", value=default_lng, format="%f")

    st.markdown("---")

    # Pre-fill from STORMGLASS_API_KEY in .env (empty string if not set)
    _env_key = os.getenv("STORMGLASS_API_KEY", "")
    api_key = st.text_input(
        "StormGlass API Key (optional)",
        value=_env_key,
        type="password",
        help="Set STORMGLASS_API_KEY in a .env file to avoid typing it each session.",
    )

    with st.expander("Advanced options"):
        hours = st.slider("Hours to fetch / display", min_value=1, max_value=72, value=24)
        use_live = st.checkbox("Fetch live data (StormGlass)", value=bool(api_key))
        units = st.radio("Units", ["Metric (m, m/s)", "Nautical (ft, knots)"], index=0)
        smoothing = st.slider("Smoothing window (points)", min_value=0, max_value=6, value=0)

    st.markdown("---")
    st.caption("Built with Streamlit + scikit-learn. Model loaded from `marine_model.pkl`.")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = get_model()

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("🔮 Predict marine condition", key="predict_button"):
    # 1. Fetch data
    df = None
    if use_live and api_key:
        df = fetch_live_data(lat, lng, api_key, hours)

    if df is None or df.empty:
        st.info("ℹ️ Using sample data for prediction.")
        df = get_sample_data()

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Predict
    X_latest = df[FEATURE_COLS].iloc[-1:]

    if model is None:
        st.error("❌ No model loaded. Make sure `marine_model.pkl` exists in the project folder.")
        st.stop()

    prediction = model.predict(X_latest)

    # 4. Confidence
    proba_text = ""
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_latest)[0]
            top_idx = np.argmax(proba)
            top_conf = proba[top_idx]
            top_class = model.classes_[top_idx]
            proba_text = f"{top_conf * 100:.0f}% confidence for {top_class}"
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Layout — left: KPIs | right: map + readings
    # ---------------------------------------------------------------------------
    left_col, right_col = st.columns([1.3, 1])
    last = df.iloc[-1]

    with left_col:
        st.markdown("### 🔵 Prediction")
        st.metric(label="Predicted Condition", value=str(prediction[0]), delta=proba_text)

        # Risk score
        try:
            wave = float(last["Wave Height (m)"])
            wind = float(last["Wind Speed (m/s)"])
            risk = (0.6 * min(wave / 6.0, 1.0) + 0.4 * min(wind / 20.0, 1.0)) * 100
            st.metric(
                label="Estimated Risk Score (0–100)",
                value=f"{risk:.0f}",
                delta=f"Wave {wave:.1f} m  |  Wind {wind:.1f} m/s",
            )
        except Exception:
            pass

    with right_col:
        st.markdown("### 📍 Location & latest readings")
        try:
            map_df = pd.DataFrame({"lat": [lat], "lon": [lng]})
            st.map(map_df, zoom=6)
        except Exception:
            try:
                view = pdk.ViewState(latitude=float(lat), longitude=float(lng), zoom=6)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lat": lat, "lon": lng}],
                    get_position="[lon, lat]",
                    get_radius=50000,
                    get_color=[0, 120, 255, 140],
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
            except Exception:
                pass

        st.write(f"🌊 Wave: **{last['Wave Height (m)']:.2f} m**")
        st.write(f"💨 Wind: **{last['Wind Speed (m/s)']:.2f} m/s**")
        st.write(f"🌀 Swell: **{last['Swell Height (m)']:.2f} m** / **{last['Swell Period (s)']:.0f} s**")

    st.caption("Condition derived from wave, wind, and swell parameters.")

    # ---------------------------------------------------------------------------
    # Safety guidance
    # ---------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("⚠️ Forecast guidance & safety tips"):
        severity = prediction[0].lower()
        wave_val = float(last["Wave Height (m)"])
        wind_val = float(last["Wind Speed (m/s)"])

        if severity == "rough" or wave_val > 3 or wind_val > 15:
            st.error("🚨 High caution: conditions are rough. Avoid small vessels.")
        elif severity == "moderate" or wave_val > 1.5 or wind_val > 8:
            st.warning("⚠️ Moderate conditions: experienced crews should prepare.")
        else:
            st.success("✅ Calm conditions — suitable for most small craft.")

        st.markdown("**Tactical tips:**")
        st.write("- Monitor local forecasts; marine weather can change quickly.")
        st.write("- Avoid small craft when wave height or wind speed rises significantly.")
        if proba is not None:
            st.write(
                "Model probabilities: "
                + ", ".join(f"{c}: {p * 100:.0f}%" for c, p in zip(model.classes_, proba))
            )

    # ---------------------------------------------------------------------------
    # Trend charts
    # ---------------------------------------------------------------------------
    st.subheader("📈 Trends")
    render_trend_chart(df, units, smoothing)

    # ---------------------------------------------------------------------------
    # Raw data table + CSV download
    # ---------------------------------------------------------------------------
    st.subheader("📋 Hourly data sample")
    display_cols = ["Time", "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)", "Swell Period (s)"]
    st.dataframe(df[display_cols].tail(12).reset_index(drop=True))

    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download data as CSV",
        data=csv,
        file_name=f"marine_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.subheader("🧠 Model Feature Importance")
        render_feature_importance(model)
