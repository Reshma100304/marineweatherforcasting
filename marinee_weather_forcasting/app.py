import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import pydeck as pdk
import joblib
import os
import warnings
from datetime import datetime
from textwrap import dedent

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # fallback: ignore user-level warnings so the UI stays clean
    warnings.filterwarnings("ignore")

# --- Page Setup ---
st.set_page_config(page_title="🌊 Marine Weather Predictor", layout="wide", initial_sidebar_state="expanded")

_MAIN_PAGE_HEADER = "🌊 Marine Weather Predictor"
st.title(_MAIN_PAGE_HEADER)
st.markdown(
    "Use a real-time StormGlass API key to fetch live oceanic parameters or pick built-in presets. The trained model predicts a simple condition label: Calm, Moderate or Rough. "
)

# --- Tiny style polish (colors + spacing) ---
st.markdown(
    """
    <style>
    .reportview-container .main header {background: linear-gradient(90deg,#0f172a,#0f2a4a); padding: 18px 24px; border-radius:10px}
    .stApp { background: linear-gradient(180deg, #021124 0%, #061A2B 100%); color: #e6f0ff; }
    .stButton>button { background: #0ea5a0; color: white; border-radius:6px }
    .stMetric-value { font-size: 28px !important }
    .streamlit-expanderHeader { color: #cfeffd }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Input Section ---
# --- Sidebar for controls ---
with st.sidebar:
    st.header("Controls")

    presets = {
        "San Francisco, USA": (37.7749, -122.4194),
        "Sydney, AU": (-33.8688, 151.2093),
        "Cape Town, ZA": (-33.9249, 18.4241),
        "Mumbai, IN": (19.0760, 72.8777),
        "Honolulu, US-HI": (21.3069, -157.8583),
    }

    preset_name = st.selectbox("Choose a location preset", ["San Francisco, USA", "Sydney, AU", "Cape Town, ZA", "Mumbai, IN", "Honolulu, US-HI"])
    lat, lng = st.number_input("Latitude", value=presets[preset_name][0], format="%f"), st.number_input("Longitude", value=presets[preset_name][1], format="%f")

    st.markdown("---")

    api_key = st.text_input("StormGlass API Key (optional)", type="password", help="Provide your StormGlass API key to fetch real data. If left empty we'll use sample data.")

    with st.expander("Advanced options"):
        hours = st.slider("Hours of forecast to fetch / display", min_value=1, max_value=72, value=24)
        use_live = st.checkbox("Fetch live data (StormGlass)", value=bool(api_key))
        units = st.radio("Units", options=["Metric (m, m/s)", "Nautical (ft, knots)"], index=0)
        smoothing = st.slider("Smoothing window (moving average, points)", min_value=0, max_value=6, value=0)

    st.markdown("---")
    st.caption("Built using Streamlit + scikit-learn. Model is loaded from `marine_model.pkl`.")

# --- Load Model Safely ---
model_path = os.path.join(os.path.dirname(__file__), "marine_model.pkl")

@st.cache_resource
def load_model(path: str):
    """Load the saved model once and cache it for quick repeated predictions."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Failed loading model file: %s" % e)
        return None

model = load_model(model_path)

# --- Prediction Section ---
if st.button("Predict marine condition", key="predict_button"):
    df = None

    # Try fetching real-time data if user opted in and provided an API key
    if use_live and api_key:
        url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lng}&params=waveHeight,windSpeed,swellHeight,swellPeriod&start={int(datetime.now().timestamp()) - (hours * 3600)}&end={int(datetime.now().timestamp())}"
        with st.spinner("Fetching live data from StormGlass..."):
            response_raw = requests.get(url, headers={"Authorization": api_key})

        if response_raw.status_code != 200:
            st.warning(f"⚠️ Could not fetch data: {response_raw.status_code} - {response_raw.text}")
        else:
            try:
                response = response_raw.json()
                df = pd.json_normalize(response.get('hours', []))
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df[['time', 'waveHeight.sg', 'windSpeed.sg', 'swellHeight.sg', 'swellPeriod.sg']]
                    df.columns = ['Time', 'Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 'Swell Period (s)']
                else:
                    st.info("No hourly data returned by the API for the requested timeframe. Falling back to sample data.")

            except Exception as e:
                st.warning(f"⚠️ Error processing API data: {e}")

    # Use sample data if API fails or user didn't request live
    if df is None or df.empty:
        st.info("Using sample data for prediction.")
        df = pd.DataFrame([
            {
                'Time': pd.Timestamp.now(),
                'Wave Height (m)': 1.2,
                'Wind Speed (m/s)': 6.5,
                'Swell Height (m)': 0.8,
                'Swell Period (s)': 10
            }
        ])

    # --- Derived Features ---
    # Derived / feature engineering
    df = df.copy()
    df['wind_x'] = df['Wind Speed (m/s)'] * np.cos(np.radians(45))
    df['wind_y'] = df['Wind Speed (m/s)'] * np.sin(np.radians(45))
    df['wave_energy'] = 0.5 * 1025 * 9.81 * (df['Wave Height (m)'] ** 2)

    # --- Prediction ---
    X_latest = df[['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 'Swell Period (s)', 'wind_x', 'wind_y', 'wave_energy']].iloc[-1:]

    if model is None:
        st.error("No model loaded. Make sure `marine_model.pkl` exists in the project folder.")
    else:
        prediction = model.predict(X_latest)

        # --- Main output ---
        # Left - main KPI / right - map + details
        left_col, right_col = st.columns([1.3, 1])

        # Predicted label and confidence (if available)
        proba_text = ""
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_latest)[0]
                top_idx = np.argmax(proba)
                top_class = model.classes_[top_idx]
                top_conf = proba[top_idx]
                proba_text = f"{top_conf*100:.0f}% confidence for {top_class}"
            except Exception:
                proba_text = ""

        with left_col:
            st.markdown("### Prediction")
            st.metric(label="Predicted Condition", value=str(prediction[0]), delta=proba_text)

        with right_col:
            st.markdown("### Location & latest readings")

            # Map showing the lat/lon
            try:
                map_df = pd.DataFrame({'lat': [lat], 'lon': [lng]})
                st.map(map_df, zoom=6)
            except Exception:
                # fallback to PyDeck
                try:
                    view = pdk.ViewState(latitude=float(lat), longitude=float(lng), zoom=6)
                    layer = pdk.Layer('ScatterplotLayer', data=[{'lat': lat, 'lon': lng}], get_position='[lon, lat]', get_radius=50000, get_color=[0, 120, 255, 140])
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
                except Exception:
                    pass

            # latest numeric readings
            last = df.iloc[-1]
            st.write(f"Wave: {last['Wave Height (m)']:.2f} m")
            st.write(f"Wind: {last['Wind Speed (m/s)']:.2f} m/s")
            st.write(f"Swell: {last['Swell Height (m)']:.2f} m / {last['Swell Period (s)']:.0f}s")

        st.caption("Condition derived using wave, wind, and swell parameters.")

        # --- Forecast helper / quick advice ---
        st.markdown("---")
        with st.expander("Forecast guidance & safety tips"):
            severity = prediction[0]
            if severity.lower() == 'rough' or (last['Wave Height (m)'] > 3 or last['Wind Speed (m/s)'] > 15):
                st.error("High caution: conditions are rough. Avoid small vessels and exercise maritime safety.")
            elif severity.lower() == 'moderate' or (last['Wave Height (m)'] > 1.5 or last['Wind Speed (m/s)'] > 8):
                st.warning("Moderate conditions: experienced crews should prepare and reduce exposure.")
            else:
                st.success("Calm conditions — suitable for most small craft and recreational activities.")

            st.markdown("**Tactical tips:**")
            st.write("- Monitor local forecasts and updates; marine weather can change quickly.")
            st.write("- Avoid operating small craft when wave height or wind speed increases significantly.")
            if proba is not None:
                st.write(f"Model probabilities: {', '.join([f'{c}: {p*100:.0f}%' for c,p in zip(model.classes_, proba)])}")

        # Charts - richer view
        st.subheader("Trends")
        try:
            df_chart = df[['Time', 'Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)']].melt('Time', var_name='Parameter', value_name='Value')
            chart = alt.Chart(df_chart).mark_line().encode(
                x='Time:T',
                y='Value:Q',
                color='Parameter:N',
                tooltip=['Time:T','Parameter:N','Value:Q']
            ).interactive()
            # optionally smooth values by moving average (smoothing=0 disables)
            if smoothing and smoothing > 0:
                df_chart['Value'] = df_chart.groupby('Parameter')['Value'].transform(lambda x: x.rolling(smoothing, min_periods=1).mean())

            # convert units if user requested Nautical
            unit_map = {'Wave Height (m)': 'm', 'Wind Speed (m/s)': 'm/s', 'Swell Height (m)': 'm'}
            if units.startswith('Nautical'):
                # convert m->ft and m/s->knots
                def convert_param(p, v):
                    if 'Wave Height' in p or 'Swell Height' in p:
                        return v * 3.28084
                    if 'Wind Speed' in p:
                        return v * 1.94384
                    return v
                df_chart['Value'] = df_chart.apply(lambda r: convert_param(r['Parameter'], r['Value']), axis=1)
                # update axis labels
                unit_map = {'Wave Height (m)': 'ft', 'Wind Speed (m/s)': 'knots', 'Swell Height (m)': 'ft'}

            chart = alt.Chart(df_chart).mark_line().encode(
                x='Time:T',
                y='Value:Q',
                color='Parameter:N',
                tooltip=['Time:T','Parameter:N','Value:Q']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            df_plot = df.set_index('Time')[['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)']]
            st.line_chart(df_plot)

        # Last rows + download
        st.subheader("Hourly data sample")
        st.dataframe(df.tail(12).reset_index(drop=True))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download data as CSV", data=csv, file_name=f"marine_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')

        # Quick risk score (combined wave + wind score) — 0..100
        try:
            wave = last['Wave Height (m)']
            wind = last['Wind Speed (m/s)']
            # normalize against practical sea thresholds
            wave_score = min(wave / 6.0, 1.0)
            wind_score = min(wind / 20.0, 1.0)
            risk = (0.6 * wave_score + 0.4 * wind_score) * 100
            st.metric(label='Estimated risk score', value=f"{risk:.0f}", delta=f"Wave {wave:.1f} m | Wind {wind:.1f} m/s")
        except Exception:
            pass

        # Feature importance (if available) — helps explain which inputs drive the prediction
        try:
            if hasattr(model, 'feature_importances_'):
                st.markdown('---')
                st.subheader('Model feature importance')
                fi = pd.DataFrame({
                    'feature': ['Wave Height (m)', 'Wind Speed (m/s)', 'Swell Height (m)', 'Swell Period (s)', 'wind_x', 'wind_y', 'wave_energy'],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                bar = alt.Chart(fi).mark_bar().encode(x='importance:Q', y=alt.Y('feature:N', sort='-x'), tooltip=['feature', 'importance'])
                st.altair_chart(bar, use_container_width=True)
        except Exception:
            # don't fail if model structure differs
            pass
