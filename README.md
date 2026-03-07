# 🌊 Marine Weather Forecasting

> An AI-powered Streamlit dashboard that predicts real-time marine sea-state conditions — **Calm**, **Moderate**, or **Rough** — using ocean parameters and machine learning.

## 📖 Overview

**Marine Weather Forecaster** is an interactive prediction dashboard built for sailors, mariners, and marine researchers. It fetches live oceanic data from the [StormGlass API](https://stormglass.io/) or the free [Open-Meteo Marine API](https://open-meteo.com/) and runs it through a pre-trained scikit-learn classifier to label current sea conditions.

**Key input features used by the model:**

| Feature | Description |
|---|---|
| Wave Height (m) | Significant wave height at the selected location |
| Wind Speed (m/s) | Surface wind speed |
| Swell Height (m) | Swell wave height |
| Swell Period (s) | Swell wave period |

**Predicted output labels:** `Calm` · `Moderate` · `Rough`

---

## ✨ Features

- 🔮 **ML Prediction** — Random Forest classifier (`marine_model.pkl`) with confidence scores
- 🌐 **Live Data** — Pulls real marine forecast via [Open-Meteo](https://open-meteo.com/) (free, no key needed) or [StormGlass](https://stormglass.io/) (API key required)
- 🗺️ **Interactive Map** — Click anywhere on a Folium map to select a prediction location
- 📌 **Location Presets** — Quick-select popular marine locations (Mumbai, Chennai, Kochi, Sydney, Cape Town…)
- 🔍 **City Search** — Type any coastal city name to geocode and fetch data automatically
- 📈 **Trend Charts** — Hourly Altair charts with tooltips for wave, wind, and swell
- ⚠️ **Safety Guidance** — Contextual safety tips based on predicted condition and risk score
- 📋 **Data Table + CSV Download** — View and export the last 12 hours of hourly data
- 🧠 **Feature Importance Chart** — Visual breakdown of what drives the model's prediction
- ⚙️ **Unit Toggle** — Switch between Metric (m, m/s) and Nautical (ft, knots)
- 🎚️ **Smoothing** — Configurable moving-average smoothing on trend charts



## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `scikit-learn` | ML model (classifier) |
| `joblib` | Model serialisation / loading |
| `pandas` / `numpy` | Data manipulation |
| `requests` | HTTP calls to weather APIs |
| `altair` | Interactive trend charts |
| `pydeck` | PyDeck map layer (fallback) |
| `folium` + `streamlit-folium` | Interactive click-to-select map |
| `python-dotenv` | Load `.env` API keys |

---

## 🧠 How It Works

```
User picks a location (map / city / preset)
           │
           ▼
   Fetch marine data
   ┌──────────────────────────────────────────┐
   │  Priority 1 → StormGlass (if key given)  │
   │  Priority 2 → Open-Meteo (free, no key)  │
   │  Priority 3 → Built-in sample data       │
   └──────────────────────────────────────────┘
           │
           ▼
   Feature Engineering
   (wave height, wind speed, swell height, swell period)
           │
           ▼
   Pre-trained Random Forest Classifier
   (marine_model.pkl)
           │
           ▼
   Output: Calm / Moderate / Rough + confidence %
         + Risk Score + Safety Guidance + Charts
```

---

## 🌍 Supported Preset Locations

| Location | Lat | Lon |
|---|---|---|
| San Francisco, USA | 37.77 | -122.42 |
| Sydney, Australia | -33.87 | 151.21 |
| Cape Town, South Africa | -33.92 | 18.42 |
| Mumbai, India | 19.08 | 72.88 |
| Honolulu, Hawaii | 21.31 | -157.86 |
| Visakhapatnam, India | 17.69 | 83.22 |
| Chennai, India | 13.08 | 80.27 |
| Kochi, India | 9.93 | 76.27 |


## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss any major changes before submitting a PR.

