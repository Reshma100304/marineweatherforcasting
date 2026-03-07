#  Marine Weather Predictor Dashboard

An AI-powered Marine Weather Prediction System that predicts real-time marine conditions — Calm, Moderate, or Rough — based on ocean parameters (wave height, wind speed, swell height/period).

Built with: Python, Streamlit, scikit-learn, joblib and the StormGlass API.

---

##  Live Demo

If this repository is deployed on Streamlit Cloud you can open the live app there otherwise you can run locally.

---

##  Whats new in this branch

This update adds a clean, polished UI with:
- Sidebar presets and advanced controls (hours, live vs sample)
- Interactive Altair charts with tooltips
- Map (PyDeck / st.map) for selected coordinates
- Cached model loading and better error handling
- Clear KPIs, confidence display and tactical safety guidance
- Downloadable CSV and a nicer color/theme
