#  Marine Weather Predictor Dashboard

An AI-powered Marine Weather Prediction System that predicts real-time marine conditions — Calm, Moderate, or Rough — based on ocean parameters (wave height, wind speed, swell height/period).

Built with: Python, Streamlit, scikit-learn, joblib and the StormGlass API.

---

##  Live Demo

If this repository is deployed on Streamlit Cloud you can open the live app there. Otherwise run locally (instructions below).

---

##  Whats new in this branch

This update adds a clean, polished UI with:
- Sidebar presets and advanced controls (hours, live vs sample)
- Interactive Altair charts with tooltips
- Map (PyDeck / st.map) for selected coordinates
- Cached model loading and better error handling
- Clear KPIs, confidence display and tactical safety guidance
- Downloadable CSV and a nicer color/theme

---

##  Quick start — Run locally (Windows PowerShell)

1. Clone the repo and open the folder

```powershell
git clone https://github.com/Reshma100304/marineweatherforcasting.git
cd MarineWeatherPredictor
```

2. Create a virtual environment, activate and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the app

```powershell
streamlit run app.py
```

4. Open http://localhost:8501 in a browser (Streamlit will also print the URL to the console).

---

##  Notes

- If you want real-time data, create a free StormGlass account and enter your key in the UI; otherwise the app uses sample data.
- The ML model is saved (marine_model.pkl) — the app loads it at runtime. Scikit-learn version mismatches can cause warnings when unpickling.

---

If you'd like, I can add screenshots and a short deployment guide for Streamlit Cloud next.
