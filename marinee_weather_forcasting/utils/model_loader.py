"""
utils/model_loader.py
Handles loading the trained ML model from disk with Streamlit caching.
"""
import os
import warnings
import joblib
import streamlit as st

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    warnings.filterwarnings("ignore")


@st.cache_resource
def load_model(path: str):
    """Load the saved model once and cache it for quick repeated predictions."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Failed loading model file: %s" % e)
        return None


def get_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "marine_model.pkl"
    )
    return load_model(model_path)
