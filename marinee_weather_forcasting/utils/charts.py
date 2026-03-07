"""
utils/charts.py
Charting utilities for the Marine Weather Predictor dashboard.
"""
import pandas as pd
import altair as alt
import streamlit as st


def convert_units(df_chart: pd.DataFrame, units: str) -> pd.DataFrame:
    """Convert metric values to nautical units if requested."""
    df = df_chart.copy()
    if units.startswith("Nautical"):
        def _convert(row):
            if "Wave Height" in row["Parameter"] or "Swell Height" in row["Parameter"]:
                return row["Value"] * 3.28084   # m -> ft
            if "Wind Speed" in row["Parameter"]:
                return row["Value"] * 1.94384   # m/s -> knots
            return row["Value"]
        df["Value"] = df.apply(_convert, axis=1)
    return df


def render_trend_chart(df: pd.DataFrame, units: str, smoothing: int) -> None:
    """Render an interactive line chart of marine parameters over time."""
    try:
        df_chart = df[["Time", "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)"]].melt(
            "Time", var_name="Parameter", value_name="Value"
        )

        if smoothing > 0:
            df_chart["Value"] = df_chart.groupby("Parameter")["Value"].transform(
                lambda x: x.rolling(smoothing, min_periods=1).mean()
            )

        df_chart = convert_units(df_chart, units)

        chart = (
            alt.Chart(df_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X("Time:T", title="Time"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Parameter:N"),
                tooltip=["Time:T", "Parameter:N", "Value:Q"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        df_plot = df.set_index("Time")[["Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)"]]
        st.line_chart(df_plot)


def render_feature_importance(model) -> None:
    """Render a horizontal bar chart of model feature importances if available."""
    try:
        if not hasattr(model, "feature_importances_"):
            return
        fi = pd.DataFrame({
            "feature": [
                "Wave Height (m)", "Wind Speed (m/s)", "Swell Height (m)",
                "Swell Period (s)", "wind_x", "wind_y", "wave_energy",
            ],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        bar = (
            alt.Chart(fi)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", "importance"],
                color=alt.value("#0ea5a0"),
            )
        )
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        pass
