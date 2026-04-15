import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Household Energy Advisor", page_icon="⚡", layout="wide")

# ==============================
# Paths
# ==============================
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "model_package"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "rf_household_forecast_timeaware.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARIFF_HOURLY_PATH = DATA_DIR / "cleaned_tariff_hourly.csv"
TARIFF_SUMMARY_PATH = DATA_DIR / "cleaned_tariff_plan_summary.csv"
SAMPLE_USAGE_PATH = DATA_DIR / "sample_user_usage.csv"


# ==============================
# Loaders
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_cols():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_tariff_hourly():
    df = pd.read_csv(TARIFF_HOURLY_PATH)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["unit_rate_eur_kwh"] = pd.to_numeric(df["unit_rate_eur_kwh"], errors="coerce")
    return df


@st.cache_data
def load_tariff_summary():
    return pd.read_csv(TARIFF_SUMMARY_PATH)


@st.cache_data
def load_usage_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    else:
        df["temp"] = 12.0

    df = df.dropna(subset=["datetime", "demand"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ==============================
# Forecast Feature Builder
# ==============================
def build_feature_row(current_dt, current_demand, temp, lag_1, lag_2, lag_3, lag_24):
    hour = current_dt.hour
    dayofweek = current_dt.weekday()
    month = current_dt.month

    return pd.DataFrame([{
        "demand": float(current_demand),
        "temp": float(temp),
        "hour": float(hour),
        "dayofweek": float(dayofweek),
        "month": float(month),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_3": float(lag_3),
        "lag_24": float(lag_24),
        "rolling_mean_3": (lag_1 + lag_2 + lag_3) / 3,
        "rolling_mean_6": (current_demand + lag_1 + lag_2 + lag_3 + lag_24 + lag_1) / 6
    }])


def predict_next_demand(model, feature_cols, feature_df):
    aligned = feature_df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = 0.0
    aligned = aligned[feature_cols]
    return float(model.predict(aligned)[0])


# ==============================
# Supplier Cost Comparison
# ==============================
def compare_supplier_costs_30min(usage_df, tariff_df, standing_charge_type="urban"):
    usage = usage_df.copy()
    usage["hour"] = usage["datetime"].dt.hour

    results = []
    plan_keys = tariff_df["plan_key"].dropna().unique()

    for plan_key in plan_keys:
        plan_rates = tariff_df[tariff_df["plan_key"] == plan_key].copy()

        merged = usage.merge(
            plan_rates[[
                "plan_key",
                "supplier",
                "plan_name",
                "meter_type",
                "tariff_variant",
                "hour",
                "unit_rate_eur_kwh",
                "standing_charge_urban",
                "standing_charge_rural"
            ]],
            on="hour",
            how="left"
        )

        if merged["unit_rate_eur_kwh"].isna().all():
            continue

        merged["interval_cost"] = merged["demand"] * merged["unit_rate_eur_kwh"]
        energy_cost = merged["interval_cost"].sum()

        if standing_charge_type == "rural":
            annual_charge = pd.to_numeric(
                merged["standing_charge_rural"], errors="coerce"
            ).dropna().iloc[0]
        else:
            annual_charge = pd.to_numeric(
                merged["standing_charge_urban"], errors="coerce"
            ).dropna().iloc[0]

        daily_charge = annual_charge / 365
        days_covered = max(1, usage["datetime"].dt.date.nunique())

        standing_cost = daily_charge * days_covered
        total_cost = energy_cost + standing_cost

        first_row = merged.iloc[0]

        results.append({
            "supplier": first_row["supplier"],
            "plan_name": first_row["plan_name"],
            "meter_type": first_row["meter_type"],
            "tariff_variant": first_row["tariff_variant"],
            "total_cost_eur": round(total_cost, 2),
            "energy_cost_eur": round(energy_cost, 2),
            "standing_cost_eur": round(standing_cost, 2),
        })

    return pd.DataFrame(results).sort_values("total_cost_eur").reset_index(drop=True)


# ==============================
# Forecast Horizon
# ==============================
def iterative_forecast_30min(model, feature_cols, usage_df, horizon_steps=48):
    df = usage_df.copy().sort_values("datetime").reset_index(drop=True)

    if len(df) < 25:
        raise ValueError("Need at least 25 rows of usage data.")

    history = list(df["demand"].values)
    current_dt = pd.Timestamp(df["datetime"].iloc[-1])
    temp = float(df["temp"].iloc[-1])

    forecasts = []

    for _ in range(horizon_steps):
        next_dt = current_dt + pd.Timedelta(minutes=30)

        feat = build_feature_row(
            next_dt,
            history[-1],
            temp,
            history[-1],
            history[-2],
            history[-3],
            history[-24]
        )

        pred = predict_next_demand(model, feature_cols, feat)

        forecasts.append({
            "datetime": next_dt,
            "predicted_demand": pred
        })

        history.append(pred)
        current_dt = next_dt

    return pd.DataFrame(forecasts)


# ==============================
# Load Resources
# ==============================
model = load_model()
feature_cols = load_feature_cols()
metadata = load_metadata()
tariff_hourly = load_tariff_hourly()
tariff_summary = load_tariff_summary()


# ==============================
# UI
# ==============================
st.title("⚡ Household Energy Advisor")
st.caption("Forecast electricity demand, compare supplier plans, and optimize usage timing.")

tab1, tab2, tab3 = st.tabs([
    "Quick Predict",
    "Usage Forecast",
    "Supplier Comparison"
])


# ==============================
# TAB 1 — Quick Predict
# ==============================
with tab1:
    st.subheader("Quick Predict")
    st.write("Enter simple current values to estimate the next 30-minute household demand.")

    col1, col2, col3 = st.columns(3)

    with col1:
        demand = st.number_input("Current demand", min_value=0.0, value=1.8, step=0.1)
        temp = st.number_input("Temperature", value=10.2, step=0.1)

    with col2:
        hour = st.number_input("Hour", min_value=0, max_value=23, value=18)
        dayofweek = st.number_input("Day of week (0=Mon)", min_value=0, max_value=6, value=2)

    with col3:
        month = st.number_input("Month", min_value=1, max_value=12, value=4)

    st.caption("Recent lag values are handled automatically in the background to keep this form simple.")

    if st.button("Predict Next Demand"):
        # Build timestamp from user inputs
        current_dt = pd.Timestamp(year=2026, month=int(month), day=1, hour=int(hour), minute=0)

        # Auto-filled lag assumptions for simple user mode
        lag_1 = demand
        lag_2 = demand
        lag_3 = demand
        lag_24 = demand

        feature_df = build_feature_row(
            current_dt=current_dt,
            current_demand=demand,
            temp=temp,
            lag_1=lag_1,
            lag_2=lag_2,
            lag_3=lag_3,
            lag_24=lag_24
        )

        # Override dayofweek and month directly from user input
        feature_df["dayofweek"] = float(dayofweek)
        feature_df["month"] = float(month)

               pred = predict_next_demand(model, feature_cols, feature_df)

        cheapest_hour = (
            tariff_hourly.groupby("hour")["unit_rate_eur_kwh"]
            .mean()
            .sort_values()
            .index[0]
        )

        st.success(f"Predicted next 30-minute demand: {pred:.3f}")

        q1, q2 = st.columns(2)
        q1.metric("Predicted Next Demand", f"{pred:.3f}")
        q2.metric("Suggested Low-Cost Hour", f"{int(cheapest_hour):02d}:00")

        st.info(
            f"Low-cost usage suggestion: flexible activities may be cheaper around {int(cheapest_hour):02d}:00."
        )


# ==============================
# TAB 2 — Usage Forecast
# ==============================
with tab2:
    st.subheader("Usage Upload & Forecast")

    uploaded_file = st.file_uploader("Upload usage CSV", type=["csv"])

    if uploaded_file:
        usage_df = load_usage_csv(uploaded_file)

        st.dataframe(usage_df.head())

        horizon = st.selectbox("Forecast horizon", ["24 hours", "7 days"])

        if st.button("Run Forecast"):
            steps = 48 if horizon == "24 hours" else 336
            forecast_df = iterative_forecast_30min(
                model,
                feature_cols,
                usage_df,
                horizon_steps=steps
            )

            fig = px.line(
                forecast_df,
                x="datetime",
                y="predicted_demand",
                title="Forecast Demand"
            )

            st.plotly_chart(fig, use_container_width=True)


# ==============================
# TAB 3 — Supplier Comparison
# ==============================
with tab3:
    st.subheader("Supplier Comparison")

    uploaded_compare = st.file_uploader("Upload usage CSV for supplier comparison", type=["csv"], key="compare")

    if uploaded_compare:
        usage_df = load_usage_csv(uploaded_compare)

        comparison_df = compare_supplier_costs_30min(
            usage_df,
            tariff_hourly,
            standing_charge_type="urban"
        )

        st.dataframe(comparison_df)

                if not comparison_df.empty:
            best = comparison_df.iloc[0]

            st.markdown("### Best Recommendation Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Cheapest Supplier", best["supplier"])
            c2.metric("Best Plan", best["plan_name"])
            c3.metric("Estimated Total Cost (€)", f"{best['total_cost_eur']:.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Estimated Energy Cost (€)", f"{best['energy_cost_eur']:.2f}")
            c5.metric("Standing Cost (€)", f"{best['standing_cost_eur']:.2f}")
            c6.metric("Meter Type", str(best["meter_type"]))


            fig = px.bar(
                comparison_df.head(10),
                x="supplier",
                y="total_cost_eur",
                color="plan_name",
                title="Supplier Cost Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)


# ==============================
# Footer
# ==============================
with st.expander("Model Metadata"):
    st.json(metadata)
