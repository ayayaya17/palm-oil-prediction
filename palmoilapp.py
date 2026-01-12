# app.py
# ✅ Streamlit app (FAST DEMO): loads saved artifacts + (optional) merged dataset for dashboard
# Tabs: Dashboard, Model Comparison, Prediction (inputs + outcome together)

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Paths (edit if your folders differ)
# ----------------------------
ART_DIR = "artifacts"
RESULTS_PATH = os.path.join(ART_DIR, "results.csv")
MODEL_PATH = os.path.join(ART_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_names.json")

# Optional merged dataset for Dashboard (EDA)
DEFAULT_MERGED_DATA_PATH = os.path.join("data", "final_merged_palm_oil_dataset.csv")

# Expected columns for the EDA dashboard (only if you use merged dataset)
COL_DATE = "Date"
COL_PRICE = "Price"
COL_PROD = "Index Production"
COL_EXPORT = "Export Number (in Tonnes)"
COL_PRECIP = "Precip"
OPTIONAL_COLS = ["Temp", "Humidity", "USD"]

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Palm Oil Price Prediction App", layout="wide")
st.title("🌴 Palm Oil Price Forecasting Dashboard for Malaysia ")

# ----------------------------
# Utility helpers
# ----------------------------
def stop_with_error(msg: str):
    st.error(msg)
    st.stop()

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

@st.cache_data(show_spinner=False)
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_feature_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        feats = json.load(f)
    if not isinstance(feats, list) or not feats:
        raise ValueError("feature_names.json must be a non-empty list of feature names.")
    return feats

def infer_best_model_name(results_df: pd.DataFrame) -> str:
    if "Model" not in results_df.columns:
        return "Best Model"
    if "RMSE" in results_df.columns:
        return str(results_df.sort_values("RMSE", ascending=True).iloc[0]["Model"])
    if "R-squared" in results_df.columns:
        return str(results_df.sort_values("R-squared", ascending=False).iloc[0]["Model"])
    if "R2" in results_df.columns:
        return str(results_df.sort_values("R2", ascending=False).iloc[0]["Model"])
    return str(results_df.iloc[0]["Model"])

def pick_best_row(results_df: pd.DataFrame) -> pd.Series:
    if "RMSE" in results_df.columns:
        return results_df.sort_values("RMSE", ascending=True).iloc[0]
    if "MAE" in results_df.columns:
        return results_df.sort_values("MAE", ascending=True).iloc[0]
    if "R-squared" in results_df.columns:
        return results_df.sort_values("R-squared", ascending=False).iloc[0]
    if "R2" in results_df.columns:
        return results_df.sort_values("R2", ascending=False).iloc[0]
    return results_df.iloc[0]

# ----------------------------
# Load artifacts (REQUIRED)
# ----------------------------
missing = [p for p in [RESULTS_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not file_exists(p)]
if missing:
    stop_with_error(
        "Missing artifact files:\n\n"
        + "\n".join([f"- {m}" for m in missing])
        + "\n\nFix: run your training code ONCE to generate the `artifacts/` folder."
    )

results_df = load_results(RESULTS_PATH)
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
feature_names = load_feature_names(FEATURES_PATH)

best_model_name = infer_best_model_name(results_df)
best_row = pick_best_row(results_df)

# ----------------------------
# Session state
# ----------------------------
if "single_pred_value" not in st.session_state:
    st.session_state.single_pred_value = None
if "single_pred_inputs" not in st.session_state:
    st.session_state.single_pred_inputs = None
if "batch_pred_df" not in st.session_state:
    st.session_state.batch_pred_df = None

# ----------------------------
# Dashboard helpers (optional merged dataset)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_merged_dataset(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file) if isinstance(path_or_file, str) else pd.read_csv(path_or_file)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).sort_values(COL_DATE).reset_index(drop=True)

    for c in [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + OPTIONAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def compute_feature_defaults(merged_csv_path: str, feature_names: list[str]) -> dict:
    df = pd.read_csv(merged_csv_path)
    # Convert columns to numeric when possible
    for c in feature_names:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Mean defaults (fallback to 0.0 if missing)
    defaults = {}
    for c in feature_names:
        if c in df.columns:
            m = df[c].mean(skipna=True)
            defaults[c] = float(m) if pd.notnull(m) else 0.0
        else:
            defaults[c] = 0.0
    return defaults

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month"] = df[COL_DATE].dt.to_period("M").dt.to_timestamp()
    agg = {COL_PRICE: "mean", COL_PROD: "mean", COL_EXPORT: "sum", COL_PRECIP: "mean"}
    for c in OPTIONAL_COLS:
        if c in df.columns:
            agg[c] = "mean"
    return df.groupby("Month", as_index=False).agg(agg)

def line_plot(x, y, title, y_label):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def scatter_plot(x, y, title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, s=24, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def corr_heatmap(df: pd.DataFrame, cols: list[str], title: str):
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(corr.values)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Prediction helpers
# ----------------------------
def predict_one(input_dict: dict) -> float:
    X_one = pd.DataFrame([input_dict])[feature_names]
    X_scaled = scaler.transform(X_one)
    return float(model.predict(X_scaled)[0])

# ----------------------------
# Tabs (Outcome merged into Prediction tab ✅)
# ----------------------------
tab_dash, tab_compare, tab_pred = st.tabs(["📊 Dashboard", "🏆 Model Comparison", "🧠 Prediction"])

# ============================================================
# TAB 1: DASHBOARD (optional merged dataset)
# ============================================================
with tab_dash:
    st.subheader("📊 Dashboard : Monthly Trends")

    merged_df = None

    try:
        merged_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load merged dataset : {e}")
        merged_df = None

    if merged_df is not None:
        required = [COL_DATE, COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP]
        miss = [c for c in required if c not in merged_df.columns]
        if miss:
            st.error("Merged dataset missing:\n" + "\n".join([f"- {m}" for m in miss]))
        else:
            df_monthly = to_monthly(merged_df)

            min_m = df_monthly["Month"].min()
            max_m = df_monthly["Month"].max()
            start_date, end_date = st.date_input(
                "Month range",
                value=(min_m.date(), max_m.date()),
                min_value=min_m.date(),
                max_value=max_m.date(),
            )
            mask = (df_monthly["Month"].dt.date >= start_date) & (df_monthly["Month"].dt.date <= end_date)
            df_m = df_monthly.loc[mask].copy()

            if df_m.empty:
                st.warning("No data in that range.")
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Months", f"{len(df_m):,}")
                k2.metric("Average Price ", f"RM {df_m[COL_PRICE].mean():,.2f}")
                k3.metric("Average Production", f"{df_m[COL_PROD].mean():,.2f}")
                k4.metric("Total Export", f"{df_m[COL_EXPORT].sum():,.0f}")

                st.divider()
                st.markdown("### Palm Oil Monthly Price Trend")
                line_plot(df_m["Month"], df_m[COL_PRICE], "Palm Oil Price (Monthly Mean)", "Price (in RM)")

                st.markdown("### Production Monthly Index Trend")
                line_plot(df_m["Month"], df_m[COL_PROD], "Production Index (Monthly Mean)", "Production Index")

                st.markdown("### Export Monthly Volume Trend")
                line_plot(df_m["Month"], df_m[COL_EXPORT], "Export Volume (Monthly Total)", "Export Number (in Tonnes)")

                st.divider()
                st.markdown("### Price-Production Relationships")
                scatter_plot(df_m[COL_PROD], df_m[COL_PRICE], "Palm Oil Price vs Production (Monthly)", "Production Index", "Price (in RM)")

                st.markdown("### Rainfall-Prodcution Relationships")
                scatter_plot(df_m[COL_PRECIP], df_m[COL_PROD], "Rainfall vs Production (Monthly)", "Rainfall", "Production Index")

                st.divider()
                st.markdown("### Feature Correlation Overview (Monthly)")
                heat_cols = [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + [c for c in OPTIONAL_COLS if c in df_m.columns]
                corr_heatmap(df_m.dropna(subset=heat_cols), heat_cols, "Monthly Feature Correlation Matrix")

                with st.expander("View Monthly Dataset"):
                    st.dataframe(df_m, use_container_width=True)

# ============================================================
# TAB 2: MODEL COMPARISON
# ============================================================
with tab_compare:
    st.subheader("🏆 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model_name)

    if "RMSE" in results_df.columns:
        c2.metric("RMSE", f"{float(best_row.get('RMSE', np.nan)):.4f}")
    if "MAE" in results_df.columns:
        c3.metric("MAE", f"{float(best_row.get('MAE', np.nan)):.4f}")

    if "R-squared" in results_df.columns:
        c4.metric("R²", f"{float(best_row.get('R-squared', np.nan)):.4f}")
    elif "R2" in results_df.columns:
        c4.metric("R²", f"{float(best_row.get('R2', np.nan)):.4f}")

    if "Model" in results_df.columns and "RMSE" in results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(results_df["Model"], results_df["RMSE"])
        ax.set_title("RMSE by Model (Lower is better)")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig, clear_figure=True)

# ============================
# TAB 3: PREDICTION (Inputs first, horizon below)
# ============================
with tab_pred:
    st.subheader("🧠 Palm Oil Price Estimation")

    # Big visible model banner
    st.markdown(
        f"""
        <div style="padding:12px 14px; border-radius:14px; background:#f3f6ff; border:1px solid #dbe4ff;">
          <div style="font-size:20px; font-weight:850;">Model used: {best_model_name}</div>
          <div style="font-size:12.5px; opacity:0.8;">
            Short-term estimation under assumed stable conditions (trained on 2020–2022).
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # Defaults
    defaults = compute_feature_defaults(DEFAULT_MERGED_DATA_PATH, feature_names)
    FORCE_ZERO_DEFAULTS = {"Index Production", "Export Number (in Tonnes)"}

    # ----------------------------
    # 1) Inputs FIRST
    # ----------------------------
    st.markdown("### Set Parameters for Single Prediction")
    st.caption("Default values use the dataset mean (except Production & Export start at 0). You can adjust any value.")

    inputs = {}
    cols = st.columns(2, gap="large")

    for i, feat in enumerate(feature_names):
        # Remove Year from UI (we fix it internally)
        if feat.lower() == "year":
            continue

        # Force 0 for selected fields, else mean
        if feat in FORCE_ZERO_DEFAULTS:
            default_val = 0.0
        else:
            default_val = float(defaults.get(feat, 0.0))

        with cols[i % 2]:
            inputs[feat] = st.number_input(
                feat,
                value=default_val,
                step=0.1,
                key=f"single_{feat}"
            )

    # Fix Year internally (avoid long-term extrapolation)
    if "Year" in feature_names:
        inputs["Year"] = 2022

    st.write("")

    # ----------------------------
    # 2) Horizon BELOW inputs
    # ----------------------------
    st.markdown("### Prediction Horizon (Days Ahead)")
    days_ahead = st.slider(
        "Days ahead",
        min_value=1,
        max_value=14,   # ✅ recommend 7–14 max
        value=1,
        step=1
    )
    st.caption(
        "This is a short-term interpretation label. The model assumes conditions remain similar unless you modify inputs."
    )

    # ----------------------------
    # Buttons
    # ----------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        do_pred = st.button("Predict Price ✅")
    with c2:
        reset = st.button("Reset output")

    if reset:
        st.session_state.single_pred_value = None
        st.session_state.single_pred_inputs = None
        st.session_state.pred_meta = None

    if do_pred:
        try:
            pred = predict_one(inputs)
            st.session_state.single_pred_value = pred
            st.session_state.single_pred_inputs = inputs
            st.session_state.pred_meta = {"days_ahead": days_ahead}
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ----------------------------
    # Outcome
    # ----------------------------
    st.divider()
    st.markdown("### Outcome")

    if st.session_state.single_pred_value is None:
        st.info("No prediction yet. Adjust values and click Predict.")
    else:
        meta = st.session_state.pred_meta or {"days_ahead": days_ahead}
        st.metric(
            f"Predicted Palm Oil Price per Tonne (in {meta['days_ahead']} day(s))",
            f"RM {st.session_state.single_pred_value:,.2f}"
        )
        with st.expander("Show inputs used"):
            st.json(st.session_state.single_pred_inputs)
