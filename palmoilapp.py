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
st.set_page_config(page_title="Palm Oil ML App", layout="wide")
st.title("🌴 Palm Oil ML Dashboard (Fast Demo)")

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
    st.subheader("📊 Dashboard (Monthly EDA)")
    st.caption("This tab uses a merged dataset CSV. If you don’t have it, you can still use Model Comparison + Prediction.")

    merged_df = None
    cA, cB = st.columns([1, 1])
    with cA:
        use_local = st.checkbox("Use local merged dataset", value=True)
    with cB:
        uploaded = st.file_uploader("Or upload merged dataset CSV", type=["csv"])

    try:
        if use_local and file_exists(DEFAULT_MERGED_DATA_PATH):
            merged_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH)
            st.success(f"Loaded: {DEFAULT_MERGED_DATA_PATH}")
        elif uploaded is not None:
            merged_df = load_merged_dataset(uploaded)
            st.success("Loaded uploaded merged dataset.")
        else:
            st.info("No merged dataset loaded. Put it at `data/final_merged_palm_oil_dataset.csv` or upload it here.")
    except Exception as e:
        st.error(f"Failed to load merged dataset: {e}")
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
                k2.metric("Avg Price", f"{df_m[COL_PRICE].mean():,.2f}")
                k3.metric("Avg Production", f"{df_m[COL_PROD].mean():,.2f}")
                k4.metric("Total Export", f"{df_m[COL_EXPORT].sum():,.0f}")

                st.divider()
                st.markdown("### 1) Price Over Time (Monthly Mean)")
                line_plot(df_m["Month"], df_m[COL_PRICE], "Palm Oil Price (Monthly Mean)", "Price")

                st.markdown("### 2) Production Over Time (Monthly Mean)")
                line_plot(df_m["Month"], df_m[COL_PROD], "Index Production (Monthly Mean)", "Index Production")

                st.markdown("### 3) Export Over Time (Monthly Total)")
                line_plot(df_m["Month"], df_m[COL_EXPORT], "Export Volume (Monthly Total)", "Export Number (in Tonnes)")

                st.divider()
                st.markdown("### 4) Relationship: Price vs Production")
                scatter_plot(df_m[COL_PROD], df_m[COL_PRICE], "Price vs Production (Monthly)", "Index Production", "Price")

                st.markdown("### 5) Relationship: Rainfall vs Production")
                scatter_plot(df_m[COL_PRECIP], df_m[COL_PROD], "Rainfall vs Production (Monthly)", "Precip", "Index Production")

                st.divider()
                st.markdown("### 6) Correlation Heatmap (Monthly)")
                heat_cols = [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + [c for c in OPTIONAL_COLS if c in df_m.columns]
                corr_heatmap(df_m.dropna(subset=heat_cols), heat_cols, "Correlation Matrix (Monthly)")

                with st.expander("Preview monthly data"):
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

# ============================================================
# TAB 3: PREDICTION (Inputs + Outcome in SAME TAB ✅)
# ============================================================
with tab_pred:
    st.subheader("🧠 Prediction (Input + Outcome)")
    st.caption(f"Model used: {best_model_name} (loaded from artifacts)")

    mode = st.radio("Mode", ["Single prediction", "Batch prediction (CSV)"], horizontal=True)

    if mode == "Single prediction":
        st.markdown("### Enter feature values")
        inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                inputs[feat] = st.number_input(feat, value=0.0, step=0.1, key=f"single_{feat}")

        c1, c2 = st.columns([1, 1])
        with c1:
            do_pred = st.button("Predict Price ✅")
        with c2:
            reset = st.button("Reset output")

        if reset:
            st.session_state.single_pred_value = None
            st.session_state.single_pred_inputs = None

        if do_pred:
            try:
                pred = predict_one(inputs)
                st.session_state.single_pred_value = pred
                st.session_state.single_pred_inputs = inputs
                st.session_state.batch_pred_df = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.divider()
        st.markdown("### Outcome")
        if st.session_state.single_pred_value is None:
            st.info("No prediction yet. Fill values and click Predict.")
        else:
            st.metric("Predicted Palm Oil Price", f"{st.session_state.single_pred_value:,.2f}")
            with st.expander("Show inputs used"):
                st.json(st.session_state.single_pred_inputs)

    else:
        st.markdown("### Upload CSV for batch prediction")
        st.caption("CSV must include ALL feature columns in feature_names.json (same names).")
        up = st.file_uploader("Upload CSV", type=["csv"])

        if st.button("Reset batch output"):
            st.session_state.batch_pred_df = None

        if up is not None:
            try:
                df_in = pd.read_csv(up)
                missing_feats = [f for f in feature_names if f not in df_in.columns]
                if missing_feats:
                    st.error("Missing columns:\n" + "\n".join([f"- {m}" for m in missing_feats]))
                else:
                    X = df_in[feature_names].copy()
                    X = X.apply(pd.to_numeric, errors="coerce")

                    if X.isnull().any().any():
                        st.warning("Some values became NaN after numeric conversion. Clean/fill missing values for best results.")

                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)

                    out = df_in.copy()
                    out["Predicted_Price"] = preds

                    st.session_state.batch_pred_df = out
                    st.session_state.single_pred_value = None
                    st.session_state.single_pred_inputs = None
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

        st.divider()
        st.markdown("### Outcome")
        if st.session_state.batch_pred_df is None:
            st.info("No batch results yet. Upload a CSV to generate predictions.")
        else:
            st.success("Batch predictions ready ✅")
            st.dataframe(st.session_state.batch_pred_df, use_container_width=True)

            csv_bytes = st.session_state.batch_pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
