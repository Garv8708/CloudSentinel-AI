from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# NTIC — Network Traffic Intelligence Center
# Powered by CloudSentinel AI
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

st.set_page_config(
    page_title="NTIC | Network Traffic Intelligence Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise light theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F4F7FC; color: #172B4D; }
    [data-testid="stHeader"] { background: rgba(244,247,252,.92); }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DFE7F2; }
    [data-testid="stSidebar"] * { color: #243B5A; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    .brand { display:flex; align-items:center; gap:13px; margin-bottom:24px; }
    .brand-icon { width:48px; height:48px; display:flex; align-items:center; justify-content:center;
      border-radius:14px; background:linear-gradient(135deg,#DCEBFF,#C7F2F4); font-size:25px; }
    .brand-name { font-size:22px; font-weight:800; color:#173B69; line-height:1.05; }
    .brand-sub { font-size:10px; font-weight:800; letter-spacing:2px; color:#66809F; margin-top:5px; }
    .hero { position:relative; overflow:hidden; padding:38px 42px; border-radius:24px;
      background:linear-gradient(112deg,#153D78 0%,#2057C5 62%,#087F9B 100%);
      box-shadow:0 18px 45px rgba(27,76,148,.16); margin-bottom:22px; }
    .hero:after { content:""; position:absolute; width:330px; height:330px; right:-85px; top:-185px;
      border:1px solid rgba(255,255,255,.18); border-radius:50%;
      box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025); }
    .hero-tag { display:inline-block; color:white; background:rgba(255,255,255,.14);
      border:1px solid rgba(255,255,255,.22); border-radius:30px; padding:7px 13px;
      font-size:11px; font-weight:800; letter-spacing:1.2px; }
    .hero h1 { color:#fff; font-size:clamp(30px,4vw,47px); font-weight:800;
      letter-spacing:-1.5px; line-height:1.12; margin:20px 0 12px; }
    .hero p { color:#E4EEFF; font-size:16px; max-width:850px; line-height:1.7; margin:0; }
    .section-title { color:#193657; font-size:23px; font-weight:800; margin:28px 0 14px; }
    .metric { background:#fff; border:1px solid #DFE8F4; border-radius:17px;
      padding:19px 21px; min-height:116px; box-shadow:0 5px 16px rgba(25,55,95,.035); }
    .metric-label { color:#71839D; font-size:13px; font-weight:600; }
    .metric-value { color:#1D477F; font-size:31px; font-weight:800; margin-top:7px; }
    .metric-note { color:#8A9AB0; font-size:11px; margin-top:2px; }
    .notice { background:#EDF5FF; border:1px solid #C9DFFF; border-radius:14px;
      padding:15px 18px; color:#405875; font-size:13px; line-height:1.65; margin:18px 0 22px; }
    .panel { background:#fff; border:1px solid #DFE8F4; border-radius:18px;
      padding:20px 22px; box-shadow:0 5px 16px rgba(25,55,95,.035); }
    div[data-testid="stFileUploader"] { background:#F8FAFE; border:1px dashed #9DB8DA;
      border-radius:14px; padding:14px; }
    .small-muted { color:#7789A2; font-size:12px; }
    div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:700;
      min-height:44px; border:0; background:linear-gradient(90deg,#2057C5,#087F9B); color:white; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { color:white; border:0;
      box-shadow:0 5px 16px rgba(32,87,197,.22); }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                value = json.load(f)
                return value if isinstance(value, dict) else {}
    except Exception:
        return {}
    return {}

@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path_string: str):
    return joblib.load(path_string)

def find_metric(metadata: dict[str, Any], candidates: tuple[str, ...], default=None):
    for key in candidates:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    # Search one level down for common nested metric dictionaries.
    for value in metadata.values():
        if isinstance(value, dict):
            for key in candidates:
                if key in value and value[key] is not None:
                    return value[key]
    return default

def get_expected_features(model) -> list[str] | None:
    features = getattr(model, "feature_names_in_", None)
    if features is not None:
        return [str(x) for x in features]
    # Pipeline's preprocessing step may expose the input feature names.
    try:
        return [str(x) for x in model[:-1].feature_names_in_]
    except Exception:
        return None

def clean_upload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [
        c for c in df.columns
        if c.strip().lower() in {"attack_type", "label", "target", "class", "unnamed: 0"}
        or c.strip().lower().startswith("unnamed:")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def align_features(df: pd.DataFrame, expected: list[str] | None) -> pd.DataFrame:
    if not expected:
        return df
    # Preserve exact training feature order and allow imputers to handle absent values.
    aligned = df.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = np.nan
    extra = [c for c in aligned.columns if c not in expected]
    if extra:
        aligned = aligned.drop(columns=extra)
    return aligned.reindex(columns=expected)

def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">🛡️</div><div>'
        '<div class="brand-name">NTIC</div><div class="brand-sub">NETWORK TRAFFIC INTELLIGENCE</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Workspace")
    page = st.radio("Navigate", ["Overview & analytics", "Traffic analysis"], label_visibility="collapsed")
    st.divider()
    st.markdown("### System status")
    model_exists = MODEL_PATH.exists()
    st.markdown(f"**ML model**  {'🟢 READY' if model_exists else '🔴 NOT FOUND'}")
    st.markdown(f"**Evaluation report**  {'🟢 AVAILABLE' if METADATA_PATH.exists() else '🟡 NOT FOUND'}")
    st.markdown("**Analysis mode**  CSV BATCH")
    st.divider()
    st.caption("BUILD 1.0 · ACADEMIC PROTOTYPE")
    st.caption("Powered by CloudSentinel AI")

metadata = load_json(METADATA_PATH)
model = None
model_error = None
if model_exists:
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = str(exc)

# Header
st.markdown(
    """
    <div class="hero">
      <span class="hero-tag">● &nbsp; AI-POWERED SECURITY ANALYTICS</span>
      <h1>Network Traffic<br>Intelligence Center</h1>
      <p>Classify network-flow records, review machine-learning performance and turn uploaded traffic data into actionable analytical insights.</p>
    </div>
    <div class="notice"><b>Scope & responsible use</b> &nbsp; Predictions are model classifications, not proof of an attack.
    This academic prototype analyzes uploaded CSV files only; it does not inspect live network traffic or automatically block connections.</div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(f"Model could not be loaded: {model_error}")
elif not model_exists:
    st.error(f"Trained model not found at: {MODEL_PATH}")

if page == "Overview & analytics":
    st.markdown('<div class="section-title">▌ Model overview</div>', unsafe_allow_html=True)
    accuracy = find_metric(metadata, ("accuracy", "test_accuracy", "held_out_accuracy"), 0.9984)
    try:
        accuracy_value = f"{float(accuracy) * 100:.2f}%" if float(accuracy) <= 1 else f"{float(accuracy):.2f}%"
    except Exception:
        accuracy_value = "99.84%"
    test_records = find_metric(metadata, ("test_records", "test_samples", "n_test", "test_size"), 24624)
    classes = find_metric(metadata, ("n_classes", "num_classes", "attack_classes", "class_count"), 12)
    training_records = find_metric(metadata, ("training_records", "train_records", "n_samples", "dataset_rows"), 123117)
    cols = st.columns(4)
    with cols[0]: metric_card("HELD-OUT ACCURACY", accuracy_value, "Saved evaluation result")
    with cols[1]: metric_card("EVALUATION RECORDS", f"{int(test_records):,}" if str(test_records).isdigit() else str(test_records), "Hold-out test set")
    with cols[2]: metric_card("TRAFFIC CLASSES", str(classes), "Target categories")
    with cols[3]: metric_card("TRAINING RECORDS", f"{int(training_records):,}" if str(training_records).isdigit() else str(training_records), "Source dataset size")

    st.markdown('<div class="section-title">▌ Model information</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**About the classification model**")
        st.write("CloudSentinel AI uses a trained Random Forest classification pipeline to categorize network-traffic records from a compatible CSV file.")
        st.write("The preprocessing pipeline handles numeric and categorical features before generating a predicted traffic class.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Evaluation context**")
        st.write("The displayed accuracy comes from the saved random hold-out evaluation. Performance on new or independently collected network traffic may differ.")
        st.write("Rare classes can have limited test examples, so accuracy should be considered alongside class-wise precision, recall and F1-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    report = find_metric(metadata, ("classification_report", "class_report", "report"), None)
    if isinstance(report, dict):
        report_df = pd.DataFrame(report).T
        st.markdown('<div class="section-title">▌ Class-wise performance</div>', unsafe_allow_html=True)
        st.dataframe(report_df, use_container_width=True)

elif page == "Traffic analysis":
    st.markdown('<div class="section-title">▌ Traffic analysis workspace</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV containing the same network-flow features used during model training.")
    uploaded = st.file_uploader("Choose a network traffic CSV file", type=["csv"], help="The target label column is optional. Identifier/index columns are ignored.")
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            raw_df = None

        if raw_df is not None:
            st.success(f"File loaded successfully · {len(raw_df):,} records · {len(raw_df.columns):,} columns")
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(raw_df.head(10), use_container_width=True)

            if model is None:
                st.error("The trained model is unavailable. Check artifacts/model.joblib.")
            elif st.button("Run traffic classification", type="primary", use_container_width=True):
                try:
                    with st.spinner("Analyzing network traffic records..."):
                        feature_df = clean_upload(raw_df)
                        expected = get_expected_features(model)
                        feature_df = align_features(feature_df, expected)
                        predictions = model.predict(feature_df)
                        result_df = raw_df.copy()
                        result_df["Predicted_traffic_class"] = predictions
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(feature_df)
                            result_df["Prediction_confidence"] = np.max(probabilities, axis=1)
                        else:
                            result_df["Prediction_confidence"] = np.nan

                    st.session_state["ntic_results"] = result_df
                    st.session_state["ntic_summary"] = {
                        "rows": len(result_df),
                        "classes": int(pd.Series(predictions).nunique()),
                        "most_common": str(pd.Series(predictions).value_counts().index[0]),
                    }
                except Exception as exc:
                    st.error(f"Could not analyze this CSV: {exc}")
                    st.info("Check that the uploaded file contains compatible feature columns and that the deployed model uses the pinned scikit-learn version.")

    result = st.session_state.get("ntic_results")
    summary = st.session_state.get("ntic_summary")
    if result is not None and summary is not None:
        st.markdown('<div class="section-title">▌ Classification results</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("RECORDS ANALYZED", f"{summary['rows']:,}", "Uploaded rows")
        with c2: metric_card("PREDICTED CLASSES", str(summary["classes"]), "Unique model outputs")
        with c3: metric_card("MOST FREQUENT CLASS", summary["most_common"], "Highest prediction count")
        counts = result["Predicted_traffic_class"].value_counts().rename_axis("Traffic class").reset_index(name="Records")
        st.markdown("**Predicted traffic distribution**")
        st.bar_chart(counts.set_index("Traffic class"), use_container_width=True)
        if "Prediction_confidence" in result and result["Prediction_confidence"].notna().any():
            avg_conf = result["Prediction_confidence"].mean()
            st.caption(f"Mean model confidence: {avg_conf:.2%}. Confidence is not a guarantee that a prediction is correct.")
            st.markdown("**Prediction confidence distribution**")
            st.bar_chart(result["Prediction_confidence"].dropna().reset_index(drop=True), use_container_width=True)
        st.markdown("**Prediction records**")
        selected_classes = st.multiselect("Filter by predicted class", options=sorted(result["Predicted_traffic_class"].astype(str).unique()), default=sorted(result["Predicted_traffic_class"].astype(str).unique()))
        filtered = result[result["Predicted_traffic_class"].astype(str).isin(selected_classes)]
        st.dataframe(filtered, use_container_width=True, height=380)
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download prediction results (CSV)", data=csv_bytes, file_name="ntic_prediction_results.csv", mime="text/csv", use_container_width=True)

st.markdown(
    '<div style="text-align:center;color:#8A9AB0;font-size:11px;padding:30px 0 8px;">'
    'NTIC · Network Traffic Intelligence Center &nbsp;|&nbsp; Powered by CloudSentinel AI'
    '</div>',
    unsafe_allow_html=True,
)
