
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import METADATA_PATH, MODEL_PATH, MAX_UPLOAD_MB
from src.predict import predict_frame


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CloudSentinel AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CloudSentinel AI")
st.caption(
    "Network traffic classification, machine learning "
    "and security analytics prototype"
)

st.warning(
    "This tool analyzes uploaded dataset records. A prediction is not "
    "proof of an attack. It does not monitor live networks or block traffic."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Project status")

    st.write(
        "Model:",
        "Ready" if MODEL_PATH.exists() else "Not trained",
    )

    st.write(
        "Evaluation report:",
        "Available" if METADATA_PATH.exists() else "Not available",
    )

    st.divider()
    st.subheader("Workflow")
    st.write("1. Train the ML model")
    st.write("2. Upload a compatible CSV")
    st.write("3. Analyze predictions")
    st.write("4. Review charts and export results")


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
if METADATA_PATH.exists():
    try:
        meta = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        st.header("Model evaluation")

        accuracy = meta.get("accuracy", 0)
        test_rows = meta.get("test_rows", 0)
        classes = meta.get("classes", [])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Held-out accuracy",
            f"{accuracy:.2%}",
        )
        col2.metric(
            "Test records",
            f"{test_rows:,}",
        )
        col3.metric(
            "Attack classes",
            len(classes),
        )

        st.caption(
            "These metrics come from the saved random hold-out evaluation. "
            "They do not guarantee performance on new real-world traffic."
        )

        report = meta.get("classification_report", {})
        report_rows = []

        for label, values in report.items():
            if (
                isinstance(values, dict)
                and "precision" in values
            ):
                report_rows.append(
                    {
                        "Class": label,
                        "Precision": values["precision"],
                        "Recall": values["recall"],
                        "F1-score": values["f1-score"],
                        "Support": int(values["support"]),
                    }
                )

        if report_rows:
            st.subheader("Class-wise performance")

            report_df = pd.DataFrame(report_rows)
            st.dataframe(
                report_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Precision": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Recall": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "F1-score": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
            )

        cm = meta.get("confusion_matrix")
        labels = meta.get("confusion_matrix_labels")

        if cm and labels:
            st.subheader("Confusion matrix")

            cm_df = pd.DataFrame(
                cm,
                index=labels,
                columns=labels,
            )

            fig_cm = px.imshow(
                cm_df,
                x=labels,
                y=labels,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={
                    "x": "Predicted class",
                    "y": "Actual class",
                    "color": "Records",
                },
                title="Actual vs predicted classes",
            )

            fig_cm.update_layout(
                height=650,
                xaxis_tickangle=-45,
            )

            st.plotly_chart(
                fig_cm,
                width="stretch",
            )

    except Exception as exc:
        st.error(f"Could not load evaluation report: {exc}")

else:
    st.info(
        "No evaluation report found. Train the model using README.md instructions."
    )


# --------------------------------------------------
# CSV PREDICTION
# --------------------------------------------------
st.divider()
st.header("Analyze network traffic")
st.write(
    "Upload a CSV containing the same input features used during model training."
)

uploaded = st.file_uploader(
    "Upload network traffic CSV",
    type=["csv"],
)

if uploaded:
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )
        st.stop()

    try:
        raw = pd.read_csv(uploaded, low_memory=False)

        st.subheader("Uploaded data preview")

        st.dataframe(
            raw.head(10),
            width="stretch",
        )

        st.caption(
            f"{len(raw):,} records · {len(raw.columns)} columns"
        )

        if st.button(
            "Run predictions",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Analyzing network traffic..."):
                output = predict_frame(raw)

            if output.empty:
                st.warning("The prediction output contains no records.")
                st.stop()

            if "predicted_class" not in output.columns:
                st.error(
                    "The prediction output does not contain "
                    "the expected 'predicted_class' column."
                )
                st.stop()

            # ------------------------------------------
            # PREDICTION SUMMARY
            # ------------------------------------------
            st.success("Prediction complete.")

            predictions = output["predicted_class"].astype(str)
            counts = (
                predictions.value_counts()
                .rename_axis("Attack class")
                .reset_index(name="Records")
            )

            counts["Percentage"] = (
                counts["Records"] / len(output) * 100
            )

            total_records = len(output)
            unique_classes = predictions.nunique()
            top_class = counts.iloc[0]["Attack class"]
            top_count = int(counts.iloc[0]["Records"])
            top_percentage = top_count / total_records * 100

            st.subheader("Prediction summary")

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Records analyzed",
                f"{total_records:,}",
            )
            k2.metric(
                "Predicted classes",
                unique_classes,
            )
            k3.metric(
                "Most frequent prediction",
                str(top_class),
                f"{top_percentage:.1f}% of records",
            )

            if unique_classes == 1:
                st.warning(
                    "All records received the same predicted class. "
                    "This is not necessarily an error, but verify the "
                    "input data and predictions before drawing conclusions."
                )

            # ------------------------------------------
            # CLASS DISTRIBUTION
            # ------------------------------------------
            st.subheader("Predicted class distribution")

            fig_dist = px.bar(
                counts.sort_values("Records"),
                x="Records",
                y="Attack class",
                orientation="h",
                text="Records",
                hover_data={"Percentage": ":.2f"},
                title="Number of records per predicted class",
                color="Records",
                color_continuous_scale="Blues",
            )

            fig_dist.update_layout(
                height=max(400, 45 * len(counts)),
                yaxis_title="Predicted class",
                xaxis_title="Number of records",
                coloraxis_showscale=False,
            )

            st.plotly_chart(
                fig_dist,
                width="stretch",
            )

            st.dataframe(
                counts,
                width="stretch",
                hide_index=True,
                column_config={
                    "Percentage": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )

            # ------------------------------------------
            # CONFIDENCE DISTRIBUTION
            # ------------------------------------------
            if "prediction_confidence" in output.columns:
                st.subheader("Prediction confidence")

                confidence = pd.to_numeric(
                    output["prediction_confidence"],
                    errors="coerce",
                ).dropna()

                if not confidence.empty:
                    st.metric(
                        "Mean confidence",
                        f"{confidence.mean():.2%}",
                    )

                    fig_conf = px.histogram(
                        confidence,
                        nbins=20,
                        title="Distribution of model confidence scores",
                        labels={
                            "value": "Confidence score",
                            "count": "Records",
                        },
                    )

                    st.plotly_chart(
                        fig_conf,
                        width="stretch",
                    )

                    st.caption(
                        "Model confidence is not the probability that a "
                        "record represents a real attack."
                    )

            # ------------------------------------------
            # PREDICTION RESULTS
            # ------------------------------------------
            st.subheader("Prediction results")

            selected_class = st.selectbox(
                "Filter by predicted class",
                ["All classes"] + counts["Attack class"].tolist(),
            )

            if selected_class == "All classes":
                filtered = output
            else:
                filtered = output[
                    output["predicted_class"].astype(str)
                    == selected_class
                ]

            st.caption(
                f"Showing {len(filtered):,} of {len(output):,} records"
            )

            st.dataframe(
                filtered.head(1000),
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------
            # CSV EXPORT
            # ------------------------------------------
            st.download_button(
                "Download all predictions as CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="cloudsentinel_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

            st.info(
                "Review suspicious predictions with appropriate security "
                "tools. This prototype does not confirm incidents or "
                "automatically take protective action."
            )

    except Exception as exc:
        st.error(f"Could not analyze this CSV: {exc}")
        st.info(
            "Check that the model is trained and the CSV has the expected "
            "feature names and compatible data."
        )

else:
    st.info("Upload a CSV file to begin analysis.")
