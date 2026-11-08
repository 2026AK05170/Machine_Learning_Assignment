from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_OPTIONS = {
    "Logistic Regression": {
        "file": "logistic_regression.joblib",
        "short": "Linear baseline",
        "note": "A transparent linear model after feature standardisation.",
    },
    "Decision Tree": {
        "file": "decision_tree.joblib",
        "short": "Rule-based",
        "note": "Learns a set of interpretable if-then decision rules.",
    },
    "KNN": {
        "file": "knn.joblib",
        "short": "Neighbour voting",
        "note": "Classifies a sample using nearby measurements in scaled feature space.",
    },
    "Gaussian Naive Bayes": {
        "file": "gaussian_naive_bayes.joblib",
        "short": "Probability model",
        "note": "Estimates class probabilities under a Gaussian feature assumption.",
    },
    "RBF SVM": {
        "file": "rbf_svm.joblib",
        "short": "Non-linear boundary",
        "note": "Uses a radial basis function kernel to model a non-linear boundary.",
    },
}

FEATURE_HELP = {
    "mean radius": "Average distance from centre to the boundary points.",
    "mean texture": "Variation in grayscale values of the image.",
    "mean perimeter": "Average tumour boundary length.",
    "mean area": "Average area enclosed by the tumour boundary.",
    "mean smoothness": "Local variation in radius lengths.",
}


st.set_page_config(
    page_title="OncoScope | Model Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
              radial-gradient(circle at 92% 7%, #d7f6ed 0, transparent 21rem),
              radial-gradient(circle at 4% 12%, #e5eeff 0, transparent 18rem),
              #f8fbfa;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102c3b 0%, #173e4d 100%);
        }
        [data-testid="stSidebar"] * { color: #f4fbfb !important; }
        .hero {
            background: linear-gradient(118deg, #103b4a 0%, #18726c 54%, #4fa78f 100%);
            padding: 2.1rem 2.3rem; border-radius: 22px; color: white;
            box-shadow: 0 12px 30px rgba(18, 78, 83, .20); margin-bottom: 1.25rem;
        }
        .hero h1 { margin: 0; font-size: 2.15rem; letter-spacing: -.04em; }
        .hero p { margin: .45rem 0 0; color: #d9f3ed; font-size: 1.02rem; }
        .eyebrow { color: #9be2d1; font-size: .76rem; font-weight: 700; letter-spacing: .12em; }
        .model-pill {
            display: inline-block; padding: .32rem .68rem; border-radius: 99px;
            background: #dff5ef; color: #0e6258; font-weight: 650; font-size: .82rem;
        }
        .section-note {
            border-left: 4px solid #46a993; background: #edf8f5; padding: .8rem 1rem;
            border-radius: 0 10px 10px 0; color: #28574f; margin: .5rem 0 1rem;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.83); border: 1px solid #e2ece9;
            border-radius: 14px; padding: .8rem; box-shadow: 0 3px 10px rgba(27,75,69,.05);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 9px; border: 0; font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def read_csv(file_or_path):
    return pd.read_csv(file_or_path)


def encode_diagnosis(series: pd.Series) -> pd.Series:
    """Convert accepted diagnosis labels to the project convention."""
    if pd.api.types.is_numeric_dtype(series):
        encoded = pd.to_numeric(series, errors="coerce")
    else:
        encoded = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"malignant": 1, "benign": 0, "m": 1, "b": 0, "1": 1, "0": 0})
        )
    if encoded.isna().any() or not set(encoded.unique()).issubset({0, 1}):
        raise ValueError("Diagnosis values must be Malignant/Benign, M/B, or 1/0.")
    return encoded.astype(int)


def metric_cards(y_true, y_pred, probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, probability) if y_true.nunique() == 2 else None,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


with st.sidebar:
    st.markdown("## ◉ OncoScope")
    st.caption("Breast Cancer Model Lab")
    st.divider()
    selected_name = st.selectbox("Select a saved classifier", list(MODEL_OPTIONS))
    st.markdown(f"<span class='model-pill'>{MODEL_OPTIONS[selected_name]['short']}</span>", unsafe_allow_html=True)
    st.caption(MODEL_OPTIONS[selected_name]["note"])
    st.divider()
    decision_threshold = st.slider(
        "Malignant probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Predictions at or above this probability are labelled malignant. This changes only the decision rule, not the trained model.",
    )
    st.caption("Target encoding: Malignant = 1 · Benign = 0")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">INTERACTIVE EVALUATION DASHBOARD</div>
      <h1>OncoScope Model Lab</h1>
      <p>Compare saved classifiers on held-out diagnostic data without retraining.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

evaluate_tab, explore_tab, compare_tab = st.tabs(
    ["Evaluate a model", "Data guide", "Five-model comparison"]
)

with evaluate_tab:
    st.markdown(
        "<div class='section-note'>Upload the generated <b>test_data.csv</b>. "
        "It must contain all 30 feature columns and the <b>diagnosis</b> target column.</div>",
        unsafe_allow_html=True,
    )
    local_test_path = ROOT / "test_data.csv"
    source_mode = st.radio(
        "Data source",
        ["Upload a CSV", "Use included test_data.csv"],
        horizontal=True,
        disabled=not local_test_path.exists(),
    )
    uploaded_file = None
    if source_mode == "Upload a CSV":
        uploaded_file = st.file_uploader("Choose test CSV", type=["csv"])

    test_df = None
    if source_mode == "Use included test_data.csv" and local_test_path.exists():
        test_df = read_csv(local_test_path)
    elif uploaded_file is not None:
        test_df = read_csv(uploaded_file)

    if test_df is None:
        st.info("Choose a source and provide a test CSV to start the evaluation.")
    else:
        model_path = MODEL_DIR / MODEL_OPTIONS[selected_name]["file"]
        if not model_path.exists():
            st.error(
                f"The required file {model_path.name} is missing. "
                "Run the notebook or train_models.py before deploying the app."
            )
            st.stop()
        if "diagnosis" not in test_df.columns:
            st.error("The uploaded CSV has no diagnosis column.")
            st.stop()

        model = load_model(str(model_path))
        X_test = test_df.drop(columns=["diagnosis"])
        expected_features = list(getattr(model, "feature_names_in_", X_test.columns))
        missing = sorted(set(expected_features) - set(X_test.columns))
        extra = sorted(set(X_test.columns) - set(expected_features))
        if missing or extra:
            st.error(
                "The CSV columns do not match the saved model. "
                f"Missing: {missing or 'none'} | Extra: {extra or 'none'}"
            )
            st.stop()

        try:
            y_true = encode_diagnosis(test_df["diagnosis"])
        except ValueError as error:
            st.error(str(error))
            st.stop()

        X_test = X_test[expected_features]
        probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        y_pred = (probability >= decision_threshold).astype(int)
        metrics = metric_cards(y_true, y_pred, probability)

        preview_col, health_col = st.columns([2.2, 1])
        with preview_col:
            st.markdown("#### Data preview")
            st.dataframe(test_df.head(8), use_container_width=True, height=255)
        with health_col:
            st.markdown("#### Evaluation data")
            st.metric("Rows", len(test_df))
            st.metric("Features checked", len(expected_features))
            malignant_share = (y_true.mean() * 100).round(1)
            st.metric("Malignant cases", f"{malignant_share}%")

        st.markdown("#### Performance at selected threshold")
        metric_columns = st.columns(6)
        for column, (label, value) in zip(metric_columns, metrics.items()):
            column.metric(label, "—" if value is None else f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5.3, 4.3))
            sns.heatmap(
                matrix, annot=True, fmt="d", cmap="GnBu", cbar=False, square=True,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
                linewidths=.5, linecolor="white", ax=ax,
            )
            ax.set_xlabel("Predicted diagnosis")
            ax.set_ylabel("Actual diagnosis")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("#### ROC curve")
            fig, ax = plt.subplots(figsize=(5.3, 4.3))
            if y_true.nunique() == 2:
                fpr, tpr, _ = roc_curve(y_true, probability, pos_label=1)
                ax.plot(fpr, tpr, color="#177c71", linewidth=2.8, label=f"AUC = {metrics['ROC-AUC']:.3f}")
                ax.plot([0, 1], [0, 1], "--", color="#96aaa6", label="Random baseline")
                ax.legend(loc="lower right", frameon=False)
            else:
                ax.text(.5, .5, "ROC curve needs both classes.", ha="center", va="center")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Probability quality (threshold independent)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        report_col, download_col = st.columns([1.35, 1])
        with report_col:
            st.markdown("#### Classification report")
            st.code(
                classification_report(
                    y_true, y_pred,
                    target_names=["Benign (0)", "Malignant (1)"],
                    zero_division=0,
                )
            )
        with download_col:
            st.markdown("#### Download predictions")
            prediction_export = test_df.copy()
            prediction_export["malignant_probability"] = probability.round(6)
            prediction_export["predicted_diagnosis"] = pd.Series(y_pred).map({0: "Benign", 1: "Malignant"})
            st.download_button(
                "Download evaluated CSV",
                data=prediction_export.to_csv(index=False).encode("utf-8"),
                file_name=f"evaluated_{selected_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
            st.caption("Contains the original rows, malignant probability, and selected-threshold prediction.")

with explore_tab:
    st.subheader("How to read this project")
    st.markdown(
        """
        - **Dataset:** 569 diagnostic samples with 30 numerical measurements.
        - **Positive class:** malignant is encoded as 1, so recall is the fraction of malignant cases detected.
        - **ROC-AUC:** evaluates the ranking quality of probabilities across thresholds.
        - **MCC:** uses all four confusion-matrix outcomes and complements accuracy when classes are not perfectly balanced.
        - **Threshold slider:** lets you explore the precision/recall trade-off without fitting the model again.
        """
    )
    st.markdown("#### Example feature meanings")
    feature_guide = pd.DataFrame(
        [{"Feature": name, "Meaning": description} for name, description in FEATURE_HELP.items()]
    )
    st.dataframe(feature_guide, use_container_width=True, hide_index=True)

with compare_tab:
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = read_csv(comparison_path)
        st.subheader("Original five-model comparison")
        st.caption("These are the results generated with the default 0.50 decision threshold.")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

        selected_metric = st.selectbox(
            "Rank models by",
            ["F1", "Recall", "ROC-AUC", "MCC", "Accuracy", "Precision"],
            key="ranking_metric",
        )
        ranking = comparison.sort_values(selected_metric, ascending=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.1))
        sns.barplot(data=ranking, x=selected_metric, y="Model", palette="crest", ax=ax)
        ax.set_title(f"Model ranking by {selected_metric}")
        ax.set_xlim(0, max(1.0, ranking[selected_metric].max() + .05))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Run the notebook first to create results/model_comparison.csv.")
