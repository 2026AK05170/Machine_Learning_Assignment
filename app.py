from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd, seaborn as sns, streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
ROOT = Path(__file__).resolve().parent
OPTIONS = {"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","KNN":"knn.joblib","Gaussian Naive Bayes":"gaussian_naive_bayes.joblib","RBF SVM":"rbf_svm.joblib"}
st.set_page_config(page_title="Breast Cancer Model Explorer", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Classification Model Explorer")
st.caption("Malignant = 1, Benign = 0. The app evaluates saved models and never retrains them.")
name = st.sidebar.selectbox("Choose a model", list(OPTIONS))
uploaded = st.file_uploader("Upload test_data.csv", type="csv")
if uploaded is None: st.info("Upload the test_data.csv created by the notebook."); st.stop()
df = pd.read_csv(uploaded); st.dataframe(df.head(), use_container_width=True)
if "diagnosis" not in df: st.error("CSV must have a diagnosis column."); st.stop()
model_path = ROOT / "models" / OPTIONS[name]
if not model_path.exists(): st.error("Model file not found. Run the notebook or train_models.py first."); st.stop()
y = df["diagnosis"].astype(str).str.strip().str.lower().map({"malignant":1,"benign":0,"m":1,"b":0,"1":1,"0":0})
if y.isna().any(): st.error("Diagnosis must be Malignant/Benign, M/B, or 1/0."); st.stop()
model = joblib.load(model_path); X = df.drop(columns="diagnosis")
expected = list(getattr(model, "feature_names_in_", X.columns))
if set(expected) != set(X.columns): st.error("Feature columns do not match the saved model."); st.stop()
pred = model.predict(X[expected]); prob = model.predict_proba(X[expected])[:, list(model.classes_).index(1)]
metrics = {"Accuracy":accuracy_score(y,pred),"ROC-AUC":roc_auc_score(y,prob),"Precision":precision_score(y,pred,zero_division=0),"Recall":recall_score(y,pred,zero_division=0),"F1":f1_score(y,pred,zero_division=0),"MCC":matthews_corrcoef(y,pred)}
st.subheader("Results: " + name)
for box, (label, value) in zip(st.columns(3)*2, metrics.items()): box.metric(label, f"{value:.4f}")
left, right = st.columns(2)
with left:
 fig, ax = plt.subplots(); sns.heatmap(confusion_matrix(y,pred,labels=[0,1]),annot=True,fmt="d",cmap="Blues",cbar=False,xticklabels=["Benign","Malignant"],yticklabels=["Benign","Malignant"],ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)
with right:
 fpr,tpr,_ = roc_curve(y,prob); fig,ax = plt.subplots(); ax.plot(fpr,tpr,label=f"AUC={metrics['ROC-AUC']:.3f}"); ax.plot([0,1],[0,1],"k--"); ax.legend(); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); st.pyplot(fig)
st.subheader("Classification report"); st.code(classification_report(y,pred,target_names=["Benign (0)","Malignant (1)"],zero_division=0))
comparison = ROOT / "results" / "model_comparison.csv"
if comparison.exists(): st.subheader("Five-model comparison"); st.dataframe(pd.read_csv(comparison).round(4),use_container_width=True)
