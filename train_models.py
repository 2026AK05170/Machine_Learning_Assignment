from pathlib import Path
import joblib, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
for folder in (DATA, MODELS, RESULTS): folder.mkdir(exist_ok=True)
cancer = load_breast_cancer(as_frame=True)
X = cancer.data.copy()
y = (cancer.target == 0).astype(int)  # Malignant=1, Benign=0
assert X.shape == (569, 30) and y.nunique() == 2
full = X.copy(); full["diagnosis"] = y.map({1: "Malignant", 0: "Benign"})
full.to_csv(DATA / "breast_cancer_wisconsin_diagnostic.csv", index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
held_out = X_test.copy(); held_out["diagnosis"] = y_test.map({1: "Malignant", 0: "Benign"})
held_out.to_csv(ROOT / "test_data.csv", index=False)
models = {
 "Logistic Regression": Pipeline([("scaler", StandardScaler()),("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
 "Decision Tree": Pipeline([("classifier", DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42))]),
 "KNN": Pipeline([("scaler", StandardScaler()),("classifier", KNeighborsClassifier(n_neighbors=5))]),
 "Gaussian Naive Bayes": Pipeline([("scaler", StandardScaler()),("classifier", GaussianNB())]),
 "RBF SVM": Pipeline([("scaler", StandardScaler()),("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
}
names = dict(zip(models, ["logistic_regression.joblib", "decision_tree.joblib", "knn.joblib", "gaussian_naive_bayes.joblib", "rbf_svm.joblib"]))
rows = []
for name, pipeline in models.items():
 pipeline.fit(X_train, y_train); pred = pipeline.predict(X_test)
 prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
 rows.append({"Model": name, "Accuracy": accuracy_score(y_test,pred), "ROC-AUC": roc_auc_score(y_test,prob), "Precision": precision_score(y_test,pred,zero_division=0), "Recall": recall_score(y_test,pred,zero_division=0), "F1": f1_score(y_test,pred,zero_division=0), "MCC": matthews_corrcoef(y_test,pred)})
 joblib.dump(pipeline, MODELS / names[name])
comparison = pd.DataFrame(rows).sort_values("F1", ascending=False)
comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
print(comparison.to_string(index=False))
