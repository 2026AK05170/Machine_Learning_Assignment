# Breast Cancer Classification — ML Assignment 2

## a. Problem statement
This project compares five classifiers for predicting whether a breast-cancer sample is malignant or benign. The Streamlit app evaluates uploaded test data using saved pipelines; it does not retrain a model.

## b. Dataset description
The Breast Cancer Wisconsin (Diagnostic) dataset is bundled with scikit-learn and attributed to the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic. It has 569 instances, 30 numerical inputs, 212 malignant and 357 benign cases, with no missing data. This project uses Malignant = 1 and Benign = 0, with a stratified 80/20 split and random_state=42.

## c. GitHub repository link
Replace after publishing: [https://github.com/y/your-repository-name](https://github.com/2026AK05170/Machine_Learning_Assignment)

## d. Models used and comparison
Models: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and RBF SVM. Preprocessing is inside relevant pipelines to avoid test-data leakage.

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| RBF SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Decision Tree | 0.9211 | 0.9676 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |

All values are from the held-out test set and saved in results/model_comparison.csv.

## e. Streamlit App
Link: https://breastcancerclassifications.streamlit.app

## Observations
- RBF SVM: Accuracy=0.974, ROC-AUC=0.995, Precision=1.000, Recall=0.929, F1=0.963, MCC=0.944. Results are from the same held-out test set.
- Logistic Regression: Accuracy=0.965, ROC-AUC=0.996, Precision=0.975, Recall=0.929, F1=0.951, MCC=0.925. Results are from the same held-out test set.
- KNN: Accuracy=0.956, ROC-AUC=0.982, Precision=0.974, Recall=0.905, F1=0.938, MCC=0.906. Results are from the same held-out test set.
- Gaussian Naive Bayes: Accuracy=0.921, ROC-AUC=0.989, Precision=0.923, Recall=0.857, F1=0.889, MCC=0.829. Results are from the same held-out test set.
- Decision Tree: Accuracy=0.921, ROC-AUC=0.968, Precision=0.946, Recall=0.833, F1=0.886, MCC=0.830. Results are from the same held-out test set.
- Overall winner by malignant-class F1: RBF SVM (0.963). This result is for the fixed held-out split; a larger study should use cross-validation.
- Recall measures malignant cases detected. MCC uses all four confusion-matrix cells and complements accuracy.

## Run locally
1. Install packages with pip install -r requirements.txt.
2. Run python train_models.py.
3. Run streamlit run app.py.
4. Upload test_data.csv, choose a model, and inspect six metrics, a confusion matrix, ROC curve, and classification report.

## Repository contents
Notebook, train_models.py, app.py, requirements.txt, README.md, test_data.csv, data/breast_cancer_wisconsin_diagnostic.csv, five model files, and results files are included.
