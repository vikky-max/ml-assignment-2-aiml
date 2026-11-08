"""Train five classifiers on the Wisconsin Diagnostic Breast Cancer dataset."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
    "K-Nearest Neighbors": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
}
rows = []
filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / filenames[name])
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    })
pd.DataFrame(rows).to_csv(ROOT / "model_metrics.csv", index=False)
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv(ROOT / "test_data.csv", index=False)
print(pd.DataFrame(rows).set_index("Model").round(4))
