# Breast Cancer Classification — Machine Learning Assignment 2

## Problem statement
Build and compare five supervised classification algorithms for predicting whether a breast mass is benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset.

## Dataset
- Source: `sklearn.datasets.load_breast_cancer`
- 569 instances and 30 numeric features
- Target: malignant (0) and benign (1)
- Stratified 80/20 train-test split with random state 42

## Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation metrics
Accuracy, ROC-AUC, precision, recall, F1-score and Matthews correlation coefficient (MCC).

## Run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload a CSV containing the 30 feature columns and a `target` column to evaluate the saved models.

## Deployment
Deploy this repository using Streamlit Community Cloud and set the main file path to `app.py`.

## Repository and live app
- GitHub: https://github.com/vikky-max/ml-assignment-2-aiml
- Streamlit app: Add the deployed app URL here.
