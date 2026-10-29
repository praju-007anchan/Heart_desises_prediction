"""Heart Disease Model Training, Tuning, and Evaluation Pipeline.

This script trains multiple machine learning classifiers on the UCI Heart Disease
dataset, performs hyperparameter tuning using RandomizedSearchCV and GridSearchCV,
computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, 
5-Fold Cross Validation), and exports a unified models bundle for the Streamlit app.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

DATA_PATH = "heart-disease.csv"
BUNDLE_PATH = "models_bundle.pkl"
LEGACY_MODEL_PATH = "heart_disease_model.pkl"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def load_data(filepath: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and validate dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    X = df[FEATURE_NAMES]
    y = df["target"]
    return X, y

def train_and_tune_all():
    """Execute complete model training, tuning, and evaluation suite."""
    print("=" * 70)
    print("CardioCare ML Engine: Model Training & Evaluation Suite")
    print("=" * 70)

    X, y = load_data()
    print(f"[+] Loaded dataset: {X.shape[0]} patient records, {X.shape[1]} clinical features")
    print(f"[+] Class distribution: Class 0 (No Disease)={sum(y==0)}, Class 1 (Disease)={sum(y==1)}")

    # 80/20 Stratified Split for fair generalization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Define Baseline Candidates
    baseline_models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
        "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))]),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    baseline_metrics = {}
    print("\n--- Phase 1: Baseline Models Evaluation ---")
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        baseline_metrics[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "CV_Accuracy_Mean": round(float(np.mean(cv_scores)), 4),
            "CV_Accuracy_Std": round(float(np.std(cv_scores)), 4)
        }
        print(f"[*] {name:24s} | Acc: {acc:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | CV: {np.mean(cv_scores):.4f}")

    # 2. Hyperparameter Tuning
    print("\n--- Phase 2: Systematic Hyperparameter Tuning ---")
    tuned_models = {}
    tuning_logs = {}

    # 2.1 Logistic Regression Tuning (GridSearchCV)
    print("[*] Tuning Logistic Regression via GridSearchCV...")
    lr_grid = {
        "C": [0.01, 0.05, 0.1, 0.2, 0.23357, 0.5, 1.0, 5.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    lr_gs = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), lr_grid, cv=cv, scoring="f1", n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    best_lr = lr_gs.best_estimator_
    tuned_models["Tuned Logistic Regression"] = best_lr
    tuning_logs["Logistic Regression"] = {
        "method": "GridSearchCV (Stratified 5-Fold)",
        "param_grid": lr_grid,
        "best_params": lr_gs.best_params_,
        "best_cv_score": round(float(lr_gs.best_score_), 4)
    }
    print(f"    Best Params: {lr_gs.best_params_} (F1 Score: {lr_gs.best_score_:.4f})")

    # 2.2 Random Forest Tuning (RandomizedSearchCV)
    print("[*] Tuning Random Forest via RandomizedSearchCV...")
    rf_param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 3, 5, 8],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }
    rf_rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_dist, n_iter=20, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
    rf_rs.fit(X_train, y_train)
    best_rf = rf_rs.best_estimator_
    tuned_models["Tuned Random Forest"] = best_rf
    tuning_logs["Random Forest"] = {
        "method": "RandomizedSearchCV (20 iterations, 5-Fold)",
        "param_grid": rf_param_dist,
        "best_params": rf_rs.best_params_,
        "best_cv_score": round(float(rf_rs.best_score_), 4)
    }
    print(f"    Best Params: {rf_rs.best_params_} (F1 Score: {rf_rs.best_score_:.4f})")

    # 2.3 KNN Tuning (GridSearchCV for k and metric)
    print("[*] Tuning K-Nearest Neighbors...")
    knn_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    knn_grid = {
        "knn__n_neighbors": list(range(3, 21, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }
    knn_gs = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring="f1", n_jobs=-1)
    knn_gs.fit(X_train, y_train)
    best_knn = knn_gs.best_estimator_
    tuned_models["Tuned K-Nearest Neighbors"] = best_knn
    tuning_logs["K-Nearest Neighbors"] = {
        "method": "GridSearchCV with StandardScaler",
        "param_grid": knn_grid,
        "best_params": knn_gs.best_params_,
        "best_cv_score": round(float(knn_gs.best_score_), 4)
    }
    print(f"    Best Params: {knn_gs.best_params_} (F1 Score: {knn_gs.best_score_:.4f})")

    # 2.4 Support Vector Machine
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42, C=1.0, kernel="rbf"))])
    svm_pipe.fit(X_train, y_train)
    tuned_models["Support Vector Machine"] = svm_pipe

    # 2.5 Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    tuned_models["Gradient Boosting"] = gb_model

    # 3. Comprehensive Final Evaluation on Test Set & Cross Validation
    print("\n--- Phase 3: Comprehensive Final Model Benchmarking ---")
    final_metrics = {}
    detailed_eval = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        final_metrics[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "CV_Accuracy_Mean": round(float(np.mean(cv_scores)), 4),
            "CV_Accuracy_Std": round(float(np.std(cv_scores)), 4)
        }

        detailed_eval[name] = {
            "confusion_matrix": cm,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        }
        print(f"[#] {name:28s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # Feature Importance (from Random Forest & Logistic Regression)
    feature_importance = {
        "Random_Forest": {feat: float(imp) for feat, imp in zip(FEATURE_NAMES, best_rf.feature_importances_)},
        "Logistic_Regression_Coeffs": {feat: float(c) for feat, c in zip(FEATURE_NAMES, best_lr.coef_[0])}
    }

    # Package Bundle
    bundle = {
        "models": tuned_models,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "detailed_eval": detailed_eval,
        "tuning_logs": tuning_logs,
        "feature_importance": feature_importance,
        "feature_names": FEATURE_NAMES,
        "best_overall_model_name": "Tuned Logistic Regression" if final_metrics["Tuned Logistic Regression"]["F1-Score"] >= final_metrics["Tuned Random Forest"]["F1-Score"] else "Tuned Random Forest",
        "dataset_summary": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(FEATURE_NAMES),
            "positive_cases": int(sum(y == 1)),
            "negative_cases": int(sum(y == 0))
        }
    }

    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[+] Successfully saved unified models bundle to: {BUNDLE_PATH}")

    # Also update heart_disease_model.pkl with the best Logistic Regression model to preserve backward compatibility
    with open(LEGACY_MODEL_PATH, "wb") as f:
        pickle.dump(best_lr, f)
    print(f"[+] Successfully refreshed legacy single model: {LEGACY_MODEL_PATH}")

    return bundle

if __name__ == "__main__":
    train_and_tune_all()
