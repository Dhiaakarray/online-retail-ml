
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/processed/customer_features.parquet")
MODEL_DIR = Path("models")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary_total", "avg_order_value",
    "distinct_products", "tenure_days", "num_returns", "return_rate",
]
LOG_FEATURES = ["monetary_total", "frequency"]
CATEGORICAL_FEATURES = ["country_group"]
TARGET = "churned"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)

    df["country_group"] = np.where(df["country"] == "United Kingdom", "UK", "Other")
    return df


def build_preprocessor() -> ColumnTransformer:

    log_then_scale = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])
    plain_scale = Pipeline([("scale", StandardScaler())])

    plain_numeric = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]

    return ColumnTransformer([
        ("log_scale", log_then_scale, LOG_FEATURES),
        ("scale", plain_scale, plain_numeric),
        ("country", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        print(f"{name:20s} | acc={metrics['accuracy']:.3f} "
              f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")


    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"]
    print(f"\nBest model by F1: {best_name}")

    final_pipe = fitted[best_name]
    if best_name == "random_forest":
        param_grid = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [8, 10, 15, None],
        }
    elif best_name == "xgboost":
        param_grid = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
        }
    elif best_name == "decision_tree":
        param_grid = {"model__max_depth": [4, 6, 8, 10]}
    else:
        param_grid = {"model__C": [0.01, 0.1, 1, 10]}

    search = GridSearchCV(final_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned_pipe = search.best_estimator_
    tuned_metrics = evaluate(f"{best_name}_tuned", tuned_pipe, X_test, y_test)
    print(f"\nAfter tuning: {tuned_metrics}")
    print(f"Best params: {search.best_params_}")


    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipe, MODEL_DIR / "churn_model.joblib")

    model_info = {
        "model_type": best_name,
        "best_params": search.best_params_,
        "metrics": tuned_metrics,
        "all_candidate_results": results,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    print(f"\nSaved final model to {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Saved model info to {MODEL_DIR / 'model_info.json'}")


if __name__ == "__main__":
    main()
