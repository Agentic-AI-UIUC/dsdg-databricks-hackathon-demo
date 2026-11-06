import numpy as np
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(num_cols, cat_cols, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)),
    ]
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols), ("cat", Pipeline(cat_steps), cat_cols)])


def candidate_models(task, seed):
    """name -> (estimator, needs_scaling)"""
    if task == "classification":
        return {
            "logistic_regression": (LogisticRegression(max_iter=1000), True),
            "random_forest": (RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed), False),
            "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=200, random_state=seed), False),
        }
    return {
        "ridge_regression": (Ridge(alpha=1.0), True),
        "random_forest": (RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed), False),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=200, random_state=seed), False),
    }


def higher_is_better(metric):
    return metric not in ("rmse", "mae")


def score(model, X, y, metric, positive_label=None):
    if metric == "roc_auc":
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return metrics.roc_auc_score(y == positive_label, proba[:, list(model.classes_).index(positive_label)])
        return metrics.roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_)
    pred = model.predict(X)
    return {
        "accuracy": lambda: metrics.accuracy_score(y, pred),
        "f1": lambda: metrics.f1_score(y, pred, average="macro"),
        "rmse": lambda: float(np.sqrt(metrics.mean_squared_error(y, pred))),
        "mae": lambda: metrics.mean_absolute_error(y, pred),
        "r2": lambda: metrics.r2_score(y, pred),
    }[metric]()
