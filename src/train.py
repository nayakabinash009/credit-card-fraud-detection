"""
train.py
────────
Model training helpers for the Credit Card Fraud Detection project.
Supports Logistic Regression, Random Forest, and XGBoost classifiers.

Usage
-----
    from src.train import train_all_models
    models = train_all_models(X_train, y_train)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ─── Individual trainers ──────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Train a Logistic Regression baseline model.

    Notes
    -----
    max_iter=1000 is used to ensure convergence on large SMOTE-resampled sets.
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    print("[train] Logistic Regression – done.")
    return model


def train_random_forest(
    X_train, y_train,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
):
    """
    Train a Random Forest ensemble classifier.

    Parameters
    ----------
    n_estimators : number of decision trees (default 100)
    n_jobs       : -1 uses all available CPU cores
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    print(f"[train] Random Forest ({n_estimators} trees) – done.")
    return model


def train_xgboost(
    X_train, y_train,
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Train an XGBoost gradient-boosting classifier.

    Parameters
    ----------
    n_estimators : number of boosting rounds (default 100)
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("[train] XGBoost – done.")
    return model


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """
    Train all three classifiers and return them in a labelled dict.

    Returns
    -------
    {
        'Logistic Regression': <fitted model>,
        'Random Forest'      : <fitted model>,
        'XGBoost'            : <fitted model>,
    }
    """
    print("\n──── Training Models ─────────────────────────────────────────")
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train, random_state),
        'Random Forest'      : train_random_forest(X_train, y_train, random_state=random_state),
        'XGBoost'            : train_xgboost(X_train, y_train, random_state=random_state),
    }
    print("─────────────────────────────────────────────────────────────\n")
    return models
