"""
evaluate.py
───────────
Evaluation helpers for the Credit Card Fraud Detection project.
Generates classification reports, confusion matrices, ROC curves,
feature importance charts, and a summary comparison table.

Usage
-----
    from src.evaluate import evaluate_all_models, plot_roc_curves
    results = evaluate_all_models(models, X_test, y_test)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Single-model helpers ────────────────────────────────────────────────────

def print_report(name: str, y_test, y_pred) -> None:
    """Print a full sklearn classification report."""
    print(f"\n{'═' * 50}")
    print(f"  {name}")
    print('═' * 50)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))


def plot_confusion_matrix(name: str, y_test, y_pred, ax=None) -> None:
    """Plot a single confusion matrix on *ax* (or create its own figure)."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud'])
    if ax is None:
        fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name)


# ─── Multi-model helpers ─────────────────────────────────────────────────────

def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every model in *models* and return a summary DataFrame.

    Parameters
    ----------
    models  : dict  { model_name: fitted_estimator }
    X_test  : feature matrix (unseen test set)
    y_test  : true labels

    Returns
    -------
    results_df : DataFrame with Accuracy, Precision, Recall, F1-Score, AUC-ROC
    """
    summary = []

    # ── Per-model reports ────────────────────────────────────────────────────
    preds_dict = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        preds_dict[name] = y_pred
        print_report(name, y_test, y_pred)
        summary.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall'   : round(recall_score(y_test, y_pred), 4),
            'F1-Score' : round(f1_score(y_test, y_pred, zero_division=0), 4),
            'AUC-ROC'  : round(roc_auc_score(y_test, y_pred), 4),
        })

    results_df = pd.DataFrame(summary)

    # ── Confusion matrices side-by-side ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        plot_confusion_matrix(name, y_test, y_pred, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=150)
    plt.show()

    # ── ROC curves ──────────────────────────────────────────────────────────
    plot_roc_curves(models, X_test, y_test)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n──── Model Comparison Summary ────────────────────────────────")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        os.path.join(os.path.dirname(__file__), '..', 'results', 'model_comparison.csv'),
        index=False
    )

    return results_df


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    """Overlay ROC curves for all models on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title('ROC Curves – All Models')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names, top_n: int = 15) -> None:
    """
    Bar chart of the top-N feature importances for tree-based models.
    Silently skips models that do not expose `feature_importances_`.
    """
    if not hasattr(model, 'feature_importances_'):
        print("[plot_feature_importance] Model has no feature_importances_ attribute – skipped.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top.plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150)
    plt.show()
