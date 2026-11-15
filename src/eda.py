"""
eda.py
──────
Exploratory Data Analysis (EDA) for the Credit Card Fraud Detection project.
Generates and saves all diagnostic plots described in Chapter 3 of the report.

Usage
-----
    from src.eda import run_eda
    run_eda(df)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Bar chart showing the severe class imbalance between
    legitimate (0) and fraudulent (1) transactions.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Class'].value_counts()
    colors = ['steelblue', 'tomato']
    counts.plot(kind='bar', color=colors, ax=ax, edgecolor='white')
    ax.set_title('Class Distribution (0 = Legit, 1 = Fraud)', fontsize=13)
    ax.set_xlabel('Class')
    ax.set_ylabel('Transaction Count')
    ax.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=150)
    plt.show()
    print("[eda] class_distribution.png saved.")


def plot_amount_distribution(df: pd.DataFrame, xlim: int = 1000) -> None:
    """
    Side-by-side histograms of transaction amounts for each class.
    Helps identify whether fraud concentrates at specific price points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df[df['Class'] == 0], x='Amount',
                 bins=60, color='steelblue', ax=axes[0])
    axes[0].set_title('Amount Distribution – Legitimate')
    axes[0].set_xlim(0, xlim)
    axes[0].set_xlabel('Transaction Amount (USD)')

    sns.histplot(data=df[df['Class'] == 1], x='Amount',
                 bins=60, color='tomato', ax=axes[1])
    axes[1].set_title('Amount Distribution – Fraud')
    axes[1].set_xlim(0, xlim)
    axes[1].set_xlabel('Transaction Amount (USD)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'amount_distribution.png'), dpi=150)
    plt.show()
    print("[eda] amount_distribution.png saved.")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Heatmap of the full feature correlation matrix, highlighting
    which PCA components (V1–V28) correlate most with the fraud label.
    """
    corr = df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', center=0,
                linewidths=0.3, annot=False, square=False)
    plt.title('Feature Correlation Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'), dpi=150)
    plt.show()
    print("[eda] correlation_matrix.png saved.")

    # Top correlated features with Class
    top = corr['Class'].drop('Class').abs().sort_values(ascending=False).head(10)
    print("\nTop 10 features correlated with Class:")
    print(top.to_string())


def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    """
    Boxplots comparing the Amount distribution (including outliers)
    for legitimate vs. fraudulent transactions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df[df['Class'] == 0]['Amount'].plot(
        kind='box', ax=axes[0], color='steelblue',
        title='Amount – Legitimate', vert=True
    )
    axes[0].set_ylabel('Transaction Amount (USD)')

    df[df['Class'] == 1]['Amount'].plot(
        kind='box', ax=axes[1], color='tomato',
        title='Amount – Fraud', vert=True
    )
    axes[1].set_ylabel('Transaction Amount (USD)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'outlier_boxplot.png'), dpi=150)
    plt.show()
    print("[eda] outlier_boxplot.png saved.")


def run_eda(df: pd.DataFrame) -> None:
    """
    Run the full EDA pipeline: class balance, amount distributions,
    correlation matrix, and outlier detection.
    """
    print("\n──── Exploratory Data Analysis ───────────────────────────────")
    fraud_pct = df['Class'].mean() * 100
    print(f"  Total records   : {len(df):,}")
    print(f"  Fraud records   : {df['Class'].sum():,}  ({fraud_pct:.4f}%)")
    print(f"  Legit records   : {(df['Class'] == 0).sum():,}")
    print("─────────────────────────────────────────────────────────────\n")

    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_correlation_matrix(df)
    plot_outlier_boxplots(df)
    print("\n[eda] All EDA plots generated and saved to results/figures/")
