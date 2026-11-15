"""
preprocess.py
─────────────
Data loading, scaling, splitting, and SMOTE resampling for the
Credit Card Fraud Detection project (CSE 2793 – ML Workshop 1).

Usage
-----
    from src.preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess('data/creditcard.csv')
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ─── Public API ───────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and print a quick summary."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Shape        : {df.shape}")
    print(f"[load_data] Missing vals : {df.isnull().sum().sum()}")
    fraud_pct = df['Class'].mean() * 100
    print(f"[load_data] Fraud %      : {fraud_pct:.4f}%")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the 'Amount' column with StandardScaler and drop 'Time'.

    Parameters
    ----------
    df : raw DataFrame (must contain 'Amount' and 'Time' columns)

    Returns
    -------
    df : transformed DataFrame with 'Amount' scaled and 'Time' removed
    """
    df = df.copy()
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df.drop(columns=['Time'], inplace=True)
    print("[scale_features] 'Amount' scaled; 'Time' dropped.")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[split_data] Train : {len(X_train):,}  |  Fraud in train: {y_train.sum()}")
    print(f"[split_data] Test  : {len(X_test):,}   |  Fraud in test : {y_test.sum()}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE **only** to training data to balance the class distribution.

    Returns
    -------
    X_resampled, y_resampled
    """
    print(f"[apply_smote] Before – Legit: {(y_train == 0).sum():,}  "
          f"Fraud: {(y_train == 1).sum():,}")

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"[apply_smote] After  – Legit: {(y_res == 0).sum():,}  "
          f"Fraud: {(y_res == 1).sum():,}")
    return X_res, y_res


def load_and_preprocess(
    filepath: str,
    test_size: float = 0.20,
    random_state: int = 42,
    apply_smote_flag: bool = True
):
    """
    End-to-end convenience function:
        1. Load  →  2. Scale  →  3. Split  →  4. SMOTE (optional)

    Returns
    -------
    X_train (or X_train_res), X_test, y_train (or y_train_res), y_test
    """
    df = load_data(filepath)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    return X_train, X_test, y_train, y_test
