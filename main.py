"""
main.py
───────
End-to-end entry point for the Credit Card Fraud Detection project.
Runs preprocessing → EDA → model training → evaluation in one command.

Usage
-----
    python main.py
    python main.py --data data/creditcard.csv --no-smote
"""

import argparse
import os

from src.preprocess import load_data, scale_features, split_data, apply_smote
from src.eda        import run_eda
from src.train      import train_all_models
from src.evaluate   import evaluate_all_models, plot_feature_importance


def parse_args():
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection – ML Pipeline'
    )
    parser.add_argument(
        '--data', type=str, default='data/creditcard.csv',
        help='Path to the creditcard.csv dataset (default: data/creditcard.csv)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.20,
        help='Fraction of data reserved for testing (default: 0.20)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-smote', action='store_true',
        help='Disable SMOTE oversampling (train on raw imbalanced data)'
    )
    parser.add_argument(
        '--skip-eda', action='store_true',
        help='Skip EDA plots (useful for fast runs in CI)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate dataset path ─────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"\n[ERROR] Dataset not found at: {args.data}")
        print("  Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("  Then place creditcard.csv in the data/ directory.\n")
        raise SystemExit(1)

    # ── Step 1: Load & EDA ────────────────────────────────────────────────────
    df = load_data(args.data)
    if not args.skip_eda:
        run_eda(df)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    df_scaled = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=args.test_size, random_state=args.seed
    )

    if not args.no_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=args.seed)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    models = train_all_models(X_train, y_train, random_state=args.seed)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    results = evaluate_all_models(models, X_test, y_test)

    # ── Step 5: Feature importance for Random Forest ──────────────────────────
    plot_feature_importance(
        models['Random Forest'],
        feature_names=X_test.columns,
        top_n=15
    )

    print("\n✓ Pipeline complete.  Results saved to results/\n")
    return results


if __name__ == '__main__':
    main()
