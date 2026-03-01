"""
Step 4: Stacked Generalization Ensemble (Algorithmic Contribution)

Uses the probability outputs of 7 trained base models as meta-features
for a second-level Logistic Regression. This is a novel variant of
stacked generalization specifically designed for tennis match prediction.

Flow:
  1. Load all 7 base models (trained in models_step2.py)
  2. Generate meta-features: P(Player_1 wins) from each base model
  3. Train a meta-learner (LogReg) on validation-set meta-features
  4. Evaluate the stacked ensemble on the test set
  5. Compare vs. best single model
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from baseline import time_split, brier_score, get_feature_cols


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "log_loss": log_loss(y_true, y_prob_clip, labels=[0, 1]),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score(y_true, y_prob),
    }


def build_meta_features(models_dict, X):
    """Build meta-feature matrix: one column per base model's P(y=1)."""
    meta = np.zeros((len(X), len(models_dict)))
    names = []
    for i, (name, model) in enumerate(sorted(models_dict.items())):
        meta[:, i] = model.predict_proba(X)[:, 1]
        names.append(name)
    return meta, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="outputs/step1_features.parquet")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Load data -----
    print("Loading features...")
    df = pd.read_parquet(args.features)
    train, val, test = time_split(df)

    feature_cols = joblib.load(os.path.join(args.out_dir, "feature_cols.joblib"))

    for split_df in [train, val, test]:
        split_df.dropna(subset=["rank_diff"], inplace=True)

    X_train = train[feature_cols].copy()
    y_train = train["target"].values
    X_val = val[feature_cols].copy()
    y_val = val["target"].values
    X_test = test[feature_cols].copy()
    y_test = test["target"].values

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # ----- Load base models -----
    models_dir = os.path.join(args.out_dir, "models")
    base_models = {}
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith(".joblib"):
            name = fname.replace("best_", "").replace(".joblib", "").replace("_", " ").title()
            base_models[name] = joblib.load(os.path.join(models_dir, fname))

    print(f"Loaded {len(base_models)} base models: {list(base_models.keys())}")

    # ----- Build meta-features -----
    print("\nBuilding meta-features from base model predictions...")
    meta_val, meta_names = build_meta_features(base_models, X_val)
    meta_test, _ = build_meta_features(base_models, X_test)
    meta_train, _ = build_meta_features(base_models, X_train)

    print(f"Meta-feature matrix shape: val={meta_val.shape}, test={meta_test.shape}")
    print(f"Meta-feature columns: {meta_names}")

    # Show correlation between base model predictions
    print("\n--- Base Model Prediction Correlations (Validation) ---")
    corr_df = pd.DataFrame(meta_val, columns=meta_names).corr()
    print(corr_df.round(3).to_string())

    # ----- Train meta-learner -----
    print("\n--- Training Meta-Learner (Stacked Ensemble) ---")

    # Try multiple regularization strengths for the meta-learner
    best_meta_ll = float("inf")
    best_meta_model = None
    best_meta_C = None

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        meta_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        meta_clf.fit(meta_val, y_val)
        meta_prob_val = meta_clf.predict_proba(meta_val)[:, 1]
        val_m = compute_metrics(y_val, meta_prob_val)
        print(f"  C={C:6.2f}  val_LL={val_m['log_loss']:.4f}  "
              f"val_AUC={val_m['auc']:.4f}  val_Acc={val_m['accuracy']:.4f}")

        if val_m["log_loss"] < best_meta_ll:
            best_meta_ll = val_m["log_loss"]
            best_meta_model = meta_clf
            best_meta_C = C

    print(f"\n  ** Best meta-learner: C={best_meta_C} (val_LL={best_meta_ll:.4f})")

    # ----- Evaluate stacked ensemble on TEST -----
    print("\n" + "=" * 60)
    print("STACKED ENSEMBLE — TEST SET EVALUATION")
    print("=" * 60)

    meta_prob_test = best_meta_model.predict_proba(meta_test)[:, 1]
    stack_metrics = compute_metrics(y_test, meta_prob_test)

    # Load best single model for comparison
    best_single = joblib.load(os.path.join(args.out_dir, "best_model.joblib"))
    single_prob_test = best_single.predict_proba(X_test)[:, 1]
    single_metrics = compute_metrics(y_test, single_prob_test)

    print(f"\n  {'Model':30s}  {'LL':>8s}  {'AUC':>8s}  {'Acc':>8s}  {'Brier':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Best Single (LogReg)':30s}  {single_metrics['log_loss']:8.4f}  "
          f"{single_metrics['auc']:8.4f}  {single_metrics['accuracy']:8.4f}  "
          f"{single_metrics['brier']:8.4f}")
    print(f"  {'STACKED ENSEMBLE':30s}  {stack_metrics['log_loss']:8.4f}  "
          f"{stack_metrics['auc']:8.4f}  {stack_metrics['accuracy']:8.4f}  "
          f"{stack_metrics['brier']:8.4f}")

    # Improvement
    ll_delta = single_metrics["log_loss"] - stack_metrics["log_loss"]
    auc_delta = stack_metrics["auc"] - single_metrics["auc"]
    acc_delta = stack_metrics["accuracy"] - single_metrics["accuracy"]
    print(f"\n  Improvement:  LL {ll_delta:+.4f}  AUC {auc_delta:+.4f}  Acc {acc_delta:+.4f}")

    # ----- Meta-learner coefficients (interpretability) -----
    print("\n--- Meta-Learner Coefficients ---")
    meta_lr = best_meta_model.named_steps["lr"]
    coefs = meta_lr.coef_[0]
    for name, coef in sorted(zip(meta_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s}  coef = {coef:+.4f}")

    # ----- Save results -----
    results = {
        "stacked_ensemble": stack_metrics,
        "best_single": single_metrics,
        "meta_learner_C": best_meta_C,
        "base_models": meta_names,
    }

    # Save stacked model
    joblib.dump(best_meta_model, os.path.join(args.out_dir, "stacked_model.joblib"))

    # Save comparison CSV
    comp_df = pd.DataFrame([
        {"model": "Best Single (LogReg)", **single_metrics},
        {"model": "Stacked Ensemble", **stack_metrics},
    ])
    comp_df.to_csv(os.path.join(args.out_dir, "step4_stacking_results.csv"), index=False)

    # Save meta-learner coefficients
    coef_df = pd.DataFrame({
        "base_model": meta_names,
        "coefficient": coefs,
    }).sort_values("coefficient", ascending=False, key=abs)
    coef_df.to_csv(os.path.join(args.out_dir, "step4_meta_coefficients.csv"), index=False)

    print(f"\nSaved: stacked_model.joblib, step4_stacking_results.csv, step4_meta_coefficients.csv")
    print("\nDone. The stacked ensemble combines the wisdom of all 7 base learners.")


if __name__ == "__main__":
    main()
