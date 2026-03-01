import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from baseline import time_split, brier_score, get_feature_cols


def eval_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    bs = brier_score(y_true, y_prob)
    return {"log_loss": ll, "auc": auc, "accuracy": acc, "brier": bs}


def bootstrap_ci(y_true, y_prob, n_iterations=1000, ci=95):
    """Bootstrap CI on pre-computed predictions (no re-fitting)."""
    metrics = {"log_loss": [], "auc": [], "accuracy": [], "brier": []}
    n_size = len(y_true)
    rng = np.random.RandomState(42)

    print(f"Running {n_iterations} bootstrap iterations...")
    for _ in tqdm(range(n_iterations)):
        indices = rng.randint(0, n_size, n_size)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        res = eval_metrics(y_true_boot, y_prob_boot)
        for k, v in res.items():
            metrics[k].append(v)

    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p

    ci_results = {}
    for k, v in metrics.items():
        v = np.array(v)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            ci_results[k] = (float("nan"), float("nan"), float("nan"))
        else:
            ci_results[k] = (np.mean(v), np.percentile(v, lower_p), np.percentile(v, upper_p))

    return ci_results


def plot_calibration(models_dict, X, y, out_path):
    """Calibration curve for multiple models."""
    plt.figure(figsize=(8, 7))

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives", fontsize=12)
    plt.title("Calibration Plot (Test Set)", fontsize=14)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved calibration plot -> {out_path}")


def plot_roc(models_dict, X, y, out_path):
    """ROC curve for multiple models."""
    plt.figure(figsize=(8, 7))

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve (Test Set)", fontsize=14)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ROC plot -> {out_path}")


def plot_feature_importance(model, feature_cols, out_path, top_n=15):
    """Feature importance bar chart for tree-based models."""
    # Extract the classifier from the pipeline
    clf = model.named_steps.get("clf", None)
    if clf is None:
        print("Cannot extract clf from pipeline, skipping feature importance")
        return

    if not hasattr(clf, "feature_importances_"):
        print(f"Model {type(clf).__name__} has no feature_importances_, skipping")
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], color="#4C72B0")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1], fontsize=10)
    plt.xlabel("Feature Importance (Gini)", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances ({type(clf).__name__})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="outputs/step1_features.parquet")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    train, val, test = time_split(df)

    feature_cols = joblib.load(os.path.join(args.out_dir, "feature_cols.joblib"))

    for split_df in [train, val, test]:
        split_df.dropna(subset=["rank_diff"], inplace=True)

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_test = test["target"].values

    medians = X_train.median(numeric_only=True)
    X_test = X_test.fillna(medians)

    # Load all saved models
    models_dir = os.path.join(args.out_dir, "models")
    best_model = joblib.load(os.path.join(args.out_dir, "best_model.joblib"))

    models_dict = {}
    if os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.endswith(".joblib"):
                name = fname.replace("best_", "").replace(".joblib", "").replace("_", " ").title()
                models_dict[name] = joblib.load(os.path.join(models_dir, fname))

    if not models_dict:
        models_dict = {"Best Model": best_model}

    # =============================================
    # 1) Calibration Plot (all models)
    # =============================================
    print("\n--- Calibration Plot ---")
    plot_calibration(models_dict, X_test, y_test,
                     os.path.join(args.out_dir, "calibration_plot.png"))

    # =============================================
    # 2) ROC Curve (all models)
    # =============================================
    print("\n--- ROC Curve ---")
    plot_roc(models_dict, X_test, y_test,
             os.path.join(args.out_dir, "roc_curve.png"))

    # =============================================
    # 3) Feature Importance (tree-based models)
    # =============================================
    print("\n--- Feature Importance ---")
    for name, model in models_dict.items():
        clf = model.named_steps.get("clf", None)
        if clf and hasattr(clf, "feature_importances_"):
            safe = name.lower().replace(" ", "_")
            plot_feature_importance(
                model, feature_cols,
                os.path.join(args.out_dir, f"feature_importance_{safe}.png"))

    # =============================================
    # 4) Bootstrap CI (95%) on Test — Best Model
    # =============================================
    print("\n--- Bootstrap CI (95%) on Test Set ---")
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    ci_res = bootstrap_ci(y_test, y_prob_test, n_iterations=2000)

    print("\n--- FINAL METRICS WITH 95% CI ---")
    output_lines = []
    for k in ["log_loss", "auc", "accuracy", "brier"]:
        mean, lo, hi = ci_res[k]
        line = f"{k:>12s}: {mean:.4f}  (95% CI: {lo:.4f} – {hi:.4f})"
        output_lines.append(line)
        print(line)

    with open(os.path.join(args.out_dir, "step3_test_metrics_with_ci.txt"), "w") as f:
        f.write("Best Model Bootstrap CI on Test Set (2000 iterations)\n")
        f.write("=" * 55 + "\n")
        f.write("\n".join(output_lines))

    print("\nDone. All evaluation outputs saved.")


if __name__ == "__main__":
    main()
