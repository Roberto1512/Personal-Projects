import argparse
import os
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from baseline import time_split, brier_score, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------
# Evaluation helpers
# ----------------------------
def compute_metrics(y_true, y_prob):
    """Compute all four metrics for a set of predictions."""
    y_pred = (y_prob >= 0.5).astype(int)
    y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "log_loss": log_loss(y_true, y_prob_clip, labels=[0, 1]),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score(y_true, y_prob),
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test: are two classifiers significantly different?
    Returns chi2 statistic and p-value.
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A correct, B wrong; c = A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0  # identical predictions

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


# ----------------------------
# Hyperparameter grids (small, reasoned)
# ----------------------------
HP_GRIDS = {
    "Logistic Regression": [
        {"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0},
    ],
    "k-NN": [
        {"n_neighbors": 30}, {"n_neighbors": 50}, {"n_neighbors": 100}, {"n_neighbors": 200},
    ],
    "Decision Tree": [
        {"max_depth": 3}, {"max_depth": 5}, {"max_depth": 8}, {"max_depth": 12},
    ],
    "Random Forest": [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ],
    "SVM Linear": [
        {"C": 0.01}, {"C": 0.1}, {"C": 1.0},
    ],
    "SVM RBF": [
        {"C": 0.1, "gamma": "scale"}, {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"}, {"C": 1.0, "gamma": 0.01},
    ],
    "Naive Bayes": [
        {"var_smoothing": 1e-9}, {"var_smoothing": 1e-7}, {"var_smoothing": 1e-5},
    ],
}


def build_model(name, params):
    """Factory: build a Pipeline for each model family + hyperparameters."""
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=params["C"], max_iter=2000, n_jobs=-1)),
        ])
    elif name == "k-NN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=params["n_neighbors"], n_jobs=-1)),
        ])
    elif name == "Decision Tree":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)),
        ])
    elif name == "Random Forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42, n_jobs=-1)),
        ])
    elif name == "SVM Linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=params["C"], probability=True,
                        random_state=42, max_iter=10000)),
        ])
    elif name == "SVM RBF":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                        probability=True, random_state=42, max_iter=10000)),
        ])
    elif name == "Naive Bayes":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=params["var_smoothing"])),
        ])
    else:
        raise ValueError(f"Unknown model: {name}")


# ----------------------------
# Feature groups for ablation
# ----------------------------
ABLATION_GROUPS = {
    "A: Rank + Surface only": lambda cols: [
        c for c in cols if c in [
            "rank_diff", "pts_diff", "round_order", "series_importance", "best_of"
        ] or c.startswith("Surface_") or c.startswith("Court_")
    ],
    "B: A + Form/Streak/H2H": lambda cols: [
        c for c in cols if c in [
            "rank_diff", "pts_diff", "round_order", "series_importance", "best_of",
            "p1_last5", "p2_last5", "p1_streak", "p2_streak",
            "h2h_p1_wins", "h2h_p2_wins", "form_diff", "streak_diff", "h2h_diff",
        ] or c.startswith("Surface_") or c.startswith("Court_")
    ],
    "C: B + Elo": lambda cols: [
        c for c in cols if c in [
            "rank_diff", "pts_diff", "round_order", "series_importance", "best_of",
            "p1_last5", "p2_last5", "p1_streak", "p2_streak",
            "h2h_p1_wins", "h2h_p2_wins", "form_diff", "streak_diff", "h2h_diff",
            "elo_1", "elo_2", "elo_diff",
        ] or c.startswith("Surface_") or c.startswith("Court_")
    ],
    "D: All features": lambda cols: cols,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="outputs/step1_features.parquet")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)

    train, val, test = time_split(df)
    print(f"Split sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    feature_cols = get_feature_cols(df)

    # Basic cleaning: rows must have rank_diff
    for split_df in [train, val, test]:
        split_df.dropna(subset=["rank_diff"], inplace=True)

    X_train_full = train[feature_cols].copy()
    y_train = train["target"].values
    X_val_full = val[feature_cols].copy()
    y_val = val["target"].values
    X_test_full = test[feature_cols].copy()
    y_test = test["target"].values

    # Impute missing values with train medians (no leakage)
    medians = X_train_full.median(numeric_only=True)
    X_train_full = X_train_full.fillna(medians)
    X_val_full = X_val_full.fillna(medians)
    X_test_full = X_test_full.fillna(medians)

    # =============================================
    # PART 1: Hyperparameter Tuning on Validation
    # =============================================
    print("\n" + "=" * 60)
    print("PART 1: HYPERPARAMETER TUNING (Validation Set)")
    print("=" * 60)

    all_grid_results = []
    best_per_family = {}

    for family_name, hp_list in HP_GRIDS.items():
        print(f"\n--- {family_name} ---")
        best_ll = float("inf")
        best_entry = None

        for params in hp_list:
            label = f"{family_name} {params}"
            try:
                model = build_model(family_name, params)
                model.fit(X_train_full, y_train)

                train_metrics = compute_metrics(y_train, model.predict_proba(X_train_full)[:, 1])
                val_metrics = compute_metrics(y_val, model.predict_proba(X_val_full)[:, 1])

                row = {
                    "family": family_name,
                    "params": str(params),
                    "train_ll": train_metrics["log_loss"],
                    "train_auc": train_metrics["auc"],
                    "train_acc": train_metrics["accuracy"],
                    "train_brier": train_metrics["brier"],
                    "val_ll": val_metrics["log_loss"],
                    "val_auc": val_metrics["auc"],
                    "val_acc": val_metrics["accuracy"],
                    "val_brier": val_metrics["brier"],
                }
                all_grid_results.append(row)

                print(f"  {params}  ->  val_LL={val_metrics['log_loss']:.4f}  "
                      f"val_AUC={val_metrics['auc']:.4f}  val_Acc={val_metrics['accuracy']:.4f}")

                if val_metrics["log_loss"] < best_ll:
                    best_ll = val_metrics["log_loss"]
                    best_entry = {"model": model, "params": params, "metrics": val_metrics, "label": label}

            except Exception as e:
                print(f"  {params}  ->  FAILED: {e}")

        if best_entry:
            best_per_family[family_name] = best_entry
            print(f"  ** Best: {best_entry['params']}  (val_LL={best_ll:.4f})")

    # Save grid search results
    grid_df = pd.DataFrame(all_grid_results)
    grid_path = os.path.join(args.out_dir, "step2_val_grid_results.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"\nSaved grid results -> {grid_path}")

    # =============================================
    # PART 2: Best Model Per Family — Summary
    # =============================================
    print("\n" + "=" * 60)
    print("PART 2: BEST MODEL PER FAMILY (sorted by val Log-Loss)")
    print("=" * 60)

    summary_rows = []
    for fam, entry in sorted(best_per_family.items(), key=lambda x: x[1]["metrics"]["log_loss"]):
        m = entry["metrics"]
        print(f"  {fam:25s}  LL={m['log_loss']:.4f}  AUC={m['auc']:.4f}  "
              f"Acc={m['accuracy']:.4f}  Brier={m['brier']:.4f}  params={entry['params']}")
        summary_rows.append({
            "model": fam, "best_params": str(entry["params"]),
            "val_ll": m["log_loss"], "val_auc": m["auc"],
            "val_acc": m["accuracy"], "val_brier": m["brier"],
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "step2_val_results.csv"), index=False)

    # Overall best
    overall_best_fam = min(best_per_family, key=lambda k: best_per_family[k]["metrics"]["log_loss"])
    overall_best = best_per_family[overall_best_fam]
    print(f"\n*** Overall best: {overall_best_fam} {overall_best['params']}  "
          f"(val_LL={overall_best['metrics']['log_loss']:.4f}) ***")

    # Save best model + all family-best models
    best_model_path = os.path.join(args.out_dir, "best_model.joblib")
    joblib.dump(overall_best["model"], best_model_path)
    print(f"Saved best model -> {best_model_path}")

    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for fam, entry in best_per_family.items():
        safe_name = fam.lower().replace(" ", "_").replace("-", "")
        joblib.dump(entry["model"], os.path.join(models_dir, f"best_{safe_name}.joblib"))

    # Save feature cols
    joblib.dump(feature_cols, os.path.join(args.out_dir, "feature_cols.joblib"))

    # =============================================
    # PART 3: Test Evaluation (One-Shot)
    # =============================================
    print("\n" + "=" * 60)
    print("PART 3: TEST EVALUATION (One-Shot)")
    print("=" * 60)

    test_rows = []
    for fam, entry in sorted(best_per_family.items(), key=lambda x: x[1]["metrics"]["log_loss"]):
        model = entry["model"]
        y_prob = model.predict_proba(X_test_full)[:, 1]
        m = compute_metrics(y_test, y_prob)
        print(f"  {fam:25s}  LL={m['log_loss']:.4f}  AUC={m['auc']:.4f}  "
              f"Acc={m['accuracy']:.4f}  Brier={m['brier']:.4f}")
        test_rows.append({"model": fam, **m})

    pd.DataFrame(test_rows).to_csv(
        os.path.join(args.out_dir, "step2_test_results.csv"), index=False)

    # =============================================
    # PART 4: McNemar's Test (Rank Baseline vs Best)
    # =============================================
    print("\n" + "=" * 60)
    print("PART 4: McNEMAR'S TEST (Rank Baseline vs Best Model)")
    print("=" * 60)

    # Rank baseline predictions on test
    rd_test = test["rank_diff"].fillna(0).values
    y_pred_rank = (rd_test > 0).astype(int)
    y_pred_best = overall_best["model"].predict(X_test_full)

    chi2_stat, p_val = mcnemar_test(y_test, y_pred_rank, y_pred_best)
    print(f"  McNemar chi2 = {chi2_stat:.4f},  p-value = {p_val:.6f}")
    if p_val < 0.05:
        print("  -> Statistically significant difference at alpha=0.05")
    else:
        print("  -> No significant difference at alpha=0.05")

    # =============================================
    # PART 5: Feature Ablation Study
    # =============================================
    print("\n" + "=" * 60)
    print("PART 5: FEATURE ABLATION (using best model family: {})".format(overall_best_fam))
    print("=" * 60)

    ablation_rows = []
    for group_name, selector in ABLATION_GROUPS.items():
        sub_cols = selector(feature_cols)
        if not sub_cols:
            print(f"  {group_name}: no columns matched, skipping")
            continue

        X_tr = X_train_full[sub_cols]
        X_v = X_val_full[sub_cols]
        X_te = X_test_full[sub_cols]

        model = build_model(overall_best_fam, overall_best["params"])
        model.fit(X_tr, y_train)

        val_m = compute_metrics(y_val, model.predict_proba(X_v)[:, 1])
        test_m = compute_metrics(y_test, model.predict_proba(X_te)[:, 1])

        print(f"  {group_name:30s}  ({len(sub_cols):2d} feats)  "
              f"val_LL={val_m['log_loss']:.4f}  test_LL={test_m['log_loss']:.4f}  "
              f"test_AUC={test_m['auc']:.4f}")

        ablation_rows.append({
            "feature_group": group_name,
            "n_features": len(sub_cols),
            "val_ll": val_m["log_loss"], "val_auc": val_m["auc"],
            "test_ll": test_m["log_loss"], "test_auc": test_m["auc"],
            "test_acc": test_m["accuracy"], "test_brier": test_m["brier"],
        })

    pd.DataFrame(ablation_rows).to_csv(
        os.path.join(args.out_dir, "step2_ablation_results.csv"), index=False)

    print("\nDone. Run evaluation.py for Bootstrap CI + Calibration + ROC + Feature Importance.")


if __name__ == "__main__":
    main()
