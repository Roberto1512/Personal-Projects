import argparse
import os
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


# ----------------------------
# Utilities
# ----------------------------
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def safe_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce")


def normalize_str(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip()


# ----------------------------
# Elo
# ----------------------------
@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_base: float = 32.0
    scale: float = 400.0


def elo_expected(r_a: float, r_b: float, scale: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))


def elo_update(r_a: float, r_b: float, s_a: float, cfg: EloConfig, k: float) -> tuple[float, float]:
    e_a = elo_expected(r_a, r_b, cfg.scale)
    e_b = 1.0 - e_a
    r_a_new = r_a + k * (s_a - e_a)
    r_b_new = r_b + k * ((1.0 - s_a) - e_b)
    return r_a_new, r_b_new


def series_k_multiplier(series: str) -> float:
    """Scale the Elo K-factor by tournament prestige."""
    s = str(series).strip()
    if s == "Grand Slam":
        return 1.25
    if s == "Masters 1000":
        return 1.15
    if s == "ATP 500":
        return 1.05
    return 1.0


# ----------------------------
# Feature building (NO ODDS)
# ----------------------------
def build_features(df: pd.DataFrame, start_year_eval: int = 2006) -> pd.DataFrame:
    df = df.copy()

    # Parse date, sort
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date", "Tournament"]).reset_index(drop=True)

    # Normalize strings (player names sometimes have weird spaces)
    for c in ["Player_1", "Player_2", "Winner", "Series", "Court", "Surface", "Round"]:
        df[c] = normalize_str(df[c])

    # Replace -1 placeholders with NaN (true missing)
    for c in ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]:
        if c in df.columns:
            df[c] = safe_float(df[c])
            df.loc[df[c] <= 0, c] = np.nan  # ranks/pts/odds use -1

    # Target (1 if Player_1 won)
    df["target"] = (df["Winner"] == df["Player_1"]).astype(int)

    # Basic diffs (pre-match) — keep diffs only; drop raw ranks/pts to reduce multicollinearity
    df["rank_diff"] = df["Rank_2"] - df["Rank_1"]     # >0 means Player_1 has better (lower) rank
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]

    # Round order (numeric encoding)
    round_order = {
        "1st Round": 1,
        "2nd Round": 2,
        "3rd Round": 3,
        "4th Round": 4,
        "Quarterfinals": 5,
        "Quarter Finals": 5,
        "Semifinals": 6,
        "The Final": 7,
        "Final": 7,
        "Round Robin": 2,
    }
    df["round_order"] = df["Round"].map(round_order).fillna(1).astype(int)

    # Series importance (numeric)
    series_map = {"Grand Slam": 4, "Masters 1000": 3, "ATP 500": 2, "ATP250": 1, "International": 1}
    df["series_importance"] = df["Series"].map(series_map).fillna(1).astype(int)

    # Best of (3 vs 5 sets — important tactical feature)
    if "Best of" in df.columns:
        df["best_of"] = safe_float(df["Best of"]).fillna(3).astype(int)
    else:
        df["best_of"] = 3  # default

    # ---- Save raw Series for Elo K-factor BEFORE one-hot encoding destroys it ----
    raw_series = df["Series"].copy()

    # One-hot categorical (pre-match)
    df = pd.get_dummies(df, columns=["Court", "Surface", "Series"], drop_first=False)

    # Historical features (pre-match) computed strictly chronological
    last_n = 5
    history = defaultdict(lambda: {"last": deque(maxlen=last_n), "streak": 0})
    h2h = defaultdict(lambda: [0, 0])  # pair_sorted -> [wins_of_pair[0], wins_of_pair[1]]
    elo_cfg = EloConfig()
    elo_rating = defaultdict(lambda: elo_cfg.base_rating)

    # init cols
    df["p1_last5"] = 0.5
    df["p2_last5"] = 0.5
    df["p1_streak"] = 0
    df["p2_streak"] = 0
    df["h2h_p1_wins"] = 0
    df["h2h_p2_wins"] = 0
    df["elo_1"] = elo_cfg.base_rating
    df["elo_2"] = elo_cfg.base_rating
    df["elo_diff"] = 0.0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Building historical features"):
        p1 = row["Player_1"]
        p2 = row["Player_2"]
        winner = row["Winner"]

        # pre-match last5 + streak
        p1_last = history[p1]["last"]
        p2_last = history[p2]["last"]
        df.at[i, "p1_last5"] = float(np.mean(p1_last)) if len(p1_last) else 0.5
        df.at[i, "p2_last5"] = float(np.mean(p2_last)) if len(p2_last) else 0.5
        df.at[i, "p1_streak"] = int(history[p1]["streak"])
        df.at[i, "p2_streak"] = int(history[p2]["streak"])

        # pre-match h2h
        pair = tuple(sorted([p1, p2]))
        wins = h2h[pair]
        if pair[0] == p1:
            df.at[i, "h2h_p1_wins"] = wins[0]
            df.at[i, "h2h_p2_wins"] = wins[1]
        else:
            df.at[i, "h2h_p1_wins"] = wins[1]
            df.at[i, "h2h_p2_wins"] = wins[0]

        # pre-match elo
        r1 = float(elo_rating[p1])
        r2 = float(elo_rating[p2])
        df.at[i, "elo_1"] = r1
        df.at[i, "elo_2"] = r2
        df.at[i, "elo_diff"] = r1 - r2

        # ---- Update AFTER the match ----
        if winner == p1:
            s1 = 1.0
            res1, res2 = 1, 0
        elif winner == p2:
            s1 = 0.0
            res1, res2 = 0, 1
        else:
            # weird row -> skip updates
            continue

        # update lastN
        history[p1]["last"].append(res1)
        history[p2]["last"].append(res2)

        # update streak
        history[p1]["streak"] = history[p1]["streak"] + 1 if res1 == 1 else 0
        history[p2]["streak"] = history[p2]["streak"] + 1 if res2 == 1 else 0

        # update h2h
        if pair[0] == p1:
            wins[0] += int(res1 == 1)
            wins[1] += int(res2 == 1)
        else:
            wins[0] += int(res2 == 1)
            wins[1] += int(res1 == 1)
        h2h[pair] = wins

        # update elo — use the SAVED raw_series (not the destroyed column)
        k = elo_cfg.k_base * series_k_multiplier(raw_series.iloc[i])
        r1_new, r2_new = elo_update(r1, r2, s1, elo_cfg, k=k)
        elo_rating[p1] = r1_new
        elo_rating[p2] = r2_new

    # comparative diffs (pre-match)
    df["form_diff"] = df["p1_last5"] - df["p2_last5"]
    df["streak_diff"] = df["p1_streak"] - df["p2_streak"]
    df["h2h_diff"] = df["h2h_p1_wins"] - df["h2h_p2_wins"]

    # Filter evaluation years (keep full history computed, but evaluate on chosen range)
    df["Year"] = df["Date"].dt.year.astype(int)
    df = df[df["Year"] >= start_year_eval].reset_index(drop=True)

    # Drop raw rank/pts columns to reduce multicollinearity with rank_diff/pts_diff
    for c in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def time_split(df: pd.DataFrame):
    # Train 2006-2021, Val 2022-2023, Test 2024-2025
    train = df[(df["Year"] >= 2006) & (df["Year"] <= 2021)].copy()
    val = df[(df["Year"] >= 2022) & (df["Year"] <= 2023)].copy()
    test = df[(df["Year"] >= 2024) & (df["Year"] <= 2025)].copy()
    return train, val, test


def evaluate_split(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    bs = brier_score(y_true, y_prob)
    print(f"\n[{name}]  Acc={acc:.4f}  AUC={auc:.4f}  LogLoss={ll:.4f}  Brier={bs:.4f}")


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature columns, excluding identifiers, target, odds, and leakage."""
    drop_cols = {
        "Tournament", "Player_1", "Player_2", "Winner", "Score",
        "Date", "Round", "Best of"
    }
    feature_cols = [c for c in df.columns if c not in drop_cols | {"target", "Year"}]

    # Remove raw odds columns (Setting A: no odds)
    for c in ["Odd_1", "Odd_2", "odds_diff", "prob_1", "prob_2"]:
        if c in feature_cols:
            feature_cols.remove(c)

    return feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/atp_tennis.csv")
    ap.add_argument("--out", type=str, default="outputs/step1_features.parquet")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, low_memory=False, dtype={"Odd_2": "string"})
    df = build_features(raw, start_year_eval=2006)

    # Save features for next steps
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved feature table -> {out_path}  (rows={len(df):,}, cols={df.shape[1]})")

    train, val, test = time_split(df)
    print(f"Split sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    feature_cols = get_feature_cols(df)

    # Basic cleaning: rows must have rank_diff at least
    for split_df in [train, val, test]:
        split_df.dropna(subset=["rank_diff"], inplace=True)

    X_train = train[feature_cols].copy()
    y_train = train["target"].values
    X_val = val[feature_cols].copy()
    y_val = val["target"].values
    X_test = test[feature_cols].copy()
    y_test = test["target"].values

    # Impute remaining NaNs with median computed on TRAIN only (no leakage)
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # ----------------------------
    # Baseline: rank only
    # Predict P1 wins if Rank_1 < Rank_2  <=> rank_diff > 0
    # ----------------------------
    def rank_baseline(split_df: pd.DataFrame):
        y_true = split_df["target"].values
        rd = split_df["rank_diff"].fillna(0).values
        y_pred = (rd > 0).astype(int)
        y_prob = np.where(rd == 0, 0.5, np.where(rd > 0, 0.65, 0.35))
        return y_true, y_prob, y_pred

    for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        yt, yp, yhat = rank_baseline(split)
        evaluate_split(f"Rank baseline - {name}", yt, yp, yhat)

    # ----------------------------
    # Model 1: Logistic Regression (course-aligned)
    # ----------------------------
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    clf.fit(X_train, y_train)
    for name, X, y in [("TRAIN", X_train, y_train), ("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        evaluate_split(f"LogReg - {name}", y, prob, pred)

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print("\nDone. Next step: run models_step2.py for full model comparison + tuning.")


if __name__ == "__main__":
    main()