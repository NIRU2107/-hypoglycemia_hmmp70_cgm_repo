"""
Logistic regression risk model for imminent hypoglycemia within 60 minutes.

This script:
1. Loads CGM data from a CSV (dt, glucose, patient_pid).
2. Labels sustained hypoglycemia events (glucose <= 70 mg/dL for >= 15 minutes).
3. Builds the primary outcome event_next60:
   imminent hypoglycemia within the next 60 minutes.
4. Constructs tabular features per timepoint:
   - CGM level
   - 5-min, 15-min, and 30-min deltas
   - Recent volatility (rolling std over 30 and 60 minutes)
   - Temporal encodings (hour-of-day sine/cosine, day-of-week)
5. Trains a calibrated logistic regression model (Platt scaling).
6. Outputs risk scores and labels for all timepoints to a CSV.

Dependencies:
    pandas
    numpy
    tqdm
    scikit-learn

Usage:
    python logistic_risk_pipeline.py \
        --input_csv combined_cgm_fixed.csv0hio11.gz \
        --output_csv logistic_risk_scores_with_events.csv.gz
"""

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


def load_cgm(path):
    """
    Load CGM data.

    Expected columns:
        dt: timestamp
        glucose: CGM value (mg/dL)
        patient_pid: patient identifier

    Returns:
        df: DataFrame sorted by patient_pid and dt.
    """
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["patient_pid", "dt"]).reset_index(drop=True)
    return df


def label_sustained_hypo_events(df, threshold=70.0, min_len=3):
    """
    Label sustained hypoglycemia events per patient.

    Hypoglycemia at a point:
        is_hypo = (glucose <= threshold)

    Sustained event:
        at least min_len contiguous is_hypo points.

    For each qualifying run:
        - Mark the *first* point of the run as event_onset = True.

    Returns:
        df_events: DataFrame with columns:
            is_hypo (bool)
            event_onset (bool)
    """
    df = df.copy()
    df = df.sort_values(["patient_pid", "dt"])

    df["is_hypo"] = df["glucose"] <= threshold

    def _label_runs(group):
        run_id = (group["is_hypo"] != group["is_hypo"].shift()).cumsum()
        group["run_id"] = run_id
        return group

    df = df.groupby("patient_pid", group_keys=False).apply(_label_runs)

    # Run length per (patient, run_id)
    run_lengths = df.groupby(["patient_pid", "run_id"])["is_hypo"].transform("sum")

    df["event_onset"] = False
    mask_event_runs = (df["is_hypo"]) and (run_lengths >= min_len)

    # First index in each run
    first_in_run = df.groupby(["patient_pid", "run_id"]).cumcount() == 0
    df.loc[mask_event_runs & first_in_run, "event_onset"] = True

    df = df.drop(columns=["run_id"])
    return df


def label_event_next60(df, window_minutes=60):
    """
    Label imminent hypoglycemia within next `window_minutes`.

    event_next60 at time t is True if the *next* event_onset for that patient
    occurs within window_minutes after t.

    Args:
        df: DataFrame with columns dt, patient_pid, event_onset (bool).
        window_minutes: lookahead window in minutes.

    Returns:
        df_out: DataFrame with added column event_next60 (bool).
    """
    df = df.copy()
    df = df.sort_values(["patient_pid", "dt"])
    df["dt"] = pd.to_datetime(df["dt"])

    df["event_next60"] = False

    patients = df["patient_pid"].unique()
    window = np.timedelta64(window_minutes, "m")

    for pid in tqdm(patients, desc="Labeling event_next60 by patient"):
        g = df[df["patient_pid"] == pid]
        idx = g.index.values
        times = g["dt"].values
        onset_mask = g["event_onset"].values

        event_idx = np.where(onset_mask)[0]
        if event_idx.size == 0:
            continue

        event_times = times[event_idx]
        labels = np.zeros(len(g), dtype=bool)

        for i in range(len(g)):
            t = times[i]
            j = np.searchsorted(event_times, t, side="right")
            if j >= len(event_times):
                continue
            if event_times[j] - t <= window:
                labels[i] = True

        df.loc[idx, "event_next60"] = labels

    return df


def build_features(df):
    """
    Build tabular features for logistic regression.

    Features:
        glucose
        glucose_delta_5m         (t - t-1, per patient)
        delta_15m                (t - t-3, per patient)
        delta_30m                (t - t-6, per patient)
        roll_std_30m             (rolling std over 6 samples ~30 min)
        roll_std_60m             (rolling std over 12 samples ~60 min)
        hour_sin, hour_cos       (cyclic hour-of-day)
        dow                      (day of week 0-6)

    Returns:
        df_feat: DataFrame with features + label event_next60.
        feature_cols: list of feature column names.
    """
    df = df.copy()
    df = df.sort_values(["patient_pid", "dt"])
    df["dt"] = pd.to_datetime(df["dt"])

    # 5-min deltas
    df["glucose_delta_5m"] = df.groupby("patient_pid")["glucose"].diff()

    # 15m (3 samples) and 30m (6 samples) deltas
    df["delta_15m"] = df.groupby("patient_pid")["glucose"].diff(3)
    df["delta_30m"] = df.groupby("patient_pid")["glucose"].diff(6)

    # Rolling volatility
    df["roll_std_30m"] = (
        df.groupby("patient_pid")["glucose"]
        .rolling(window=6, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["roll_std_60m"] = (
        df.groupby("patient_pid")["glucose"]
        .rolling(window=12, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Temporal features
    df["hour"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.weekday

    # Cyclic encoding
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)

    feature_cols = [
        "glucose",
        "glucose_delta_5m",
        "delta_15m",
        "delta_30m",
        "roll_std_30m",
        "roll_std_60m",
        "hour_sin",
        "hour_cos",
        "dow",
    ]

    # Drop rows missing features or label
    df_feat = df.dropna(subset=feature_cols + ["event_next60"]).reset_index(drop=True)

    return df_feat, feature_cols


def train_calibrated_logistic(df_feat, feature_cols, seed=42):
    """
    Train a calibrated logistic regression model to predict event_next60.

    Patient-wise split:
        60% train, 20% val, 20% test (approx).

    Calibration:
        Platt scaling (sigmoid) via CalibratedClassifierCV with 3-fold CV.

    Returns:
        model: fitted CalibratedClassifierCV instance.
        scaler: fitted StandardScaler.
        splits: dict with keys 'train', 'val', 'test' containing (X, y, df).
        metrics: dict with AUCs and Brier scores per split.
    """
    from collections import OrderedDict

    df_model = df_feat.copy()
    patients = df_model["patient_pid"].unique()

    # Patient-wise splits
    train_p, test_p = train_test_split(
        patients, test_size=0.2, random_state=seed
    )
    train_p, val_p = train_test_split(
        train_p, test_size=0.25, random_state=seed
    )  # 0.25 of 0.8 -> 0.2 overall

    splits = OrderedDict()
    for name, pids in [("train", train_p), ("val", val_p), ("test", test_p)]:
        g = df_model[df_model["patient_pid"].isin(pids)]
        X = g[feature_cols].values
        y = g["event_next60"].astype(int).values
        splits[name] = (X, y, g)

    X_train, y_train, df_train = splits["train"]
    X_val, y_val, df_val = splits["val"]
    X_test, y_test, df_test = splits["test"]

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    # Base logistic regression
    base_logit = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    # Calibrated using Platt scaling
    calibrated_clf = CalibratedClassifierCV(
        base_logit,
        cv=3,
        method="sigmoid",
    )
    calibrated_clf.fit(X_train_std, y_train)

    # Probabilities
    p_train = calibrated_clf.predict_proba(X_train_std)[:, 1]
    p_val = calibrated_clf.predict_proba(X_val_std)[:, 1]
    p_test = calibrated_clf.predict_proba(X_test_std)[:, 1]

    # Metrics
    metrics = {}
    metrics["auc_train"] = roc_auc_score(y_train, p_train)
    metrics["auc_val"] = roc_auc_score(y_val, p_val)
    metrics["auc_test"] = roc_auc_score(y_test, p_test)

    metrics["brier_train"] = brier_score_loss(y_train, p_train)
    metrics["brier_val"] = brier_score_loss(y_val, p_val)
    metrics["brier_test"] = brier_score_loss(y_test, p_test)

    return calibrated_clf, scaler, splits, metrics


def attach_risk_scores(df_feat, feature_cols, model, scaler):
    """
    Attach calibrated logistic regression risk scores to df_feat.

    Adds:
        lr_risk: P(event_next60 = 1 | features) in [0,1].
    """
    df_out = df_feat.copy()
    X_all = df_out[feature_cols].values
    X_all_std = scaler.transform(X_all)
    df_out["lr_risk"] = model.predict_proba(X_all_std)[:, 1]
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Logistic regression risk model for imminent hypoglycemia (next 60 minutes)."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CGM CSV (e.g., combined_cgm_fixed.csv0hio11.gz)",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV with risk scores and labels (e.g., logistic_risk_scores_with_events.csv.gz)",
    )
    args = parser.parse_args()

    # 1) Load CGM
    df = load_cgm(args.input_csv)

    # 2) Label sustained hypoglycemia events
    df = label_sustained_hypo_events(df, threshold=70.0, min_len=3)

    # 3) Label imminent events within 60 minutes
    df = label_event_next60(df, window_minutes=60)

    # 4) Build features
    df_feat, feature_cols = build_features(df)

    # 5) Train calibrated logistic regression
    model, scaler, splits, metrics = train_calibrated_logistic(df_feat, feature_cols)

    print("Logistic regression performance:")
    print("AUC train:", metrics["auc_train"])
    print("AUC val:", metrics["auc_val"])
    print("AUC test:", metrics["auc_test"])
    print("Brier train:", metrics["brier_train"])
    print("Brier val:", metrics["brier_val"])
    print("Brier test:", metrics["brier_test"])

    # 6) Attach risk scores to full feature dataframe
    df_risk = attach_risk_scores(df_feat, feature_cols, model, scaler)

    # 7) Save for reviewers
    print("Saving risk scores and labels to", args.output_csv)
    df_risk.to_csv(args.output_csv, index=False, compression="infer")
    print("Done.")


if __name__ == "__main__":
    main()
