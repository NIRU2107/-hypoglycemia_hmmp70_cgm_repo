"""
Stacked Ensemble: HMM + Logistic Regression with Patient-Level Stacking.

This script:
1. Loads a per-time CGM dataset that already includes:
   - dt
   - patient_pid
   - event_next60     (primary outcome: imminent hypoglycemia within 60 minutes)
   - hmm_risk         (base HMM risk score, e.g. P70_t)
   - lr_risk          (base logistic regression risk score)

2. Performs patient-level K-fold cross-validation to generate
   OUT-OF-FOLD (OOF) base-model predictions for stacking:
   - For each fold, train shallow logistic meta-learner on all other folds,
     using [hmm_risk, lr_risk] to predict event_next60.
   - Predict on the held-out patients to obtain OOF ensemble predictions.

3. Trains a final logistic regression meta-learner on ALL patients
   using the full base-model predictions.

4. Evaluates the ensemble on:
   - Internal cohort (cross-validated metrics using OOF predictions).
   - Optionally, an external cohort (if provided).

5. Saves out:
   - A CSV with per-time predictions and labels for the internal cohort.
   - Model coefficients for the meta-learner.

Dependencies:
    pandas
    numpy
    scikit-learn

Usage examples:

    # Internal cohort only:
    python stacked_ensemble_pipeline.py \
        --internal_csv internal_base_risks.csv.gz \
        --output_csv internal_stacked_ensemble_predictions.csv.gz

    # Internal + external:
    python stacked_ensemble_pipeline.py \
        --internal_csv internal_base_risks.csv.gz \
        --external_csv external_base_risks.csv.gz \
        --output_csv internal_stacked_ensemble_predictions.csv.gz \
        --external_output_csv external_stacked_ensemble_predictions.csv.gz
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

RANDOM_STATE = 42
N_FOLDS = 5

def load_base_risks(path):
    """
    Load data containing:
        dt
        patient_pid
        event_next60 (0/1)
        hmm_risk
        lr_risk

    Returns:
        df: DataFrame sorted by patient_pid, dt
    """
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["patient_pid", "dt"]).reset_index(drop=True)

    required_cols = ["dt", "patient_pid", "event_next60", "hmm_risk", "lr_risk"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df

def patient_level_folds(pids, n_splits=5, random_state=42):
    """
    Generate patient-level folds using KFold on unique patient IDs.

    Args:
        pids: array-like of patient IDs
        n_splits: number of folds
        random_state: seed

    Returns:
        list of (train_pids, val_pids) tuples.
    """
    unique_pids = np.unique(pids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in kf.split(unique_pids):
        train_p = unique_pids[train_idx]
        val_p = unique_pids[val_idx]
        folds.append((train_p, val_p))

    return folds

def fit_meta_learner(X, y):
    """
    Fit a shallow logistic regression meta-learner.

    Args:
        X: numpy array (n_samples, 2) -> [hmm_risk, lr_risk]
        y: labels (0/1)

    Returns:
        model: fitted LogisticRegression
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    clf.fit(X, y)
    return clf

def generate_oof_ensemble_predictions(df_internal, n_folds=N_FOLDS):
    """
    Perform patient-level stacking to generate OOF ensemble predictions
    on the internal cohort.

    Args:
        df_internal: DataFrame with columns:
            patient_pid, event_next60, hmm_risk, lr_risk

    Returns:
        df_oof: same as df_internal with an added column:
            ensemble_oof (float) – OOF prediction of meta-learner
        meta_models: list of meta-learners (one per fold)
    """
    df = df_internal.copy()
    df = df.sort_values(["patient_pid", "dt"]).reset_index(drop=True)

    pids = df["patient_pid"].values
    folds = patient_level_folds(pids, n_splits=n_folds, random_state=RANDOM_STATE)

    df["ensemble_oof"] = np.
    meta_models = []

    for fold_idx, (train_p, val_p) in enumerate(folds):
        print(f"Fold {fold_idx + 1}/{n_folds}")

        train_mask = df["patient_pid"].isin(train_p)
        val_mask = df["patient_pid"].isin(val_p)

        df_train = df[train_mask]
        df_val = df[val_mask]

        X_train = df_train[["hmm_risk", "lr_risk"]].values
        y_train = df_train["event_next60"].astype(int).values

        X_val = df_val[["hmm_risk", "lr_risk"]].values

        meta_model = fit_meta_learner(X_train, y_train)
        meta_models.append(meta_model)

        df.loc[val_mask, "ensemble_oof"] = meta_model.predict_proba(X_val)[:, 1]

    return df, meta_models

def train_final_meta_learner(df_internal):
    """
    Train a final meta-learner on ALL internal data.

    Args:
        df_internal: DataFrame with hmm_risk, lr_risk, event_next60

    Returns:
        final_model: LogisticRegression fitted on all samples
    """
    X_all = df_internal[["hmm_risk", "lr_risk"]].values
    y_all = df_internal["event_next60"].astype(int).values
    final_model = fit_meta_learner(X_all, y_all)
    return final_model

def evaluate_predictions(y_true, p_pred, name="model"):
    """
    Compute AUC and Brier score.

    Args:
        y_true: ground-truth labels (0/1)
        p_pred: predicted probabilities
        name: label for printing

    Returns:
        metrics: dict with auc, brier
    """
    auc = roc_auc_score(y_true, p_pred)
    brier = brier_score_loss(y_true, p_pred)
    print(f"{name}: AUC={auc:.3f}, Brier={brier:.4f}")
    return {"auc": auc, "brier": brier}

def apply_meta_model(df, model, col_name="ensemble_risk"):
    """
    Apply the final meta-learner to a dataframe.

    Args:
        df: DataFrame with hmm_risk, lr_risk
        model: LogisticRegression meta-learner
        col_name: output column name

    Returns:
        df_out: copy of df with new column col_name
    """
    df_out = df.copy()
    X = df_out[["hmm_risk", "lr_risk"]].values
    df_out[col_name] = model.predict_proba(X)[:, 1]
    return df_out

def main():
    parser = argparse.ArgumentParser(description="Stacked ensemble of HMM and logistic regression risks.")
    parser.add_argument(
        "--internal_csv",
        required=True,
        help="Internal cohort CSV (dt, patient_pid, event_next60, hmm_risk, lr_risk, ...)",
    )
    parser.add_argument(
        "--external_csv",
        default=None,
        help="Optional external cohort CSV with same columns for independent validation.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV for internal cohort with OOF and final ensemble predictions.",
    )
    parser.add_argument(
        "--external_output_csv",
        default=None,
        help="Optional output CSV for external cohort with ensemble predictions.",
    )
    args = parser.parse_args()

    # 1) Load internal base risks
    df_int = load_base_risks(args.internal_csv)

    # 2) Generate OOF ensemble predictions via patient-level stacking
    df_int_oof, meta_models = generate_oof_ensemble_predictions(df_int, n_folds=N_FOLDS)

    # 3) Evaluate internal performance using OOF ensemble predictions
    mask_oof = df_int_oof["ensemble_oof"].notna()
    y_int = df_int_oof.loc[mask_oof, "event_next60"].astype(int).values
    p_int_oof = df_int_oof.loc[mask_oof, "ensemble_oof"].values
    internal_metrics = evaluate_predictions(y_int, p_int_oof, name="Stacked Ensemble (OOF internal)")

    # 4) Train final meta-learner on all internal data
    final_meta = train_final_meta_learner(df_int)

    # 5) Apply final ensemble to internal cohort (for clean final scores)
    df_int_final = apply_meta_model(df_int_oof, final_meta, col_name="ensemble_risk")

    # 6) Evaluate internal again using final model (optional)
    y_int_all = df_int_final["event_next60"].astype(int).values
    p_int_final = df_int_final["ensemble_risk"].values
    final_metrics = evaluate_predictions(y_int_all, p_int_final, name="Stacked Ensemble (final internal)")

    # 7) Save internal predictions
    print("Saving internal stacked ensemble predictions to", args.output_csv)
    df_int_final.to_csv(args.output_csv, index=False, compression="infer")

    # 8) If external cohort provided, apply the final meta-learner
    if args.external_csv is not None and args.external_output_csv is not None:
        print("Loading external cohort from", args.external_csv)
        df_ext = load_base_risks(args.external_csv)
        df_ext_pred = apply_meta_model(df_ext, final_meta, col_name="ensemble_risk")

        # If external has labels, evaluate
        if "event_next60" in df_ext_pred.columns:
            y_ext = df_ext_pred["event_next60"].astype(int).values
            p_ext = df_ext_pred["ensemble_risk"].values
            _ = evaluate_predictions(y_ext, p_ext, name="Stacked Ensemble (external)")

        print("Saving external stacked ensemble predictions to", args.external_output_csv)
        df_ext_pred.to_csv(args.external_output_csv, index=False, compression="infer")

    print("Done.")

if __name__ == "__main__":
    main()
