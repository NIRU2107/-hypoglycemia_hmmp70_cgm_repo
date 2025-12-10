"""
HMM-based hypoglycemia risk model (P70_t) for CGM data.

This script:
1. Loads CGM data from a CSV (dt, glucose, patient_pid).
2. Computes 5-minute glucose deltas.
3. Trains a 4-state Gaussian HMM on [glucose, 5-minute delta].
4. Computes real-time filtering posteriors and per-time P(glucose <= 70) = P70_t.
5. Labels sustained hypoglycemia events (>= 15 minutes, i.e., >= 3 contiguous samples <= 70).
6. Constructs primary outcome labels: imminent hypoglycemia within the next 60 minutes.
7. Saves a merged file with CGM, labels, and HMM risk scores.

Dependencies:
- pandas
- numpy
- scipy
- tqdm
- hmmlearn

Usage:
    python hmm_p70_pipeline.py \
        --input_csv combined_cgm_fixed.csv0hio11.gz \
        --output_csv hmm_p70_series_with_events.csv.gz
"""

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import norm
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM


def load_cgm(path):
    """
    Load CGM data.

    Expected columns:
        dt: timestamp (string or datetime)
        glucose: CGM value (mg/dL)
        patient_pid: patient identifier

    Returns:
        df (DataFrame) with dt as datetime and sorted by patient_pid, dt.
    """
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["patient_pid", "dt"]).reset_index(drop=True)
    return df


def add_5min_delta(df):
    """
    Compute 5-minute glucose delta per patient.

    Adds:
        glucose_delta_5m: difference in glucose vs previous sample per patient.
    """
    df = df.copy()
    df["glucose_delta_5m"] = df.groupby("patient_pid")["glucose"].diff()
    return df


def fit_hmm(feature_df, n_states=4, random_state=0, n_iter=100):
    """
    Fit a Gaussian HMM on [glucose, glucose_delta_5m].

    Args:
        feature_df: DataFrame with columns ['dt', 'glucose', 'glucose_delta_5m', 'patient_pid']
                   (no NaNs in these columns).
        n_states: number of hidden states.
        random_state: RNG seed for reproducibility.
        n_iter: max EM iterations.

    Returns:
        hmm_model: fitted GaussianHMM instance.
        X: numpy array of shape (N, 2) with features.
        seq_lengths: array of sequence lengths per patient.
    """
    feature_df = feature_df.sort_values(["patient_pid", "dt"]).copy()
    X = feature_df[["glucose", "glucose_delta_5m"]].values
    seq_lengths = feature_df.groupby("patient_pid").size().values

    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )

    hmm_model.fit(X, lengths=seq_lengths)
    return hmm_model, X, seq_lengths


def compute_state_p70(hmm_model, threshold=70.0):
    """
    Compute state-wise probability of glucose <= threshold.

    Assumes emission is 2D Gaussian: [glucose, delta].
    Uses the marginal normal distribution of the glucose dimension.
    """
    means = hmm_model.means_          # shape (n_states, 2)
    covars = hmm_model.covars_        # shape (n_states, 2, 2)

    mu_glucose = means[:, 0]
    sigma_glucose = np.sqrt(covars[:, 0, 0])

    z = (threshold - mu_glucose) / sigma_glucose
    state_p70 = norm.cdf(z)
    return state_p70


def forward_filtering_posteriors(hmm_model, X, seq_lengths):
    """
    Compute filtering posteriors gamma_t(s) = P(z_t = s | x_1:t)
    using the forward algorithm in log-space.

    Args:
        hmm_model: fitted GaussianHMM.
        X: feature matrix (N, d), concatenated sequences.
        seq_lengths: sequence lengths per patient.

    Returns:
        gamma_all: (N, n_states) array of posteriors for each time, each state.
    """
    n_states = hmm_model.n_components
    N = X.shape[0]

    log_startprob = np.log(hmm_model.startprob_)
    log_transmat = np.log(hmm_model.transmat_)

    gamma_all = np.zeros((N, n_states))
    start_idx = 0

    for L in tqdm(seq_lengths, desc="Forward filtering by sequence"):
        end_idx = start_idx + L
        X_seq = X[start_idx:end_idx]

        # Log-likelihoods per state per time
        log_lik = hmm_model._compute_log_likelihood(X_seq)

        alpha_log = np.zeros_like(log_lik)

        # t = 0
        alpha_log[0] = log_startprob + log_lik[0]

        # Forward recursion
        for t in range(1, L):
            for j in range(n_states):
                alpha_log[t, j] = log_lik[t, j] + logsumexp(
                    alpha_log[t - 1] + log_transmat[:, j]
                )

        # Normalize to get posteriors gamma_t(s)
        for t in range(L):
            log_norm = logsumexp(alpha_log[t])
            gamma_all[start_idx + t] = np.exp(alpha_log[t] - log_norm)

        start_idx = end_idx

    return gamma_all


def label_hypo_events(df, threshold=70.0, min_points=3):
    """
    Label sustained hypoglycemia events per patient.

    Hypoglycemia definition:
        glucose <= threshold
        sustained for at least `min_points` contiguous samples.

    Adds:
        is_hypo: per-sample boolean
        run_id: ID of contiguous hypo/non-hypo runs per patient
        event_onset: True only for the first sample of runs that satisfy the
                     sustained hypoglycemia criterion.
    """
    df = df.sort_values(["patient_pid", "dt"]).copy()
    df["is_hypo"] = df["glucose"] <= threshold

    def _label_runs(group):
        run_id = (group["is_hypo"] != group["is_hypo"].shift()).cumsum()
        group["run_id"] = run_id
        return group

    df = df.groupby("patient_pid", group_keys=False).apply(_label_runs)

    # Run length (number of hypo samples in each run)
    run_lengths = df.groupby(["patient_pid", "run_id"])["is_hypo"].transform("sum")

    df["event_onset"] = False
    mask_event_runs = (df["is_hypo"]) and (run_lengths >= min_points)

    first_in_run = df.groupby(["patient_pid", "run_id"]).cumcount() == 0
    df.loc[mask_event_runs & first_in_run, "event_onset"] = True

    return df


def label_imminent_events(df, horizon_minutes=60):
    """
    For each sample, label whether a hypoglycemia event will start within
    the next `horizon_minutes`.

    Uses `event_onset` per patient.

    Adds:
        event_next60 (or event_next{horizon}): True if the next event_onset
        is within `horizon_minutes` of the current time.
    """
    df = df.sort_values(["patient_pid", "dt"]).copy()
    df["dt"] = pd.to_datetime(df["dt"])

    col_name = "event_next" + str(horizon_minutes)
    df[col_name] = False

    sixty = np.timedelta64(horizon_minutes, "m")

    for pid in tqdm(df["patient_pid"].unique(), desc="Labeling imminent events"):
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
            if event_times[j] - t <= sixty:
                labels[i] = True

        df.loc[idx, col_name] = labels

    return df


def main():
    parser = argparse.ArgumentParser(
        description="HMM-based P70 risk model for CGM data."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CGM CSV (e.g., combined_cgm_fixed.csv0hio11.gz).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV with P70_t and event labels.",
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=4,
        help="Number of hidden states in the Gaussian HMM (default: 4).",
    )

    args = parser.parse_args()

    print("Loading CGM data from", args.input_csv)
    df_cgm = load_cgm(args.input_csv)

    print("Computing 5-minute glucose deltas")
    df_cgm = add_5min_delta(df_cgm)

    # Prepare features for HMM (drop rows without delta)
    feature_df = df_cgm.dropna(subset=["glucose", "glucose_delta_5m"]).copy()

    print("Fitting Gaussian HMM with", args.n_states, "states")
    hmm_model, X, seq_lengths = fit_hmm(
        feature_df, n_states=args.n_states, random_state=0, n_iter=100
    )

    print("Computing state-wise P(glucose <= 70)")
    state_p70 = compute_state_p70(hmm_model, threshold=70.0)

    print("Running forward filtering for posteriors")
    gamma_all = forward_filtering_posteriors(hmm_model, X, seq_lengths)

    print("Computing P70_t as posterior-weighted mixture")
    P70_all = np.dot(gamma_all, state_p70)
    feature_df["P70_t"] = P70_all

    # Label events and imminent events on the full CGM dataframe
    print("Labeling sustained hypoglycemia events (event_onset)")
    cgm_events_df = label_hypo_events(
        df_cgm, threshold=70.0, min_points=3
    )

    print("Labeling imminent events within 60 minutes (event_next60)")
    cgm_events_df = label_imminent_events(cgm_events_df, horizon_minutes=60)

    # Merge P70_t back onto cgm_events_df
    print("Merging P70_t with event labels")
    p70_df = feature_df[
        ["dt", "patient_pid", "glucose", "glucose_delta_5m", "P70_t"]
    ].copy()

    merged_df = pd.merge(
        cgm_events_df[
            ["dt", "patient_pid", "glucose", "is_hypo", "event_onset", "event_next60"]
        ],
        p70_df,
        on=["dt", "patient_pid", "glucose"],
        how="left",
    )

    merged_df = merged_df.sort_values(["patient_pid", "dt"]).reset_index(drop=True)

    print("Saving output to", args.output_csv)
    merged_df.to_csv(args.output_csv, index=False, compression="infer")

    print("Done.")


if __name__ == "__main__":
    main()
