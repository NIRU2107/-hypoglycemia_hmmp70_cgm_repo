# HMM-based P70 Hypoglycemia Risk Model

This repository implements a **Gaussian HMM-based hypoglycemia risk model** that estimates, at each CGM timepoint, the probability that current glucose is at or below 70 mg/dL (P70_t), and labels imminent hypoglycemia within the next 60 minutes.

## Overview

The pipeline:

1. Loads CGM data with columns:
   - `dt`: timestamp
   - `glucose`: CGM glucose (mg/dL)
   - `patient_pid`: patient identifier

2. Computes 5-minute glucose deltas per patient:
   - `glucose_delta_5m = glucose_t - glucose_{t-1}`

3. Trains a **4-state Gaussian HMM** on 2D observations:
   - `x_t = [glucose_t, glucose_delta_5m_t]`

4. For each hidden state `s`, computes:
   - `p70(s) = P(glucose <= 70 | z_t = s)`
     using the marginal normal distribution of the glucose dimension.

5. Uses the **forward algorithm** to compute **filtering posteriors**:
   - `gamma_t(s) = P(z_t = s | x_{1:t})`

6. Defines the real-time HMM P70 risk as:
   - `P70_t = sum_s gamma_t(s) * p70(s)`

7. Labels **sustained hypoglycemia events**:
   - Hypoglycemia event: `glucose <= 70 mg/dL` sustained for ≥ 15 minutes
     (≥ 3 contiguous 5-minute samples).
   - `event_onset = True` at the first sample of each such run.

8. Labels **imminent events within 60 minutes**:
   - `event_next60 = True` at time `t` if the next `event_onset` occurs within the next 60 minutes.

9. Outputs a merged CSV with:
   - CGM values
   - Event labels (`is_hypo`, `event_onset`, `event_next60`)
   - HMM risk score (`P70_t`)

## Usage
## Logistic Regression Risk Model (Imminent Hypoglycemia, 60 Minutes)

The script `logistic_risk_pipeline.py` trains a **calibrated logistic regression** model to predict imminent hypoglycemia within the next 60 minutes.

Outcome:
- `event_next60 = 1` if a sustained hypoglycemia event (glucose <= 70 mg/dL for >= 15 min) will begin within the next 60 minutes.
- `event_next60 = 0` otherwise.

Features per timepoint:
- Current CGM: `glucose`
- Short-term dynamics:
  - `glucose_delta_5m`
  - `delta_15m` (t - t-3)
  - `delta_30m` (t - t-6)
- Volatility:
  - `roll_std_30m` (6-sample rolling std)
  - `roll_std_60m` (12-sample rolling std)
- Temporal encodings:
  - `hour_sin`, `hour_cos` (cyclic hour-of-day)
  - `dow` (day-of-week)

Model:
- Logistic regression with `lbfgs` solver.
- Calibration via Platt scaling (`CalibratedClassifierCV`, method="sigmoid", 3-fold CV).
- Patient-wise train/val/test splits to avoid information leakage.

** Ensemble Overview**

This project combines two complementary risk models—a Gaussian HMM and a logistic regression model—into a single stacked ensemble** that predicts imminent hypoglycemia within the next 60 minutes. At each CGM timepoint, the HMM produces a probabilistic risk score based on latent glucose dynamics, and the logistic regression model produces a calibrated risk score based on tabular features (current CGM level, short-term deltas, recent volatility, and temporal encodings). These two base risk scores are then fed into a shallow logistic regression **meta-learner** that learns how to optimally weight and combine them into a single ensemble risk score.

To avoid optimistic bias, we use **patient-level stacking**. Patients are split into K folds. For each fold, the base-model predictions used to train the meta-learner come only from patients whose data were held out during base-model training. Concretely, for each fold we (1) train the meta-learner on all other folds’ base-model scores and labels, and (2) generate out-of-fold ensemble predictions for the held-out patients. This yields an unbiased, cross-validated view of ensemble performance on the internal cohort, while still allowing us to fit a final meta-learner on all internal patients once the stacking process is complete.

All models (HMM, logistic regression base model, and the stacked ensemble meta-learner) are **trained and tuned exclusively on the internal cohort**. The external cohort is never used during model training or selection; it is reserved strictly for independent validation of the final ensemble. On the external cohort, we apply the fixed, trained meta-learner to the HMM and logistic base risk scores to obtain ensemble risks and evaluate generalization.



Top layer: **Data and Base Models**  
CGM time series enter two parallel branches: one into the HMM (producing `hmm_risk`) and one into the logistic regression model (producing `lr_risk`). Each branch outputs a calibrated probability of imminent hypoglycemia within 60 minutes.

Middle layer: **Patient-Level Stacking**  
Patients are split into K folds. For each fold, we use base-model scores from the other K−1 folds to train a shallow logistic regression meta-learner on `[hmm_risk, lr_risk] → event_next60`, then generate out-of-fold ensemble predictions for the held-out patients. This produces unbiased ensemble predictions across all internal patients.

Bottom layer: **Final Ensemble and External Validation**  
Using all internal patients’ base-model scores and labels, we refit the meta-learner once to obtain the final stacked model. This final model is then applied to the external cohort’s base-model scores to generate ensemble risk scores for independent validation.
