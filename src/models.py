# -*- coding: utf-8 -*-
"""
models.py
---------
Model training, evaluation, and SIR compartmental modeling.

Contains:
  - train_random_forest()   : RF regressor with time-based split
  - train_xgboost()         : XGBoost regressor
  - train_risk_classifier() : RF multi-class outbreak risk classifier
  - sir_model()             : SIR differential equations
  - detect_waves()          : epidemic wave detection via peak finding
  - fit_sir_wave()          : fit SIR to a single epidemic wave

Model selection rationale:
  Random Forest chosen over LSTM because:
  - Dataset is wide (21 features) relative to per-country length
  - Time series is non-stationary (multiple waves, different parameters)
  - RF gives interpretable feature importance for biological insight
  - LSTM would require substantially more data per country

Time-based split is NON-NEGOTIABLE for time series:
  - Random split = data leakage (future rows seen during training)
  - We train on pre-2022 data, test on 2022 onwards
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — GradientBoostingRegressor will be used as fallback.")

# ── Train/test split cutoff ───────────────────────────────────────────────────
CUTOFF_DATE = pd.Timestamp("2022-01-01")


# ══════════════════════════════════════════════════════════════════════════════
# ML FORECASTING MODELS
# ══════════════════════════════════════════════════════════════════════════════

def time_based_split(model_df, feature_cols, target_col="Target_7d"):
    """
    Split dataset by date — not randomly.
    Training: all data before CUTOFF_DATE
    Testing : all data from CUTOFF_DATE onwards

    Parameters
    ----------
    model_df     : pd.DataFrame  filtered dataset
    feature_cols : list of str   feature column names
    target_col   : str           target column name

    Returns
    -------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    train_df = model_df[model_df["Date"] < CUTOFF_DATE]
    test_df  = model_df[model_df["Date"] >= CUTOFF_DATE]

    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols].fillna(0).astype(float)
    y_test  = test_df[target_col]

    print(f"Train samples : {len(X_train):,}  (up to {CUTOFF_DATE.date()})")
    print(f"Test samples  : {len(X_test):,}   ({CUTOFF_DATE.date()} onwards)")

    return X_train, X_test, y_train, y_test, train_df, test_df


def evaluate_regressor(y_true, y_pred, model_name="Model"):
    """Print and return regression evaluation metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{model_name:25s} | MAE: {mae:>12,.0f} | RMSE: {rmse:>12,.0f} | R²: {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest regressor for 7-day case forecasting.

    Returns
    -------
    model   : fitted RandomForestRegressor
    metrics : dict  {mae, rmse, r2}
    """
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    # Convert to numpy to avoid XGBoost/pandas dtype compatibility issues
    model.fit(X_train.to_numpy(), y_train)
    preds   = model.predict(X_test.to_numpy())
    metrics = evaluate_regressor(y_test, preds, "Random Forest")
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost (or GradientBoosting fallback) regressor.

    Returns
    -------
    model      : fitted model
    metrics    : dict  {mae, rmse, r2}
    model_name : str
    """
    X_tr = X_train.to_numpy()
    X_te = X_test.to_numpy()

    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)
        model_name = "XGBoost"
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        print("\nTraining GradientBoostingRegressor (XGBoost not available)...")
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
        )
        model.fit(X_tr, y_train)
        model_name = "GradientBoosting"

    preds   = model.predict(X_te)
    metrics = evaluate_regressor(y_test, preds, model_name)
    return model, metrics, model_name


def train_risk_classifier(df, feature_cols, cutoff_date=CUTOFF_DATE):
    """
    Train Random Forest multi-class risk classifier.
    Classes: 0=Low, 1=Medium, 2=High, 3=Critical

    Returns
    -------
    clf        : fitted RandomForestClassifier
    clf_report : str  classification report
    """
    print("\nTraining Risk Classifier...")
    clf_df    = df[feature_cols + ["Risk_Label", "Date"]].dropna()
    clf_train = clf_df[clf_df["Date"] < cutoff_date]
    clf_test  = clf_df[clf_df["Date"] >= cutoff_date]

    X_tr = clf_train[feature_cols].fillna(0).astype(float).to_numpy()
    X_te = clf_test[feature_cols].fillna(0).astype(float).to_numpy()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, clf_train["Risk_Label"])
    preds = clf.predict(X_te)

    RISK_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    report = classification_report(
        clf_test["Risk_Label"], preds,
        target_names=[RISK_NAMES[i] for i in range(4)],
    )
    print("Risk Classifier Report:")
    print(report)
    return clf, report


# ══════════════════════════════════════════════════════════════════════════════
# SIR COMPARTMENTAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def sir_model(y, t, beta, gamma):
    """
    SIR differential equations.

    S → Susceptible (can be infected)
    I → Infected    (currently infectious)
    R → Recovered   (immune or deceased)

    dS/dt = -β × S × I / N
    dI/dt =  β × S × I / N - γ × I
    dR/dt =  γ × I

    R₀ = β / γ  (basic reproduction number)
    """
    S, I, R = y
    N = S + I + R
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]


def detect_waves(country_df, prominence=0.3, distance=30):
    """
    Detect epidemic waves using peak detection on 7-day smoothed cases.

    Why per-wave SIR instead of full-timeline:
    Full-timeline fitting averages over multiple waves with different β/γ,
    yielding R₀ ≈ 1. Wave-1 fitting isolates the biological transmission
    rate before interventions changed behaviour → realistic R₀ = 2.5–3.0.

    Parameters
    ----------
    country_df : pd.DataFrame  single country data with Cases_7d_Avg
    prominence : float         minimum peak prominence (fraction of max)
    distance   : int           minimum days between peaks

    Returns
    -------
    starts : list of int  wave start indices
    ends   : list of int  wave end indices
    peaks  : np.array     peak indices
    """
    cases = country_df["Cases_7d_Avg"].fillna(0).values
    if len(cases) < 30:
        return [0], [len(cases) - 1], np.array([])

    max_val = cases.max()
    if max_val == 0:
        return [0], [len(cases) - 1], np.array([])

    norm   = cases / max_val
    peaks, _ = find_peaks(norm, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        return [0], [len(cases) - 1], np.array([])

    starts, ends = [], []
    for i, peak in enumerate(peaks):
        starts.append(peaks[i - 1] if i > 0 else 0)
        ends.append(peaks[i + 1] if i < len(peaks) - 1 else len(cases) - 1)

    return starts, ends, peaks


def fit_sir_wave(country, df, population, wave_idx=0):
    """
    Fit SIR model to a specific epidemic wave using scipy curve_fit.

    Parameters
    ----------
    country    : str    country name
    df         : pd.DataFrame  full dataset
    population : float  country population
    wave_idx   : int    which wave to fit (0 = first wave)

    Returns
    -------
    dict with keys: beta, gamma, r0, wave_df, country
    or None if fitting fails
    """
    country_df = df[
        (df["Country/Region"] == country) & (df["Confirmed"] > 100)
    ].sort_values("Date").reset_index(drop=True)

    if len(country_df) < 30:
        return None

    starts, ends, _ = detect_waves(country_df)
    if wave_idx >= len(starts):
        wave_idx = 0

    ws, we   = starts[wave_idx], ends[wave_idx]
    wave_df  = country_df.iloc[ws:we + 1].reset_index(drop=True)

    if len(wave_df) < 14:
        return None

    N      = population
    I0     = max(float(wave_df["New_Cases"].iloc[0]), 1)
    R0_est = float(wave_df["Confirmed"].iloc[0])
    S0     = N - I0 - R0_est
    t      = np.arange(len(wave_df))
    I_obs  = wave_df["Cases_7d_Avg"].fillna(0).values

    def ode_fit(t_arr, beta, gamma):
        sol = odeint(sir_model, [S0, I0, R0_est], t_arr, args=(beta, gamma))
        return sol[:, 1]

    try:
        popt, _ = curve_fit(
            ode_fit, t, I_obs,
            p0=[0.3, 0.1],
            bounds=([0.05, 0.01], [2.0, 0.9]),
            maxfev=8000,
        )
        beta_fit, gamma_fit = popt
        r0 = beta_fit / gamma_fit
        return {
            "country": country,
            "beta": beta_fit,
            "gamma": gamma_fit,
            "r0": r0,
            "wave_df": wave_df,
            "S0": S0, "I0": I0, "R0_init": R0_est,
        }
    except Exception:
        return None


def fit_sir_multiple_countries(countries_pop, df):
    """
    Fit SIR wave model for multiple countries.

    Parameters
    ----------
    countries_pop : dict  {country_name: population}
    df            : pd.DataFrame

    Returns
    -------
    results : dict  {country_name: sir_result_dict}
    """
    results = {}
    print("\nFitting SIR wave models:")
    for country, pop in countries_pop.items():
        result = fit_sir_wave(country, df, pop, wave_idx=0)
        if result:
            results[country] = result
            tag = "Explosive" if result["r0"] > 3 else \
                  "High"      if result["r0"] > 2 else \
                  "Moderate"  if result["r0"] > 1.5 else "Low"
            print(f"  {country:20s}  R\u2080 = {result['r0']:.2f}  [{tag}]")
        else:
            print(f"  {country:20s}  SIR fit did not converge")
    return results


if __name__ == "__main__":
    from data_pipeline import load_and_prepare_all
    from feature_engineering import engineer_features, get_feature_list, add_risk_labels

    df = load_and_prepare_all()
    df = engineer_features(df)
    df = add_risk_labels(df)

    feature_cols = get_feature_list(df)
    model_df     = df[feature_cols + ["Target_7d", "Date", "Country/Region"]].dropna()
    model_df     = model_df[model_df["Confirmed"] > 100]

    X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
        model_df, feature_cols
    )
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    POPS = {"India": 1.38e9, "US": 3.31e8, "Brazil": 2.12e8}
    sir_results = fit_sir_multiple_countries(POPS, df)
