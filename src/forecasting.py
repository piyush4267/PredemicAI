# -*- coding: utf-8 -*-
"""
forecasting.py
--------------
30-day iterative forecast with Monte Carlo confidence bands.

How it works:
  1. Take the last known data row for a country
  2. Feed into the trained model → predict 7 days ahead
  3. Update the row with the prediction as new "current"
     (shift all lag features forward, recalculate Rₜ and growth rate)
  4. Repeat 4× → 28-day projection
  5. Run 50 times with ±2% feature noise → confidence band

Limitation:
  Each step compounds prediction error. Confidence band widens
  significantly at day 21+. Treat as directional, not point-exact.
"""

import numpy as np
import pandas as pd


def _update_row(row, pred_conf, run_conf):
    """
    Update a feature row after a prediction step.
    Shifts lag features forward and recalculates derived metrics.

    Parameters
    ----------
    row       : pd.Series  current feature row
    pred_conf : float      predicted cumulative confirmed cases
    run_conf  : float      current cumulative confirmed cases

    Returns
    -------
    updated row : pd.Series
    """
    pred_conf = max(pred_conf, run_conf)   # cumulative cases cannot decrease
    pred_new  = max(pred_conf - run_conf, 0)

    row = row.copy()
    row["Confirmed"]      = pred_conf
    row["Cases_Lag_14d"]  = row.get("Cases_Lag_7d",  pred_new)
    row["Cases_Lag_7d"]   = row.get("Cases_Lag_3d",  pred_new)
    row["Cases_Lag_3d"]   = row.get("Cases_Lag_1d",  pred_new)
    row["Cases_Lag_1d"]   = pred_new
    row["New_Cases"]      = pred_new
    row["Cases_7d_Avg"]   = pred_new

    # Recalculate growth rate
    if run_conf > 0:
        row["Growth_Rate_7d"] = float(
            np.clip((pred_new / (run_conf * 7 / 30 + 1)), -1, 10)
        )

    # Recalculate Rₜ proxy
    row["Rt_Proxy"] = float(
        np.clip((pred_new / (row.get("Cases_7d_Avg", 1) + 1)), 0, 5)
    )

    return row, pred_conf


def forecast_single_run(last_row, feature_cols, model, n_days=28, noise_scale=0.0):
    """
    Single iterative forecast run.

    Parameters
    ----------
    last_row     : pd.Series  last known data row for a country
    feature_cols : list       feature column names (must match model training)
    model        : fitted sklearn/xgboost model
    n_days       : int        number of days to forecast (must be multiple of 7)
    noise_scale  : float      fractional noise added to features (0 = clean run)

    Returns
    -------
    predicted_values : list of float  cumulative case predictions at each step
    """
    run_row  = last_row.copy()
    run_conf = float(last_row["Confirmed"])
    values   = []

    for _ in range(0, n_days, 7):
        fv = run_row[feature_cols].fillna(0).astype(float).values
        if noise_scale > 0:
            fv = fv * (1 + np.random.normal(0, noise_scale, len(fv)))
        pred_conf = float(model.predict(fv.reshape(1, -1))[0])
        values.append(pred_conf)
        run_row, run_conf = _update_row(run_row, pred_conf, run_conf)

    return values


def forecast_country(country, df, model, feature_cols, n_days=28, n_runs=50):
    """
    30-day iterative forecast with Monte Carlo confidence band.

    Parameters
    ----------
    country      : str          country name
    df           : pd.DataFrame full dataset
    model        : fitted model
    feature_cols : list         exact feature list used at training time
    n_days       : int          forecast horizon in days (default 28)
    n_runs       : int          Monte Carlo runs for confidence band

    Returns
    -------
    dict with keys:
        dates    : list of pd.Timestamp  forecast dates
        median   : np.array              median prediction per step
        lower    : np.array              10th percentile (lower CI)
        upper    : np.array              90th percentile (upper CI)
        last_date: pd.Timestamp          last historical date
        last_val : float                 last historical confirmed count
    or None if country not found
    """
    country_df = df[df["Country/Region"] == country].sort_values("Date")
    if len(country_df) == 0:
        print(f"  {country} not found in dataset")
        return None

    last_row  = country_df.iloc[-1].copy()
    last_date = last_row["Date"]
    last_val  = float(last_row["Confirmed"])

    all_runs = []
    for run in range(n_runs):
        noise = 0.02 if run > 0 else 0.0   # first run is clean median
        values = forecast_single_run(
            last_row, feature_cols, model,
            n_days=n_days, noise_scale=noise
        )
        all_runs.append(values)

    runs_arr = np.array(all_runs)   # shape: (n_runs, n_steps)
    steps    = list(range(0, n_days, 7))
    dates    = [last_date + pd.Timedelta(days=s + 7) for s in steps]

    return {
        "country":   country,
        "dates":     dates,
        "median":    np.median(runs_arr, axis=0),
        "lower":     np.percentile(runs_arr, 10, axis=0),
        "upper":     np.percentile(runs_arr, 90, axis=0),
        "last_date": last_date,
        "last_val":  last_val,
    }


def forecast_multiple_countries(countries, df, model, feature_cols, n_days=28):
    """
    Run forecast for multiple countries and print summary.

    Parameters
    ----------
    countries    : list of str  country names
    df           : pd.DataFrame
    model        : fitted model
    feature_cols : list
    n_days       : int

    Returns
    -------
    results : dict  {country_name: forecast_dict}
    """
    results = {}
    print(f"\n30-Day Forecasts ({n_days} days, 80% CI):")
    print("-" * 65)

    for country in countries:
        result = forecast_country(country, df, model, feature_cols, n_days)
        if result:
            results[country] = result
            last = result["last_val"]
            med  = result["median"][-1]
            lo   = result["lower"][-1]
            hi   = result["upper"][-1]
            chg  = ((med - last) / (last + 1)) * 100
            print(
                f"  {country:15s}: {last/1e6:.2f}M -> {med/1e6:.2f}M "
                f"(+{chg:.1f}%) | CI: [{lo/1e6:.2f}M - {hi/1e6:.2f}M]"
            )

    return results


if __name__ == "__main__":
    from data_pipeline import load_and_prepare_all
    from feature_engineering import engineer_features, get_feature_list, add_risk_labels
    from models import (
        time_based_split, train_random_forest
    )

    df = load_and_prepare_all()
    df = engineer_features(df)
    df = add_risk_labels(df)

    feature_cols = get_feature_list(df)
    model_df     = df[feature_cols + ["Target_7d", "Date", "Country/Region"]].dropna()
    model_df     = model_df[model_df["Confirmed"] > 100]

    X_train, X_test, y_train, y_test, _, _ = time_based_split(
        model_df, feature_cols
    )
    rf_model, _ = train_random_forest(X_train, y_train, X_test, y_test)

    results = forecast_multiple_countries(
        ["India", "US", "Brazil"], df, rf_model, feature_cols
    )
