# -*- coding: utf-8 -*-
"""
feature_engineering.py
-----------------------
Creates epidemiologically meaningful features for outbreak prediction.

Every feature has a biological justification:
- Rₜ proxy     : fundamental epidemic control parameter (WHO standard)
- Growth rate  : acceleration of spread — key outbreak signal
- Doubling time: log(2)/log(1+r) — standard public health threshold
- CFR          : case fatality rate — healthcare stress indicator
- Lag features : capture incubation period dynamics (COVID median = 5 days)
- 7-day avg    : standard epidemiological smoothing for reporting noise
"""

import numpy as np
import pandas as pd


# ── Feature configuration ─────────────────────────────────────────────────────

LAG_DAYS = [1, 3, 7, 14]   # incubation-aligned lag windows

FORECAST_FEATURES = [
    "Confirmed", "New_Cases", "Cases_7d_Avg",
    "Cases_Lag_1d", "Cases_Lag_3d", "Cases_Lag_7d", "Cases_Lag_14d",
    "Growth_Rate_7d", "Doubling_Time", "Rt_Proxy", "CFR",
    "Day_of_Week", "Month",
    "stringency_index", "people_fully_vaccinated_per_hundred",
    "new_tests_smoothed_per_thousand",
    "population_density", "median_age", "aged_65_older",
    "hospital_beds_per_thousand", "gdp_per_capita",
]


def engineer_features(df):
    """
    Add all epidemiological features to the dataset.

    Parameters
    ----------
    df : pd.DataFrame  merged JHU + OWID data

    Returns
    -------
    df : pd.DataFrame  with 21 new feature columns + targets
    """
    df = df.copy()
    grp = df.groupby("Country/Region")

    # ── Daily new cases and deaths ────────────────────────────────────────────
    df["New_Cases"]  = grp["Confirmed"].diff().clip(lower=0)
    df["New_Deaths"] = grp["Deaths"].diff().clip(lower=0)

    # ── 7-day rolling averages (standard epidemiological smoothing) ───────────
    # Removes day-of-week reporting artifacts common in COVID data
    df["Cases_7d_Avg"] = grp["New_Cases"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["Deaths_7d_Avg"] = grp["New_Deaths"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # ── Lag features (incubation period dynamics) ─────────────────────────────
    # COVID-19 median incubation = 5 days → 7d lag is biologically optimal
    # 14d lag covers the full contact-tracing window
    for lag in LAG_DAYS:
        df[f"Cases_Lag_{lag}d"] = grp["New_Cases"].transform(
            lambda x: x.shift(lag)
        )

    # ── Growth rate (7-day) ───────────────────────────────────────────────────
    # (today - 7 days ago) / (7 days ago + 1)
    # Positive = outbreak accelerating, negative = containment succeeding
    cases_7d_lag = grp["Cases_7d_Avg"].transform(lambda x: x.shift(7))
    df["Growth_Rate_7d"] = (
        (df["Cases_7d_Avg"] - cases_7d_lag) / (cases_7d_lag + 1)
    ).clip(-1, 10)

    # ── Doubling time (days) ──────────────────────────────────────────────────
    # log(2) / log(1 + daily_growth_rate)
    # 7-day doubling = public health emergency threshold (CDC standard)
    daily_growth = grp["Confirmed"].transform(
        lambda x: x.pct_change().clip(0, 2)
    )
    df["Doubling_Time"] = np.where(
        daily_growth > 0,
        np.log(2) / np.log(1 + daily_growth),
        np.nan,
    )
    df["Doubling_Time"] = df["Doubling_Time"].clip(1, 365)

    # ── Case Fatality Rate (CFR) ──────────────────────────────────────────────
    # Deaths / Confirmed — rising CFR signals healthcare system stress
    # or a shift to a more virulent variant
    df["CFR"] = (df["Deaths"] / (df["Confirmed"] + 1)).clip(0, 0.3)

    # ── Rₜ proxy (Effective Reproduction Number) ──────────────────────────────
    # 7-day avg cases / 7-day avg cases from 7 days prior
    # Rₜ > 1 → epidemic expanding, Rₜ < 1 → epidemic shrinking
    # Standard WHO outbreak monitoring metric
    df["Rt_Proxy"] = (
        df["Cases_7d_Avg"]
        / (grp["Cases_7d_Avg"].transform(lambda x: x.shift(7)) + 1)
    ).clip(0, 5)

    # ── Temporal features (day-of-week reporting patterns) ───────────────────
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"]       = df["Date"].dt.month

    # ── Forecast targets ──────────────────────────────────────────────────────
    # TARGET: cumulative confirmed cases 7 and 14 days from now
    df["Target_7d"]  = grp["Confirmed"].transform(lambda x: x.shift(-7))
    df["Target_14d"] = grp["Confirmed"].transform(lambda x: x.shift(-14))

    # Drop rows missing key features or targets
    df = df.dropna(subset=["Cases_Lag_14d", "Target_7d"])

    return df


def get_feature_list(df):
    """
    Return list of available features from FORECAST_FEATURES
    that actually exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    features : list of str
    """
    features = [f for f in FORECAST_FEATURES if f in df.columns]
    print(f"Features available : {len(features)} / {len(FORECAST_FEATURES)}")
    return features


def assign_risk_label(row):
    """
    Multi-factor outbreak risk classification.
    Mirrors WHO multi-indicator approach — avoids single-metric thresholds.

    Scoring:
        Rₜ proxy:          1pt (>1.0), 2pt (>1.5), 3pt (>2.0)
        Growth rate (7d):  1pt (>5%),  2pt (>20%), 3pt (>50%)
        Cases per million: 1pt (>20),  2pt (>100), 3pt (>500)

    Returns
    -------
    int : 0=Low, 1=Medium, 2=High, 3=Critical
    """
    rt  = row.get("Rt_Proxy", 1)
    gr  = row.get("Growth_Rate_7d", 0)
    avg = row.get("Cases_7d_Avg", 0)
    pop = max(row.get("population", 1e6), 1)
    cases_per_million = (avg / pop) * 1e6

    score = 0

    if rt > 2.0:        score += 3
    elif rt > 1.5:      score += 2
    elif rt > 1.0:      score += 1

    if gr > 0.5:        score += 3
    elif gr > 0.2:      score += 2
    elif gr > 0.05:     score += 1

    if cases_per_million > 500:    score += 3
    elif cases_per_million > 100:  score += 2
    elif cases_per_million > 20:   score += 1

    if score >= 7:      return 3   # Critical
    elif score >= 5:    return 2   # High
    elif score >= 2:    return 1   # Medium
    else:               return 0   # Low


RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def add_risk_labels(df):
    """
    Add Risk_Label column to dataframe using multi-factor scoring.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame  with Risk_Label column added
    """
    df = df.copy()
    df["Risk_Label"] = df.apply(assign_risk_label, axis=1)
    df["Risk_Label_Text"] = df["Risk_Label"].map(RISK_LABELS)
    print("Risk labels assigned.")
    print(df["Risk_Label_Text"].value_counts())
    return df


if __name__ == "__main__":
    from data_pipeline import load_and_prepare_all
    df_raw = load_and_prepare_all()
    df     = engineer_features(df_raw)
    df     = add_risk_labels(df)
    feats  = get_feature_list(df)
    print(f"\nSample features:\n{df[feats].head()}")
