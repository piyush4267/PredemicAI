# -*- coding: utf-8 -*-
"""
data_pipeline.py
----------------
Loads and preprocesses JHU COVID-19 and OWID datasets.
Handles wide-to-long transformation, country aggregation,
and merging of primary and secondary datasets.
"""

import pandas as pd
import numpy as np


# ── Dataset URLs ──────────────────────────────────────────────────────────────

JHU_CONFIRMED_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

JHU_DEATHS_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_deaths_global.csv"
)

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)


# ── OWID columns to extract ───────────────────────────────────────────────────

OWID_DYNAMIC_COLS = [
    "location", "date",
    "stringency_index",
    "people_fully_vaccinated_per_hundred",
    "new_tests_smoothed_per_thousand",
    "reproduction_rate",
]

OWID_STATIC_COLS = [
    "location",
    "population",
    "population_density",
    "median_age",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "human_development_index",
]


def load_jhu_data():
    """
    Load JHU confirmed cases and deaths datasets.

    Returns
    -------
    df_confirmed : pd.DataFrame  (wide format)
    df_deaths    : pd.DataFrame  (wide format)
    """
    print("Loading JHU confirmed cases...")
    df_confirmed = pd.read_csv(JHU_CONFIRMED_URL)

    print("Loading JHU deaths...")
    df_deaths = pd.read_csv(JHU_DEATHS_URL)

    print(f"  JHU confirmed shape : {df_confirmed.shape}")
    print(f"  JHU deaths shape    : {df_deaths.shape}")
    return df_confirmed, df_deaths


def load_owid_data():
    """
    Load Our World in Data COVID-19 dataset.

    Returns
    -------
    owid_raw : pd.DataFrame
    """
    print("Loading OWID dataset...")
    owid_raw = pd.read_csv(OWID_URL, low_memory=False)
    print(f"  OWID shape : {owid_raw.shape}")
    return owid_raw


def jhu_wide_to_long(df, value_name):
    """
    Convert JHU wide-format time series to long format.
    Aggregates sub-national provinces to country level.

    Parameters
    ----------
    df         : pd.DataFrame  JHU wide format
    value_name : str           column name for values e.g. 'Confirmed'

    Returns
    -------
    df_country : pd.DataFrame  long format aggregated by country
    """
    df_long = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date",
        value_name=value_name,
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"])

    # Aggregate sub-national rows to country level
    df_country = (
        df_long
        .groupby(["Country/Region", "Date"])[value_name]
        .sum()
        .reset_index()
    )
    df_country = df_country.sort_values(
        ["Country/Region", "Date"]
    ).reset_index(drop=True)

    return df_country


def merge_jhu_datasets(df_confirmed_raw, df_deaths_raw):
    """
    Convert both JHU datasets to long format and merge.

    Returns
    -------
    df : pd.DataFrame  with columns [Country/Region, Date, Confirmed, Deaths]
    """
    df_cases  = jhu_wide_to_long(df_confirmed_raw, "Confirmed")
    df_deaths = jhu_wide_to_long(df_deaths_raw, "Deaths")

    df = df_cases.merge(df_deaths, on=["Country/Region", "Date"], how="left")
    print(f"  Merged JHU shape : {df.shape}")
    return df


def prepare_owid_features(owid_raw):
    """
    Extract dynamic (time-varying) and static (demographic) features from OWID.

    Returns
    -------
    owid_dynamic : pd.DataFrame  time-varying features per country per date
    owid_static  : pd.DataFrame  one row per country with demographic features
    """
    # Filter to available columns
    dynamic_cols = [c for c in OWID_DYNAMIC_COLS if c in owid_raw.columns]
    static_cols  = [c for c in OWID_STATIC_COLS  if c in owid_raw.columns]

    owid_dynamic = owid_raw[dynamic_cols].copy()
    owid_dynamic["date"] = pd.to_datetime(owid_dynamic["date"])

    # Static demographics — last non-null value per country
    owid_static = (
        owid_raw[static_cols]
        .groupby("location")
        .last()
        .reset_index()
    )

    return owid_dynamic, owid_static


def merge_with_owid(df, owid_dynamic, owid_static):
    """
    Merge JHU data with OWID dynamic and static features.
    Uses left join to preserve all JHU records.
    Forward-fills dynamic columns within each country in time order.

    Parameters
    ----------
    df           : pd.DataFrame  JHU long format
    owid_dynamic : pd.DataFrame  time-varying OWID features
    owid_static  : pd.DataFrame  demographic OWID features

    Returns
    -------
    df : pd.DataFrame  enriched with OWID features
    """
    # Merge dynamic features
    df = df.merge(
        owid_dynamic,
        left_on=["Country/Region", "Date"],
        right_on=["location", "date"],
        how="left",
    )
    df = df.drop(columns=["location", "date"], errors="ignore")

    # Merge static demographic features
    df = df.merge(
        owid_static,
        left_on="Country/Region",
        right_on="location",
        how="left",
    )
    df = df.drop(columns=["location"], errors="ignore")

    # Sort before forward-fill to prevent future data leaking backwards
    df = df.sort_values(["Country/Region", "Date"])

    dynamic_fill = [
        "stringency_index",
        "people_fully_vaccinated_per_hundred",
        "new_tests_smoothed_per_thousand",
        "reproduction_rate",
    ]
    dynamic_fill = [c for c in dynamic_fill if c in df.columns]

    df[dynamic_fill] = (
        df.groupby("Country/Region")[dynamic_fill]
        .transform(lambda x: x.ffill())
    )
    df[dynamic_fill] = df[dynamic_fill].fillna(0)

    # Fill static demographic NaNs with median
    static_num_cols = [
        "population", "population_density", "median_age",
        "aged_65_older", "gdp_per_capita",
        "hospital_beds_per_thousand", "human_development_index",
    ]
    for col in static_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Final merged shape : {df.shape}")
    return df


def load_and_prepare_all():
    """
    Full pipeline: load both datasets, transform, merge, and return.

    Returns
    -------
    df : pd.DataFrame  ready for feature engineering
    """
    df_confirmed_raw, df_deaths_raw = load_jhu_data()
    owid_raw = load_owid_data()

    df = merge_jhu_datasets(df_confirmed_raw, df_deaths_raw)
    owid_dynamic, owid_static = prepare_owid_features(owid_raw)
    df = merge_with_owid(df, owid_dynamic, owid_static)

    print("\nData pipeline complete.")
    print(f"Countries : {df['Country/Region'].nunique()}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Records   : {len(df):,}")
    return df


if __name__ == "__main__":
    df = load_and_prepare_all()
    print(df.head())
