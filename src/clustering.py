# -*- coding: utf-8 -*-
"""
clustering.py
-------------
Hotspot detection via K-Means clustering on per-capita burden features.

Key improvement over naive clustering:
  Previous approach used absolute case counts → all clusters looked identical
  because large countries dominated.

  This implementation uses CASES PER MILLION as the primary feature, which:
  - Separates high-burden from low-burden countries correctly
  - Makes cluster assignments epidemiologically meaningful
  - Allows meaningful comparison across countries of different sizes

Cluster interpretation:
  Very High Burden → countries with highest cases/million + low vaccination
  High Burden      → significant outbreak, moderate vaccination
  Moderate Burden  → controlled outbreak, higher vaccination
  Low Burden       → minimal cases/million, high vaccination or early control
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Features used for clustering ─────────────────────────────────────────────
# Per-capita features are essential — absolute counts conflate
# population size with epidemic severity

CLUSTER_FEATURES = [
    "Peak_Growth_Rate",       # how fast did it spread at worst
    "Avg_Rt",                 # average transmission over the pandemic
    "Cases_Per_Million",      # total per-capita burden ← key feature
    "Peak_Cases_Per_Million", # acute per-capita severity at peak
    "Avg_Stringency",         # average policy response
    "Avg_Vaccination",        # vaccination coverage achieved
    "Population_Density",     # proxy for transmission rate β
    "Median_Age",             # proxy for infection fatality rate
]

CLUSTER_LABELS = {
    0: "Very High Burden",
    1: "High Burden",
    2: "Moderate Burden",
    3: "Low Burden",
}

CLUSTER_COLORS = {
    "Very High Burden": "#e74c3c",
    "High Burden":      "#e67e22",
    "Moderate Burden":  "#f1c40f",
    "Low Burden":       "#2ecc71",
}


def build_country_profiles(df):
    """
    Aggregate per-country epidemic trajectory statistics.
    All metrics normalised per capita where appropriate.

    Parameters
    ----------
    df : pd.DataFrame  engineered feature dataset

    Returns
    -------
    profiles : pd.DataFrame  one row per country
    """
    profiles = df.groupby("Country/Region").agg(
        Peak_Growth_Rate   = ("Growth_Rate_7d", "max"),
        Avg_Rt             = ("Rt_Proxy", "mean"),
        Max_Cases_7d_Avg   = ("Cases_7d_Avg", "max"),
        Avg_Stringency     = ("stringency_index", "mean"),
        Avg_Vaccination    = ("people_fully_vaccinated_per_hundred", "mean"),
        Population_Density = ("population_density", "last"),
        Median_Age         = ("median_age", "last"),
        Total_Confirmed    = ("Confirmed", "max"),
        Population         = ("population", "last"),
    ).dropna()

    # ── Per-capita features ───────────────────────────────────────────────────
    profiles["Cases_Per_Million"] = (
        profiles["Total_Confirmed"] / (profiles["Population"] + 1) * 1e6
    ).clip(0, 1e6)

    profiles["Peak_Cases_Per_Million"] = (
        profiles["Max_Cases_7d_Avg"] / (profiles["Population"] + 1) * 1e6
    ).clip(0, 1e5)

    print(f"Country profiles built: {len(profiles)} countries")
    return profiles


def run_elbow_analysis(profiles, k_range=range(2, 9)):
    """
    Compute inertia for different K values to find optimal K.

    Parameters
    ----------
    profiles : pd.DataFrame  country profiles
    k_range  : range         K values to test

    Returns
    -------
    inertias : list of float
    """
    available_features = [f for f in CLUSTER_FEATURES if f in profiles.columns]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[available_features])

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"  K={k}  inertia={km.inertia_:,.0f}")

    return inertias


def cluster_countries(profiles, n_clusters=4):
    """
    Apply K-Means clustering to country epidemic profiles.
    Assigns meaningful burden labels based on cases per million.

    Parameters
    ----------
    profiles   : pd.DataFrame  country profiles from build_country_profiles()
    n_clusters : int           number of clusters (default 4)

    Returns
    -------
    profiles : pd.DataFrame  with Cluster and Cluster_Label columns added
    scaler   : fitted StandardScaler (for future transform)
    """
    available_features = [f for f in CLUSTER_FEATURES if f in profiles.columns]
    print(f"\nClustering on features: {available_features}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[available_features])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles = profiles.copy()
    profiles["Cluster"] = km.fit_predict(X_scaled)

    # Label clusters by cases per million (highest burden = label 0)
    burden_order = (
        profiles.groupby("Cluster")["Cases_Per_Million"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    label_map = {
        cluster_id: CLUSTER_LABELS[i]
        for i, cluster_id in enumerate(burden_order)
    }
    profiles["Cluster_Label"] = profiles["Cluster"].map(label_map)

    # Print cluster summary
    print("\nCluster Summary:")
    print("-" * 80)
    for label in CLUSTER_LABELS.values():
        grp = profiles[profiles["Cluster_Label"] == label]
        if len(grp) == 0:
            continue
        examples = ", ".join(grp.nlargest(3, "Cases_Per_Million").index.tolist())
        print(
            f"  {label:20s} | {len(grp):3d} countries | "
            f"Avg cases/M: {grp['Cases_Per_Million'].mean():8,.0f} | "
            f"Avg vacc: {grp['Avg_Vaccination'].mean():5.1f}% | "
            f"e.g. {examples}"
        )

    return profiles, scaler


def get_country_cluster(country, profiles):
    """
    Get cluster information for a specific country.

    Parameters
    ----------
    country  : str
    profiles : pd.DataFrame  clustered profiles

    Returns
    -------
    dict with cluster info or None
    """
    if country not in profiles.index:
        return None
    row = profiles.loc[country]
    return {
        "country":       country,
        "cluster_label": row.get("Cluster_Label", "Unknown"),
        "cases_per_million": row.get("Cases_Per_Million", 0),
        "avg_vaccination":   row.get("Avg_Vaccination", 0),
        "peak_growth_rate":  row.get("Peak_Growth_Rate", 0),
        "median_age":        row.get("Median_Age", 0),
    }


if __name__ == "__main__":
    from data_pipeline import load_and_prepare_all
    from feature_engineering import engineer_features, add_risk_labels

    df       = load_and_prepare_all()
    df       = engineer_features(df)
    df       = add_risk_labels(df)
    profiles = build_country_profiles(df)

    print("\nElbow analysis:")
    inertias = run_elbow_analysis(profiles)

    profiles, scaler = cluster_countries(profiles, n_clusters=4)

    for country in ["India", "US", "Brazil", "New Zealand", "Japan"]:
        info = get_country_cluster(country, profiles)
        if info:
            print(f"\n{country}: {info['cluster_label']} | "
                  f"Cases/M: {info['cases_per_million']:,.0f} | "
                  f"Vacc: {info['avg_vaccination']:.1f}%")
