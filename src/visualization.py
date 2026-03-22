# -*- coding: utf-8 -*-
"""
visualization.py
----------------
All plotting functions for the EpidemicAI project.

Contains:
  - plot_cases_trends()        : daily new cases for top countries
  - plot_rt_over_time()        : Rₜ proxy over time
  - plot_risk_distribution()   : stacked area chart of risk levels
  - plot_feature_importance()  : RF feature importance bar chart
  - plot_actual_vs_predicted() : model validation chart
  - plot_sir_wave_fit()        : SIR model fit per country
  - plot_wave_detection()      : epidemic waves with Rₜ overlay
  - plot_forecast()            : 30-day forecast with confidence band
  - plot_cluster_scatter()     : cluster analysis scatter plots
  - plot_demographic_heatmap() : vulnerability bubble chart + correlation matrix
  - plot_global_risk_map()     : interactive Plotly choropleth
  - plot_risk_dashboard()      : 4-panel Plotly dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.integrate import odeint

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed — interactive maps will be skipped.")

# ── Style settings ────────────────────────────────────────────────────────────
plt.rcParams["figure.dpi"]    = 120
plt.rcParams["figure.figsize"] = (14, 5)
sns.set_theme(style="whitegrid", palette="muted")

RISK_COLORS = {
    "Critical": "#e74c3c",
    "High":     "#e67e22",
    "Medium":   "#3498db",
    "Low":      "#2ecc71",
}

COUNTRY_COLORS = {
    "India":          "#00d2ff",
    "US":             "#ff3b5c",
    "Brazil":         "#ffaa00",
    "United Kingdom": "#00ffcc",
    "South Africa":   "#a78bfa",
    "Germany":        "#f9e2af",
}


# ══════════════════════════════════════════════════════════════════════════════
# TREND CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_cases_trends(df, countries=None, figsize=(14, 5)):
    """Plot 7-day average new cases for selected countries."""
    if countries is None:
        latest = df.groupby("Country/Region")["Confirmed"].max()
        countries = latest.nlargest(4).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    for country in countries:
        subset = df[df["Country/Region"] == country]
        color  = COUNTRY_COLORS.get(country, None)
        ax.plot(
            subset["Date"], subset["Cases_7d_Avg"] / 1e6,
            label=country, linewidth=2, color=color
        )
        ax.fill_between(
            subset["Date"], subset["Cases_7d_Avg"] / 1e6,
            alpha=0.06, color=color
        )

    ax.set_title("Daily New Cases — 7-Day Average (millions)", fontweight="bold")
    ax.set_ylabel("New cases per day (M)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    return fig


def plot_rt_over_time(df, countries=None, figsize=(14, 5)):
    """Plot Rₜ proxy over time with threshold line at 1.0."""
    if countries is None:
        countries = ["India", "US", "Brazil", "United Kingdom"]

    fig, ax = plt.subplots(figsize=figsize)
    for country in countries:
        subset = df[
            (df["Country/Region"] == country) & (df["Confirmed"] > 1000)
        ]
        if len(subset) == 0:
            continue
        smooth = subset["Rt_Proxy"].rolling(14).mean()
        color  = COUNTRY_COLORS.get(country, None)
        ax.plot(subset["Date"], smooth, label=country, linewidth=2, color=color)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               alpha=0.6, label="R\u209c = 1.0 (threshold)")
    ax.set_title("Effective Reproduction Number R\u209c Over Time", fontweight="bold")
    ax.set_ylabel("R\u209c proxy")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    return fig


def plot_risk_distribution(df, figsize=(14, 5)):
    """Stacked area chart of number of countries per risk level over time."""
    from feature_engineering import RISK_LABELS

    risk_time = (
        df.groupby(["Date", "Risk_Label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=RISK_LABELS)
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors  = [RISK_COLORS.get(l, "gray") for l in ["Low", "Medium", "High", "Critical"]
               if l in risk_time.columns]
    cols    = [l for l in ["Low", "Medium", "High", "Critical"] if l in risk_time.columns]

    risk_time[cols].plot.area(ax=ax, color=colors, alpha=0.8, linewidth=0)
    ax.set_title("Countries by Outbreak Risk Level Over Time", fontweight="bold")
    ax.set_ylabel("Number of countries")
    ax.legend(loc="upper left", title="Risk level")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_cols, top_n=15, figsize=(12, 6)):
    """Horizontal bar chart of Random Forest feature importance."""
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors  = [
        "#e74c3c" if v > 0.2 else "#e67e22" if v > 0.05 else "#3498db"
        for v in importance.tail(top_n).values
    ]
    importance.tail(top_n).plot.barh(ax=ax, color=colors)
    ax.set_title(f"Random Forest — Feature Importance (Top {top_n})", fontweight="bold")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(test_df, model, feature_cols,
                              country="India", figsize=(12, 5)):
    """Plot actual vs predicted cases for a country in the test set."""
    c_test = test_df[test_df["Country/Region"] == country].copy()
    if len(c_test) == 0:
        print(f"  {country} not in test set")
        return None

    X = c_test[feature_cols].fillna(0).astype(float).to_numpy()
    c_test["Predicted"] = model.predict(X)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(c_test["Date"], c_test["Target_7d"] / 1e6,
            label="Actual", color="#00d2ff", linewidth=2.5)
    ax.plot(c_test["Date"], c_test["Predicted"] / 1e6,
            label="Predicted", color="#ffaa00",
            linestyle="--", linewidth=2)
    ax.fill_between(c_test["Date"],
                    c_test["Target_7d"] / 1e6,
                    c_test["Predicted"] / 1e6,
                    alpha=0.08, color="orange")
    ax.set_title(f"Actual vs Predicted — {country} (Test Set)", fontweight="bold")
    ax.set_ylabel("Confirmed cases (millions)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIR MODEL
# ══════════════════════════════════════════════════════════════════════════════

def plot_sir_wave_fit(sir_results, figsize=(18, 4)):
    """Plot SIR wave-1 fit for multiple countries."""
    from models import sir_model

    countries = list(sir_results.keys())
    fig, axes = plt.subplots(1, len(countries), figsize=figsize)
    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):
        res = sir_results[country]
        if res is None:
            ax.set_title(f"{country}\nNo convergence")
            ax.axis("off")
            continue

        wdf  = res["wave_df"]
        t    = np.arange(len(wdf))
        sol  = odeint(sir_model, [res["S0"], res["I0"], res["R0_init"]],
                      t, args=(res["beta"], res["gamma"]))

        ax.fill_between(wdf["Date"], wdf["Cases_7d_Avg"] / 1e3,
                        alpha=0.2, color="#00d2ff")
        ax.plot(wdf["Date"], wdf["Cases_7d_Avg"] / 1e3,
                label="Observed", color="#00d2ff", linewidth=2)
        ax.plot(wdf["Date"], sol[:, 1] / 1e3,
                label="SIR fit", color="#ff3b5c",
                linestyle="--", linewidth=1.8)

        ax.set_title(
            f"{country}\n"
            f"R\u2080={res['r0']:.2f}  "
            f"\u03b2={res['beta']:.3f}  "
            f"\u03b3={res['gamma']:.3f}",
            fontsize=9
        )
        ax.set_ylabel("Cases (thousands)")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    plt.suptitle("SIR Model — Wave 1 Fit (Realistic R\u2080 Estimates)",
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_wave_detection(df, countries=None, figsize=(16, 10)):
    """Plot epidemic waves with Rₜ overlay for multiple countries."""
    from scipy.signal import find_peaks

    if countries is None:
        countries = ["India", "US", "United Kingdom", "South Africa"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    wave_colors = ["#e74c3c", "#e67e22", "#9b59b6",
                   "#27ae60", "#1abc9c", "#f39c12"]

    for ax, country in zip(axes, countries):
        cdf = df[
            (df["Country/Region"] == country) & (df["Confirmed"] > 100)
        ].sort_values("Date").reset_index(drop=True)

        if len(cdf) == 0:
            ax.set_title(f"{country} — no data")
            continue

        cases = cdf["Cases_7d_Avg"].fillna(0).values
        max_v = cases.max()

        if max_v > 0:
            peaks, _ = find_peaks(cases / max_v, prominence=0.2, distance=30)
        else:
            peaks = np.array([])

        ax.fill_between(cdf["Date"], cases / 1e3, alpha=0.2, color="#00d2ff")
        ax.plot(cdf["Date"], cases / 1e3, color="#00d2ff",
                linewidth=1.8, label="7-day avg")

        for i, peak in enumerate(peaks):
            color = wave_colors[i % len(wave_colors)]
            ax.axvline(cdf["Date"].iloc[peak], color=color,
                       linestyle="--", linewidth=1.2, alpha=0.8)
            ax.annotate(
                f"Wave {i+1}\n{cdf['Date'].iloc[peak].strftime('%b %Y')}",
                (cdf["Date"].iloc[peak], cases[peak] / 1e3),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=color, fontweight="bold",
            )

        ax2 = ax.twinx()
        ax2.plot(cdf["Date"], cdf["Rt_Proxy"].rolling(14).mean(),
                 color="#ff3b5c", linewidth=1, alpha=0.5, linestyle=":")
        ax2.axhline(1.0, color="#ff3b5c", linewidth=0.8,
                    linestyle="--", alpha=0.4)
        ax2.set_ylabel("R\u209c (right axis)", color="#ff3b5c", fontsize=8)
        ax2.set_ylim(0, 4)
        ax2.tick_params(axis="y", labelcolor="#ff3b5c", labelsize=7)

        ax.set_title(f"{country} — {len(peaks)} wave(s) detected",
                     fontweight="bold")
        ax.set_ylabel("New cases (thousands/day)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Epidemic Wave Detection with R\u209c Overlay",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast(forecast_results, df, countries=None, figsize=(18, 5)):
    """Plot 30-day forecast with confidence band for multiple countries."""
    if countries is None:
        countries = list(forecast_results.keys())[:3]

    fig, axes = plt.subplots(1, len(countries), figsize=figsize)
    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):
        if country not in forecast_results:
            continue
        res = forecast_results[country]

        hist = df[df["Country/Region"] == country].sort_values("Date").tail(90)
        color = COUNTRY_COLORS.get(country, "#00d2ff")

        ax.plot(hist["Date"], hist["Confirmed"] / 1e6,
                color=color, linewidth=2, label="Historical")

        ld = res["last_date"]
        lv = res["last_val"] / 1e6
        fd = [ld] + list(res["dates"])

        ax.plot(fd, [lv] + list(res["median"] / 1e6),
                color="white", linewidth=2.5,
                linestyle="--", label="30-day forecast")
        ax.fill_between(
            fd,
            [lv] + list(res["lower"] / 1e6),
            [lv] + list(res["upper"] / 1e6),
            color="white", alpha=0.1, label="80% confidence band"
        )
        ax.axvline(ld, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(
            f"{res['median'][-1]/1e6:.1f}M",
            (fd[-1], res["median"][-1] / 1e6),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, color="tomato",
        )
        ax.set_title(f"{country} — 30-Day Forecast", fontweight="bold")
        ax.set_ylabel("Confirmed cases (M)")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

    plt.suptitle("30-Day Iterative Forecast with 80% Confidence Band",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING & DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_cluster_scatter(profiles, figsize=(16, 6)):
    """Two scatter plots showing cluster separation."""
    from clustering import CLUSTER_COLORS

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for label, grp in profiles.groupby("Cluster_Label"):
        color = CLUSTER_COLORS.get(label, "gray")
        axes[0].scatter(
            grp["Avg_Vaccination"],
            grp["Cases_Per_Million"] / 1000,
            label=label, alpha=0.75, s=45, color=color
        )
        axes[1].scatter(
            grp["Median_Age"],
            grp["Peak_Growth_Rate"].clip(0, 5),
            label=label, alpha=0.75, s=45, color=color
        )

    for country in ["US", "India", "Brazil", "New Zealand", "Japan", "S.Africa"]:
        row = profiles[profiles.index == country]
        if len(row):
            axes[0].annotate(
                country,
                (row["Avg_Vaccination"].values[0],
                 row["Cases_Per_Million"].values[0] / 1000),
                fontsize=7, alpha=0.85
            )

    axes[0].set_xlabel("Average vaccination coverage (%)")
    axes[0].set_ylabel("Total cases per million (thousands)")
    axes[0].set_title("Clusters: Vaccination vs Cases Per Capita")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Median age")
    axes[1].set_ylabel("Peak 7-day growth rate")
    axes[1].set_title("Clusters: Age Demographics vs Growth Rate")
    axes[1].legend(fontsize=8)

    plt.suptitle("Improved Hotspot Clustering — 4 Epidemic Burden Groups",
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_demographic_heatmap(profiles, latest_complete, figsize=(16, 6)):
    """Demographic vulnerability bubble chart + correlation heatmap."""
    vuln_df = profiles.copy().reset_index()
    vuln_df = vuln_df.merge(
        latest_complete[["Country/Region", "Risk_Label_Text"]],
        on="Country/Region", how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for risk_level, color in RISK_COLORS.items():
        sub = vuln_df[vuln_df["Risk_Label_Text"] == risk_level]
        if len(sub) == 0:
            continue
        sizes = (sub["Total_Confirmed"].clip(0, 5e7) / 5e7 * 400 + 20)
        axes[0].scatter(
            sub["Median_Age"],
            sub["Population_Density"].clip(0, 1000),
            s=sizes, alpha=0.6, color=color,
            label=risk_level, edgecolors="white", linewidth=0.5
        )

    for country in ["US", "India", "Japan", "Nigeria", "Brazil", "Italy"]:
        row = vuln_df[vuln_df["Country/Region"] == country]
        if len(row):
            axes[0].annotate(
                country,
                (row["Median_Age"].values[0],
                 min(row["Population_Density"].values[0], 1000)),
                fontsize=7, alpha=0.85
            )

    axes[0].set_xlabel("Median age")
    axes[0].set_ylabel("Population density (per km²)")
    axes[0].set_title("Demographic Vulnerability\n"
                       "(bubble size = total cases, colour = risk level)")
    axes[0].legend(title="Risk level", fontsize=8)

    corr_cols = [
        "Cases_Per_Million", "Avg_Vaccination", "Avg_Stringency",
        "Population_Density", "Median_Age", "Avg_Rt", "Peak_Growth_Rate"
    ]
    corr_cols   = [c for c in corr_cols if c in vuln_df.columns]
    corr_matrix = vuln_df[corr_cols].corr()
    short_labels = ["Cases/M", "Vaccination", "Stringency",
                    "Pop density", "Median age", "Avg Rₜ", "Peak growth"]
    short_labels = short_labels[:len(corr_cols)]

    im = axes[1].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    axes[1].set_xticks(range(len(corr_cols)))
    axes[1].set_yticks(range(len(corr_cols)))
    axes[1].set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    axes[1].set_yticklabels(short_labels, fontsize=9)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            axes[1].text(
                j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
            )
    axes[1].set_title("Epidemiological Factor Correlations")

    plt.suptitle("Demographic Vulnerability & Factor Correlation Analysis",
                 fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY MAPS
# ══════════════════════════════════════════════════════════════════════════════

def plot_global_risk_map(latest_complete):
    """Interactive Plotly choropleth — continuous risk score."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available — skipping interactive map.")
        return None

    hover_cols = {
        "Confirmed":          ":,.0f",
        "Predicted_7d_Cases": ":,.0f",
        "Rt_Proxy":           ":.2f",
        "Growth_Rate_7d":     ":.3f",
        "stringency_index":   ":.1f",
        "people_fully_vaccinated_per_hundred": ":.1f",
    }
    hover_data = {
        k: v for k, v in hover_cols.items() if k in latest_complete.columns
    }

    fig = px.choropleth(
        latest_complete,
        locations="Country/Region",
        locationmode="country names",
        color="Risk_Score",
        hover_name="Country/Region",
        hover_data=hover_data,
        color_continuous_scale="Reds",
        title="Global COVID-19 Outbreak Risk — 7-Day Growth Index",
        labels={"Risk_Score": "Growth Risk Index"},
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type="natural earth"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_risk_classification_map(latest_complete):
    """Interactive Plotly choropleth — discrete risk level."""
    if not PLOTLY_AVAILABLE:
        return None

    RISK_COLOR_MAP = {
        "Low":      "#27ae60",
        "Medium":   "#f39c12",
        "High":     "#e67e22",
        "Critical": "#c0392b",
    }
    valid = ["Low", "Medium", "High", "Critical"]
    latest_complete = latest_complete.copy()
    latest_complete["Risk_Label_Text"] = latest_complete[
        "Risk_Label_Text"
    ].where(latest_complete["Risk_Label_Text"].isin(valid), "Low")

    fig = px.choropleth(
        latest_complete,
        locations="Country/Region",
        locationmode="country names",
        color="Risk_Label_Text",
        hover_name="Country/Region",
        color_discrete_map=RISK_COLOR_MAP,
        category_orders={"Risk_Label_Text": valid},
        title="Global COVID-19 Outbreak Risk Classification (Latest Data)",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type="natural earth"),
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="Outbreak Risk",
    )
    return fig


def plot_risk_dashboard(latest_complete):
    """4-panel Plotly dashboard."""
    if not PLOTLY_AVAILABLE:
        return None

    top_risk = latest_complete.nlargest(20, "Risk_Score").copy()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Top 20 Countries by Risk Score",
            "Vaccination vs Risk Score",
            "Stringency Index vs Rₜ Proxy",
            "Predicted vs Current Confirmed Cases",
        ]
    )

    fig.add_trace(go.Bar(
        x=top_risk["Risk_Score"].round(3),
        y=top_risk["Country/Region"],
        orientation="h",
        marker=dict(color=top_risk["Risk_Score"],
                    colorscale="Reds", showscale=False),
        showlegend=False,
    ), row=1, col=1)

    if "people_fully_vaccinated_per_hundred" in latest_complete.columns:
        fig.add_trace(go.Scatter(
            x=latest_complete["people_fully_vaccinated_per_hundred"].fillna(0),
            y=latest_complete["Risk_Score"],
            mode="markers",
            marker=dict(size=5,
                        color=latest_complete["Risk_Class"].astype(int),
                        colorscale="RdYlGn_r", opacity=0.6),
            text=latest_complete["Country/Region"],
            showlegend=False,
        ), row=1, col=2)

    if "stringency_index" in latest_complete.columns:
        fig.add_trace(go.Scatter(
            x=latest_complete["stringency_index"].fillna(0),
            y=latest_complete["Rt_Proxy"].fillna(1),
            mode="markers",
            marker=dict(size=5,
                        color=latest_complete["Risk_Class"].astype(int),
                        colorscale="RdYlGn_r", opacity=0.6),
            text=latest_complete["Country/Region"],
            showlegend=False,
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=np.log1p(latest_complete["Confirmed"].fillna(0)),
        y=np.log1p(latest_complete["Predicted_7d_Cases"].fillna(0)),
        mode="markers",
        marker=dict(size=5,
                    color=latest_complete["Risk_Class"].astype(int),
                    colorscale="RdYlGn_r", opacity=0.6),
        text=latest_complete["Country/Region"],
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(title_text="Epidemic Outbreak Risk Dashboard", height=700)
    fig.update_xaxes(title_text="Vaccination coverage (%)", row=1, col=2)
    fig.update_xaxes(title_text="Stringency Index",         row=2, col=1)
    fig.update_xaxes(title_text="log(Current Confirmed)",   row=2, col=2)
    fig.update_yaxes(title_text="Risk Score",               row=1, col=2)
    fig.update_yaxes(title_text="Rₜ Proxy",                 row=2, col=1)
    fig.update_yaxes(title_text="log(Predicted 7-day)",     row=2, col=2)
    return fig
