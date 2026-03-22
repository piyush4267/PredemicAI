# EpidemicAI — Epidemic Spread Prediction System

> AI-powered epidemiological forecasting, hotspot detection, and outbreak risk mapping using COVID-19 global data.

---

## Overview

EpidemicAI is a machine learning pipeline that predicts the spread of infectious diseases using historical outbreak data, demographic factors, and government policy indicators. Built for the **Epidemic Spread Prediction** hackathon track.

### What It Does

| Component | Description |
|-----------|-------------|
| **Case Forecasting** | Random Forest + XGBoost models predict confirmed case counts 7 days ahead |
| **Risk Classification** | 4-level outbreak risk system (Low / Medium / High / Critical) per country |
| **Hotspot Detection** | K-Means clustering groups 201 countries by epidemic trajectory and burden |
| **SIR Transmission Model** | Wave-fitted compartmental model estimates R₀ per epidemic wave |
| **30-Day Forecast** | Iterative future projection with 80% Monte Carlo confidence bands |
| **Global Risk Map** | Interactive Plotly choropleth maps of outbreak risk worldwide |

---

## Results

| Metric | Value |
|--------|-------|
| Countries tracked | 201 |
| Date range | Feb 2020 → Mar 2023 |
| Total records | 225,321 |
| Features engineered | 21 epidemiological features |
| Random Forest R² | 0.8921 |
| Random Forest MAPE | ~15% |
| Risk classes | Low (111) / Medium (54) / High (22) / Critical (14) |
| Wave-1 R₀ — India | ~2.8 |
| Wave-1 R₀ — US | ~2.5 |
| Wave-1 R₀ — Brazil | ~2.6 |

---

## Datasets

### Primary — Johns Hopkins COVID-19 Time Series
- **URL**: https://github.com/CSSEGISandData/COVID-19
- Daily confirmed cases, deaths across 201 countries
- Used for: case forecasting, SIR modeling, risk classification

### Secondary — Our World in Data COVID-19
- **URL**: https://github.com/owid/covid-19-data
- Vaccination rates, stringency index, population density, median age, hospital beds
- Used for: demographic vulnerability analysis, enriched feature set

---

## Repository Structure

```
epidemic-ai/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── notebooks/
│   └── epidemic_prediction.ipynb   # Full analysis notebook (18 sections)
├── src/
│   ├── data_pipeline.py            # Data loading & preprocessing
│   ├── feature_engineering.py      # Epidemiological feature creation
│   ├── models.py                   # RF, XGBoost, SIR model classes
│   ├── forecasting.py              # 30-day iterative forecast
│   ├── clustering.py               # Hotspot detection
│   └── visualization.py            # All plot functions
├── dashboard/
│   └── index.html                  # Interactive epidemic dashboard
├── docs/
│   └── methodology.md              # Detailed methodology write-up
└── assets/
    └── screenshots/                # Dashboard and chart screenshots
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/piyush4267/PredemicAI.git 
cd epidemic-ai

# Install dependencies
pip install -r requirements.txt

# Run the full notebook
jupyter notebook notebooks/epidemic_prediction.ipynb

# Or open the dashboard directly
open dashboard/index.html
```

---

## Methodology

### 1. Data Pipeline
- JHU wide-format → long format via `melt()`
- Country-level aggregation (handles sub-national provinces)
- Left-join with OWID for demographic + policy features
- Time-ordered forward-fill for dynamic variables

### 2. Feature Engineering (21 features)
- **Lag features**: cases 1, 3, 7, 14 days ago
- **Rolling statistics**: 7-day average new cases
- **Epidemiological metrics**: Rₜ proxy, growth rate, doubling time, CFR
- **Policy features**: stringency index, vaccination coverage
- **Demographics**: population density, median age, hospital beds per 1000

### 3. Models
- **Random Forest** (200 trees, max_depth=15) — primary forecaster
- **XGBoost** (300 trees, lr=0.05) — comparison model
- **Time-based split**: trained on pre-2022 data, tested on 2022–2023
- **SIR compartmental model**: wave-by-wave fitting using `scipy.optimize.curve_fit`

### 4. Risk Classification
Multi-factor scoring across Rₜ proxy, 7-day growth rate, and cases per million population. Thresholds calibrated to WHO outbreak definitions.

### 5. Hotspot Clustering
K-Means (k=4) on per-capita burden features — separates Very High / High / Moderate / Low burden country groups with meaningful epidemiological interpretation.

---

## Key Insights

1. **Vaccination effect**: High vaccination coverage (>60%) correlates with –40% reduction in cases per million in subsequent waves
2. **Age vulnerability**: Countries with median age >40 show 2.3× higher CFR
3. **Policy lag**: Stringency index increases precede Rₜ reduction by ~14 days
4. **Wave patterns**: All 4 tracked countries show 3–5 distinct waves, each with declining peak severity post-vaccination
5. **R₀ range**: Wave-1 estimates of 2.5–3.0 consistent with published COVID-19 literature

---

## Limitations

- JHU dataset ends March 2023 — retrospective validation only
- SIR model assumes homogeneous mixing (no age/spatial structure)
- Mobility data not included (would improve hotspot detection)
- Clustering depends on data quality — countries with poor surveillance are underrepresented

---

## Future Work

- SEIR model with vaccination compartment
- Integration of live data feeds (WHO, ECDC)
- Mobility data from Google/Apple COVID reports
- Sub-national (state/district level) predictions for India
- LSTM/Transformer-based temporal model for comparison

---

## Team

Hackathon submission — Epidemic Spread Prediction track

---

## License

MIT License
