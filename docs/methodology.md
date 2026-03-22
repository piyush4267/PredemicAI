# Methodology — EpidemicAI

## Biological & Epidemiological Reasoning

### Why These Features?

Every feature in this model has a grounding in infectious disease epidemiology:

**Rₜ (Effective Reproduction Number)**
The single most important metric in outbreak monitoring. When Rₜ > 1, each infected person infects more than one other — the outbreak grows. When Rₜ < 1, it shrinks. We estimate Rₜ as the ratio of 7-day average cases to 7-day average cases from 7 days prior. This is a standard proxy used by the WHO and national health agencies.

**Growth Rate (7-day)**
Captures the acceleration or deceleration of spread. A growth rate of 0.2 means cases are growing 20% week-over-week — a signal of emerging outbreak. Epidemiologically important because it reflects changes in transmission faster than cumulative counts.

**Doubling Time**
Derived from the daily growth rate using log(2)/log(1+r). A doubling time of 7 days is a public health emergency; 30+ days is manageable. Used by CDC and WHO for outbreak severity classification.

**Case Fatality Rate (CFR)**
Deaths / confirmed cases. A rising CFR signals healthcare system stress or a more virulent variant. A falling CFR often reflects improved treatment or a shift to younger infected demographics.

**Lag Features (1, 3, 7, 14 days)**
Epidemiologically, these capture the incubation period dynamics. COVID-19 has a median incubation of 5 days — so the 7-day lag is particularly biologically meaningful. The 14-day lag captures a full two-incubation-period window, standard in contact tracing protocols.

### Why SIR Per Wave?

A classic SIR model assumes a fully susceptible population and a single transmission event. COVID-19 had multiple distinct waves driven by:
- New variants (Delta, Omicron) with different transmission properties
- Waning immunity between waves
- Policy changes between waves

Fitting SIR across the full timeline averages over these dynamics and produces R₀ ≈ 1 (the long-run average). Fitting per-wave isolates the biological transmission parameter for each epidemiological event. Wave 1 R₀ estimates of 2.5–3.0 are consistent with peer-reviewed COVID-19 literature (Liu et al., 2020; Sanche et al., 2020).

### Why Vaccination Rate as a Feature?

Vaccination reduces susceptibility (S in SIR) and infectiousness. Countries with high vaccination coverage at the time of a new wave show measurably lower Rₜ values. Including `people_fully_vaccinated_per_hundred` as a feature allows the model to account for this biological protection when forecasting cases — a direct link between immunological knowledge and model design.

### Why Stringency Index?

Government interventions (lockdowns, masking mandates, travel restrictions) reduce effective contact rates — the β parameter in SIR. The Oxford COVID-19 Government Response Tracker's stringency index (0–100) captures this. There is typically a 10–14 day lag between policy implementation and measurable case reduction (matching the incubation + reporting delay window), which is why we use current stringency rather than a leading indicator.

### Why Demographic Features?

**Median age**: COVID-19 IFR increases exponentially with age (Levin et al., 2020). Countries with older populations have higher clinical burden per case, and age also correlates with pre-existing condition prevalence.

**Population density**: Contact rate (β in SIR) scales with density. Urban high-density areas show faster early exponential growth.

**Hospital beds per thousand**: A proxy for healthcare capacity. When this is low relative to case burden, CFR rises due to care rationing — a known dynamic from the Italian and Brazilian outbreak peaks.

---

## Model Selection Rationale

**Random Forest over LSTM**: Tree-based models outperform sequence models when:
1. The time series has non-stationarity (multiple waves with different parameters)
2. The dataset is wide (many features) relative to its length per country
3. Interpretability is required (feature importance)

For a global cross-country model with only ~1000 rows per country but 21 features, Random Forest is the appropriate choice. LSTM would require substantially more per-country data to avoid overfitting.

**XGBoost as comparison**: XGBoost's gradient boosting handles feature interactions differently from Random Forest's bagging. Running both and comparing allows us to identify whether the predictive signal comes from individual features (RF) or from feature interactions (XGBoost).

**Time-based split (not random)**: This is non-negotiable for time series. A random split allows the model to see future rows of country A while predicting past rows of country A — this is data leakage that artificially inflates R². We train on all data before 2022-01-01 and test on 2022 onwards, simulating genuine out-of-sample prediction.

---

## Risk Classification Logic

The 4-level risk system (Low/Medium/High/Critical) is derived from three independent signals:

1. **Rₜ proxy**: Scores 1–3 points depending on whether Rₜ > 1.0, > 1.5, or > 2.0
2. **7-day growth rate**: Scores 1–3 points depending on whether growth > 5%, > 20%, or > 50%
3. **Cases per million**: Scores 1–3 points depending on whether burden > 20, > 100, or > 500 per million

Total scores of 0–1 = Low, 2–4 = Medium, 5–6 = High, 7–9 = Critical.

This mirrors the WHO's multi-indicator approach to outbreak severity assessment, which deliberately avoids single-metric thresholds that can be gamed by reporting differences between countries.

---

## Hotspot Detection

Clustering is performed on per-capita burden features rather than absolute case counts, because absolute counts conflate population size with epidemic severity. A country with 1 million cases and 1.4 billion people (India, 714 cases/million) is in a different epidemiological situation than a country with 100,000 cases and 500,000 people (2,000 cases/million).

The four cluster features that drive differentiation:
- Cases per million (total burden)
- Peak cases per million (acute severity)
- Average vaccination coverage (protection level)
- Median age (population vulnerability)

---

## References

- Liu Y et al. (2020). The reproductive number of COVID-19 is higher compared to SARS coronavirus. *J Travel Med*.
- Sanche S et al. (2020). High contagiousness and rapid spread of severe acute respiratory syndrome coronavirus 2. *Emerg Infect Dis*.
- Levin AT et al. (2020). Assessing the age specificity of infection fatality rates for COVID-19. *Eur J Epidemiol*.
- Hale T et al. (2021). A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker). *Nature Human Behaviour*.
