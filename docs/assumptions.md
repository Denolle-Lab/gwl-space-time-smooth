# Assumptions Register

Living document. Update whenever a new assumption is introduced or an existing one
is revisited. Each entry is tagged with impact severity and a mitigation strategy.

## Data Assumptions

| # | Assumption | Severity | Mitigation |
|---|-----------|----------|------------|
| A1 | Wells are representative of the surrounding 1 km grid cell | Medium | Weight by well density; flag cells with no wells within 50 km as low-confidence |
| A2 | Confined and unconfined aquifers are not distinguished in the spatial model | High | Filter wells with `well_depth_va` > 150 m (500 ft); add aquifer type code as categorical covariate where available |
| A3 | `lev_status_cd` blank entries are assumed to be valid static measurements | Low | Conservative — blank is the most common code for routine measurements |
| A4 | NGVD29 sites are excluded rather than datum-corrected | Medium | VERTCON correction is available but adds complexity; document excluded site count |
| A5 | Monthly median aggregation adequately represents water table for months with irregular sampling | Low | No gap filling — sparse months remain absent; `coverage_fraction` and `max_gap_months` flags let downstream code exclude gappy sites |
| A6 | Land surface altitude (`alt_va`) from NWIS is accurate to ±1 m | Medium | Cross-check a sample of sites against 3DEP DEM; flag outliers |

## Modeling Assumptions

| # | Assumption | Severity | Mitigation |
|---|-----------|----------|------------|
| M1 | WTE is a smooth function of topography, climate, soils, and geology at 1 km scale | Medium | Validate variograms; check that WTE gradient < DEM gradient |
| M2 | Temporal anomalies are driven primarily by precipitation anomalies and large-scale storage changes (GRACE) | Medium | Include PDSI, soil moisture as additional temporal covariates |
| M3 | Spatial stationarity holds within HUC-2 regions for variogram estimation | Medium | Compute regional variograms; use non-stationary methods if sill is absent |
| M4 | Random Forest / XGBoost feature importance reflects physical causation | Low | It doesn't — document that importance is associative, not causal |

## Changelog

| Date | Change |
|------|--------|
| _today_ | Initial assumptions register created |
