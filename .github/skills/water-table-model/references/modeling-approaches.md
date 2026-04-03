# Modeling Approaches Reference

## Decision Tree

```
Is the domain < 1 aquifer system (~500 km)?
├─ Yes → Start with spatial-only RF/XGBoost baseline (Stage 1 below)
│         Add temporal component (Stage 2) once spatial model is validated
└─ No (CONUS) → Must partition by HUC-2 or aquifer type for variogram estimation
                 Non-stationary GP or spatially-varying coefficient model

Do you have > 10 years of monthly records at most sites?
├─ Yes → Two-stage spatial + temporal model (recommended)
└─ No  → Static / climatological model only; document as limitation

Is the user asking for deep learning end-to-end?
├─ Yes → See §Deep Learning below — caution them first
└─ No  → RF/XGBoost baseline, then GP or LSTM anomaly model
```

---

## Stage 1 — Spatial Baseline (Climatological WTE)

**Target**: long-term median WTE per site (units: meters NAVD88).

**Model**: `RandomForestRegressor` or `XGBRegressor` / `LGBMRegressor` (all in `pixi.toml`)

**Features** (all reprojected to EPSG:5070, 1 km):
```
elevation_m, slope_deg, twi, precip_mean_mm, et_mean_mm,
soil_ksat, soil_clay_frac, aquifer_cd (encoded), nlcd_class (encoded),
dist_to_stream_m, lat_5070, lon_5070
```

**Critical: Use spatial cross-validation, not random CV.**

Random CV for spatial interpolation is deeply wrong — nearby wells share spatial autocorrelation,
so held-out wells are not truly independent. Random CV inflates R² by 0.1–0.3 in typical datasets.

Recommended: **spatial block CV** via `verde` (in `pixi.toml`):
```python
from verde import BlockShuffleSplit
splitter = BlockShuffleSplit(spacing=200_000, n_splits=5, random_state=0)  # 200 km blocks
for train_idx, test_idx in splitter.split(coords, extra_coords_name="region"):
    ...
```

Also acceptable: **leave-one-cluster-out** grouped by HUC-4 basin.

**Hyperparameter tuning**: Use `sklearn.model_selection.cross_val_score` with the spatial splitter,
not `GridSearchCV` with random splits.

---

## Stage 2 — Temporal Anomalies

**Target**: `WTE_anomaly(site, t) = WTE_obs(site, t) − WTE_spatial_baseline(site)`

### Option A: GP Regression (recommended for small domains, < 500 sites)

```python
import gpytorch
# RBF kernel for time + linear kernel for trend; fit per spatial cluster of sites
```

Pros: principled uncertainty, handles irregular sampling, extrapolation is honest (widens CIs).
Cons: O(n³) in number of time steps × sites; use sparse GP (gpytorch.models.ExactGP with
inducing points) for large datasets.

### Option B: LSTM / Transformer (recommended for CONUS, dense training data)

- Input: time series of monthly covariates (precip anomaly, GRACE TWSA, soil moisture, PDSI)
- Architecture: LSTM with 2–3 layers, dropout 0.2, teacher forcing
- Train on all sites jointly with site embeddings for site-level random effects
- **Do not** train purely on gridded inputs without well observations as anchor

### Option C: Linear Mixed Model (fast baseline, interpretable)

```python
from sklearn.linear_model import Ridge
# Or use statsmodels MixedLM for proper random intercepts per site
```

Pros: fast, interpretable coefficients, easy to diagnose.
Cons: cannot capture nonlinear anomaly responses (e.g., threshold recharge).

### Spatial Interpolation of Anomaly Field

At each time step, interpolate the site anomalies to the grid:
- **Kriging** (`pykrige.OrdinaryKriging`): theoretically optimal for Gaussian RF; slow at CONUS scale
- **Verde** (`verde.ScipyGridder` or `verde.Spline`): faster, appropriate for large domains
- **Nearest-neighbor baseline**: always compute as sanity check; any interpolation method
  that can't beat nearest-neighbor is wrong

---

## Uncertainty Quantification

### Quantile Regression Forest (recommended for Stage 1)

```python
from sklearn.ensemble import GradientBoostingRegressor
low = GradientBoostingRegressor(loss="quantile", alpha=0.1)
high = GradientBoostingRegressor(loss="quantile", alpha=0.9)
```

Or use `lightgbm` with `objective="quantile"`.

### Conformal Prediction (model-agnostic, rigorous coverage guarantees)

```python
from mapie.regression import MapieRegressor  # mapie is in pixi.toml
mapie = MapieRegressor(estimator=rf, method="plus", cv=spatial_cv_splitter)
y_pred, y_pi = mapie.predict(X_test, alpha=0.1)  # 90% PI
```

**Important**: Use the spatial CV splitter (not random) when calibrating conformal PIs —
otherwise coverage guarantees don't hold for spatially correlated data.

### Well-Density Mask

Flag cells where the nearest observation is > 50 km as low-confidence:
```python
from scipy.spatial import cKDTree
tree = cKDTree(well_coords_5070)
dist, _ = tree.query(grid_coords_5070)
confidence_mask = (dist <= 50_000).astype(np.uint8)  # 50 km in meters
```

Deliver as a companion raster alongside predictions. Never suppress this mask in publications.

---

## Deep Learning

If deep learning is required, do **not** train a model that maps gridded covariates → WTD without
well observations as a constraint. That produces a covariate interpolation, not a groundwater model.

### Preferred DL Approach: Physics-Informed Neural Network (PINN)

Embed the 1D Boussinesq equation as a soft constraint in the loss function:
```
L = L_data (fit to well observations) + λ * L_physics (residual of PDE)
```

### Acceptable DL Approach: Neural Operator

Use FNO (Fourier Neural Operator) or DeepONet to learn the solution operator of the
groundwater flow equation. These generalize better than standard PINNs to unseen forcing.

### Red Flags in DL for WTD

- Training primarily on GRACE or reanalysis grids → reproduces gridding artifacts
- No well observations in the loss function → not a groundwater model
- Predicting finer than the coarsest input → structure below coarsest input resolution is hallucinated
- No spatial CV → R² is inflated by spatial autocorrelation

---

## Diagnostics Checklist (Run Before Publishing)

- [ ] Spatial CV RMSE < nearest-neighbor baseline
- [ ] Residuals have no spatial trend (Moran's I ≈ 0 in test set)
- [ ] WTE prediction nowhere exceeds land surface elevation (DTW < 0 cells < 0.1%)
- [ ] WTE gradient < DEM gradient everywhere (predictions smoother than topography)
- [ ] No straight-line discontinuities visible in predicted WTE raster
- [ ] Conformal prediction intervals achieve nominal coverage on spatial test blocks
- [ ] Well-density mask delivered as companion raster
