# Data Manifest

Record of all downloaded datasets with provenance. Update after each download.

## USGS NWIS Groundwater Levels

| Field | Value |
|-------|-------|
| Source | USGS NWIS Web Services via `dataretrieval` Python package |
| URL | `https://waterservices.usgs.gov/nwis/gwlevels/` |
| Query | All CONUS states, site type GW, 2000-01-01 to present |
| Download date | _pending_ |
| Script | `src/data/download_nwis.py` |
| Raw location | `data/raw/nwis/` |
| Checksum | See `data/raw/nwis/download_log.json` |
| Notes | State-by-state download with checkpointing |

## Comparison / Physics Prior Datasets

### Ma 2025 — HydroGEN Ensemble WTD

| Field | Value |
|-------|-------|
| Source | Princeton HydroGEN via `hf_hydrodata` Python package |
| Reference | Ma et al. (2025), *ERL* |
| URL | `https://hydrogen.princeton.edu` |
| Dataset name | `ma_2025` |
| Variables | `water_table_depth` (median, 50th pct.), `wtd_uncertainty` (ensemble spread) |
| Native grid | CONUS2, LCC 24.14 m, delivered as WGS84 mosaic |
| Raw files | `data/comparison/WT2-ma_wtd_50.tif`, `data/comparison/wtd_uncertainty_mosaic_wgs84.tif` |
| Retrieval script | `1-HydroGEN Retrieval.ipynb` (in notebooks/; uses `hf_hydrodata`) |
| Alignment script | `src/features/align_hydrogen.py` — reprojects to EPSG:5070, 1 km |
| Aligned outputs | `data/processed/hydrogen_wtd_prior_1km.tif`, `data/processed/hydrogen_wtd_uncertainty_1km.tif` |
| Role in model | Physics prior WTE(x,y) for EDK baseline; σ_physics uncertainty layer |
| Notes | DTW sign: positive = below surface. WTE_prior = DEM − Ma2025_DTW. Uncertainty is ensemble spread (confirm IQR vs σ in notebooks/02_hydrogen_eda.ipynb) |

## Covariates

_To be filled as each covariate is downloaded._

| Dataset | Source | Resolution | Download date | Script | Location |
|---------|--------|-----------|---------------|--------|----------|
| DEM | 3DEP / MERIT Hydro | 30m / 90m | _pending_ | _TBD_ | `data/raw/dem/` |
| PRISM Precip | PRISM Climate Group | 4 km monthly | _pending_ | _TBD_ | `data/raw/prism/` |
| gSSURGO Soils | USDA NRCS | 30m | _pending_ | _TBD_ | `data/raw/soils/` |
| NLCD | MRLC | 30m | _pending_ | _TBD_ | `data/raw/nlcd/` |
| NHDPlus | USGS | Vector | _pending_ | _TBD_ | `data/raw/nhdplus/` |
| GRACE | JPL PO.DAAC | 0.5° monthly | _pending_ | _TBD_ | `data/raw/grace/` |
