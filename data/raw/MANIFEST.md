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
