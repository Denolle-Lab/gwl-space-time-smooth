# Water Table Model — CONUS Monthly (2000–present)

Reproducible, observation-anchored model of the water table (depth-to-groundwater)
across the contiguous United States at monthly temporal resolution.

## Scope

| Parameter | Decision | Rationale |
|-----------|----------|-----------|
| **Spatial domain** | CONUS (contiguous US) | Full national coverage; ~50 states queried via NWIS |
| **Temporal resolution** | Monthly | Pragmatic sweet spot given USGS measurement cadence |
| **Temporal extent** | 2000-01-01 → present | Captures GRACE era (2002+), modern monitoring network |
| **Target variable** | Water table elevation (WTE, m NAVD88) internally; deliver DTW (m below surface) | WTE is smoother for interpolation; DTW is what users want |
| **Output grid** | 1 km (~30 arc-sec), EPSG:5070 (NAD83 CONUS Albers) | Standard for national hydrologic products |
| **Delivery CRS** | EPSG:4326 (WGS84 geographic) | Interoperability |

## Core Philosophy

Real wells first. Physics-informed covariates second. ML/statistical learning third.
Never trust a gridded product until validated against held-out observations.

## Reproducing

```bash
# 1. Create environment
mamba env create -f environment.yml
conda activate wtm

# 2. Download and QC well data
make data

# 3. Download covariates
make covariates

# 4. Run EDA
make eda

# 5. Train and validate
make train
make validate
```

## Key Documents

- [`docs/assumptions.md`](docs/assumptions.md) — All simplifying assumptions with severity ratings
- [`docs/limitations.md`](docs/limitations.md) — Known limitations, updated continuously
- [`docs/validation_report.md`](docs/validation_report.md) — Validation metrics and artifact analysis
- [`docs/literature_review.md`](docs/literature_review.md) — Annotated bibliography

## License

- Code: MIT
- Data products: CC-BY-4.0
