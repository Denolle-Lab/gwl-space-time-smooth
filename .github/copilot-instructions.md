# gwl-space-time-smooth — Workspace Instructions

Reproducible, observation-anchored model of water table elevation (WTE) and
depth-to-groundwater (DTW) across CONUS at monthly resolution (2000–present).

**Core philosophy**: Real wells first. Physics-informed covariates second. ML third.
Never trust a gridded product until validated against held-out observations.

## Environment

**Use pixi, not conda/mamba directly.**

```bash
pixi install          # set up environment (first time or after pixi.toml changes)
pixi run lab          # start JupyterLab
pixi shell            # activate environment in terminal
```

- `pixi.toml` is the canonical dependency file. `environment.yml` is kept for reference only.
- Always run Python via `pixi run python` (or `pixi run` for Makefile tasks) — bare `python` resolves to the system interpreter.
- `richdem` must stay in `[dependencies]` (conda), not `[pypi-dependencies]` — it fails to compile from source on macOS clang.

## Build & Pipeline

```bash
make data       # Download raw NWIS GW levels (state-by-state, checkpointed)
make qc         # QC + monthly aggregation → data/processed/
make dem        # Download MERIT Hydro DEM → data/raw/dem/merit_hydro_1km_5070.tif
make grid       # Build canonical 1 km CONUS grid → data/processed/conus_grid_1km.nc
make baseline   # Co-krige WTE baseline (MM1 + DEM) → data/processed/baseline_*.tif
make anomalies  # Krige monthly anomaly fields → data/processed/gwl_*.zarr
make eda        # Execute EDA notebook → HTML
make train      # Placeholder: model training
make validate   # Placeholder: validation pipeline
make clean      # Remove processed outputs (keeps raw downloads)
make clean-all  # Remove everything including raw downloads
```

## Data Layout & Provenance

```
data/raw/nwis/            ← one parquet per state (git-ignored)
                            download_log.json ← checkpoint + provenance (git-tracked)
data/raw/dem/             ← MERIT Hydro tiles + merit_hydro_1km_5070.tif (git-ignored)
data/raw/MANIFEST.md      ← dataset registry (git-tracked)
data/processed/           ← QC'd parquets, GeoTIFFs, Zarr archives (git-ignored)
data/comparison/          ← held-out comparison datasets (git-ignored)
```

**Rule**: Data files (`.parquet`, `.tif`, `.nc`, `.csv`, …) are never committed.
Provenance files (`download_log.json`, `MANIFEST.md`, `src/` scripts) are always committed.
Update `data/raw/MANIFEST.md` after adding any new dataset.

## Key Conventions

**Target variables**
- Model internally in **WTE (m NAVD88)** — smoother for interpolation.
- Deliver as **DTW (m below surface)** — what users expect.
- Output grid: 1 km, EPSG:5070 (NAD83 CONUS Albers); deliver in EPSG:4326.

**QC chain** (`src/data/qc_nwis.py`) — do not add steps without updating docstring:
1. Filter bad `lev_status_cd` codes (pumping, dry, obstructed, etc.)
2. Drop sites with < N measurements
3. Flag deep wells (> 150 m) as likely confined — exclude or tag
4. Convert feet → meters
5. Compute WTE from DTW + `alt_va` (land surface altitude)
6. Exclude NGVD29 datum sites (VERTCON correction deferred — see A4 in `docs/assumptions.md`)
7. Aggregate to monthly medians per site
8. Compute per-site temporal coverage statistics (`coverage_fraction`, `max_gap_months`); flag sparse sites (`is_sparse_timeseries`) and sites with long gaps (`has_long_gap`) — **no gap filling, no interpolation**

**Modeling**
- Interpolation: GStatSim co-kriging MM1 (DEM as secondary variable) for the spatial baseline;
  ordinary kriging for monthly anomaly fields. See `src/models/interpolate_baseline.py` and
  `src/models/interpolate_anomalies.py`.
- NST (Normal Score Transform via `QuantileTransformer`) is applied before any kriging step;
  inverse NST applied to all outputs.
- Variograms fitted per HUC-2 region (18 regions) using `scikit-gstat`; params saved to
  `data/processed/variogram_params_huc2.json`.
- Use well-density confidence mask alongside predictions; flag cells >50 km from nearest well.
- Karst (Edwards Plateau, Ozarks, Florida), urban, and permafrost fringe regions need explicit flags — see `docs/limitations.md`.
- Ensemble uncertainty estimates underestimate error in extrapolation regions (L13).

## Coding Conventions

- **CRS**: All spatial analysis in **EPSG:5070** (NAD83 CONUS Albers); deliver outputs in **EPSG:4326**.
- **Units**: SI throughout (meters, seconds, kg). Convert NWIS feet → meters on ingest; never store raw feet in processed files.
- **Time**: UTC only. Use `pandas.Timestamp` or `numpy.datetime64`; no bare strings.
- **Paths**: `pathlib.Path` everywhere in `src/`; no `os.path` string concatenation.
- **Logging**: Use the `logging` module in all `src/` modules; no `print()` statements.
- **Style**: NumPy-style docstrings and type hints on every public function.

## Documentation

- [`docs/assumptions.md`](../docs/assumptions.md) — severity-tagged assumptions register; update when adding any new assumption
- [`docs/limitations.md`](../docs/limitations.md) — known limitations; update continuously
- New scripts: add a row to `data/raw/MANIFEST.md` for any newly downloaded dataset

## Modeling Workflow

For the full phase-by-phase workflow (scope → data → EDA → modeling → validation → packaging), invoke the `water-table-model` skill.
