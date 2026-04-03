# Water Table Model — CONUS Monthly (2000–present)

Reproducible, observation-anchored model of the water table (depth-to-groundwater, DTW)
and water table elevation (WTE) across the contiguous United States at 1 km spatial
resolution and monthly temporal resolution (2000–present).

> **Core philosophy**: Real wells first. Physics-informed covariates second. ML third.
> Never trust a gridded product until validated against held-out observations.

---

## Goals

This project builds a first-order, physics-grounded, smoothly varying gridded groundwater
level product from USGS NWIS well observations. Unlike prior products (Fan et al. 2013,
GRACE downscaling), every cell prediction traces back to real well measurements, not
satellite retrievals or simulated recharge. Target outputs are:

- `gwl_wte.zarr` — monthly water table elevation (m NAVD88), 1 km EPSG:5070
- `gwl_dtw.zarr` — monthly depth to groundwater (m below surface), same grid
- `gwl_anomaly.zarr` — departure from site long-term median (WTE anomaly, m)
- `baseline_wte_m.tif` — single long-term median WTE field from co-kriging + DEM

All grids carry a `well_density_mask.tif`; cells > 50 km from the nearest usable well are
set to NaN and should be reported as unobserved.

---

## Scope

| Parameter | Decision | Rationale |
|-----------|----------|-----------|
| **Spatial domain** | CONUS (contiguous US) | Full national coverage; ~50 states queried via NWIS |
| **Temporal resolution** | Monthly | Pragmatic sweet spot given USGS measurement cadence |
| **Temporal extent** | 2000-01-01 → present | Captures GRACE era (2002+), modern monitoring network |
| **Target variable** | WTE (m NAVD88) internally; deliver DTW (m below surface) | WTE is smoother for interpolation; DTW is what users want |
| **Output grid** | 1 km, EPSG:5070 (NAD83 CONUS Albers) | Standard for national hydrologic products |
| **Delivery CRS** | EPSG:4326 (WGS84 geographic) | Interoperability |
| **Interpolation method** | GStatSim co-kriging MM1 (DEM as secondary) + ordinary kriging for anomalies | Geostatistically rigorous; preserves variogram statistics via SGS |

---

## Repository Structure

```
.
├── pixi.toml                  ← canonical dependency file (use `pixi install`)
├── Makefile                   ← pipeline entry points
├── README.md                  ← this file
│
├── src/
│   ├── data/
│   │   ├── download_nwis.py   ← USGS NWIS well download (state-by-state, checkpointed)
│   │   ├── qc_nwis.py         ← QC chain + monthly aggregation (no gap filling)
│   │   └── download_dem.py    ← MERIT Hydro DEM download → 1 km EPSG:5070 mosaic
│   ├── features/
│   │   └── compute_grid.py    ← canonical CONUS 1 km grid definition + helpers
│   └── models/
│       ├── interpolate_baseline.py  ← spatial WTE baseline via co-kriging MM1 + DEM
│       └── interpolate_anomalies.py ← monthly anomaly fields via ordinary kriging → Zarr
│
├── notebooks/
│   └── 01_eda.ipynb           ← exploratory data analysis (figs saved to figures/eda/)
│
├── data/                      ← all data files are git-ignored
│   ├── raw/
│   │   ├── nwis/              ← one parquet per state + download_log.json
│   │   ├── dem/               ← MERIT Hydro tiles + merit_hydro_1km_5070.tif
│   │   └── MANIFEST.md        ← dataset registry (git-tracked)
│   └── processed/             ← QC'd parquets, GeoTIFFs, Zarr archives
│
├── docs/
│   ├── assumptions.md         ← severity-tagged assumptions register
│   └── limitations.md         ← known limitations (update continuously)
│
└── .github/
    ├── copilot-instructions.md        ← workspace-level Copilot coding rules
    └── skills/water-table-model/
        ├── SKILL.md                   ← Copilot skill for this domain
        └── references/                ← modeling reference docs for the skill
```

**Data provenance rule**: `.parquet`, `.tif`, `.nc`, `.zarr`, `.csv` files are never
committed. Only `download_log.json`, `MANIFEST.md`, and `src/` scripts are tracked.

---

## Reproducing

### 1. Environment

This project uses [pixi](https://prefix.dev/docs/pixi/) for reproducible environments.
Do **not** use `conda` or `pip` directly.

```bash
pixi install       # first time, or after pixi.toml changes
pixi run lab       # launch JupyterLab
pixi shell         # activate the environment in a shell
```

### 2. Pipeline

Run targets in order. Each target is idempotent (safe to re-run; skips already-done work
where possible).

```bash
make data          # Download raw NWIS GW levels (state-by-state, checkpointed)
make qc            # QC filtering + monthly aggregation → data/processed/
make dem           # Download MERIT Hydro DEM → data/raw/dem/merit_hydro_1km_5070.tif
make grid          # Build canonical 1 km CONUS grid metadata → data/processed/conus_grid_1km.nc
make baseline      # Co-krige WTE baseline with DEM (MM1) → baseline_*.tif
make anomalies     # Krige monthly anomalies → gwl_*.zarr
make eda           # Execute EDA notebook → HTML report
make clean         # Remove processed outputs (keeps raw downloads)
make clean-all     # Remove everything including raw downloads
```

`make data` is the slowest step (~hours for CONUS; checkpointed per state).
All other targets complete in minutes on a laptop once `make dem` has run.

### 3. Outputs

| File | Description |
|------|-------------|
| `data/processed/nwis_sites_clean.parquet` | QC-passed well sites with per-site statistics |
| `data/processed/nwis_gwlevels_monthly.parquet` | Monthly median WTE/DTW per site |
| `data/raw/dem/merit_hydro_1km_5070.tif` | MERIT Hydro DEM, 1 km EPSG:5070 |
| `data/processed/baseline_wte_m.tif` | Long-term median WTE (co-kriging MM1 + DEM) |
| `data/processed/baseline_dtw_m.tif` | Long-term median DTW = DEM − WTE |
| `data/processed/baseline_std_m.tif` | Uncertainty (σ across SGS realisations) |
| `data/processed/well_density_mask.tif` | Boolean: 1 = within 50 km of a usable well |
| `data/processed/gwl_anomaly.zarr` | Monthly WTE anomaly from site median (m) |
| `data/processed/gwl_wte.zarr` | Monthly WTE = baseline + anomaly (m NAVD88) |
| `data/processed/gwl_dtw.zarr` | Monthly DTW = DEM − WTE (m, positive = below surface) |

---

## AI-Assisted Workflow: Copilot Instructions and Skills

This repository uses GitHub Copilot with custom workspace instructions and a domain skill
to accelerate development. Understanding these files lets you steer or extend the AI
assistance.

### How it works

```
.github/copilot-instructions.md      ← always loaded; sets baseline coding rules
.github/skills/water-table-model/
    SKILL.md                         ← loaded on-demand for domain-specific work
    references/
        data-sources.md              ← NWIS API patterns, field-name mappings
        modeling-approaches.md       ← modeling decision tree, spatial CV caveats
        literature-review-protocol.md ← key papers, literature scan workflow
```

Copilot reads `.github/copilot-instructions.md` for every conversation in this workspace.
The `SKILL.md` is loaded automatically when you ask questions related to groundwater
modeling (keywords listed in the skill frontmatter).

### Workspace instructions (`.github/copilot-instructions.md`)

This file sets project-wide conventions that Copilot must follow for all code in this
repository. Edit it to change:

- **Environment rules** — which package manager to use, Python version constraints, any
  banned imports.
- **Pipeline targets** — which `make` targets exist and what they do; keep this in sync
  with the Makefile.
- **Data layout** — directory names, git-tracked vs. git-ignored files.
- **Coding conventions** — CRS choice, unit conventions (always SI), path handling
  (`pathlib.Path`, no `os.path`), logging (no `print()`), docstring style.
- **QC chain** — the ordered steps in `qc_nwis.py`; any change to the QC chain must be
  reflected here and in the module docstring.

**When to edit**: whenever you add a new pipeline stage, change a directory convention, or
introduce a new data source.

### Skill file (`.github/skills/water-table-model/SKILL.md`)

The skill provides deep domain knowledge activated when you describe groundwater or
water-table tasks. Edit the skill to:

- Add new modeling approaches (e.g., change from GStatSim kriging to a different method).
- Update the trigger keywords in the YAML frontmatter (`description: >`) so Copilot
  activates the skill for new task formulations.
- Link additional reference documents under `references/` for complex sub-topics
  (e.g., add a `covariate-processing.md` when developing the covariates pipeline).

**When to edit**: whenever the modeling approach changes, or when you want Copilot to
favour a particular library or algorithm. The skill is versioned in git alongside the
code it governs.

### Reference documents (`.github/skills/water-table-model/references/`)

Detailed lookup tables and decision trees the skill can point to. Add a new file here
(and a link in `SKILL.md`) when a topic is complex enough to warrant its own reference.

| File | Content |
|------|---------|
| `data-sources.md` | NWIS REST API patterns, available fields, covariate dataset URLs |
| `modeling-approaches.md` | Interpolation method decision tree, spatial CV protocol |
| `literature-review-protocol.md` | Key papers to check, how to scan new literature |

### Quick-reference: what to edit and where

| You want to… | Edit this file |
|---|---|
| Change the Python environment or add a dependency | `pixi.toml` |
| Add a new `make` target | `Makefile` + `pixi.toml [tasks]` + `.github/copilot-instructions.md` |
| Change a coding convention (CRS, units, style) | `.github/copilot-instructions.md` |
| Change the interpolation method | `.github/skills/water-table-model/SKILL.md` + `src/models/` |
| Change the QC chain | `src/data/qc_nwis.py` + docstring + `.github/copilot-instructions.md` |
| Add a new covariate dataset | `src/data/` + `data/raw/MANIFEST.md` + `references/data-sources.md` |
| Add domain knowledge (new paper, new algorithm) | `.github/skills/water-table-model/references/` + `SKILL.md` link |
| Change model assumptions | `docs/assumptions.md` + `.github/copilot-instructions.md` |

---

## Key Documents

- [`docs/assumptions.md`](docs/assumptions.md) — All simplifying assumptions with severity ratings
- [`docs/limitations.md`](docs/limitations.md) — Known limitations, updated continuously
- [`data/raw/MANIFEST.md`](data/raw/MANIFEST.md) — Registry of all raw datasets with provenance

---

## Citation

If you use outputs from this pipeline, please cite:
- **USGS NWIS**: [https://waterdata.usgs.gov/nwis/gw](https://waterdata.usgs.gov/nwis/gw)
- **MERIT Hydro**: Yamazaki, D., et al. (2019). MERIT Hydro. *Water Resources Research*, 55, 5053–5073.
- **GStatSim**: MacKie, E., et al. (2022). GStatSim (1.0). Zenodo. https://doi.org/10.5281/zenodo.7230276

## License

- Code: MIT
- Data products: CC-BY-4.0
