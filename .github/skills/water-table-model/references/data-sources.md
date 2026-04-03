# Data Sources Reference

## USGS NWIS Groundwater Levels

**REST API base URL**: `https://waterservices.usgs.gov/nwis/gwlevels/`

Key query parameters:
| Parameter | Value | Notes |
|-----------|-------|-------|
| `stateCd` | e.g., `CA` | One state per request |
| `siteType` | `GW` | Groundwater only |
| `startDT` | `2000-01-01` | GRACE era start |
| `format` | `rdb` or `json` | rdb is tab-delimited, easier to parse |

**Python wrapper** (preferred): `dataretrieval` (in `pixi.toml`)
```python
import dataretrieval.nwis as nwis
data, meta = nwis.get_gwlevels(stateCd="CA", startDT="2000-01-01")
```

**Site metadata**: `https://waterservices.usgs.gov/nwis/site/`
```python
sites, meta = nwis.get_info(stateCd="CA", siteType="GW")
```

### Critical Fields

| Field | Description | Notes |
|-------|-------------|-------|
| `site_no` | USGS site number | 8–15 digit string; use as string, not int |
| `dec_lat_va` | Decimal latitude (NAD83) | |
| `dec_long_va` | Decimal longitude (NAD83) | |
| `lev_dt` | Measurement date | Parse as UTC |
| `lev_va` | Water level below land surface (feet) | Convert to meters on ingest |
| `lev_status_cd` | Measurement status | See status code table below |
| `well_depth_va` | Well depth (feet) | Flag > 500 ft as likely confined |
| `alt_va` | Land surface altitude (feet, NAVD88) | Convert to meters; use for WTE = alt_va_m − lev_va_m |
| `alt_datum_cd` | Vertical datum | Must be NAVD88; flag/exclude NGVD29 |
| `aquifer_cd` | Aquifer code | Useful categorical covariate |

### `lev_status_cd` Values to Filter Out

| Code | Meaning |
|------|---------|
| `P` | Affected by pumping |
| `D` | Dry |
| `O` | Obstructed |
| `X` | Measurement failed |
| `F` | Frozen |
| `E` | Estimated |
| `R` | Revegetating (recovery) |
| Blank | Routine static measurement — **keep** |

---

## Covariates

### Topography — USGS 3DEP / MERIT Hydro

- **3DEP 1/3 arc-sec (~10 m) DEM**: `https://pubs.usgs.gov/ds/1276/` or via `py3dep`
- **MERIT Hydro (~90 m, hydrologically conditioned)**: `http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/`
  - Pre-computed TWI available — download `twi_*.tif` tiles
  - Use this for TWI rather than computing from scratch unless matching a finer DEM
- **TWI computation** (if computing from scratch with `richdem`):
  ```python
  import richdem as rd
  dem = rd.LoadGDAL("dem.tif")
  rd.FillDepressions(dem, in_place=True)
  accum = rd.FlowAccumulation(dem, method="Dinf")
  slope = rd.TerrainAttribute(dem, attrib="slope_radians")
  twi = np.log(accum / np.tan(slope + 1e-6))
  ```

### Climate — PRISM / gridMET

- **PRISM** (4 km monthly, CONUS): `https://prism.oregonstate.edu/`
  - Access via `prism` Python package or direct FTP
  - Variables: `ppt` (precipitation), `tmean`, `tmax`, `tmin`, `tdmean`, `vpdmax`
- **gridMET** (4 km daily → aggregate to monthly): `https://www.climateengine.org/`
  - Access via OpenDAP or `intake` catalog: `http://thredds.northwestknowledge.net:8080/thredds/`

### Soils — gSSURGO / SoilGrids

- **gSSURGO** (CONUS, 30 m): `https://gdg.sc.egov.usda.gov/` — requires USDA Geospatial Data Gateway account
  - Key attributes: `ksat` (saturated hydraulic conductivity), `awc` (available water capacity), `claytotal`
- **SoilGrids 2.0** (global, 250 m): `https://soilgrids.org/` — no registration required
  - Access via `owslib` WCS or REST API

### Geology — USGS Aquifer Maps

- **Principal Aquifer Systems**: `https://water.usgs.gov/GIS/metadata/usgswrd/XML/aquiferov.xml`
- **GLHYMPS v2** (global hydraulic properties): `https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/TTJNIU`
- **GLiM** (global lithological map): `https://www.geo.uni-hamburg.de/en/geologie/forschung/aquifer.html`

### Land Cover — NLCD

- **NLCD 2021** (30 m, CONUS): `https://www.mrlc.gov/data/nlcd-2021-land-cover-conus`
  - Access via `pystac-client` + Planetary Computer (`planetary-computer` in `pixi.toml`)
  - Key classes: 21–24 (developed/urban), 81–82 (agriculture), 41–43 (forest)

### GRACE TWSA

- **GRACE/GRACE-FO RL06** monthly TWSA grids (~300 km native): NASA PO.DAAC
  - `https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3`
  - Access via `earthaccess` Python package
  - Use mascon solution (not spherical harmonics) — less post-processing needed
  - **Reminder**: any structure finer than ~300 km is hallucinated by downscaling; use as a
    large-scale temporal anomaly driver only

### Streamlines — NHDPlus HR

- **NHDPlus High Resolution** (vector): `https://www.usgs.gov/national-hydrography/nhdplus-high-resolution`
  - Download by HUC-4 or HUC-8; access via `nhd` Python package or direct download
  - Compute distance to nearest stream per grid cell using `geopandas.sjoin_nearest` or raster proximity

---

## Comparison-Only Products (Never Use as Ground Truth)

| Product | URL | Known Issues |
|---------|-----|--------------|
| Fan et al. (2013) global WTD | `https://igb.rc.fas.harvard.edu/` | Staircase artifacts at tile boundaries; poor in mountains |
| de Graaf et al. (2017) | PCR-GLOBWB outputs | 5 arcmin resolution smooths all structure |
| GRACE-downscaled WTD | Various | Finer-than-300 km structure is hallucinated |

Store all comparison downloads in `data/comparison/` — never mix with `data/processed/`.
