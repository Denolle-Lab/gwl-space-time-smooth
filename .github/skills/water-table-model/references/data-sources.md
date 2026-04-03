# Data Sources Reference

## USGS NWIS Groundwater Levels

> **⚠️ API MIGRATION (2026-02-01)**: The legacy `/nwis/gwlevels/` endpoint was
> decommissioned on 2026-02-01 and now returns an HTML redirect. Use the new
> OGC API endpoint below instead. `dataretrieval.nwis.get_gwlevels()` is **broken**.

### New OGC API (field-measurements)

**Items endpoint**: `https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items`

Key query parameters:
| Parameter | Value | Notes |
|-----------|-------|-------|
| `monitoring_location_id` | `USGS-{site_no},...` | Comma-separated list, up to ~50 at once |
| `parameter_code` | `72019` | Depth to water level, ft below land surface |
| `datetime` | `2000-01-01/..` | RFC3339 interval; `..` means open end |
| `limit` | `1000` | Records per page (max) |
| `f` | `json` | Always set for programmatic access |
| `api_key` | from env var | Required for >few requests/hr; free signup at https://api.waterdata.usgs.gov/signup/ |

**Response format**: GeoJSON FeatureCollection; paginate via `"rel": "next"` link.

**Critical fields in `feature.properties`**:
| New field | Old field | Notes |
|-----------|-----------|-------|
| `monitoring_location_id` | `site_no` | Strip `"USGS-"` prefix to get site number |
| `time` | `lev_dt` + `lev_tm` | ISO8601 with timezone; slice `[:10]` for date |
| `value` | `lev_va` | Depth in feet (string → coerce to float) |
| `qualifier` | `lev_status_cd` | Array of strings; see mapping table below |
| `unit_of_measure` | `lev_unit_cd` | Usually `"ft"` |
| `observing_procedure_code` | `lev_meth_cd` | Method code (T=tape, R=recorder, etc.) |
| `vertical_datum` | (was in sites) | Vertical datum name string |
| `approval_status` | — | `"Approved"` or `"Provisional"` |
| `geometry.coordinates` | `dec_long_va`, `dec_lat_va` | [lon, lat] in EPSG:4326 |

**Qualifier → lev_status_cd mapping**:
| New qualifier string | Old code | Action |
|---------------------|----------|--------|
| `"Static"` | `""` (blank) | **Keep** |
| `"Pumping"` | `"P"` | Drop |
| `"Dry"` | `"D"` | Drop |
| `"Flowing"` | `"F"` | Drop |
| `"Obstructed"` | `"O"` | Drop |
| `"Recently pumped"` | `"R"` | Drop |
| `"Other"` | `"Z"` | Drop |

**Site metadata**: Still uses `dataretrieval.nwis.get_info()` (working as of 2026)
```python
import dataretrieval.nwis as nwis
sites, meta = nwis.get_info(stateCd="10", siteType="GW", siteOutput="expanded", hasDataTypeCd="gw")
```

**Set API key** (recommended for bulk downloads):
```bash
export USGS_API_KEY="your_key_here"
```
Then `download_nwis.py` will pick it up automatically via `os.environ.get("USGS_API_KEY")`.

**Documentation**:
- OGC API overview: `https://api.waterdata.usgs.gov/docs/ogcapi/`
- Migration guide: `https://api.waterdata.usgs.gov/docs/ogcapi/migration`
- OpenAPI/Swagger: `https://api.waterdata.usgs.gov/ogcapi/v0/openapi`

### Site Metadata Fields (from `dataretrieval.nwis.get_info()`, unchanged)

| Field | Description | Notes |
|-------|-------------|-------|
| `site_no` | USGS site number | 8–15 digit string; use as string, not int |
| `dec_lat_va` | Decimal latitude (NAD83) | |
| `dec_long_va` | Decimal longitude (NAD83) | |
| `well_depth_va` | Well depth (feet) | Flag > 500 ft as likely confined |
| `alt_va` | Land surface altitude (feet, NAVD88) | Convert to meters; WTE = alt_va_m − dtw_m |
| `alt_datum_cd` | Vertical datum | Must be NAVD88; flag/exclude NGVD29 |
| `aquifer_cd` | Aquifer code | Useful categorical covariate |

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
