.PHONY: data qc dem grid baseline anomalies covariates eda train validate clean clean-all all pilot pilot-qc pilot-grid pilot-eda hydrogen pilot-baseline uncertainty-stack

# === Configuration ===
START_DATE := 2026-01-01
RAW_DIR := data/raw/nwis
RAW_DEM_DIR := data/raw/dem
PROCESSED_DIR := data/processed
MONTHLY_PARQUET := $(PROCESSED_DIR)/nwis_gwlevels_monthly.parquet
SITES_PARQUET := $(PROCESSED_DIR)/nwis_sites_clean.parquet
DEM_TIF := $(RAW_DEM_DIR)/merit_hydro_1km_5070.tif
GRID_NC := $(PROCESSED_DIR)/conus_grid_1km.nc
BASELINE_WTE := $(PROCESSED_DIR)/baseline_wte_m.tif

# Pacific Northwest pilot scope
PILOT_STATES := WA OR ID
# EPSG:5070 bbox for PNW: left bottom right top (metres)
# Covers all WA+OR wells + 100 km buffer.  Grid: 900 km wide × 1100 km tall.
PNW_BBOX := -2300000 2200000 -1400000 3300000
PILOT_GRID_NC := $(PROCESSED_DIR)/bbox_grid_1km.nc
HYDROGEN_WTD := $(PROCESSED_DIR)/hydrogen_wtd_prior_1km.tif
HYDROGEN_UNC := $(PROCESSED_DIR)/hydrogen_wtd_uncertainty_1km.tif
PILOT_BASELINE_WTE := $(PROCESSED_DIR)/baseline_wte_m.tif
PILOT_BASELINE_STD := $(PROCESSED_DIR)/baseline_kriging_std_m.tif

# === Targets ===

all: data qc dem grid baseline anomalies

## Download raw NWIS groundwater data (state-by-state, checkpointed)
data:
	pixi run python -m src.data.download_nwis \
		--start-date $(START_DATE) \
		--output-dir $(RAW_DIR)

## Run QC filtering and monthly aggregation
qc: $(RAW_DIR)/download_log.json
	pixi run python -m src.data.qc_nwis \
		--input-dir $(RAW_DIR) \
		--output $(MONTHLY_PARQUET)

## Download MERIT Hydro DEM and reproject to 1 km EPSG:5070
dem:
	pixi run python -m src.data.download_dem \
		--output-dir $(RAW_DEM_DIR)

## Build canonical 1 km CONUS grid definition
grid: $(DEM_TIF)
	pixi run python -m src.features.compute_grid \
		--dem $(DEM_TIF) \
		--output-dir $(PROCESSED_DIR)

## Interpolate spatial baseline (co-kriging MM1 with DEM)
baseline: $(SITES_PARQUET) $(DEM_TIF)
	pixi run python -m src.models.interpolate_baseline \
		--sites $(SITES_PARQUET) \
		--dem $(DEM_TIF) \
		--output-dir $(PROCESSED_DIR)

## Interpolate monthly anomaly fields via ordinary kriging
anomalies: $(BASELINE_WTE) $(MONTHLY_PARQUET)
	pixi run python -m src.models.interpolate_anomalies \
		--monthly $(MONTHLY_PARQUET) \
		--sites $(SITES_PARQUET) \
		--baseline-wte $(BASELINE_WTE) \
		--dem $(DEM_TIF) \
		--output-dir $(PROCESSED_DIR)

## Download covariates (DEM, PRISM, soils, etc.) — placeholder
covariates:
	@echo "TODO: Implement covariate download scripts"
	@echo "  - PRISM climate → data/raw/prism/"
	@echo "  - gSSURGO soils → data/raw/soils/"
	@echo "  - NLCD land cover → data/raw/nlcd/"
	@echo "  - NHDPlus streams → data/raw/nhdplus/"

## Run EDA notebook — placeholder
eda:
	pixi run jupyter nbconvert --execute notebooks/01_eda.ipynb --to html

## Train model — placeholder
train:
	@echo "TODO: Implement model training pipeline"

## Run validation and artifact detection — placeholder
validate:
	@echo "TODO: Implement validation pipeline"

## PNW pilot — QC (WA + OR only — ID has no level records yet)
pilot-qc: $(RAW_DIR)/download_log.json
	pixi run python -m src.data.qc_nwis \
		--input-dir $(RAW_DIR) \
		--output $(MONTHLY_PARQUET) \
		--states $(PILOT_STATES)

## PNW pilot — build regional 1 km grid from bbox (no DEM required)
pilot-grid:
	pixi run python -m src.features.compute_grid \
		--bbox $(PNW_BBOX) \
		--output-dir $(PROCESSED_DIR)

## PNW pilot — run EDA notebook
pilot-eda:
	pixi run jupyter nbconvert --execute notebooks/01_eda.ipynb --to html

## PNW pilot — align HydroGEN TIFs to the 1 km EPSG:5070 grid
hydrogen: $(PILOT_GRID_NC)
	pixi run python -m src.features.align_hydrogen \
		--wtd data/comparison/WT2-ma_wtd_50.tif \
		--unc data/comparison/wtd_uncertainty_mosaic_wgs84.tif \
		--grid $(PILOT_GRID_NC) \
		--output-dir $(PROCESSED_DIR)

## PNW pilot — EDK spatial baseline (HydroGEN prior + kriged residuals)
pilot-baseline: $(SITES_PARQUET) $(DEM_TIF) $(HYDROGEN_WTD)
	pixi run python -m src.models.interpolate_baseline \
		--sites $(SITES_PARQUET) \
		--dem $(DEM_TIF) \
		--hydrogen-wtd $(HYDROGEN_WTD) \
		--output-dir $(PROCESSED_DIR)

## PNW pilot — krige monthly WTE anomaly fields onto the bbox grid
## Default: 5 km pilot grid, last 36 months.  Full run: GRID_STEP=1 N_MONTHS=0
GRID_STEP    ?= 5
PILOT_MONTHS ?= 36
pilot-anomalies: $(PILOT_GRID_NC) $(SITES_PARQUET) $(MONTHLY_PARQUET)
	pixi run python -m src.models.pilot_temporal \
		--monthly $(MONTHLY_PARQUET) \
		--sites $(SITES_PARQUET) \
		--grid $(PILOT_GRID_NC) \
		--hydrogen-wtd $(HYDROGEN_WTD) \
		--states $(PILOT_STATES) \
		--grid-step $(GRID_STEP) \
		--n-months $(PILOT_MONTHS) \
		--output-dir $(PROCESSED_DIR)

## Combine physics + EDK uncertainty into a single explainable stack
uncertainty-stack: $(PILOT_BASELINE_STD) $(HYDROGEN_UNC)
	pixi run python -m src.evaluation.uncertainty_stack \
		--physics $(HYDROGEN_UNC) \
		--edk-std $(PILOT_BASELINE_STD) \
		--mask $(PROCESSED_DIR)/well_density_mask.tif \
		--output-dir $(PROCESSED_DIR)

## PNW pilot — full pipeline: download WA/OR/ID → QC → grid → EDA
## (baseline + anomalies still need the DEM; run `make dem` first for those)
pilot: pilot-qc pilot-grid pilot-eda hydrogen pilot-baseline uncertainty-stack

## Clean processed data (keeps raw downloads)
clean:
	rm -f $(MONTHLY_PARQUET) $(SITES_PARQUET) $(PROCESSED_DIR)/qc_report.json

## Nuclear clean (removes everything including raw downloads)
clean-all: clean
	rm -rf $(RAW_DIR)
