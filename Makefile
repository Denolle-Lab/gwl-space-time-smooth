.PHONY: data qc dem grid baseline anomalies covariates eda train validate clean clean-all all

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

## Clean processed data (keeps raw downloads)
clean:
	rm -f $(MONTHLY_PARQUET) $(SITES_PARQUET) $(PROCESSED_DIR)/qc_report.json

## Nuclear clean (removes everything including raw downloads)
clean-all: clean
	rm -rf $(RAW_DIR)
