.PHONY: data qc covariates eda train validate clean all

# === Configuration ===
START_DATE := 2000-01-01
RAW_DIR := data/raw/nwis
PROCESSED_DIR := data/processed
MONTHLY_PARQUET := $(PROCESSED_DIR)/nwis_gwlevels_monthly.parquet
SITES_PARQUET := $(PROCESSED_DIR)/nwis_sites_clean.parquet

# === Targets ===

all: data qc

## Download raw NWIS groundwater data (state-by-state, checkpointed)
data:
	python -m src.data.download_nwis \
		--start-date $(START_DATE) \
		--output-dir $(RAW_DIR)

## Run QC filtering and monthly aggregation
qc: $(RAW_DIR)/download_log.json
	python -m src.data.qc_nwis \
		--input-dir $(RAW_DIR) \
		--output $(MONTHLY_PARQUET)

## Download covariates (DEM, PRISM, soils, etc.) — placeholder
covariates:
	@echo "TODO: Implement covariate download scripts"
	@echo "  - DEM (3DEP or MERIT Hydro) → data/raw/dem/"
	@echo "  - PRISM climate → data/raw/prism/"
	@echo "  - gSSURGO soils → data/raw/soils/"
	@echo "  - NLCD land cover → data/raw/nlcd/"
	@echo "  - NHDPlus streams → data/raw/nhdplus/"

## Run EDA notebook — placeholder
eda:
	jupyter nbconvert --execute notebooks/01_eda.ipynb --to html

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
