# Literature Review Protocol

## Goal

Build a working knowledge of the state of the art in continental-scale groundwater level
modeling, understand the failure modes of existing products, and document assumptions and
limitations that are well-established in the literature.

---

## Scan Order

Work through sources in this order — stop when new papers stop providing new insights.

### 1. USGS Water Resources Publications (Open Access)

Search: `https://pubs.usgs.gov/` → Topic: Groundwater → filter date 2010–present

Priority types:
- **Professional Papers** (PP) — comprehensive regional groundwater assessments
- **Scientific Investigations Reports** (SIR) — state/aquifer-level monitoring and modeling
- **Fact Sheets** — useful for understanding monitoring network design

Key search terms: `"water table" CONUS`, `"groundwater level" national`, `"depth to water"`

### 2. AGU Water Resources Research (WRR)

`https://agupubs.onlinelibrary.wiley.com/journal/19447973`

Search titles/abstracts for: `"water table depth" OR "depth to groundwater" AND ("machine learning" OR "interpolation" OR "national" OR "continental")`

Most AGU papers now have ESSOAr preprints — check `https://essopenarchive.org/` for open access.

### 3. AGU Geophysical Research Letters (GRL)

Same search terms; GRL favors shorter, higher-impact results. Many GRACE-based studies here.

### 4. HydroShare

`https://www.hydroshare.org/` → search "water table depth" or "groundwater level"

Look for:
- Published datasets (may be usable as additional validation data)
- Jupyter notebooks demonstrating modeling approaches
- Model outputs from other groups (download to `data/comparison/`)

### 5. Google Scholar Forward/Backward Citation Chain

Start from the anchor papers below. For each:
- Read the **Cited by** list (forward citations) — find the most-cited follow-up papers
- Read the **References** section (backward) — follow key methodological citations

---

## Anchor Papers (Must-Read)

| Paper | Year | Why Essential |
|-------|------|--------------|
| Fan, Li & Miguez-Macho — "Global patterns of groundwater table depth" — *Science* | 2013 | The most-cited global WTD product; know its limitations **intimately** — staircase artifacts, poor performance in mountains and arid regions, interpolation across data voids |
| Fan et al. — "Hillslope hydrology in global change research and Earth system modeling" — *Water Resources Research* | 2019 | Fan group's follow-up; discusses scale dependence and model limitations |
| de Graaf et al. — "A global-scale two-layer transient groundwater model" — *Advances in Water Resources* | 2015 | First global dynamic GW model; 5 arcmin resolution |
| de Graaf et al. — "A global-scale groundwater model for the analysis of future water scarcity" — *Global Environmental Change* | 2017 | Extension; used as comparison product |
| Jasechko & Perrone — "Global groundwater wells at risk of running dry" — *Science* | 2021 | Uses GWDB well data; useful for understanding well database quality issues |
| Maxwell & Condon — "Connections between groundwater flow and transpiration partitioning" — *Science* | 2016 | WTD–topography connections at continental scale; ParFlow-CLM |
| Condon & Maxwell — "Evaluating the relationship between topography and groundwater using outputs from a continental-scale integrated hydrology model" — *Water Resources Research* | 2015 | TWI as WTD predictor; physical basis for the covariate |
| Gleeson et al. — "The global volume and distribution of modern groundwater" — *Nature Geoscience* | 2016 | Global aquifer characterization; useful for geology covariate design |
| Martens et al. (GLEAM) | 2017 | Evapotranspiration product used as covariate |

### ML-Specific Papers to Review

| Paper | Notes |
|-------|-------|
| Shen et al. — LSTM for hydrology | Foundation for applying sequence models to well time series |
| Kratzert et al. — "Rainfall-runoff modelling using Long Short-Term Memory" | Demonstrates LSTM generalization; methodology transferable |
| Nearing et al. — "What role does hydrological science play in the age of machine learning?" | Honest assessment of ML limitations in hydrology |
| Recent WRR papers on "groundwater" + "random forest" or "XGBoost" | Search for current state of the art |

---

## Review Protocol

For each paper, record in a notes file:

```
## [Author Year] Short Title

**Claim**: One sentence on the main finding.
**Method**: Data sources, spatial resolution, ML/statistical approach.
**Limitations stated by authors**: What do they admit doesn't work well?
**Artifacts observed by us**: Staircase patterns? Bull's-eyes? Discontinuities?
**Relevance**: How does this inform our modeling choices?
**Citation**: Full BibTeX or DOI.
```

### Red Flags to Watch For in Published Products

When reviewing any paper that presents WTD/WTE maps:

1. **Smooth at coarse scale, noisy at fine scale** — suggests interpolation of sparse data without spatial constraints
2. **Political boundary artifacts** — training data was split or processed by jurisdiction
3. **DEM-tile artifacts** — visible grid lines at ~1° or ~5° intervals
4. **Unrealistic arid-region values** — WTD < 1 m across the Mojave or Sahara = hallucination
5. **No spatial CV** — reported R² > 0.95 with random CV for spatial interpolation = inflated

---

## Key Journals Beyond WRR/GRL

| Journal | Why |
|---------|-----|
| *Hydrogeology Journal* | Applied GW; good regional studies |
| *Journal of Hydrology* | Broad; many NWIS-based studies |
| *Ground Water* | NGWA journal; monitoring network papers |
| *Environmental Research Letters* | Open access; high-impact short papers |
| *Nature Water* | New (2023+); review articles on global water |

---

## Databases and Gray Literature

- **NGWMN** (National Ground-Water Monitoring Network): `https://cida.usgs.gov/ngwmn/` — inter-agency network with standardized data
- **GWDB** (Global Groundwater Database): `https://www.bgs.ac.uk/research/groundwater/international/globalGroundwaterMonitoringNetwork.html`
- **USGS National Water Census**: documents on monitoring gaps and data quality
- **State Water Board reports**: California DWR, Texas TWDB, etc. — often contain high-quality regional WTD analyses

---

## Output

Maintain a living `docs/literature_review.md` in the repo with:
- Annotated bibliography using the protocol above
- A "Key Findings" summary that informs modeling decisions
- A "Critique of Existing Products" section feeding directly into `docs/limitations.md`
