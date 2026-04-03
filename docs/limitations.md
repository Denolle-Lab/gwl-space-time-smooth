# Limitations Register

Living document. Delivered with the final data product so users understand where
the model should and should not be trusted. Update continuously.

## Known Limitations

### Data Coverage

| # | Limitation | Impact | Affected Regions |
|---|-----------|--------|-----------------|
| L1 | USGS monitoring network is spatially uneven — dense in humid East, sparse in arid West | High | Great Basin, Mojave, parts of Mountain West |
| L2 | Many wells have short records (< 5 years), limiting temporal trend estimation | Medium | Newer installations, discontinued sites |
| L3 | Seasonal bias in measurement timing — more measurements in spring/fall than winter | Medium | Northern states with winter access issues |
| L4 | Legacy wells may have uncertain vertical datum or land surface elevation | Medium | Pre-GPS era sites, especially pre-1990 |
| L5 | Pumping wells are filtered by status code, but some pumping-influenced wells may have blank status codes | Medium | Agricultural regions (High Plains, Central Valley, Mississippi Embayment) |

### Model Limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| L6 | Predictions in data-sparse regions are extrapolations from covariates, not interpolations from observations | High | Well-density confidence mask delivered as companion raster |
| L7 | The model does not explicitly represent groundwater flow physics (it is statistical/ML, not a numerical groundwater model) | Medium | Document that predictions may violate flow physics in some areas |
| L8 | Karst regions (e.g., Edwards Plateau, Ozarks, Florida) have discontinuous water tables that smooth models cannot capture | High | Flag karst areas explicitly; consider excluding from national product |
| L9 | Urban areas with extensive groundwater extraction and infrastructure are poorly represented | Medium | NLCD urban fraction covariate partially captures this |
| L10 | Permafrost regions (northern CONUS fringe) have different water table dynamics not captured by temperate-zone covariates | Low | Minimal impact within CONUS; document for any extension to Alaska |

### Product Limitations

| # | Limitation | Impact |
|---|-----------|--------|
| L11 | Monthly resolution may alias sub-monthly events (storm recharge, pump drawdowns) | Low |
| L12 | 1 km grid resolution cannot capture sub-kilometer heterogeneity in shallow aquifers | Medium |
| L13 | Uncertainty estimates from tree ensembles may be underestimated in extrapolation regions | High |

## Changelog

| Date | Change |
|------|--------|
| _today_ | Initial limitations register created |
