"""
Combine the three uncertainty layers into a single explainable uncertainty stack.

Uncertainty decomposition
-------------------------
σ²_total(x, y) = σ²_physics(x, y)   from hydrogen_wtd_uncertainty_1km.tif
               + σ²_EDK(x, y)        from baseline_kriging_std_m.tif (SGS spread on residuals)

These two components are independent by construction:
  - σ_physics reflects the Ma 2025 ensemble spread due to forcing / parameter uncertainty
  - σ_EDK reflects spatial data-gap uncertainty — it is large where wells are sparse
    and the kriged correction is extrapolating

Temporal uncertainty σ_anomaly(x, y, t) is stored separately in gwl_kriging_std.zarr and is
not combined here because it has a time dimension.

Outputs
-------
data/processed/baseline_uncertainty_stack.tif   — 3-band GeoTIFF (EPSG:5070)
    Band 1: σ_physics   (m)   — Ma 2025 physics model uncertainty
    Band 2: σ_EDK       (m)   — spatial data-gap uncertainty from EDK
    Band 3: mask_50km   (0/1) — 1 = within 50 km of a well (high confidence zone)

data/processed/total_uncertainty_m.tif          — 1-band GeoTIFF (EPSG:5070)
    σ_total = sqrt(σ²_physics + σ²_EDK)  (m)

Usage:
    python -m src.evaluation.uncertainty_stack \\
        --physics  data/processed/hydrogen_wtd_uncertainty_1km.tif \\
        --edk-std  data/processed/baseline_kriging_std_m.tif \\
        --mask     data/processed/well_density_mask.tif \\
        --output-dir data/processed

If --physics is omitted (DEM-only baseline, no HydroGEN), σ_total = σ_EDK only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_CRS = CRS.from_epsg(5070)
NODATA_OUT = np.float32(-9999.0)


def _read_band(path: Path) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read band 1 of a GeoTIFF; return (array float32, transform, crs). nodata → NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype=np.float32)
        nd = src.nodata if src.nodata is not None else -9999.0
        transform = src.transform
        crs = src.crs
    arr[arr == nd] = np.nan
    return arr, transform, crs


def _save_multiband(
    arrays: list[np.ndarray],
    transform: rasterio.transform.Affine,
    path: Path,
    band_descriptions: list[str],
) -> None:
    """Write a list of 2-D float32 arrays as a multi-band GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = arrays[0].shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(arrays),
        dtype=np.float32,
        crs=TARGET_CRS,
        transform=transform,
        nodata=float(NODATA_OUT),
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for i, (arr, desc) in enumerate(zip(arrays, band_descriptions), start=1):
            out = np.where(np.isnan(arr), NODATA_OUT, arr).astype(np.float32)
            dst.write(out, i)
            dst.update_tags(i, description=desc)
    logger.info(f"Saved {len(arrays)}-band: {path}")


def _save_single(
    array: np.ndarray,
    transform: rasterio.transform.Affine,
    path: Path,
    description: str = "",
) -> None:
    """Write a single float32 2-D array as a GeoTIFF."""
    _save_multiband([array], transform, path, [description])


def build_uncertainty_stack(
    physics_path: Path | None,
    edk_std_path: Path,
    mask_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Combine uncertainty layers and write outputs.

    Parameters
    ----------
    physics_path:
        Path to ``hydrogen_wtd_uncertainty_1km.tif`` (σ_physics, m).
        May be None if no HydroGEN prior was used.
    edk_std_path:
        Path to ``baseline_kriging_std_m.tif`` (σ_EDK from SGS, m).
    mask_path:
        Path to ``well_density_mask.tif`` (1 = within 50 km of well).
    output_dir:
        Output directory.

    Returns
    -------
    dict with keys ``"stack"`` and ``"total"``, values = output Paths.
    """
    # ---- Load layers ----
    sigma_edk, transform, crs = _read_band(edk_std_path)
    height, width = sigma_edk.shape

    if crs.to_epsg() != 5070:
        raise ValueError(
            f"σ_EDK raster CRS is {crs} — expected EPSG:5070. "
            "Re-run interpolate_baseline with the aligned grid."
        )

    mask_arr, _, _ = _read_band(mask_path)

    if physics_path is not None:
        sigma_physics, _, _ = _read_band(physics_path)
        # Regrid if shapes don't match (should not happen if align_hydrogen was run)
        if sigma_physics.shape != sigma_edk.shape:
            import rasterio.warp as rwarp
            dst = np.full_like(sigma_edk, np.nan)
            with rasterio.open(physics_path) as src:
                rwarp.reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=float(NODATA_OUT),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
            dst[dst == NODATA_OUT] = np.nan
            sigma_physics = dst
            logger.warning(
                "σ_physics shape differed from σ_EDK — automatically reprojected. "
                "Run make hydrogen to ensure grids are pre-aligned."
            )
    else:
        sigma_physics = np.zeros_like(sigma_edk)
        logger.info("No physics uncertainty provided — σ_physics = 0 (DEM-only baseline)")

    # ---- Combined uncertainty ----
    sigma_total = np.sqrt(
        np.where(np.isfinite(sigma_physics), sigma_physics ** 2, 0.0)
        + np.where(np.isfinite(sigma_edk), sigma_edk ** 2, np.nan)
    )

    # ---- Diagnostics ----
    for name, arr in [("σ_physics", sigma_physics), ("σ_EDK", sigma_edk), ("σ_total", sigma_total)]:
        valid = arr[np.isfinite(arr)]
        if len(valid):
            logger.info(
                f"  {name}: median={np.median(valid):.3f} m  90th-pct={np.percentile(valid, 90):.3f} m  "
                f"max={valid.max():.3f} m"
            )

    # Fraction of cells where physics uncertainty exceeds EDK uncertainty
    both_valid = np.isfinite(sigma_physics) & np.isfinite(sigma_edk)
    if both_valid.sum() > 0:
        phys_dominant = (sigma_physics > sigma_edk)[both_valid]
        logger.info(
            f"  σ_physics > σ_EDK in {phys_dominant.mean()*100:.1f}% of co-valid cells "
            "(if high: physics model is the precision bottleneck, not well density)"
        )

    # ---- Save outputs ----
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_path = output_dir / "baseline_uncertainty_stack.tif"
    _save_multiband(
        [sigma_physics, sigma_edk, mask_arr],
        transform, stack_path,
        [
            "Band 1: sigma_physics (m) — Ma 2025 HydroGEN ensemble spread",
            "Band 2: sigma_edk (m) — spatial data-gap uncertainty (SGS on residuals)",
            "Band 3: mask_50km (0/1) — 1=within 50 km of well",
        ],
    )

    total_path = output_dir / "total_uncertainty_m.tif"
    _save_single(sigma_total, transform, total_path, "sigma_total (m) = sqrt(sigma_physics^2 + sigma_edk^2)")

    return {"stack": stack_path, "total": total_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine physics + EDK spatial uncertainty layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=Path("data/processed/hydrogen_wtd_uncertainty_1km.tif"),
        help="Ma 2025 uncertainty raster aligned to analysis grid. Omit for DEM-only baseline.",
    )
    parser.add_argument(
        "--edk-std",
        type=Path,
        default=Path("data/processed/baseline_kriging_std_m.tif"),
        help="σ_EDK from SGS realisations (output of interpolate_baseline).",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=Path("data/processed/well_density_mask.tif"),
        help="Well-density mask (1 = within 50 km of observation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        default=False,
        help="Set σ_physics = 0 even if the raster exists (for testing without HydroGEN).",
    )
    args = parser.parse_args()

    for p in [args.edk_std, args.mask]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}. Run `make pilot-baseline` first.")

    physics_path: Path | None = None
    if not args.no_physics:
        if args.physics.exists():
            physics_path = args.physics
        else:
            logger.warning(f"Physics uncertainty raster not found: {args.physics} — setting σ_physics = 0")

    outputs = build_uncertainty_stack(
        physics_path=physics_path,
        edk_std_path=args.edk_std,
        mask_path=args.mask,
        output_dir=args.output_dir,
    )
    logger.info(f"Uncertainty stack → {outputs['stack']}")
    logger.info(f"Total uncertainty → {outputs['total']}")
    logger.info(
        "Next: open baseline_uncertainty_stack.tif in QGIS; verify σ_EDK is largest far from wells."
    )


if __name__ == "__main__":
    main()
