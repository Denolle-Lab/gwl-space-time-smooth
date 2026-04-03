"""
Download USGS NWIS groundwater level data for CONUS.

Pulls groundwater levels and site metadata state-by-state using the
`dataretrieval` package, with checkpointing so interrupted runs can resume.

Usage:
    python -m src.data.download_nwis --start-date 2000-01-01 --output-dir data/raw/nwis

Output:
    data/raw/nwis/{state}_gwlevels.parquet   — one file per state
    data/raw/nwis/{state}_sites.parquet      — site metadata per state
    data/raw/nwis/download_log.json          — checkpoint / provenance log
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import dataretrieval.nwis as nwis
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# FIPS codes for the 48 CONUS states + DC
CONUS_STATES: dict[str, str] = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26",
    "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36",
    "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41",
    "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "DC": "11",
}

# Maximum retries per state on transient API failures
MAX_RETRIES = 3
RETRY_DELAY_SEC = 30


def load_checkpoint(log_path: Path) -> dict:
    """Load the download checkpoint log, or initialize a new one."""
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {"completed_states": {}, "started": datetime.now(timezone.utc).isoformat()}


def save_checkpoint(log_path: Path, log: dict) -> None:
    """Persist the checkpoint log to disk."""
    log["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def download_state_gwlevels(
    state_abbr: str,
    state_fips: str,
    start_date: str,
    output_dir: Path,
) -> dict:
    """
    Download groundwater levels + site metadata for one state.

    Returns a summary dict with record counts and status.
    """
    gw_path = output_dir / f"{state_abbr}_gwlevels.parquet"
    sites_path = output_dir / f"{state_abbr}_sites.parquet"

    # --- Groundwater levels ---
    logger.info(f"  Downloading GW levels for {state_abbr} (FIPS {state_fips})...")
    try:
        df_gw, _ = nwis.get_gwlevels(
            stateCd=state_fips,
            startDT=start_date,
        )
    except Exception as e:
        # dataretrieval returns empty DataFrame on "no data" — but may also
        # raise on genuine API errors
        logger.warning(f"  get_gwlevels failed for {state_abbr}: {e}")
        df_gw = pd.DataFrame()

    n_levels = len(df_gw)
    if n_levels > 0:
        # Reset multi-index that dataretrieval sometimes returns
        if isinstance(df_gw.index, pd.MultiIndex):
            df_gw = df_gw.reset_index()
        elif df_gw.index.name:
            df_gw = df_gw.reset_index()
        df_gw.to_parquet(gw_path, index=False)
        logger.info(f"  → {n_levels:,} GW level records saved to {gw_path.name}")
    else:
        logger.info(f"  → No GW level records for {state_abbr}")

    # --- Site metadata (expanded) ---
    logger.info(f"  Downloading site metadata for {state_abbr}...")
    try:
        df_sites, _ = nwis.get_info(
            stateCd=state_fips,
            siteType="GW",
            siteOutput="expanded",
            hasDataTypeCd="gw",
        )
    except Exception as e:
        logger.warning(f"  get_info failed for {state_abbr}: {e}")
        df_sites = pd.DataFrame()

    n_sites = len(df_sites)
    if n_sites > 0:
        if isinstance(df_sites.index, pd.MultiIndex):
            df_sites = df_sites.reset_index()
        elif df_sites.index.name:
            df_sites = df_sites.reset_index()
        df_sites.to_parquet(sites_path, index=False)
        logger.info(f"  → {n_sites:,} GW sites saved to {sites_path.name}")
    else:
        logger.info(f"  → No GW site metadata for {state_abbr}")

    return {
        "n_levels": n_levels,
        "n_sites": n_sites,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_download(start_date: str, output_dir: Path) -> None:
    """Download NWIS GW data for all CONUS states with checkpointing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_log.json"
    log = load_checkpoint(log_path)

    states_todo = [
        (abbr, fips)
        for abbr, fips in sorted(CONUS_STATES.items())
        if abbr not in log["completed_states"]
    ]

    if not states_todo:
        logger.info("All states already downloaded. Delete download_log.json to re-run.")
        return

    logger.info(
        f"Downloading {len(states_todo)} states "
        f"({len(CONUS_STATES) - len(states_todo)} already done)"
    )

    for state_abbr, state_fips in tqdm(states_todo, desc="States", unit="state"):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                summary = download_state_gwlevels(
                    state_abbr, state_fips, start_date, output_dir
                )
                log["completed_states"][state_abbr] = summary
                save_checkpoint(log_path, log)
                break
            except Exception as e:
                logger.error(
                    f"  Attempt {attempt}/{MAX_RETRIES} failed for {state_abbr}: {e}"
                )
                if attempt < MAX_RETRIES:
                    logger.info(f"  Retrying in {RETRY_DELAY_SEC}s...")
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    logger.error(f"  SKIPPING {state_abbr} after {MAX_RETRIES} failures")
                    log["completed_states"][state_abbr] = {
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    save_checkpoint(log_path, log)

        # Be polite to the USGS API
        time.sleep(2)

    # Final summary
    total_levels = sum(
        s.get("n_levels", 0) for s in log["completed_states"].values()
    )
    total_sites = sum(
        s.get("n_sites", 0) for s in log["completed_states"].values()
    )
    errors = [
        k for k, v in log["completed_states"].items() if "error" in v
    ]
    logger.info(f"\nDownload complete:")
    logger.info(f"  Total GW level records: {total_levels:,}")
    logger.info(f"  Total GW sites: {total_sites:,}")
    if errors:
        logger.warning(f"  States with errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Download USGS NWIS GW data for CONUS")
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="Start date for GW level query (default: 2000-01-01)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/nwis"),
        help="Output directory (default: data/raw/nwis)",
    )
    args = parser.parse_args()
    run_download(args.start_date, args.output_dir)


if __name__ == "__main__":
    main()
