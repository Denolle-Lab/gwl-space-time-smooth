"""
Download USGS NWIS groundwater level data for CONUS.

Pulls groundwater levels state-by-state using the USGS Water Data OGC API
(https://api.waterdata.usgs.gov/ogcapi/v0/), with checkpointing so interrupted
runs can resume.  Site metadata is fetched via dataretrieval.nwis.get_info()
which still uses the supported /site endpoint.

The legacy /nwis/gwlevels/ endpoint was decommissioned on 2026-02-01 and now
returns an HTML redirect.  This script replaces the broken dataretrieval
get_gwlevels() call with a direct paginated call to:
  https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items
filtered by parameter_code=72019 (depth-to-water below land surface, ft).

Set environment variable USGS_API_KEY to avoid anonymous rate-limit throttling.
Sign up for a free key at https://api.waterdata.usgs.gov/signup/.

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
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import dataretrieval.nwis as nwis
import pandas as pd
import requests
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

# New USGS Water Data OGC API base URL
_OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"
_FM_ITEMS = f"{_OGC_BASE}/field-measurements/items"

# USGS groundwater level parameter code (depth to water, ft below land surface)
_GW_PARAM_CODE = "72019"

# Sites per HTTP request to the OGC API (keep URL short enough for GET)
_SITE_BATCH = 50

# Records returned per page
_PAGE_SIZE = 1000

# Maximum retries per state on transient API failures
MAX_RETRIES = 3
RETRY_DELAY_SEC = 30

# Retry configuration (applies to 429, Timeout, and ConnectionError)
_MAX_RETRIES_PER_REQUEST = 6
# Base delay for exponential backoff (seconds); doubles each attempt, capped at 5 min
_BACKOFF_BASE_SEC = 15.0
# Seconds to sleep between batches (polite baseline)
_INTER_BATCH_SLEEP_SEC = 2.0
# Request timeouts: (connect_timeout_s, read_timeout_s)
# Read timeout fires if no data arrives for this many seconds (not total duration).
# A shorter read timeout surfaces stalled connections quickly instead of hanging silently.
_REQUEST_TIMEOUT = (10, 60)
# Log batch progress every N batches (set to 0 to disable)
_LOG_BATCH_EVERY = 25

# Map new API qualifier strings → legacy lev_status_cd single-letter codes.
# Keys are lower-cased versions of qualifier values returned by the OGC API.
# Values of "" mean "valid/static" and are kept by qc_nwis.py's filter.
_QUALIFIER_TO_STATUS: dict[str, str] = {
    "static": "",
    "pumping": "P",
    "dry": "D",
    "flowing": "F",
    "obstructed": "O",
    "recently pumped": "R",
    "recently flowing nearby": "E",
    "nearby recently flowing": "G",
    "injecting": "I",
    "discontinued": "N",
    "foreign substance": "V",
    "other": "Z",
}


def _qualifiers_to_status(qualifiers: list[str] | None) -> str:
    """
    Convert a list of OGC API qualifier strings to a single lev_status_cd letter.

    Returns the first non-blank mapped code found, or "" if all are "Static"/unknown.
    """
    if not qualifiers:
        return ""
    for q in qualifiers:
        code = _QUALIFIER_TO_STATUS.get(q.lower(), "")
        if code:
            return code
    return ""


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


def _get_with_backoff(
    session: requests.Session,
    url: str,
    params: dict | None,
    max_retries: int = _MAX_RETRIES_PER_REQUEST,
) -> requests.Response:
    """
    GET request with exponential backoff for transient failures.

    Retries on:
    - HTTP 429 Too Many Requests (respects ``Retry-After`` header)
    - HTTP 503 Service Unavailable
    - ``requests.Timeout`` (connect or read stall)
    - ``requests.ConnectionError`` (network drop, DNS failure)

    All other 4xx/5xx HTTP errors are raised immediately.

    Parameters
    ----------
    session:
        Shared requests.Session.
    url:
        Request URL.
    params:
        Query parameters (passed only on the first request; pagination URLs are
        already fully qualified).
    max_retries:
        Maximum number of retries before re-raising.

    Returns
    -------
    requests.Response with a 2xx status code.
    """
    delay = _BACKOFF_BASE_SEC
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = delay
                logger.warning(
                    f"  {type(exc).__name__} — waiting {wait:.0f}s then retrying "
                    f"(attempt {attempt + 1}/{max_retries}) …"
                )
                time.sleep(wait)
                delay = min(delay * 2, 300.0)
                continue
            raise

        # Handle retryable HTTP status codes
        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = delay
            else:
                wait = delay
            if attempt < max_retries:
                label = "429 Too Many Requests" if resp.status_code == 429 else "503 Service Unavailable"
                logger.warning(
                    f"  {label} — waiting {wait:.0f}s then retrying "
                    f"(attempt {attempt + 1}/{max_retries}) …"
                )
                time.sleep(wait)
                delay = min(delay * 2, 300.0)
                continue
            resp.raise_for_status()

        resp.raise_for_status()
        return resp

    # Raise the last network error if all retries were connection/timeout failures
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Exceeded retry limit")  # unreachable


def _fetch_gw_levels_for_sites(
    site_nos: list[str],
    start_date: str,
    session: requests.Session,
    api_key: str | None,
) -> list[dict]:
    """
    Paginate through OGC field-measurements for a list of site numbers.

    Parameters
    ----------
    site_nos:
        USGS site numbers (without "USGS-" prefix).
    start_date:
        ISO date string, e.g. "2001-01-01".  Fetches from this date to present.
    session:
        Shared requests.Session for connection pooling.
    api_key:
        Optional USGS API key for higher rate limits.

    Returns
    -------
    list of record dicts compatible with the legacy *_gwlevels.parquet schema.
    """
    monitoring_ids = ",".join(f"USGS-{s}" for s in site_nos)
    base_params: dict[str, str] = {
        "f": "json",
        "monitoring_location_id": monitoring_ids,
        "parameter_code": _GW_PARAM_CODE,
        "datetime": f"{start_date}/..",
        "limit": str(_PAGE_SIZE),
    }
    if api_key:
        base_params["api_key"] = api_key

    records: list[dict] = []
    url: str | None = _FM_ITEMS
    params: dict | None = base_params

    while url is not None:
        resp = _get_with_backoff(session, url, params)
        payload = resp.json()

        for feat in payload.get("features", []):
            props = feat["properties"]
            # Extract site_no by stripping "USGS-" agency prefix
            mon_id: str = props.get("monitoring_location_id", "")
            site_no = mon_id.split("-", 1)[1] if "-" in mon_id else mon_id

            # Geometry coordinates (lon, lat)
            coords = (feat.get("geometry") or {}).get("coordinates")
            lat = coords[1] if coords else None
            lon = coords[0] if coords else None

            records.append({
                "site_no": site_no,
                "agency_cd": mon_id.split("-")[0] if "-" in mon_id else "USGS",
                "lev_dt": props.get("time", "")[:10],      # date portion of ISO timestamp
                "lev_tm": props.get("time", "")[11:19],     # time portion HH:MM:SS
                "lev_tz_cd": "UTC",
                "lev_va": props.get("value"),               # depth, ft (str → float in QC)
                "lev_status_cd": _qualifiers_to_status(props.get("qualifier")),
                "lev_meth_cd": props.get("observing_procedure_code"),
                "lev_unit_cd": props.get("unit_of_measure", "ft"),
                "vertical_datum": props.get("vertical_datum"),
                "approval_status": props.get("approval_status"),
                "lat": lat,
                "lon": lon,
            })

        # Follow pagination "next" link; clear params (URL is fully qualified)
        url = None
        params = None
        for lnk in payload.get("links", []):
            if lnk.get("rel") == "next":
                url = lnk["href"]
                break

    return records


def download_state_gwlevels(
    state_abbr: str,
    state_fips: str,
    start_date: str,
    output_dir: Path,
) -> dict:
    """
    Download groundwater levels + site metadata for one state.

    GW levels are fetched from the USGS Water Data OGC API
    (field-measurements collection, parameter_code=72019).
    Site metadata is fetched via dataretrieval.nwis.get_info().

    Returns a summary dict with record counts and status.
    """
    gw_path = output_dir / f"{state_abbr}_gwlevels.parquet"
    sites_path = output_dir / f"{state_abbr}_sites.parquet"
    api_key: str | None = os.environ.get("USGS_API_KEY")

    # --- Site metadata (dataretrieval /site endpoint, still supported) ---
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

    # --- Groundwater levels (OGC field-measurements API) ---
    logger.info(f"  Downloading GW levels for {state_abbr} (FIPS {state_fips})...")

    # Use existing sites file (just downloaded or from a previous run) to get
    # the list of site numbers to query.
    if sites_path.exists():
        site_nos = pd.read_parquet(sites_path, columns=["site_no"])["site_no"].astype(str).tolist()
    else:
        logger.warning(f"  No sites file for {state_abbr} — skipping GW level download")
        return {"n_levels": 0, "n_sites": n_sites, "timestamp": datetime.now(timezone.utc).isoformat()}

    all_records: list[dict] = []
    n_batches = (len(site_nos) + _SITE_BATCH - 1) // _SITE_BATCH

    with requests.Session() as session:
        session.headers.update({"Accept": "application/geo+json"})
        for batch_idx in range(n_batches):
            batch = site_nos[batch_idx * _SITE_BATCH: (batch_idx + 1) * _SITE_BATCH]
            try:
                recs = _fetch_gw_levels_for_sites(batch, start_date, session, api_key)
                all_records.extend(recs)
            except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"  Skipping batch {batch_idx + 1}/{n_batches} after retries: {e}")
            except Exception as e:
                logger.warning(f"  Error on batch {batch_idx + 1}/{n_batches}: {e}")
            # Per-batch progress heartbeat so long states don't appear frozen
            is_last = batch_idx == n_batches - 1
            if _LOG_BATCH_EVERY > 0 and (batch_idx % _LOG_BATCH_EVERY == 0 or is_last):
                logger.info(
                    f"  [{state_abbr}] batch {batch_idx + 1}/{n_batches} "
                    f"— {len(all_records):,} records so far"
                )
            # Polite delay between batches to respect API rate limits
            if not is_last:
                time.sleep(_INTER_BATCH_SLEEP_SEC)

    n_levels = len(all_records)
    if n_levels > 0:
        df_gw = pd.DataFrame(all_records)
        df_gw["lev_va"] = pd.to_numeric(df_gw["lev_va"], errors="coerce")
        df_gw.to_parquet(gw_path, index=False)
        logger.info(f"  → {n_levels:,} GW level records saved to {gw_path.name}")
    else:
        logger.info(f"  → No GW level records for {state_abbr}")

    return {
        "n_levels": n_levels,
        "n_sites": n_sites,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_download(start_date: str, output_dir: Path, states: list[str] | None = None) -> None:
    """Download NWIS GW data for all CONUS states (or a subset) with checkpointing.

    Parameters
    ----------
    start_date:
        ISO date string, e.g. "2000-01-01".
    output_dir:
        Directory to write per-state parquets and download_log.json.
    states:
        Optional list of state abbreviations (e.g. ["WA", "OR"]).  When
        provided, only these states are downloaded; all others are skipped.
        Already-completed states are still skipped via the checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_log.json"
    log = load_checkpoint(log_path)

    # Normalise requested state list
    if states:
        unknown = [s for s in states if s.upper() not in CONUS_STATES]
        if unknown:
            raise ValueError(f"Unknown state abbreviation(s): {unknown}. Valid: {sorted(CONUS_STATES)}")
        states_universe = {s.upper(): CONUS_STATES[s.upper()] for s in states}
    else:
        states_universe = CONUS_STATES

    states_todo = [
        (abbr, fips)
        for abbr, fips in sorted(states_universe.items())
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
    parser.add_argument(
        "--states",
        nargs="+",
        metavar="STATE",
        default=None,
        help="One or more state abbreviations to download (e.g. WA OR CA). "
             "Omit to download all CONUS states.",
    )
    args = parser.parse_args()
    run_download(args.start_date, args.output_dir, states=args.states)


if __name__ == "__main__":
    main()
