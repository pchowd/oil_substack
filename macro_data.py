"""Load macro series for oil-shock analysis (prices, activity, inflation).

Data files live in ``data/macro/``. Use :func:`refresh_macro_files` to re-download
from FRED, CPB, and the IMF SDMX API.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests

MACRO_DIR = Path(__file__).resolve().parent / "data" / "macro"

BRENT_FRED_CSV = MACRO_DIR / "brent_monthly_fred.csv"
KILIAN_IGREA_FRED_CSV = MACRO_DIR / "kilian_igrea_fred.csv"
CPB_WTM_XLSX = MACRO_DIR / "cpb_world_trade_monitor.xlsx"
IMF_CPI_WCA_CSV = MACRO_DIR / "imf_cpi_wca_monthly.csv"

FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILBRENTEU"
FRED_IGREA_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IGREA"
# CPB publishes a new cumulative workbook each month (~2-month lag). Since 2025 the
# files moved from omnidownload/ to system/files/cpbmedia/ with lowercase names.
CPB_WTM_URL = (
    "https://www.cpb.nl/system/files/cpbmedia/"
    "cpb-world-trade-monitor-february-2026.xlsx"
)
IMF_CPI_WCA_API_KEY = "G001+U150+U019+U142.CPI._T.IX.M"
IMF_CPI_WCA_URL = (
    "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI_WCA"
    f"/~/{IMF_CPI_WCA_API_KEY}?c[TIME_PERIOD]=ge:2000-M01"
)

# IMF CPI_WCA aggregate codes (CL_COUNTRY in the CPI_WCA DSD).
IMF_CPI_CODES: Mapping[str, str] = {
    "world": "G001",
    "europe": "U150",
    "americas": "U019",
    "asia": "U142",
}
IMF_CPI_LABELS: Mapping[str, str] = {
    code: label for label, code in {
        "World": "G001",
        "Europe": "U150",
        "Americas": "U019",
        "Asia": "U142",
    }.items()
}

# CPB inpro_out row labels (seasonally adjusted industrial-production index, _sm).
CPB_IP_LABELS: Mapping[str, str] = {
    "world": "World",
    "united_states": "United States",
    "euro_area": "Euro Area",
    "china": "China",
    "emerging_asia": "Emerging Asia excl China",
    "latin_america": "Latin America",
}
DEFAULT_CPB_IP_REGIONS = tuple(CPB_IP_LABELS.keys())
DEFAULT_IMF_CPI_REGIONS = tuple(IMF_CPI_CODES.keys())

_CP_DATE_COL = 5  # first monthly column in inpro_out (cols 0–4 are metadata/weights)


def _fred_csv_to_series(path: Path, value_column: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    series = df.set_index("observation_date")[value_column].astype(float)
    series.index = series.index.to_period("M").to_timestamp()
    series.name = value_column.lower()
    return series.sort_index()


def load_brent_monthly(path: Path | None = None) -> pd.Series:
    """Brent spot, USD/bbl (FRED ``MCOILBRENTEU``), monthly."""
    return _fred_csv_to_series(path or BRENT_FRED_CSV, "MCOILBRENTEU").rename("brent_usd_bbl")


def load_kilian_igrea(path: Path | None = None) -> pd.Series:
    """Kilian global real economic activity index (FRED ``IGREA``), monthly."""
    return _fred_csv_to_series(path or KILIAN_IGREA_FRED_CSV, "IGREA").rename("kilian_igrea")


def _cpb_region_key(label: str) -> str | None:
    clean = re.sub(r"\s+", " ", str(label).strip())
    for key, target in CPB_IP_LABELS.items():
        if clean == target:
            return key
    return None


def parse_cpb_industrial_production(
    path: Path | None = None,
    regions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Parse CPB World Trade Monitor ``inpro_out`` (seasonally adjusted IP index)."""
    path = path or CPB_WTM_XLSX
    raw = pd.read_excel(path, sheet_name="inpro_out", header=None)

    date_cells = raw.iloc[3, _CP_DATE_COL :]
    dates = pd.to_datetime(date_cells.astype(str).str.replace("m", "-", regex=False), format="%Y-%m")

    wanted = set(regions or DEFAULT_CPB_IP_REGIONS)
    rows: dict[str, pd.Series] = {}
    for i in range(7, raw.shape[0]):
        code = raw.iloc[i, 2]
        if not isinstance(code, str) or not code.endswith("_sm"):
            continue
        key = _cpb_region_key(raw.iloc[i, 1])
        if key is None or key not in wanted:
            continue
        values = pd.to_numeric(raw.iloc[i, _CP_DATE_COL :], errors="coerce")
        rows[key] = pd.Series(values.values, index=dates, name=key)

    if not rows:
        raise ValueError(f"No CPB regions matched {sorted(wanted)} in {path}")

    out = pd.DataFrame(rows).sort_index()
    out.index.name = "date"
    return out


def load_cpb_industrial_production(
    regions: Iterable[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """CPB monthly industrial-production index by region (``inpro_out``, SA)."""
    return parse_cpb_industrial_production(path=path, regions=regions)


def yoy_inflation_from_cpi(cpi: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Year-over-year headline inflation in percent: ``100 * (CPI_t / CPI_{t-12} - 1)``."""
    if isinstance(cpi, pd.Series):
        return 100.0 * (cpi / cpi.shift(12) - 1.0)
    return 100.0 * (cpi / cpi.shift(12) - 1.0)


def parse_imf_cpi_wca_csv(path: Path | None = None) -> pd.DataFrame:
    """Parse saved IMF CPI_WCA monthly index (all-item headline, 2010=100-style index)."""
    path = path or IMF_CPI_WCA_CSV
    df = pd.read_csv(path, parse_dates=["date"])
    wide = df.pivot(index="date", columns="region", values="cpi_index")
    wide.index = wide.index.to_period("M").to_timestamp()
    wide.index.name = "date"
    return wide.sort_index()


def load_imf_cpi_index(
    regions: Iterable[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """IMF CPI_WCA monthly headline CPI index for world / Europe / Americas / Asia."""
    wide = parse_imf_cpi_wca_csv(path=path)
    wanted = list(regions or DEFAULT_IMF_CPI_REGIONS)
    missing = [r for r in wanted if r not in wide.columns]
    if missing:
        raise KeyError(f"Unknown IMF CPI regions: {missing}. Available: {list(wide.columns)}")
    return wide[wanted]


def load_imf_cpi_yoy(
    regions: Iterable[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """IMF CPI_WCA year-over-year inflation (percent), derived from monthly CPI index."""
    return yoy_inflation_from_cpi(load_imf_cpi_index(regions=regions, path=path))


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_imf_cpi_wca_monthly(save_path: Path | None = None) -> pd.DataFrame:
    """Download IMF CPI_WCA monthly CPI and write a long-format CSV."""
    save_path = save_path or IMF_CPI_WCA_CSV
    resp = requests.get(IMF_CPI_WCA_URL, headers={"Accept": "text/csv"}, timeout=120)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    raw = raw[raw["STRUCTURE_ID"].astype(str).str.contains("CPI_WCA", na=False)]

    code_to_region = {code: key for key, code in IMF_CPI_CODES.items()}
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["TIME_PERIOD"].str.replace("-M", "-", regex=False)),
            "imf_code": raw["COUNTRY"],
            "region": raw["COUNTRY"].map(code_to_region),
            "cpi_index": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
        }
    ).dropna(subset=["cpi_index", "region"])
    out = out.sort_values(["region", "date"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    return out


def refresh_macro_files() -> dict[str, Path]:
    """Re-download FRED, CPB, and IMF files into ``data/macro/``."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    brent = _download_bytes(FRED_BRENT_URL)
    BRENT_FRED_CSV.write_bytes(brent)
    paths["brent"] = BRENT_FRED_CSV

    igrea = _download_bytes(FRED_IGREA_URL)
    KILIAN_IGREA_FRED_CSV.write_bytes(igrea)
    paths["kilian_igrea"] = KILIAN_IGREA_FRED_CSV

    cpb = _download_bytes(CPB_WTM_URL)
    CPB_WTM_XLSX.write_bytes(cpb)
    paths["cpb_wtm"] = CPB_WTM_XLSX

    fetch_imf_cpi_wca_monthly()
    paths["imf_cpi"] = IMF_CPI_WCA_CSV

    return paths


def load_macro_panel(
    cpb_regions: Iterable[str] | None = None,
    imf_regions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convenience join of Brent, Kilian IGREA, CPB IP, and IMF CPI YoY on month-end."""
    panel = pd.concat(
        [
            load_brent_monthly(),
            load_kilian_igrea(),
            load_cpb_industrial_production(regions=cpb_regions).add_prefix("cpb_ip_"),
            load_imf_cpi_yoy(regions=imf_regions).add_prefix("cpi_yoy_"),
        ],
        axis=1,
    )
    panel.index.name = "date"
    return panel
