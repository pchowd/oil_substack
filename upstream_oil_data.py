"""Load upstream oil production CSV exports by calendar year."""

from __future__ import annotations

import calendar
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

JODI_KEY_COLUMNS = (
    "REF_AREA",
    "TIME_PERIOD",
    "ENERGY_PRODUCT",
    "FLOW_BREAKDOWN",
)

# JODI unit codes (see jodi-oil-wdb-item-names-ver2017.pdf).
UNIT_KBD = "KBD"  # thousand barrels per day
UNIT_KBBL = "KBBL"  # thousand barrels (monthly volume or stock level)
UNIT_KL = "KL"  # thousand kilolitres
UNIT_KTONS = "KTONS"  # thousand metric tons
UNIT_CONVBBL = "CONVBBL"  # barrels per kton conversion factor (not a quantity)

STOCK_LEVEL_FLOWS = frozenset({"CLOSTLV"})
QUANTITY_UNITS = frozenset({UNIT_KBD, UNIT_KBBL, UNIT_KL, UNIT_KTONS})
LITERS_PER_BARREL = 158.987294928
MISSING_OBS_VALUES = frozenset({"-", "N/A", "x", "..", ""})
TargetUnit = Literal["kbbl", "bbl", "kbd"]


def available_years() -> list[int]:
    """Return sorted years for which a CSV file exists in ``data/``."""
    years: list[int] = []
    for path in DATA_DIR.glob("*.csv"):
        stem = path.stem
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def _csv_path_for_year(year: int) -> Path:
    path = DATA_DIR / f"{year}.csv"
    if not path.is_file():
        available = ", ".join(str(y) for y in available_years()) or "(none)"
        raise FileNotFoundError(
            f"No upstream oil data for {year}. Available years: {available}."
        )
    return path


def load_year(year: int, **read_csv_kwargs) -> pd.DataFrame:
    """Load upstream oil production data for ``year``.

    Parameters
    ----------
    year
        Calendar year encoded in the CSV filename (e.g. ``2020`` -> ``data/2020.csv``).
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Columns: REF_AREA, TIME_PERIOD, ENERGY_PRODUCT, FLOW_BREAKDOWN,
        UNIT_MEASURE, OBS_VALUE, ASSESSMENT_CODE.
    """
    path = _csv_path_for_year(year)
    defaults = {"dtype": {"REF_AREA": "string", "TIME_PERIOD": "string"}}
    merged = {**defaults, **read_csv_kwargs}
    return pd.read_csv(path, **merged)


def load_years(years: Iterable[int], **read_csv_kwargs) -> pd.DataFrame:
    """Concatenate data for multiple years."""
    frames = [load_year(year, **read_csv_kwargs) for year in years]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_jodi_value(value: object) -> float:
    """Parse a JODI ``OBS_VALUE`` cell to float, returning NaN for missing markers."""
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
    elif pd.api.types.is_scalar(value) and pd.isna(value):
        return np.nan
    else:
        text = str(value).strip()
    if text in MISSING_OBS_VALUES:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def is_missing_obs_value(value: object) -> bool:
    """Return True for JODI missing-value markers (``-``, ``x``, ``N/A``, etc.)."""
    return np.isnan(parse_jodi_value(value))


def drop_empty_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop long-format rows with no usable quantity.

    Removes rows where ``UNIT_MEASURE`` is ``CONVBBL`` (conversion factor only)
    or ``OBS_VALUE`` is a JODI missing marker (``-``, ``x``, ``N/A``, ``..``).
    """
    required = {"UNIT_MEASURE", "OBS_VALUE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    quantity_rows = df["UNIT_MEASURE"].isin(QUANTITY_UNITS)
    has_value = ~df["OBS_VALUE"].map(is_missing_obs_value)
    return df.loc[quantity_rows & has_value].reset_index(drop=True)


def _days_in_month(time_period: str) -> int:
    year, month = map(int, time_period.split("-"))
    return calendar.monthrange(year, month)[1]


def _to_kbbl(
    *,
    flow_breakdown: str,
    days_in_month: int,
    kbb: float,
    kl: float,
    ktons: float,
    conv_bbl: float,
    kbd: float,
) -> tuple[float, str | None]:
    """Convert one wide row to thousand barrels (``KBBL`` scale).

    Conversion rules follow JODI-Oil WDB unit definitions:
    - ``KBBL``: already in thousand barrels
    - ``KL``: thousand kilolitres -> thousand barrels
    - ``KTONS`` with ``CONVBBL``: barrels = ktons * conv; then / 1000 for kbbl
    - ``KBD``: thousand barrels/day; monthly flow total = kbd * days in month
      (not used for closing-stock levels)
    """
    if not np.isnan(kbb):
        return kbb, UNIT_KBBL
    if not np.isnan(kl):
        return kl * 1000.0 / LITERS_PER_BARREL, UNIT_KL
    if not np.isnan(ktons) and not np.isnan(conv_bbl):
        return ktons * conv_bbl / 1000.0, UNIT_KTONS
    if flow_breakdown not in STOCK_LEVEL_FLOWS and not np.isnan(kbd):
        return kbd * days_in_month, UNIT_KBD
    return np.nan, None


def standardize_units(
    df: pd.DataFrame,
    *,
    target: TargetUnit = "kbbl",
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Collapse JODI long-format unit rows to a single standardized quantity.

    Parameters
    ----------
    df
        JODI primary-table data in long format (as returned by :func:`load_year`).
    target
        Output unit:

        - ``"kbbl"``: thousand barrels (JODI ``KBBL`` scale; default)
        - ``"bbl"``: barrels (= ``kbbl * 1000``)
        - ``"kbd"``: thousand barrels per day (= ``kbbl / days_in_month``;
          meaningful for flows, not stock levels)
    drop_missing
        If True (default), drop country/month/product/flow rows with no valid
        quantity in any unit (e.g. all ``OBS_VALUE`` entries are ``-`` or ``x``).

    Returns
    -------
    pandas.DataFrame
        One row per country / month / product / flow with columns
        ``REF_AREA``, ``TIME_PERIOD``, ``ENERGY_PRODUCT``, ``FLOW_BREAKDOWN``,
        ``OBS_VALUE`` (standardized), ``SOURCE_UNIT`` (which JODI unit was used),
        ``ASSESSMENT_CODE``, and ``DAYS_IN_MONTH``.
    """
    required = set(JODI_KEY_COLUMNS) | {"UNIT_MEASURE", "OBS_VALUE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    work = df.copy()
    work = work[work["UNIT_MEASURE"].isin(QUANTITY_UNITS | {UNIT_CONVBBL})]
    work["OBS_VALUE_NUM"] = work["OBS_VALUE"].map(parse_jodi_value)

    value_wide = work.pivot_table(
        index=list(JODI_KEY_COLUMNS),
        columns="UNIT_MEASURE",
        values="OBS_VALUE_NUM",
        aggfunc="first",
    )
    assessment_wide = work.pivot_table(
        index=list(JODI_KEY_COLUMNS),
        columns="UNIT_MEASURE",
        values="ASSESSMENT_CODE",
        aggfunc="first",
    )

    rows: list[dict[str, object]] = []
    for key, measure_row in value_wide.iterrows():
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(JODI_KEY_COLUMNS, key))
        days = _days_in_month(str(record["TIME_PERIOD"]))
        kbbl, source_unit = _to_kbbl(
            flow_breakdown=str(record["FLOW_BREAKDOWN"]),
            days_in_month=days,
            kbb=float(measure_row.get(UNIT_KBBL, np.nan)),
            kl=float(measure_row.get(UNIT_KL, np.nan)),
            ktons=float(measure_row.get(UNIT_KTONS, np.nan)),
            conv_bbl=float(measure_row.get(UNIT_CONVBBL, np.nan)),
            kbd=float(measure_row.get(UNIT_KBD, np.nan)),
        )

        if source_unit is None:
            obs_value = np.nan
            assessment = np.nan
        else:
            if target == "kbbl":
                obs_value = kbbl
            elif target == "bbl":
                obs_value = kbbl * 1000.0
            elif target == "kbd":
                obs_value = kbbl / days
            else:
                raise ValueError(f"Unsupported target unit: {target!r}")

            assessment = assessment_wide.loc[key, source_unit]
            if pd.isna(assessment) and source_unit == UNIT_KTONS:
                assessment = assessment_wide.loc[key, UNIT_CONVBBL]

        rows.append(
            {
                **record,
                "OBS_VALUE": obs_value,
                "SOURCE_UNIT": source_unit,
                "ASSESSMENT_CODE": assessment,
                "DAYS_IN_MONTH": days,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(list(JODI_KEY_COLUMNS)).reset_index(drop=True)
    if drop_missing:
        out = out[out["SOURCE_UNIT"].notna()].reset_index(drop=True)
    return out
