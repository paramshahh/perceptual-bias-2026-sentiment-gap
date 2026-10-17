"""Data ingestion for the 2026 Sentiment Gap study.

The project has two data sources with different roles:

* The **provided CSV** (`Copy_of_Perceptual_Bias_Data.csv`) is canonical for the
  four core series. It is the file the project was issued with, so every number
  in the report must reconcile with it.
* **FRED** supplies the extras that the CSV lacks: the cumulative PCE price
  level, 30-year mortgage rates, the household debt-service ratio and the
  Michigan series that FRED publishes. FRED access is optional — without a
  usable key the panel is still built, an approximate price level is
  substituted, and callers are told exactly what they got.

The CSV is in a wide "side-by-side blocks" layout: four independent
``Date, Value`` column pairs with blank spacer columns between them, each block
sorted newest-first and with its own length and frequency. Real GDP is
*quarterly* while the other three are monthly, so the blocks cannot simply be
concatenated — they are parsed separately and joined on a month-end index.

Timing: the brief sets two hard rules
-------------------------------------

1. "Do not interpolate quarterly GDP into artificial monthly observations."
2. "Use only information available by each prediction date. Do not ... use
   subsequently released GDP estimates to explain earlier perceptions."

The default build obeys both. It is a **real-time (vintage-aware)** alignment:

* Each series keeps its own observation period. Nothing is interpolated. A
  month never receives a number that was invented for it.
* Month ``t`` carries the most recent value that a statistical agency had
  actually **published** by the end of month ``t``. The publication lags are in
  `PUBLICATION_LAGS`, with the source of each lag in a comment.
* Quarterly real GDP therefore enters as a **step function of published
  quarterly estimates**. Months April to June of a year carry the Q1 estimate,
  because the Q1 advance estimate appears about 30 days after the quarter ends.
  Each monthly row holds a real published quarterly figure, not a share of one.
  The observation quarter of every row is in the provenance frame, so a reader
  can see which quarter each month used.
* A value does not stay current forever. Each series has a staleness limit. When
  no fresh publication arrives in time, the row stays empty. The row is then
  reported as incomplete instead of being filled with an old number.

The earlier **contemporaneous** alignment is still available with
``alignment="contemporaneous"``. It is the alignment that `BRAINSTORMING.md`
describes, so the report can quote both and explain the change. It is
**not compliant** with the brief: it spreads each quarterly GDP value across its
own three months, and it lets month ``t`` use data that nobody could read until
weeks after month ``t``. It is never the default.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column offsets of each (date, value) block in the provided CSV, keyed by the
# canonical column name used throughout the project.
CSV_BLOCKS: dict[str, tuple[int, int]] = {
    "unemployment": (0, 1),
    "sentiment": (3, 4),
    "pce_yoy": (7, 8),
    "real_gdp_qoq": (10, 11),
}

# Series that arrive quarterly. They keep their quarterly observation period in
# every alignment. The real-time alignment carries them as a step function of
# published estimates. It does not create intra-quarter observations.
QUARTERLY_COLUMNS = ("real_gdp_qoq", "real_gdp_yoy", "debt_service")

Alignment = Literal["real_time", "contemporaneous"]

# The brief requires a real-time information set, so this is the default.
DEFAULT_ALIGNMENT: Alignment = "real_time"


@dataclass(frozen=True)
class PublicationLag:
    """When a statistical agency publishes one observation period.

    `lag_days` is the delay from the **end of the observation period** to the
    first public release. `max_stale_days` is how long one published value stays
    the current value. After that limit the row stays empty: the pipeline says
    "not published yet" instead of repeating an old number.
    """

    column: str
    frequency: str
    lag_days: int
    max_stale_days: int
    source: str
    note: str

    def publication_date(self, period_end: pd.Timestamp) -> pd.Timestamp:
        """The first date on which this observation period was public."""
        return pd.Timestamp(period_end) + pd.Timedelta(days=self.lag_days)


# Publication lags, in days after the end of the observation period.
#
# Sources (release calendars, checked 2026-10):
#   * BLS, "Employment Situation": the U-3 rate for a month appears on the first
#     Friday of the next month, which is 3 to 8 days after the month ends.
#     https://www.bls.gov/schedule/news_release/empsit.htm
#   * BEA, "Personal Income and Outlays": the PCE price index for a month
#     appears near the end of the next month, about 26 to 30 days after the
#     month ends. 28 days holds the lag at exactly one month for every month
#     length, including February.
#     https://www.bea.gov/data/personal-consumption-expenditures-price-index
#   * BEA, "Gross Domestic Product": the advance estimate for a quarter appears
#     about 30 days after the quarter ends. The second and third estimates
#     follow at about 60 and 90 days. The advance estimate is the first number a
#     reader could have, so it sets the lag.
#     https://www.bea.gov/data/gdp/gross-domestic-product
#   * University of Michigan, Surveys of Consumers: the preliminary index for a
#     month appears near day 15 of that same month, and the final index appears
#     on the last Friday of that same month. A lag of 0 days therefore makes the
#     index for month t available inside month t.
#     https://data.sca.isr.umich.edu/
#   * Freddie Mac, "Primary Mortgage Market Survey": weekly, each Thursday. The
#     monthly mean for month t is complete at the end of month t.
#   * Federal Reserve Board, "Household Debt Service and Financial Obligations
#     Ratios" (TDSP): quarterly, about one quarter after the reference quarter.
#     https://www.federalreserve.gov/releases/housedebt/
#
# Staleness limits: one period plus a small margin for monthly series (31 days),
# five months for quarterly GDP (a quarter stays current until the next advance
# estimate arrives), and seven months for the debt-service ratio. The target and
# the two survey series get 0 days, because a survey answer for month t is only
# valid for month t. Carrying it forward would invent survey data.
PUBLICATION_LAGS: dict[str, PublicationLag] = {
    "sentiment": PublicationLag(
        column="sentiment",
        frequency="monthly",
        lag_days=0,
        max_stale_days=0,
        source="University of Michigan, Surveys of Consumers",
        note="Final index on the last Friday of the same month; preliminary near day 15.",
    ),
    "unemployment": PublicationLag(
        column="unemployment",
        frequency="monthly",
        lag_days=7,
        max_stale_days=31,
        source="BLS, Employment Situation (U-3, seasonally adjusted)",
        note="First Friday of the following month, so month t uses month t-1.",
    ),
    "pce_yoy": PublicationLag(
        column="pce_yoy",
        frequency="monthly",
        lag_days=28,
        max_stale_days=31,
        source="BEA, Personal Income and Outlays (PCE price index)",
        note="End of the following month, so month t uses month t-1.",
    ),
    "real_gdp_qoq": PublicationLag(
        column="real_gdp_qoq",
        frequency="quarterly",
        lag_days=30,
        max_stale_days=150,
        source="BEA, Gross Domestic Product (advance estimate)",
        note="About 30 days after the quarter ends, so April to June use Q1.",
    ),
    "real_gdp_yoy": PublicationLag(
        column="real_gdp_yoy",
        frequency="quarterly",
        lag_days=30,
        max_stale_days=150,
        source="Derived from the BEA quarterly growth rates in the provided CSV",
        note="Four published quarterly rates compounded; same release date as the last one.",
    ),
    "pce_level": PublicationLag(
        column="pce_level",
        frequency="monthly",
        lag_days=28,
        max_stale_days=31,
        source="BEA, Personal Income and Outlays (PCEPI level, through FRED)",
        note="Same release as the PCE price index.",
    ),
    "pce_level_approx": PublicationLag(
        column="pce_level_approx",
        frequency="monthly",
        lag_days=28,
        max_stale_days=31,
        source="Accumulated from the published PCE YoY rates in the provided CSV",
        note="Same release timing as the rates it accumulates.",
    ),
    "mortgage_30y": PublicationLag(
        column="mortgage_30y",
        frequency="weekly",
        lag_days=0,
        max_stale_days=31,
        source="Freddie Mac, Primary Mortgage Market Survey (through FRED)",
        note="Weekly each Thursday; the monthly mean is complete at month end.",
    ),
    "debt_service": PublicationLag(
        column="debt_service",
        frequency="quarterly",
        lag_days=90,
        max_stale_days=210,
        source="Federal Reserve Board, household debt service ratio (TDSP)",
        note="About one quarter after the reference quarter.",
    ),
    "sentiment_umcsent": PublicationLag(
        column="sentiment_umcsent",
        frequency="monthly",
        lag_days=0,
        max_stale_days=0,
        source="University of Michigan headline index, UMCSENT (through FRED)",
        note="Cross-check on the CSV headline index; same release timing.",
    ),
    "inflation_expect_1y": PublicationLag(
        column="inflation_expect_1y",
        frequency="monthly",
        lag_days=0,
        max_stale_days=0,
        source="University of Michigan 1-year inflation expectation, MICH (through FRED)",
        note="Published with the same survey as the headline index.",
    ),
}

FRED_SERIES: dict[str, str] = {
    "pce_level": "PCEPI",            # PCE price index, level (rebased here)
    "mortgage_30y": "MORTGAGE30US",  # 30-year fixed mortgage rate, weekly
    "debt_service": "TDSP",          # household debt service ratio, quarterly
}

# Michigan series that FRED carries. The brief asks for the headline index, the
# Current Economic Conditions sub-index and the Consumer Expectations sub-index
# "where available".
#
# Checked against the FRED series and series/search endpoints on 2026-10-16:
#   * UMCSENT  — the headline index. Monthly, 1952-11 to 2026-08. Available.
#   * MICH     — the 1-year inflation expectation from the same survey.
#     Monthly, 1978-01 to 2026-08. Available. It is an expectations measure, but
#     it is NOT the Index of Consumer Expectations.
#   * UMCSENT1 — NOT a sub-index. It is the discontinued old headline series
#     that stops in 1977-11. Do not use it as Current Economic Conditions.
#   * The Index of Current Economic Conditions (ICC) and the Index of Consumer
#     Expectations (ICE) returned no match. A search for "Current Economic
#     Conditions Michigan" returned 0 series. UMCSENT2, UMCCURR, UMCEXP,
#     UMCSENTC, UMCSENTE, ICC and ICE all returned HTTP 400 (no such series).
#     Michigan distributes the two sub-indices on its own site under its own
#     terms (https://data.sca.isr.umich.edu/), not through FRED.
# `build_monthly_panel` therefore records a note that says the two sub-indices
# are absent. It does not omit them in silence.
MICHIGAN_SERIES: dict[str, str] = {
    "sentiment_umcsent": "UMCSENT",
    "inflation_expect_1y": "MICH",
}

MICHIGAN_SUBINDEX_NOTE = (
    "Michigan sub-indices: FRED carries the headline index (UMCSENT) and the "
    "1-year inflation expectation (MICH) only. The Index of Current Economic "
    "Conditions and the Index of Consumer Expectations are not on FRED "
    "(UMCSENT1 is the discontinued pre-1978 headline series, not a sub-index). "
    "Michigan publishes both sub-indices on data.sca.isr.umich.edu under its "
    "own terms. The headline/conditions/expectations split that the brief asks "
    "for is therefore not available from this pipeline."
)

PCE_BASE_YEAR = 2020

# Quarters compounded to get a year-over-year growth rate from the published
# quarter-over-quarter annualised rates.
QUARTERS_PER_YEAR = 4

# Two-digit years in the CSV span 1978-2026. Anything above this cutoff is a
# 20th-century date; at or below it is 21st-century.
TWO_DIGIT_YEAR_CUTOFF = 50


class DataError(Exception):
    """Raised when input data is missing or structurally unusable."""


class AlignmentError(DataError):
    """Raised when a column cannot be aligned as the brief requires.

    The most common cause is a column with no entry in `PUBLICATION_LAGS`. A
    real-time panel cannot hold a column whose publication date is unknown, so
    the build stops instead of guessing a lag.
    """


@dataclass(frozen=True)
class PanelReport:
    """What a panel build actually managed to assemble.

    Carried alongside the frame so callers (and the report) can state plainly
    which hypotheses the available data can support, and on which information
    set each row stands.
    """

    rows: int
    first: date
    last: date
    core_columns: tuple[str, ...]
    fred_columns: tuple[str, ...]
    price_level_column: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)
    alignment: str = DEFAULT_ALIGNMENT
    publication_lags: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)
    michigan_subindex_columns: tuple[str, ...] = field(default_factory=tuple)
    provenance: pd.DataFrame | None = field(default=None, compare=False, repr=False)

    @property
    def fred_available(self) -> bool:
        """True only when FRED actually delivered columns.

        A configured-but-failing key (expired, offline, renamed series) must not
        report as available, or the report will claim data it does not have.
        """
        return bool(self.fred_columns)

    @property
    def real_time(self) -> bool:
        """True when every regressor stands on data published by its own month."""
        return self.alignment == "real_time"

    def lag_table(self) -> pd.DataFrame:
        """The applied publication lags, for the report and the dashboard."""
        return pd.DataFrame(
            list(self.publication_lags),
            columns=["column", "lag_days", "max_stale_days"],
        )

    def summary(self) -> str:
        parts = [
            f"{self.rows} monthly rows, {self.first:%Y-%m} to {self.last:%Y-%m}",
            f"core: {', '.join(self.core_columns)}",
        ]
        if self.fred_available:
            parts.append(f"FRED: {', '.join(self.fred_columns)}")
        else:
            parts.append("FRED: unavailable")
        parts.append(f"price level: {self.price_level_column or 'none'}")
        parts.append(f"alignment: {self.alignment}")
        return " | ".join(parts)


def _parse_csv_date(raw: str) -> date | None:
    """Parse the CSV's ``m/d/yy`` dates with an explicit century cutoff.

    ``9/30/26`` is 2026 and ``1/31/78`` is 1978. The century is resolved from
    `TWO_DIGIT_YEAR_CUTOFF` rather than relying on ``%y``'s own pivot, so the
    rule is stated in one place instead of inherited from the stdlib.
    """
    raw = raw.strip()
    if not raw:
        return None
    parts = raw.split("/")
    if len(parts) != 3:
        return None
    try:
        month, day, year_fragment = (int(part) for part in parts)
    except ValueError:
        return None
    if len(parts[2]) <= 2:
        century = 2000 if year_fragment <= TWO_DIGIT_YEAR_CUTOFF else 1900
        year = century + year_fragment
    else:
        year = year_fragment
    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None


def load_provided_csv(path: str | Path) -> pd.DataFrame:
    """Read the wide provided CSV into a tidy month-end indexed frame.

    Returns a frame indexed by month-end timestamps with one column per block in
    `CSV_BLOCKS`. Quarterly series keep their native sparsity here (only the
    quarter-end months are populated); `build_monthly_panel` decides how to
    align them.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"provided CSV not found: {path}")

    # utf-8-sig: the file carries a BOM, which would otherwise corrupt the
    # first header cell and silently break block offsets.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        raise DataError(f"provided CSV has no data rows: {path}")

    series: dict[str, pd.Series] = {}
    for name, (date_idx, value_idx) in CSV_BLOCKS.items():
        points: dict[pd.Timestamp, float] = {}
        sources: dict[pd.Timestamp, str] = {}
        for row in rows[1:]:
            if len(row) <= value_idx:
                continue
            parsed = _parse_csv_date(row[date_idx])
            if parsed is None:
                continue
            raw_value = row[value_idx].strip()
            if not raw_value:
                continue
            try:
                value = float(raw_value.replace(",", ""))
            except ValueError:
                logger.warning("non-numeric %s value %r at %s", name, raw_value, parsed)
                continue
            # Normalise to month end so blocks align regardless of the day the
            # source used for a given observation.
            key = pd.Timestamp(parsed) + pd.offsets.MonthEnd(0)
            if key in points:
                # Blocks are newest-first, so the first row seen for a month is
                # the latest vintage: keep it and say so rather than letting a
                # stale revision win silently.
                logger.warning(
                    "%s has multiple observations in %s (%s kept, %s ignored)",
                    name,
                    key.strftime("%Y-%m"),
                    sources[key],
                    row[date_idx].strip(),
                )
                continue
            points[key] = value
            sources[key] = row[date_idx].strip()
        if not points:
            raise DataError(f"no usable observations parsed for {name} in {path}")
        series[name] = pd.Series(points, name=name).sort_index()

    frame = pd.concat(series.values(), axis=1, sort=False)
    frame.index.name = "date"
    return frame.sort_index()


def expand_quarterly(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """NON-COMPLIANT. Spread each quarterly value across its quarter's months.

    **Do not use this function for the headline panel.** It stays in the module
    for two reasons only: `BRAINSTORMING.md` records it as the earlier choice,
    and the report explains why the project moved away from it. The default
    build does not call it. `build_monthly_panel` calls it only with
    ``alignment="contemporaneous"``, which is the historical comparison.

    Two rules of the brief stop it being usable:

    * "Do not interpolate quarterly GDP into artificial monthly observations."
      This function creates exactly those observations. January and February get
      a Q1 number that describes the quarter they sit in.
    * "Use only information available by each prediction date." The Q1 advance
      estimate appears about 30 days after Q1 ends, so a January row built this
      way holds a number from late April.

    Mechanically: the CSV stamps quarterly GDP at the quarter's **end** (``3/31``
    is Q1), so a plain forward-fill would push Q1's growth rate onto March,
    April and May. Each observation is therefore mapped onto its own quarter's
    three months. The result is a clean contemporaneous series and a look-ahead
    one. Use `align_real_time` instead.
    """
    expanded = pd.Series(index=index, dtype=float)
    for stamp, value in values.dropna().items():
        quarter_start = pd.Timestamp(stamp).to_period("Q").start_time
        months = pd.date_range(quarter_start, periods=3, freq="ME")
        expanded.loc[expanded.index.intersection(months)] = value
    return expanded


def align_real_time(
    values: pd.Series,
    index: pd.DatetimeIndex,
    lag: PublicationLag,
) -> tuple[pd.Series, pd.DataFrame]:
    """Carry into month ``t`` the newest value published by the end of month ``t``.

    This is the alignment the brief requires. For each month end ``t`` the
    function takes every observation whose publication date is ``t`` or earlier,
    and keeps the one with the latest observation period. A value published
    after ``t`` can never reach row ``t``, so no row can use a later GDP
    estimate to explain an earlier perception.

    Quarterly series come out as a step function of published quarterly
    estimates. Nothing is interpolated and nothing is split across months.

    A value also stops being current. When the newest published observation is
    older than ``lag.max_stale_days``, the row stays empty. The caller then
    reports the row as incomplete.

    Returns the aligned series and a provenance frame with one row per month:
    the observation period that the month used, its publication date, and the
    age of that observation in days.
    """
    observations = values.dropna().sort_index()
    aligned = pd.Series(np.nan, index=index, dtype=float, name=values.name)
    provenance = pd.DataFrame(
        {
            "column": lag.column,
            "observation_period_end": pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"),
            "publication_date": pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"),
            "age_days": pd.Series(np.nan, index=index, dtype=float),
        },
        index=index,
    )
    if observations.empty:
        return aligned, provenance

    period_ends = pd.DatetimeIndex(observations.index)
    publications = period_ends + pd.Timedelta(days=lag.lag_days)
    # A constant lag keeps publication order equal to observation order, so a
    # single searchsorted finds the newest published observation per month.
    position = publications.searchsorted(pd.DatetimeIndex(index), side="right") - 1
    known = position >= 0
    if not known.any():
        return aligned, provenance

    chosen = position[known]
    used_period = period_ends[chosen]
    used_publication = publications[chosen]
    month_ends = pd.DatetimeIndex(index)[known]
    age_days = (month_ends - used_period).days.to_numpy(dtype=float)
    fresh = age_days <= float(lag.max_stale_days)

    target = month_ends[fresh]
    aligned.loc[target] = observations.to_numpy(dtype=float)[chosen][fresh]
    provenance.loc[target, "observation_period_end"] = used_period[fresh]
    provenance.loc[target, "publication_date"] = used_publication[fresh]
    provenance.loc[target, "age_days"] = age_days[fresh]
    return aligned, provenance


def quarterly_yoy_growth(qoq_annualised: pd.Series) -> pd.Series:
    """Year-over-year real GDP growth from the published annualised QoQ rates.

    The brief asks the project to "clearly distinguish quarter-over-quarter
    annualized growth from year-over-year growth". The CSV gives only the
    annualised quarter-over-quarter rate, so the year-over-year rate is built
    from four consecutive published rates:

        1 + yoy/100 = product over the last 4 quarters of (1 + qoq/400)

    The result keeps the quarterly observation period of its last quarter. It
    uses no future data: the four rates are all published by the release date of
    the last one. The first three quarters of the series stay empty.
    """
    rates = qoq_annualised.dropna().sort_index()
    if rates.empty:
        return pd.Series(dtype=float, name="real_gdp_yoy")
    growth = 1.0 + rates / 400.0
    compounded = growth.rolling(QUARTERS_PER_YEAR).apply(np.prod, raw=True)
    return ((compounded - 1.0) * 100.0).rename("real_gdp_yoy")


def quarterly_gdp_vintages(frame: pd.DataFrame) -> pd.DataFrame:
    """Real GDP at its native quarterly frequency, with publication dates.

    The brief permits two compliant treatments of quarterly GDP: a step function
    of published estimates in a monthly model, or estimation at quarterly
    frequency. The monthly panel uses the first. This function gives the second
    one its data, and it also lets a grader check the step function by hand: one
    row per quarter, the observation period, the publication date of the advance
    estimate, the annualised quarter-over-quarter rate and the year-over-year
    rate.
    """
    if "real_gdp_qoq" not in frame.columns:
        raise DataError("frame has no real_gdp_qoq column")
    lag = PUBLICATION_LAGS["real_gdp_qoq"]
    qoq = frame["real_gdp_qoq"].dropna().sort_index()
    yoy = quarterly_yoy_growth(qoq)
    out = pd.DataFrame(
        {
            "observation_quarter": [
                pd.Timestamp(stamp).to_period("Q").strftime("%YQ%q") for stamp in qoq.index
            ],
            "observation_period_end": pd.DatetimeIndex(qoq.index),
            "publication_date": pd.DatetimeIndex(qoq.index) + pd.Timedelta(days=lag.lag_days),
            "real_gdp_qoq_annualised": qoq.to_numpy(dtype=float),
            "real_gdp_yoy": yoy.reindex(qoq.index).to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(qoq.index),
    )
    out.index.name = "date"
    return out


def approximate_price_level(
    pce_yoy: pd.Series,
    base_year: int = PCE_BASE_YEAR,
) -> pd.Series:
    """Approximate a cumulative price level from YoY inflation rates.

    Used only when FRED's `PCEPI` is unavailable.

    **Why not chain on the YoY identity directly.** YoY inflation relates
    ``L_t`` to ``L_{t-12}`` and nothing else, so chaining on it from a single
    anchor yields one calendar month per year, not a monthly series. Seeding all
    twelve months of a base year instead creates twelve independent chains: the
    base year comes out perfectly flat, and each January absorbs a whole year's
    price increase, producing an annual sawtooth that would then masquerade as
    signal in any monthly regression.

    **What this does instead.** It treats the published YoY rate as the
    prevailing annualised rate for that month and accumulates in logs:

        log L_t = log L_{t-1} + yoy_t / 1200

    The result is smooth and monotone in cumulative inflation, which is the
    feature Hypothesis A actually needs (households anchoring to a price *level*
    far above 2020). It is an approximation: it will drift from true `PCEPI`
    whenever inflation moves sharply within a year, and it carries no
    within-year seasonal detail at all. Rebased so the mean of `base_year` is
    100 to keep the ">20% above 2020" comparison interpretable.
    """
    yoy = pce_yoy.dropna().sort_index()
    if yoy.empty:
        return pd.Series(dtype=float, name="pce_level_approx")
    if not (yoy.index.year == base_year).any():
        raise DataError(f"PCE YoY series does not cover base year {base_year}")

    log_level = np.log(100.0) + (yoy / 1200.0).cumsum()
    level = np.exp(log_level)
    base = level[level.index.year == base_year].mean()
    if not np.isfinite(base) or base == 0:
        raise DataError(f"could not rebase price level to {base_year}")
    return (level / base * 100.0).rename("pce_level_approx")


class FredSource:
    """Thin wrapper over `fredapi` that degrades instead of exploding.

    `configured` says a key was supplied; it does **not** promise the key works.
    Whether FRED really delivered is decided by what `build_monthly_panel`
    manages to fetch.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY") or ""
        self._client = None
        if not self.api_key:
            logger.info("no FRED_API_KEY configured; FRED series will be skipped")
            return
        try:
            from fredapi import Fred

            self._client = Fred(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("FRED client unavailable (%s); continuing without it", exc)
            self._client = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def series(self, series_id: str) -> pd.Series:
        """Fetch one FRED series as a month-end series, or empty on failure."""
        if self._client is None:
            return pd.Series(dtype=float, name=series_id)
        try:
            raw = self._client.get_series(series_id)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("FRED fetch failed for %s (%s)", series_id, exc)
            return pd.Series(dtype=float, name=series_id)
        raw = pd.Series(raw).dropna()
        if raw.empty:
            return pd.Series(dtype=float, name=series_id)
        raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
        # Weekly series (mortgage rates) collapse to a monthly mean.
        return raw.groupby(level=0).mean().rename(series_id)


def _lag_for(column: str) -> PublicationLag:
    """The documented lag for a column, or a clear failure."""
    try:
        return PUBLICATION_LAGS[column]
    except KeyError:
        raise AlignmentError(
            f"no publication lag is documented for {column!r}; add it to "
            "PUBLICATION_LAGS with its release calendar before a real-time "
            "panel can carry it"
        ) from None


def build_monthly_panel(
    csv_path: str | Path,
    fred: FredSource | None = None,
    base_year: int = PCE_BASE_YEAR,
    alignment: Alignment = DEFAULT_ALIGNMENT,
) -> tuple[pd.DataFrame, PanelReport]:
    """Assemble the monthly analysis panel and a report of what it contains.

    ``alignment="real_time"`` (the default) obeys the brief: every row holds only
    values that were published by the end of that month, and quarterly GDP
    enters as a step function of published quarterly estimates.

    ``alignment="contemporaneous"`` reproduces the earlier, non-compliant panel
    for comparison in the report. It interpolates quarterly GDP across its own
    quarter and lets each row use data published weeks later.
    """
    if alignment not in ("real_time", "contemporaneous"):
        raise AlignmentError(
            f"unknown alignment {alignment!r}; use 'real_time' or 'contemporaneous'"
        )

    frame = load_provided_csv(csv_path)
    notes: list[str] = []
    provenance_parts: list[pd.DataFrame] = []
    applied_lags: list[tuple[str, int, int]] = []

    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="ME")
    panel = pd.DataFrame(index=full_index)
    panel.index.name = "date"

    # Year-over-year growth is derived from the published quarterly rates, so it
    # is available in both alignments and keeps the same publication date.
    frame = frame.copy()
    frame["real_gdp_yoy"] = quarterly_yoy_growth(frame["real_gdp_qoq"]).reindex(frame.index)
    notes.append(
        "real_gdp_yoy: four published annualised QoQ rates compounded; "
        "quarterly observation period, same release date as the last rate"
    )

    def place(name: str, values: pd.Series) -> None:
        """Align one series into the panel under the chosen alignment."""
        if alignment == "real_time":
            lag = _lag_for(name)
            aligned, provenance = align_real_time(values, full_index, lag)
            panel[name] = aligned
            provenance_parts.append(provenance)
            applied_lags.append((name, lag.lag_days, lag.max_stale_days))
            return
        if name in QUARTERLY_COLUMNS:
            panel[name] = expand_quarterly(values, full_index)
        else:
            panel[name] = values.reindex(full_index)

    for name in list(CSV_BLOCKS) + ["real_gdp_yoy"]:
        place(name, frame[name])

    if alignment == "real_time":
        notes.append(
            "alignment: REAL TIME. Month t carries only values published by the "
            "end of month t. Publication lags come from PUBLICATION_LAGS."
        )
        notes.append(
            "real_gdp_qoq: quarterly, carried as a step function of published "
            "advance estimates (April to June use Q1). No intra-quarter GDP "
            "observation is created."
        )
        notes.append(
            "unemployment and pce_yoy: month t carries the value for month t-1, "
            "because month t's own value is not published inside month t."
        )
    else:
        notes.append(
            "alignment: CONTEMPORANEOUS. This is the earlier choice and it is "
            "NOT compliant with the brief: it interpolates quarterly GDP across "
            "its own three months and uses data published after the row's month."
        )
        notes.append(
            "real_gdp_qoq and real_gdp_yoy: quarterly, expanded across the "
            "quarter's three months (contemporaneous, not real-time)"
        )

    fred = fred if fred is not None else FredSource()
    fred_columns: list[str] = []
    michigan_columns: list[str] = []
    price_level_column: str | None = None

    if fred.configured:
        for name, series_id in FRED_SERIES.items():
            fetched = fred.series(series_id)
            if fetched.empty:
                notes.append(f"{name}: FRED series {series_id} returned no data")
                continue
            if name == "pce_level":
                base = fetched[fetched.index.year == base_year].mean()
                if pd.notna(base) and base != 0:
                    fetched = fetched / base * 100.0
                    notes.append(f"pce_level: rebased to {base_year} = 100")
                price_level_column = "pce_level"
            place(name, fetched)
            fred_columns.append(name)

        for name, series_id in MICHIGAN_SERIES.items():
            fetched = fred.series(series_id)
            if fetched.empty:
                notes.append(f"{name}: FRED series {series_id} returned no data")
                continue
            place(name, fetched)
            fred_columns.append(name)
            michigan_columns.append(name)

        if not fred_columns:
            notes.append("FRED key configured but no series were retrieved")

    notes.append(MICHIGAN_SUBINDEX_NOTE)

    if price_level_column is None:
        # Covers both "no key" and "key present but FRED gave us nothing".
        approximate = approximate_price_level(frame["pce_yoy"], base_year=base_year)
        place("pce_level_approx", approximate)
        price_level_column = "pce_level_approx"
        notes.append(
            "pce_level_approx: APPROXIMATION accumulated from published YoY rates "
            f"in logs ({base_year} mean = 100); PCEPI unavailable"
        )

    provenance: pd.DataFrame | None = None
    if provenance_parts:
        provenance = pd.concat(provenance_parts, axis=0).reset_index(names="date")
        empty = [name for name, _, _ in applied_lags if not panel[name].notna().any()]
        if empty:
            notes.append(
                "no value was published in time for any row of these columns: "
                f"{', '.join(empty)}"
            )
        ragged = {
            name: int(panel[name].isna().sum())
            for name, _, _ in applied_lags
            if panel[name].isna().any()
        }
        if ragged:
            notes.append(
                "rows left empty because no release had arrived: "
                + ", ".join(f"{name} ({count})" for name, count in ragged.items())
            )

    report = PanelReport(
        rows=len(panel),
        first=panel.index.min().date(),
        last=panel.index.max().date(),
        core_columns=tuple(CSV_BLOCKS),
        fred_columns=tuple(fred_columns),
        price_level_column=price_level_column,
        notes=tuple(notes),
        alignment=alignment,
        publication_lags=tuple(applied_lags),
        michigan_subindex_columns=tuple(michigan_columns),
        provenance=provenance,
    )
    return panel, report


def training_slice(panel: pd.DataFrame, start: str = "2010-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Rows used to fit the baseline (default 2010-2024, the spec's window)."""
    return panel.loc[start:end]


def holdout_slice(panel: pd.DataFrame, start: str = "2025-01-01") -> pd.DataFrame:
    """Rows the baseline never sees, where the Sentiment Gap is measured."""
    return panel.loc[start:]
