"""Comparative regressions, Granger precedence and heteroskedasticity IV.

This module carries the econometric evidence for **Hypothesis B** (media
negativity drives part of the Sentiment Gap). It provides three instruments of
increasing identification strength, and it is written so that the *label on the
strength* travels with the number:

1. :func:`compare_models` — Models 1/2/3 on a common sample with HAC errors.
   Produces the report's Table 3. Evidence class: **association**.
2. :func:`granger_media_gap` — VAR with AIC/BIC lag selection, ADF stationarity
   checks, and Granger F-tests in *both* directions. Evidence class:
   **predictive precedence**. Not causation — see `GRANGER_CAVEAT`.
3. :func:`rigobon_sack_iv` — identification through heteroskedasticity
   (Rigobon 2003; Rigobon & Sack 2003/2004), with the rank and order conditions
   counted for the system *as actually specified*. Evidence class:
   **instrumented, conditional on a stated variance-shift assumption**.

Section 4 of the brief asks for more than those three, and the rest of the
module supplies it:

4. :func:`compare_model_ladder` — the brief's three-rung ladder: economic only,
   economic plus media, and economic plus media plus household or political
   controls (`build_household_political_controls` fetches the controls).
5. :func:`seasonality_report` — a test for monthly seasonality, with
   ``seasonal_control`` in `compare_models` to control it when it is there.
6. :func:`rolling_origin_forecast` — expanding-window, one-step-ahead
   out-of-sample RMSE and MAE per model, with a Diebold-Mariano test against
   the economic-only model. Evidence class: **predictive accuracy**.
7. :func:`coefficient_stability` — rolling coefficient paths with confidence
   bands, and a stated stability rule.
8. :func:`specification_sensitivity` — the headline coefficient recomputed
   across a documented grid of reasonable choices.
9. :func:`multiple_testing_correction` with :func:`collect_media_test_family` —
   Holm and Benjamini-Hochberg over the declared family of media tests, with
   raw and adjusted p-values side by side.
10. :func:`reverse_causality_evidence` — the brief's reverse-causality
    discussion as output: both Granger directions and the lead-lag
    correlations per media measure, scored against the brief's narratives.
11. :func:`causal_design_proposals` — the three identification designs as
    machine-readable objects (estimand, assumptions, data, threats,
    falsification), which is the brief's "detailed plan for stronger
    identification".

**Supply is not exposure.** `EXPOSURE_GAP_STATEMENT` travels on every result
object in this module as ``exposure_gap``. The corpus measures what publishers
produced. It does not measure what Michigan respondents met, and nothing here
makes the step from one to the other.

Two project-wide constraints shape every function here.

**Power is the binding constraint, not technique.** The out-of-sample gap has
19 monthly observations. A VAR at six lags or a two-regime covariance split
estimated on 19 points returns output that *looks* like a result and is not.
Every entry point therefore refuses by default (`EconometricsError`) when the
sample cannot support the requested lag structure or regime split, and
``allow_underpowered=True`` is the only way past — it still returns, but with
``underpowered=True`` and the arithmetic spelled out in ``power_warnings``.
Nothing in this module can hand back a bare float; results are dataclasses that
carry their own caveats.

**Rows are never silently dropped.** Every result object names the excluded
dates and why, in the style of `PanelReport` / `PredictionReport`.

Structural-break testing (Hypothesis C) deliberately lives elsewhere, in
`survey_diagnostics`.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

TARGET = "sentiment"

# Model 1: the naive macro baseline (mirrors `baseline_model.BASELINE_FEATURES`).
MODEL1_FEATURES: tuple[str, ...] = ("unemployment", "pce_yoy", "real_gdp_qoq")

# Model 2 adds the Hypothesis-A level/burden regressors.
# `debt_service` (FRED `TDSP`) is deliberately excluded. It is insignificant
# under HAC errors (coefficient -1.604, p = 0.256) and the series stops at
# 2026-04, so including it removes two months from the out-of-sample window
# that every other model keeps. It stays available in the panel for anyone who
# wants to test it; pass it through `model2_extra` to put it back.
MODEL2_EXTRA_FEATURES: tuple[str, ...] = ("pce_level", "mortgage_30y")

# Model 3 adds the Hypothesis-B media regressors. These are produced by
# `nlp_media_svd` and may legitimately not exist yet, so Model 3 is reported as
# *unavailable* rather than raising: a missing corpus is a state of the project,
# not a programming error.
MEDIA_FEATURES: tuple[str, ...] = (
    "media_tone",             # (pos - neg) / total, in [-1, +1]
    "topic_salience",         # SVD/LSA economic-topic share of coverage
    "political_attribution",  # share of coverage attributing outcomes politically
    "framing",                # price-LEVEL framing vs inflation-RATE framing
)

# The framing measure carries a sign predicted BEFORE estimation, which is the
# only way a framing coefficient can be evidence rather than a description.
#
# `framing` is positive when coverage frames the economy by the price LEVEL and
# negative when it frames it by the falling inflation RATE. Hypothesis A says
# households anchor to the level of prices, not to the rate of change. Under
# that account, level-framed coverage is the coverage that AGREES with how
# households already think, so:
#
#   PREDICTED SIGN: NEGATIVE. More level framing goes with lower sentiment,
#   conditional on the measured economic indicators.
#
# A positive estimate contradicts the anchoring account. An estimate
# indistinguishable from zero says the framing measure carries no information
# about the gap. Both are reportable results.
PREDICTED_SIGNS: dict[str, str] = {
    "framing": "negative",
    "media_tone": "positive",
    "topic_salience": "negative",
    "political_attribution": "negative",
}
PREDICTED_SIGN_NOTES: dict[str, str] = {
    "framing": (
        "positive framing values mean price-LEVEL framing; Hypothesis A's "
        "anchoring account predicts a NEGATIVE coefficient"
    ),
    "media_tone": (
        "higher tone is less negative coverage, so a POSITIVE coefficient is "
        "the media-negativity prediction"
    ),
    "topic_salience": (
        "more attention to the economy under a negativity account predicts a "
        "NEGATIVE coefficient"
    ),
    "political_attribution": (
        "more political attribution of economic outcomes predicts a NEGATIVE "
        "coefficient under the framing account"
    ),
}

# Columns that may stand in for `pce_level` when FRED was unavailable, in
# preference order. `data_loader` substitutes an approximation and says so; the
# comparison table has to pick it up or Model 2 would be reported as missing
# data it actually has.
PCE_LEVEL_FALLBACKS: tuple[str, ...] = ("pce_level", "pce_level_approx")

# Newey-West truncation, matching the rest of the project (monthly data).
HAC_LAGS = 12

# --- Power thresholds -------------------------------------------------------
# Below `ABSOLUTE_MIN_REGRESSION_OBS` no HAC regression is reported at all.
ABSOLUTE_MIN_REGRESSION_OBS = 24
# Observations per estimated parameter below which a fit is flagged underpowered.
MIN_OBS_PER_PARAMETER = 8
# A VAR at lag p with 2 variables estimates 2p+1 parameters per equation.
MIN_OBS_PER_VAR_PARAMETER = 3
# Each volatility regime needs at least this many observations for its own
# covariance matrix to mean anything.
MIN_REGIME_OBS = 15
DEFAULT_VOLATILITY_WINDOW = 6
DEFAULT_VOLATILITY_QUANTILE = 0.5
DEFAULT_BOOTSTRAP_DRAWS = 1000

GRANGER_CAVEAT = (
    "Granger causality is PREDICTIVE PRECEDENCE, NOT CAUSATION. A rejection "
    "says lagged values of one series improve the forecast of the other given "
    "that other series' own past. A common driver that moves both -- an "
    "election cycle, a shared shock -- can produce exactly this pattern with "
    "no causal channel at all. This is why the Rigobon-Sack IV is attempted."
)

RIGOBON_SACK_ASSUMPTION = (
    "Identification assumes the regime split shifts the variance of the MEDIA "
    "shock only, with the structural coefficients and the gap-shock variance "
    "constant across regimes. This is the Rigobon-Sack assumption, not a "
    "result: it is not testable from these two moment matrices, and it is the "
    "single weakest link in the IV estimate."
)


# --- The brief's third rung: household and political controls --------------
# Section 4 of the brief asks for three models: economic indicators only,
# economic plus media, and economic plus media plus "additional household or
# political controls". The third rung was absent from this module, so the
# ladder was not reproducible. These are the controls, with the FRED series
# that supply them. Each id was verified to return data on 2026-10-16.
HOUSEHOLD_POLITICAL_SERIES: dict[str, str] = {
    "policy_uncertainty": "USEPUINDXD",  # Economic Policy Uncertainty, daily
    "real_dpi_pc": "A229RX0",            # real disposable income per capita
    "u6_rate": "U6RATE",                 # labour underutilisation, U-6
}

# The regressors that the third rung adds. `real_dpi_pc` enters as a
# year-over-year percentage change, because the level has a strong trend and a
# trend regressor in a 116-month window absorbs the gap instead of explaining
# it. `election_proximity` is calculated here, offline, from the calendar.
HOUSEHOLD_POLITICAL_FEATURES: tuple[str, ...] = (
    "policy_uncertainty",
    "real_dpi_pc_yoy",
    "u6_rate",
    "election_proximity",
)

# US federal general elections. Election day is the Tuesday after the first
# Monday in November of each even year. The project window needs 2010 forward.
FEDERAL_ELECTION_DATES: tuple[str, ...] = (
    "2010-11-02",
    "2012-11-06",
    "2014-11-04",
    "2016-11-08",
    "2018-11-06",
    "2020-11-03",
    "2022-11-08",
    "2024-11-05",
    "2026-11-03",
)

# Columns that may stand in for a canonical regressor, in preference order.
# `data_loader` can change what it emits -- for example it must not interpolate
# quarterly GDP into monthly observations -- so every model resolves its
# regressors against what the panel actually carries and says what it used.
FEATURE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "pce_level": ("pce_level", "pce_level_approx"),
    "real_gdp_qoq": (
        "real_gdp_qoq",
        "real_gdp_qoq_rt",
        "real_gdp_qoq_vintage",
        "real_gdp_qoq_asof",
        "real_gdp_yoy",
    ),
}

# What a substitute column really is, so that the note on the result says it.
SUBSTITUTION_NOTES: dict[str, str] = {
    "pce_level_approx": (
        "an approximation accumulated from published YoY rates, not true PCEPI"
    ),
    "real_gdp_qoq_rt": (
        "a real-time (first-release) GDP series, not the current vintage"
    ),
    "real_gdp_qoq_vintage": "a stated GDP vintage, not the current estimate",
    "real_gdp_qoq_asof": "GDP as of each prediction date, not the current vintage",
    "real_gdp_yoy": (
        "year-over-year GDP growth in place of the quarter-over-quarter "
        "annualised rate; the two are different quantities"
    ),
}

# A regressor whose non-null share over the candidate rows is below this is
# dropped from the specification with a named reason, rather than silently
# deleting two thirds of the sample through `dropna`. A quarterly series that
# is correctly NOT interpolated into monthly values has a coverage near 0.33,
# so this guard is what keeps a real-time GDP column from destroying the
# monthly panel.
MIN_FEATURE_COVERAGE = 0.6

# --- Out-of-sample and stability thresholds --------------------------------
# The first training block of a rolling-origin evaluation.
DEFAULT_INITIAL_TRAIN = 60
# Fewer folds than this and an out-of-sample RMSE is a number, not a measure.
MIN_FORECAST_FOLDS = 12
# A rolling coefficient window shorter than this cannot carry HAC inference.
MIN_STABILITY_WINDOW = 36
MIN_STABILITY_WINDOWS = 5
# A seasonality test estimates 11 month dummies; below this it cannot.
MIN_SEASONALITY_OBS = 36

SEASONALITY_COVARIANCE_NOTE = (
    "The joint test uses HC3 errors, not the project's HAC(12). A Newey-West "
    "kernel and monthly dummies interact badly, because the dummies are "
    "themselves periodic with period 12: a Monte Carlo over independent "
    "normal noise (40 draws, 144 months, this module's own dummy design) "
    "rejected a true null in 65 percent of draws at HAC(12), 42 percent at "
    "HAC(6) and 7.5 percent at HC3 against a nominal 5 percent. HC3 is "
    "therefore the default, and it is conservative for a persistent series. "
    "Pass covariance='HAC' to reproduce the over-rejecting version."
)

EXPOSURE_GAP_STATEMENT = (
    "WHAT THIS MEASURES IS CONTENT SUPPLY, NOT EXPOSURE. The corpus is "
    "national economic coverage. It records what publishers produced. It does "
    "not record what Michigan respondents read, saw, remembered or believed. "
    "No step in this module measures exposure, and no coefficient here can be "
    "read as the effect of exposure on sentiment. The inferential step from "
    "supply to exposure is NOT MADE: it is left open, and section 4 of the "
    "brief requires it to be named rather than assumed."
)

TRANSPORTABILITY_REQUIREMENTS: tuple[str, ...] = (
    ("Exposure estimates: individual-level or market-level measures of which "
    "economic coverage each respondent actually met, for example panel "
    "metering, platform delivery logs, or survey recall items."),
    ("Representative sampling: an exposure sample drawn to the same frame as "
    "the Michigan survey, because a volunteer or platform sample over-counts "
    "heavy news users."),
    ("Weighting: post-stratification of the exposure sample to the Michigan "
    "frame on age, income, geography and political affiliation, which the "
    "brief names as a moderator and a confounder."),
    ("Transportability assumptions: the effect of coverage on sentiment must "
    "not vary with the variables on which the two samples differ, or the "
    "variation must be modelled. This assumption is not testable inside one "
    "sample."),
    ("Aggregation: a per-person effect transports to the national index only "
    "under a stated aggregation rule, because the index is a weighted survey "
    "statistic and not a population mean of a treatment response."),
)

REVERSE_CAUSALITY_NARRATIVES: dict[str, str] = {
    "consumers_seek_negative_content": (
        "Pessimistic consumers seek negative content. Sentiment moves first "
        "and coverage follows, through demand for confirming material. This "
        "narrative is about EXPOSURE choice, which this corpus cannot see."
    ),
    "publishers_respond_to_demand": (
        "Publishers respond to audience demand. Sentiment moves first and "
        "coverage follows, through editorial selection. This narrative is "
        "about SUPPLY, which this corpus can see."
    ),
    "common_shock": (
        "An economic shock moves coverage and sentiment together with no "
        "channel between them. This narrative predicts contemporaneous "
        "association with no precedence in either direction."
    ),
}


class EconometricsError(Exception):
    """Raised when an econometric procedure cannot be run as specified.

    Follows the project's `DataError` / `ModelError` pattern: fail with our own
    type before pandas, numpy or statsmodels can surface an internal exception
    that callers would have to know about to catch.
    """


def _require_columns(frame: pd.DataFrame, columns) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise EconometricsError(f"frame is missing required columns: {missing}")


def _label(stamp) -> str:
    """Format an index entry for a human-readable dropped-rows message."""
    if isinstance(stamp, pd.Timestamp):
        return stamp.strftime("%Y-%m")
    return str(stamp)


def _missing_reasons(subset: pd.DataFrame, dropped) -> tuple[str, ...]:
    """Name each dropped row and the columns that caused it."""
    reasons = []
    for stamp in dropped:
        row = subset.loc[stamp]
        blanks = [name for name in subset.columns if pd.isna(row[name])]
        reasons.append(f"{_label(stamp)}: missing {', '.join(blanks)}")
    return tuple(reasons)


# ---------------------------------------------------------------------------
# 1. Comparative regression framework (report Table 3)
# ---------------------------------------------------------------------------


@dataclass
class ModelFit:
    """One row of the comparison table, available or not.

    An *unavailable* model is a first-class outcome: `available=False` with
    `missing_features` populated says "the corpus isn't built yet", which is
    different from, and must not be confused with, "the media regressors did
    not help".
    """

    name: str
    requested_features: tuple[str, ...]
    features: tuple[str, ...]
    available: bool
    missing_features: tuple[str, ...] = field(default_factory=tuple)
    unavailable_reason: str | None = None
    n: int = 0
    r_squared: float = float("nan")
    adj_r_squared: float = float("nan")
    rmse: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    coefficients: pd.DataFrame | None = None
    ols_result: object | None = None
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Regressors that were requested, exist in the panel, and were still left
    # out -- a sparse quarterly column, for example. Named, with the reason.
    dropped_features: tuple[str, ...] = field(default_factory=tuple)
    drop_reasons: tuple[str, ...] = field(default_factory=tuple)
    substitutions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_parameters(self) -> int:
        return len(self.features) + 1  # + intercept

    def significant_features(self, alpha: float = 0.05) -> tuple[str, ...]:
        """Features whose HAC p-value clears `alpha` (intercept excluded).

        Used by the falsification conditions: Hypothesis B requires media tone
        to be significant here, and "insignificant" has to be readable off the
        result object rather than eyeballed from a printed table.
        """
        if self.coefficients is None:
            return ()
        frame = self.coefficients
        hits = frame.index[(frame["p_value"] < alpha) & (frame.index != "const")]
        return tuple(str(name) for name in hits)


@dataclass
class ModelComparison:
    """Models 1-3 on one common sample, plus what that sample cost.

    Fitting on a *common* sample is the whole point: R-squared and AIC are only
    comparable across specifications estimated on identical rows. Because
    Model 2's and Model 3's extra regressors are shorter series than the
    baseline's, enforcing a common sample throws rows away, and the count is
    reported rather than absorbed.
    """

    models: dict[str, ModelFit]
    common_sample_n: int
    common_start: pd.Timestamp | None
    common_end: pd.Timestamp | None
    rows_offered: int
    dropped_dates: tuple[pd.Timestamp, ...]
    dropped_reasons: tuple[str, ...]
    hac_lags: int
    target: str
    evidence_class: str = "association (no identification claimed)"
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Which of our model names answers each rung of the brief's section 4
    # ladder. A rung with the value None was not estimated, and the reason is
    # in `ladder_notes`.
    brief_ladder: dict[str, str | None] = field(default_factory=dict)
    ladder_notes: tuple[str, ...] = field(default_factory=tuple)
    seasonal_control: str = "none"
    seasonal_terms: tuple[str, ...] = field(default_factory=tuple)
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(name for name, fit in self.models.items() if fit.available)

    @property
    def underpowered(self) -> bool:
        return any(fit.underpowered for fit in self.models.values() if fit.available)

    def table(self) -> pd.DataFrame:
        """Table 3: one row per model, fit statistics side by side.

        Unavailable models keep their row with NaN statistics so the report
        shows explicitly that Model 3 was attempted, not omitted.
        """
        rows = []
        for name, fit in self.models.items():
            rows.append(
                {
                    "model": name,
                    "available": fit.available,
                    "n": fit.n if fit.available else np.nan,
                    "k": fit.n_parameters if fit.available else np.nan,
                    "r_squared": fit.r_squared,
                    "adj_r_squared": fit.adj_r_squared,
                    "rmse": fit.rmse,
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "underpowered": fit.underpowered,
                    "missing_features": ", ".join(fit.missing_features),
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def coefficient_table(self) -> pd.DataFrame:
        """Long-format coefficients with HAC standard errors and p-values."""
        frames = []
        for name, fit in self.models.items():
            if not fit.available or fit.coefficients is None:
                continue
            block = fit.coefficients.copy()
            block.index.name = "regressor"
            block = block.reset_index()
            block.insert(0, "model", name)
            frames.append(block)
        columns = [
            "model",
            "regressor",
            "coef",
            "std_err",
            "t_stat",
            "p_value",
            "ci_lower",
            "ci_upper",
        ]
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True).loc[:, columns]

    def summary(self) -> str:
        parts = [
            f"common sample n={self.common_sample_n}"
            + (
                f" ({_label(self.common_start)}..{_label(self.common_end)})"
                if self.common_start is not None
                else ""
            ),
            f"available: {', '.join(self.available_models) or 'none'}",
        ]
        unavailable = [n for n, f in self.models.items() if not f.available]
        if unavailable:
            parts.append(f"unavailable: {', '.join(unavailable)}")
        if self.dropped_dates:
            parts.append(f"{len(self.dropped_dates)} rows dropped for missing inputs")
        if self.underpowered:
            parts.append("UNDERPOWERED")
        return " | ".join(parts)


def resolve_price_level_column(frame: pd.DataFrame) -> str | None:
    """Pick the available cumulative-price-level column, exact before proxy.

    `data_loader` emits `pce_level` from FRED's PCEPI when it can and
    `pce_level_approx` when it cannot. Model 2 should use whichever exists, and
    say which, rather than silently reporting itself as missing data.
    """
    for name in PCE_LEVEL_FALLBACKS:
        if name in frame.columns:
            return name
    return None


def resolve_feature_column(frame: pd.DataFrame, name: str) -> tuple[str | None, str]:
    """Pick the column that stands for `name`, exact before documented proxy.

    Returns the column that exists and a note when a substitute was taken. A
    missing regressor returns ``(None, "")``: the caller decides whether that
    makes a model unavailable or only shorter.
    """
    for candidate in FEATURE_FALLBACKS.get(name, (name,)):
        if candidate in frame.columns:
            if candidate == name:
                return candidate, ""
            note = f"'{candidate}' used in place of '{name}'"
            detail = SUBSTITUTION_NOTES.get(candidate)
            return candidate, f"{note}: {detail}" if detail else note
    return None, ""


def column_coverage(frame: pd.DataFrame, column: str, within_span: bool = False) -> float:
    """Non-null share of a column, 0.0 for an empty frame or a missing column.

    ``within_span=True`` measures the share inside the column's own first-to-
    last valid range. The two numbers answer different questions, and the
    difference is what separates two kinds of incomplete column:

    * a series that covers a shorter PERIOD than the panel -- the media
      measures start in 2017 -- has a span coverage near 1.0. It is usable, and
      the cost is that the common sample is shorter;
    * a series that is INTERNALLY sparse -- a quarterly rate that is correctly
      not interpolated into monthly values -- has a span coverage near 0.33.
      Keeping it would delete two of every three months.

    The specification guard uses the span measure, so that a short series is
    kept and an internally sparse one is dropped by name.
    """
    if column not in frame.columns or len(frame) == 0:
        return 0.0
    series = frame[column]
    if within_span:
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            return 0.0
        series = series.loc[first:last]
        if len(series) == 0:
            return 0.0
    return float(series.notna().mean())


def _resolve_specification(
    panel: pd.DataFrame,
    requested: tuple[str, ...],
    min_coverage: float,
) -> tuple[tuple[str, ...], list[str], list[str], list[str], list[str]]:
    """Resolve requested regressors against the panel that actually exists.

    Three outcomes per regressor, and all three are named in the result:

    * present -- the column, or a documented substitute, is there and dense
      enough to use;
    * missing -- no column of that name or any fallback name exists;
    * dropped -- the column exists but its non-null share is below
      `min_coverage`, which is what a correctly non-interpolated quarterly
      series looks like in a monthly panel.
    """
    present: list[str] = []
    missing: list[str] = []
    dropped: list[str] = []
    drop_reasons: list[str] = []
    substitutions: list[str] = []
    for name in requested:
        column, note = resolve_feature_column(panel, name)
        if column is None:
            missing.append(name)
            continue
        span_coverage = column_coverage(panel, column, within_span=True)
        if span_coverage < min_coverage:
            overall = column_coverage(panel, column)
            dropped.append(column)
            drop_reasons.append(
                f"'{column}' dropped: inside its own first-to-last valid range "
                f"its non-null share is {span_coverage:.2f}, below "
                f"{min_coverage:.2f} ({overall:.2f} over all "
                f"{len(panel)} candidate rows). A quarterly series that is "
                "correctly not interpolated into monthly values looks like "
                "this; keeping it would delete most of the sample through "
                "listwise deletion. A series that merely starts later keeps a "
                "span coverage near 1.0 and is NOT dropped here."
            )
            continue
        if note:
            substitutions.append(note)
        present.append(column)
    return tuple(present), missing, dropped, drop_reasons, substitutions


def month_dummies(index: pd.DatetimeIndex, base_month: int = 1) -> pd.DataFrame:
    """Eleven monthly indicators, with `base_month` as the omitted category.

    Used to control calendar seasonality when `seasonality_report` finds it.
    The base month is omitted so the design keeps full rank with a constant.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise EconometricsError("month dummies need a DatetimeIndex")
    frame = pd.DataFrame(index=index)
    for month in range(1, 13):
        if month == base_month:
            continue
        frame[f"month_{month:02d}"] = (index.month == month).astype(float)
    return frame


def _fit_one(
    name: str,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    requested: tuple[str, ...],
    target: str,
    hac_lags: int,
    allow_underpowered: bool,
) -> ModelFit:
    """HAC-OLS one specification on an already-common sample."""
    n = len(frame)
    k = len(features) + 1
    warnings_out: list[str] = []
    notes: list[str] = [f"HAC (Newey-West) errors, maxlags={hac_lags}"]

    if n <= k:
        message = (
            f"{name}: n={n} observations cannot identify {k} parameters "
            f"({features})"
        )
        if not allow_underpowered:
            raise EconometricsError(message)
        return ModelFit(
            name=name,
            requested_features=requested,
            features=features,
            available=False,
            unavailable_reason=message,
            n=n,
            underpowered=True,
            power_warnings=(message,),
        )

    if n < ABSOLUTE_MIN_REGRESSION_OBS:
        message = (
            f"{name}: n={n} is below the minimum of {ABSOLUTE_MIN_REGRESSION_OBS} "
            f"for HAC inference; Newey-West with maxlags={hac_lags} on {n} "
            "observations has no asymptotic justification"
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " (pass allow_underpowered=True to report it anyway)"
            )
        warnings_out.append(message)

    if n < MIN_OBS_PER_PARAMETER * k:
        warnings_out.append(
            f"{name}: n={n} is fewer than {MIN_OBS_PER_PARAMETER} observations "
            f"per parameter ({k} parameters); coefficients are fragile"
        )
    if hac_lags >= n / 2:
        warnings_out.append(
            f"{name}: HAC maxlags={hac_lags} is at least half of n={n}; the "
            "long-run variance estimate is near-degenerate"
        )

    design = sm.add_constant(frame.loc[:, list(features)], has_constant="add")
    endog = frame[target]
    with warnings.catch_warnings():
        # statsmodels warns about small samples and kernel bandwidth; those are
        # exactly the conditions we are reporting ourselves, and `pyproject`
        # turns DeprecationWarning into an error.
        warnings.simplefilter("ignore")
        result = sm.OLS(endog, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}
        )

    confidence = result.conf_int()
    coefficients = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "t_stat": result.tvalues,
            "p_value": result.pvalues,
            "ci_lower": confidence.iloc[:, 0],
            "ci_upper": confidence.iloc[:, 1],
        }
    )
    residuals = np.asarray(result.resid, dtype=float)

    return ModelFit(
        name=name,
        requested_features=requested,
        features=features,
        available=True,
        n=n,
        r_squared=float(result.rsquared),
        adj_r_squared=float(result.rsquared_adj),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        aic=float(result.aic),
        bic=float(result.bic),
        coefficients=coefficients,
        ols_result=result,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        notes=tuple(notes),
    )


def compare_models(
    panel: pd.DataFrame,
    target: str = TARGET,
    model1_features: tuple[str, ...] = MODEL1_FEATURES,
    model2_extra: tuple[str, ...] = MODEL2_EXTRA_FEATURES,
    media_features: tuple[str, ...] = MEDIA_FEATURES,
    hac_lags: int = HAC_LAGS,
    allow_underpowered: bool = False,
    household_political: tuple[str, ...] = (),
    include_media_only: bool = False,
    min_coverage: float = MIN_FEATURE_COVERAGE,
    seasonal_control: str = "none",
    seasonality_alpha: float = 0.05,
) -> ModelComparison:
    """Fit Models 1, 2 and 3 on a common sample with HAC errors.

    Model 1 is the macro baseline, Model 2 adds Hypothesis A's level and
    burden regressors, Model 3 adds Hypothesis B's media regressors.

    **Missing regressors degrade, they do not crash.** The media columns come
    from `nlp_media_svd` and may not exist; `debt_service` ends early in FRED.
    Any specification whose regressors are absent is returned with
    ``available=False`` and the missing names, and is excluded from the common
    sample so that its absence does not silently shorten the models that *can*
    be fitted.

    **The common sample is the intersection across available models.** Fit
    statistics are only comparable on identical rows, so every available model
    is estimated on the same index. Rows excluded for missing inputs are named
    in ``dropped_dates`` / ``dropped_reasons``.

    **The brief's ladder.** Section 4 of the brief asks for economic only,
    economic plus media, and economic plus media plus household or political
    controls. Pass ``household_political=HOUSEHOLD_POLITICAL_FEATURES`` (or
    call `compare_model_ladder`) to add "Model 4", which is the brief's third
    rung. ``include_media_only=True`` adds "Model 2b", media regressors with no
    economic regressor, which shows how much of the fit the media series carry
    alone. Both are off by default so that the existing Table 3 is unchanged.
    The mapping from our names to the brief's rungs travels on the result as
    ``brief_ladder``.

    **Columns are read defensively.** A requested regressor can be absent
    (the model becomes unavailable), present under a documented substitute name
    (recorded in ``substitutions``), or present but too sparse to use -- which
    is what a quarterly series that is correctly *not* interpolated into
    monthly values looks like. A sparse column is dropped from the
    specification, with the coverage figure, instead of deleting most of the
    sample through listwise deletion.

    Raises `EconometricsError` when the common sample is too small for HAC
    inference, unless ``allow_underpowered=True``.
    """
    if panel.empty:
        raise EconometricsError("cannot compare models on an empty panel")
    _require_columns(panel, (target,))
    if seasonal_control not in {"none", "auto", "always"}:
        raise EconometricsError(
            "seasonal_control must be 'none', 'auto' or 'always', got "
            f"{seasonal_control!r}"
        )

    notes: list[str] = []
    ladder_notes: list[str] = []

    present_model1, missing_model1, dropped1, reasons1, subs1 = _resolve_specification(
        panel, tuple(model1_features), min_coverage
    )
    present_model2, missing_model2, dropped2, reasons2, subs2 = _resolve_specification(
        panel, tuple(model2_extra), min_coverage
    )
    present_media, missing_media, droppedm, reasonsm, subsm = _resolve_specification(
        panel, tuple(media_features), min_coverage
    )
    present_hp, missing_hp, droppedh, reasonsh, subsh = _resolve_specification(
        panel, tuple(household_political), min_coverage
    )
    notes.extend(reasons1 + reasons2 + reasonsm + reasonsh)
    notes.extend(subs1 + subs2 + subsm + subsh)

    # Seasonal control. The decision is reported, never silent.
    seasonal_terms: tuple[str, ...] = ()
    seasonal_frame: pd.DataFrame | None = None
    if seasonal_control != "none":
        add_seasonal = seasonal_control == "always"
        if seasonal_control == "auto":
            try:
                season = seasonality_report(
                    panel[target].dropna(), name=target, alpha=seasonality_alpha
                )
            except EconometricsError as exc:
                notes.append(f"seasonality not tested ({exc}); no seasonal control")
                season = None
            else:
                add_seasonal = season.seasonal
                notes.append(
                    f"seasonality test on '{target}': F={season.f_statistic:.3f}, "
                    f"p={season.p_value:.4f} -> "
                    + ("month dummies added" if add_seasonal else "no seasonal control")
                )
        if add_seasonal:
            seasonal_frame = month_dummies(panel.index)
            seasonal_terms = tuple(seasonal_frame.columns)
            notes.append(
                "monthly dummies (January omitted) enter every specification; "
                "they are controls, not regressors of interest"
            )

    specs: dict[str, tuple[tuple[str, ...], tuple[str, ...], list[str]]] = {
        "Model 1": (present_model1, tuple(model1_features), list(missing_model1)),
        "Model 2": (
            present_model1 + present_model2,
            tuple(model1_features) + tuple(model2_extra),
            list(missing_model1) + list(missing_model2),
        ),
    }
    if include_media_only:
        specs["Model 2b"] = (present_media, tuple(media_features), list(missing_media))
    specs["Model 3"] = (
        present_model1 + present_model2 + present_media,
        tuple(model1_features) + tuple(model2_extra) + tuple(media_features),
        list(missing_model1) + list(missing_model2) + list(missing_media),
    )
    if household_political:
        specs["Model 4"] = (
            present_model1 + present_model2 + present_media + present_hp,
            tuple(model1_features)
            + tuple(model2_extra)
            + tuple(media_features)
            + tuple(household_political),
            list(missing_model1)
            + list(missing_model2)
            + list(missing_media)
            + list(missing_hp),
        )
    else:
        ladder_notes.append(
            "the brief's third rung (household or political controls) was not "
            "requested; pass household_political=HOUSEHOLD_POLITICAL_FEATURES "
            "or call compare_model_ladder"
        )

    drops_by_model: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for name in specs:
        if name == "Model 1":
            these, why = dropped1, reasons1
        elif name == "Model 2":
            these, why = dropped1 + dropped2, reasons1 + reasons2
        elif name == "Model 2b":
            these, why = droppedm, reasonsm
        elif name == "Model 3":
            these, why = dropped1 + dropped2 + droppedm, reasons1 + reasons2 + reasonsm
        else:
            these = dropped1 + dropped2 + droppedm + droppedh
            why = reasons1 + reasons2 + reasonsm + reasonsh
        drops_by_model[name] = (tuple(these), tuple(why))

    models: dict[str, ModelFit] = {}
    fittable: dict[str, tuple[str, ...]] = {}
    for name, (features, requested, missing) in specs.items():
        if missing:
            reason = f"{name} not estimated: columns absent from the panel: {missing}"
            logger.info("%s", reason)
            models[name] = ModelFit(
                name=name,
                requested_features=requested,
                features=features,
                available=False,
                missing_features=tuple(missing),
                unavailable_reason=reason,
                notes=(
                    (
                        "Reported as unavailable, which is NOT the same finding "
                        "as 'these regressors did not help'."
                    ),
                ),
                dropped_features=drops_by_model[name][0],
                drop_reasons=drops_by_model[name][1],
            )
        elif not features:
            reason = (
                f"{name} not estimated: every requested regressor was dropped for "
                f"sparse coverage: {list(drops_by_model[name][0])}"
            )
            models[name] = ModelFit(
                name=name,
                requested_features=requested,
                features=(),
                available=False,
                unavailable_reason=reason,
                dropped_features=drops_by_model[name][0],
                drop_reasons=drops_by_model[name][1],
            )
        else:
            fittable[name] = features

    if not fittable:
        raise EconometricsError(
            "no model can be estimated: every specification is missing regressors "
            f"({ {n: list(m) for n, (_, _, m) in specs.items()} })"
        )

    needed = sorted({target, *(col for cols in fittable.values() for col in cols)})
    subset = panel.loc[:, needed]
    if seasonal_frame is not None:
        subset = subset.join(seasonal_frame, how="left")
    usable = subset.dropna()
    dropped = tuple(subset.index.difference(usable.index))
    reasons = _missing_reasons(subset, dropped)
    if dropped:
        logger.info(
            "common-sample construction dropped %d rows: %s",
            len(dropped),
            "; ".join(reasons[:6]),
        )
        notes.append(
            f"common sample drops {len(dropped)} of {len(subset)} rows for missing "
            "inputs; see dropped_reasons"
        )
    if usable.empty:
        raise EconometricsError(
            "no row has every regressor of every fittable model; common sample is "
            f"empty. Rows offered: {len(subset)}; dropped: {len(dropped)}"
        )

    for name, features in fittable.items():
        fit = _fit_one(
            name=name,
            frame=usable,
            features=features + seasonal_terms,
            requested=specs[name][1],
            target=target,
            hac_lags=hac_lags,
            allow_underpowered=allow_underpowered,
        )
        fit.dropped_features = drops_by_model[name][0]
        fit.drop_reasons = drops_by_model[name][1]
        if seasonal_terms:
            fit.notes = fit.notes + (
                f"{len(seasonal_terms)} monthly dummies included as controls",
            )
        models[name] = fit

    ladder: dict[str, str | None] = {
        "1. economic indicators only": (
            "Model 2" if models.get("Model 2") and models["Model 2"].available else
            ("Model 1" if models["Model 1"].available else None)
        ),
        "2. economic indicators plus media": (
            "Model 3" if models["Model 3"].available else None
        ),
        "3. economic, media, household or political controls": (
            "Model 4"
            if "Model 4" in models and models["Model 4"].available
            else None
        ),
    }
    if ladder["1. economic indicators only"] == "Model 1":
        ladder_notes.append(
            "rung 1 falls back to Model 1 because Model 2's level and burden "
            "regressors are not available"
        )
    if ladder["2. economic indicators plus media"] is None:
        ladder_notes.append(
            "rung 2 is not estimated: "
            f"{models['Model 3'].unavailable_reason or 'Model 3 unavailable'}"
        )
    if household_political and ladder[
        "3. economic, media, household or political controls"
    ] is None:
        ladder_notes.append(
            "rung 3 is not estimated: "
            f"{models.get('Model 4').unavailable_reason if 'Model 4' in models else 'Model 4 absent'}"
        )

    ordered = {name: models[name] for name in specs}
    return ModelComparison(
        models=ordered,
        common_sample_n=len(usable),
        common_start=usable.index.min(),
        common_end=usable.index.max(),
        rows_offered=len(subset),
        dropped_dates=dropped,
        dropped_reasons=reasons,
        hac_lags=hac_lags,
        target=target,
        notes=tuple(notes),
        brief_ladder=ladder,
        ladder_notes=tuple(ladder_notes),
        seasonal_control=seasonal_control,
        seasonal_terms=seasonal_terms,
    )


def predicted_sign_check(
    comparison: ModelComparison,
    model_name: str | None = None,
    media_features: tuple[str, ...] = MEDIA_FEATURES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare each media coefficient with the sign predicted in advance.

    `PREDICTED_SIGNS` states the direction each measure must take for the
    media account to hold. The table reports the predicted sign, the estimate,
    whether the sign agrees, and whether the estimate is distinguishable from
    zero at `alpha`. A coefficient of the predicted sign that is not
    distinguishable from zero is NOT evidence, and the table keeps those two
    questions in separate columns so they cannot merge.
    """
    candidates = [
        name
        for name, fit in comparison.models.items()
        if fit.available
        and fit.coefficients is not None
        and any(feature in fit.features for feature in media_features)
    ]
    chosen = model_name or (candidates[-1] if candidates else None)
    columns = [
        "regressor",
        "model",
        "predicted_sign",
        "coef",
        "p_value",
        "sign_agrees",
        "distinguishable_from_zero",
        "note",
    ]
    if chosen is None:
        return pd.DataFrame(columns=columns)
    fit = comparison.models[chosen]
    rows = []
    for feature in media_features:
        if fit.coefficients is None or feature not in fit.coefficients.index:
            continue
        row = fit.coefficients.loc[feature]
        predicted = PREDICTED_SIGNS.get(feature, "none stated")
        coefficient = float(row["coef"])
        agrees = (
            (coefficient < 0 and predicted == "negative")
            or (coefficient > 0 and predicted == "positive")
            if predicted in {"negative", "positive"}
            else False
        )
        rows.append(
            {
                "regressor": feature,
                "model": chosen,
                "predicted_sign": predicted,
                "coef": coefficient,
                "p_value": float(row["p_value"]),
                "sign_agrees": bool(agrees),
                "distinguishable_from_zero": bool(row["p_value"] < alpha),
                "note": PREDICTED_SIGN_NOTES.get(feature, ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).loc[:, columns].set_index("regressor")


def compare_model_ladder(
    panel: pd.DataFrame,
    target: str = TARGET,
    household_political: tuple[str, ...] = HOUSEHOLD_POLITICAL_FEATURES,
    include_media_only: bool = True,
    seasonal_control: str = "auto",
    **kwargs,
) -> ModelComparison:
    """The brief's section-4 ladder in one call, all three rungs.

    Rung 1 is economic indicators only (Model 2), rung 2 adds the media
    measures (Model 3) and rung 3 adds the household and political controls
    (Model 4). "Model 2b" carries the media measures with no economic
    regressor, which separates "media adds nothing" from "media carries the
    same information as the macro series". Seasonality is tested and
    controlled only if the test finds it.

    A control column that the panel does not carry makes rung 3 unavailable
    with the missing names, which is a state of the data and not an error.
    """
    return compare_models(
        panel,
        target=target,
        household_political=household_political,
        include_media_only=include_media_only,
        seasonal_control=seasonal_control,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2. VAR + Granger causality (predictive precedence)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationarityCheck:
    """ADF result for one series, before and (if needed) after differencing."""

    name: str
    adf_statistic: float
    p_value: float
    used_lags: int
    n_obs: int
    stationary: bool
    differenced: bool
    adf_statistic_after: float | None = None
    p_value_after: float | None = None
    stationary_after: bool | None = None
    note: str = ""

    @property
    def still_nonstationary(self) -> bool:
        """True when differencing was applied and ADF still fails to reject."""
        return bool(self.differenced and self.stationary_after is False)


@dataclass(frozen=True)
class GrangerTest:
    """One direction of a Granger F-test."""

    causing: str
    caused: str
    f_statistic: float
    p_value: float
    df_num: int
    df_denom: float
    lag_order: int

    @property
    def direction(self) -> str:
        return f"{self.causing} -> {self.caused}"

    def rejects(self, alpha: float = 0.05) -> bool:
        """Rejects the null of *no* predictive precedence at `alpha`."""
        return bool(self.p_value < alpha)


@dataclass
class GrangerReport:
    """Bidirectional Granger evidence, with its own interpretation attached.

    `evidence_class` and `interpretation` are part of the data, not the
    docstring, because this object is what the report and the dashboard quote.
    Reverse causality (coverage following sentiment) is a live hypothesis and a
    stated falsification condition for Hypothesis B, so both directions are
    always computed and both are always reported.
    """

    media_column: str
    gap_column: str
    lag_order: int
    lag_order_aic: int | None
    lag_order_bic: int | None
    lag_criterion: str
    max_lags_considered: int
    n_obs_input: int
    n_obs_used: int
    stationarity: dict[str, StationarityCheck]
    forward: GrangerTest | None
    reverse: GrangerTest | None
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    dropped_reasons: tuple[str, ...] = field(default_factory=tuple)
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    var_result: object | None = None
    evidence_class: str = "predictive precedence (NOT causation)"
    interpretation: str = GRANGER_CAVEAT
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def differenced_series(self) -> tuple[str, ...]:
        return tuple(
            name for name, check in self.stationarity.items() if check.differenced
        )

    def verdict(self, alpha: float = 0.05) -> str:
        """Which direction(s) survive, phrased so it cannot be over-read."""
        forward = self.forward is not None and self.forward.rejects(alpha)
        reverse = self.reverse is not None and self.reverse.rejects(alpha)
        if forward and reverse:
            body = (
                f"bidirectional predictive precedence at alpha={alpha}: feedback, "
                "so neither direction identifies a mechanism"
            )
        elif forward:
            body = (
                f"{self.media_column} precedes {self.gap_column} at alpha={alpha}; "
                "the reverse is not rejected"
            )
        elif reverse:
            body = (
                f"{self.gap_column} precedes {self.media_column} at alpha={alpha} "
                "and not the reverse -- coverage FOLLOWS sentiment, which is a "
                "stated falsification condition for Hypothesis B"
            )
        else:
            body = f"no predictive precedence in either direction at alpha={alpha}"
        if self.underpowered:
            body += " [UNDERPOWERED -- see power_warnings]"
        return body


def _adf_check(
    series: pd.Series, name: str, alpha: float, max_lags: int
) -> tuple[pd.Series, StationarityCheck]:
    """ADF-test a series and difference it once if it fails to reject.

    A unit root in either series makes the VAR's Granger F-statistic
    non-standard, so differencing is applied when ADF cannot reject -- and the
    fact that it was applied is reported, because the resulting test is then
    about changes, not levels.

    Note on the 'no look-ahead' rule: this is an in-sample descriptive test on
    a complete series, not a forecast, so using the whole series is correct
    here. Nothing fitted on the full sample is ever used to predict within it.
    """
    values = series.dropna().astype(float)
    # adfuller needs room for its own lag augmentation plus a trend term.
    usable_lags = max(0, min(max_lags, (len(values) - 4) // 2))
    if len(values) < 8:
        raise EconometricsError(
            f"{name}: {len(values)} observations is too few for an ADF test"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, used, n_obs = adfuller(
            values, maxlag=usable_lags or None, autolag="AIC"
        )[:4]
    stationary = bool(p_value < alpha)
    if stationary:
        return values, StationarityCheck(
            name=name,
            adf_statistic=float(stat),
            p_value=float(p_value),
            used_lags=int(used),
            n_obs=int(n_obs),
            stationary=True,
            differenced=False,
            note="ADF rejects a unit root; used in levels",
        )

    differenced = values.diff().dropna()
    if len(differenced) < 8:
        raise EconometricsError(
            f"{name}: non-stationary in levels and too short ({len(differenced)} "
            "rows) to test after differencing"
        )
    usable_lags_d = max(0, min(max_lags, (len(differenced) - 4) // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat_d, p_d = adfuller(
            differenced, maxlag=usable_lags_d or None, autolag="AIC"
        )[:2]
    stationary_after = bool(p_d < alpha)
    note = (
        f"ADF p={p_value:.3f} fails to reject a unit root; FIRST-DIFFERENCED. "
        "Granger results for this series concern changes, not levels."
    )
    if not stationary_after:
        note += (
            f" WARNING: after differencing ADF p={p_d:.3f} still fails to reject; "
            "the series may be I(2) or the test is simply underpowered at this n."
        )
    logger.info("%s", note)
    return differenced, StationarityCheck(
        name=name,
        adf_statistic=float(stat),
        p_value=float(p_value),
        used_lags=int(used),
        n_obs=int(n_obs),
        stationary=False,
        differenced=True,
        adf_statistic_after=float(stat_d),
        p_value_after=float(p_d),
        stationary_after=stationary_after,
        note=note,
    )


def granger_media_gap(
    frame: pd.DataFrame,
    media_column: str = "media_tone",
    gap_column: str = "gap",
    max_lags: int = 6,
    lag_criterion: str = "aic",
    alpha: float = 0.05,
    adf_alpha: float = 0.05,
    allow_underpowered: bool = False,
) -> GrangerReport:
    """VAR lag selection, ADF checks and Granger F-tests in both directions.

    `max_lags` is the *stated maximum* searched; the order is chosen by
    `lag_criterion` ("aic" or "bic") and both criteria's choices are reported so
    a reader can see whether the conclusion depends on the selection rule.

    Both directions are always tested. Hypothesis B's falsification condition
    is precisely that precedence runs predominantly from the gap to coverage,
    so a one-directional test would be assuming the answer.

    The null hypothesis is *no predictive precedence*. Rejecting it does not
    license a causal reading -- see `GRANGER_CAVEAT`, which travels on the
    result object as ``interpretation``.

    Raises `EconometricsError` when the sample cannot support `max_lags`,
    unless ``allow_underpowered=True``.
    """
    if lag_criterion not in {"aic", "bic"}:
        raise EconometricsError(
            f"lag_criterion must be 'aic' or 'bic', got {lag_criterion!r}"
        )
    if max_lags < 1:
        raise EconometricsError(f"max_lags must be at least 1, got {max_lags}")
    _require_columns(frame, (media_column, gap_column))
    if media_column == gap_column:
        raise EconometricsError(
            "media_column and gap_column must be different series"
        )

    subset = frame.loc[:, [media_column, gap_column]].astype(float)
    usable = subset.dropna()
    dropped = tuple(subset.index.difference(usable.index))
    reasons = _missing_reasons(subset, dropped)
    if dropped:
        logger.info(
            "Granger input dropped %d rows: %s", len(dropped), "; ".join(reasons[:6])
        )

    warnings_out: list[str] = []
    notes: list[str] = []

    # A VAR at p lags with 2 variables estimates 2p+1 coefficients per equation
    # from (n - p) effective observations. Check the *maximum* order, because
    # that is the model the selection step has to be able to fit.
    n_available = len(usable)
    params_at_max = 2 * max_lags + 1
    required = max_lags + MIN_OBS_PER_VAR_PARAMETER * params_at_max
    if n_available < required:
        message = (
            f"n={n_available} usable rows cannot support a VAR search up to "
            f"max_lags={max_lags}: that order estimates {params_at_max} "
            f"coefficients per equation from {n_available - max_lags} effective "
            f"observations, and this module requires at least "
            f"{MIN_OBS_PER_VAR_PARAMETER} observations per coefficient "
            f"(n >= {required}). Reduce max_lags or lengthen the window."
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " Pass allow_underpowered=True to report it anyway."
            )
        warnings_out.append(message)

    prepared: dict[str, pd.Series] = {}
    checks: dict[str, StationarityCheck] = {}
    for name in (media_column, gap_column):
        series, check = _adf_check(usable[name], name, adf_alpha, max_lags)
        prepared[name] = series
        checks[name] = check
        if check.differenced:
            notes.append(f"{name}: first-differenced before the VAR ({check.note})")
        if check.still_nonstationary:
            warnings_out.append(
                f"{name}: still non-stationary after differencing; the Granger "
                "F-test's distribution is not trustworthy here"
            )

    # sort=False: the two series may now have different lengths (one
    # differenced, one not) and pandas must not reorder the index.
    data = pd.concat(prepared, axis=1, sort=False).sort_index().dropna()
    if data.empty:
        raise EconometricsError(
            "no overlapping observations remain after stationarity differencing"
        )
    if len({check.differenced for check in checks.values()}) > 1:
        notes.append(
            "the two series were transformed differently (one differenced, one "
            "not), so the VAR mixes a level and a change -- stated because it "
            "affects how the coefficients read"
        )

    n_obs = len(data)
    effective_max = max(1, min(max_lags, (n_obs - 2) // 3))
    if effective_max < max_lags:
        warnings_out.append(
            f"lag search truncated from max_lags={max_lags} to {effective_max}: "
            f"only {n_obs} rows survive differencing"
        )

    model = VAR(data.to_numpy(dtype=float))
    aic_choice: int | None = None
    bic_choice: int | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            order = model.select_order(maxlags=effective_max)
        aic_choice = int(order.aic) if order.aic is not None else None
        bic_choice = int(order.bic) if order.bic is not None else None
    except Exception as exc:  # pragma: no cover - degenerate small samples
        warnings_out.append(f"lag-order selection failed ({exc}); defaulting to 1 lag")

    chosen = aic_choice if lag_criterion == "aic" else bic_choice
    if chosen is None or chosen < 1:
        # Both criteria can select zero lags, which has no Granger test to run.
        notes.append(
            f"{lag_criterion.upper()} selected {chosen} lags; forced to 1 because a "
            "zero-lag VAR has no lagged terms to test"
        )
        chosen = 1
    if aic_choice is not None and bic_choice is not None and aic_choice != bic_choice:
        notes.append(
            f"AIC chose {aic_choice} lags, BIC chose {bic_choice}; "
            f"{lag_criterion.upper()} used. Check both before quoting the result."
        )

    params_at_chosen = 2 * chosen + 1
    if n_obs - chosen <= params_at_chosen:
        message = (
            f"VAR({chosen}) needs more than {params_at_chosen} effective "
            f"observations per equation and has {n_obs - chosen}"
        )
        if not allow_underpowered:
            raise EconometricsError(message)
        warnings_out.append(message)
        chosen = 1
        params_at_chosen = 3
    if n_obs - chosen < MIN_OBS_PER_VAR_PARAMETER * params_at_chosen:
        warnings_out.append(
            f"VAR({chosen}): {n_obs - chosen} effective observations for "
            f"{params_at_chosen} coefficients per equation is below this "
            f"module's {MIN_OBS_PER_VAR_PARAMETER}-per-coefficient floor"
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(chosen)
    except Exception as exc:
        raise EconometricsError(f"VAR({chosen}) could not be fitted: {exc}") from exc

    names = [media_column, gap_column]

    def _test(causing_idx: int, caused_idx: int) -> GrangerTest | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outcome = fitted.test_causality(
                    caused=caused_idx, causing=[causing_idx], kind="f"
                )
        except Exception as exc:  # pragma: no cover - degenerate systems
            warnings_out.append(
                f"Granger test {names[causing_idx]} -> {names[caused_idx]} failed: {exc}"
            )
            return None
        degrees = outcome.df
        if isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            df_num, df_denom = float(degrees[0]), float(degrees[1])
        else:
            df_num, df_denom = float(chosen), float(n_obs - params_at_chosen)
        return GrangerTest(
            causing=names[causing_idx],
            caused=names[caused_idx],
            f_statistic=float(outcome.test_statistic),
            p_value=float(outcome.pvalue),
            df_num=int(df_num),
            df_denom=df_denom,
            lag_order=chosen,
        )

    forward = _test(0, 1)  # media -> gap
    reverse = _test(1, 0)  # gap -> media

    notes.append(f"lag order {chosen} selected by {lag_criterion.upper()} over 1..{effective_max}")
    notes.append(
        "both directions tested because reverse causality (coverage following "
        "sentiment) is a live hypothesis and a stated falsification condition"
    )

    return GrangerReport(
        media_column=media_column,
        gap_column=gap_column,
        lag_order=chosen,
        lag_order_aic=aic_choice,
        lag_order_bic=bic_choice,
        lag_criterion=lag_criterion,
        max_lags_considered=max_lags,
        n_obs_input=len(subset),
        n_obs_used=n_obs,
        stationarity=checks,
        forward=forward,
        reverse=reverse,
        dropped_dates=dropped,
        dropped_reasons=reasons,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        notes=tuple(notes),
        var_result=fitted,
    )


# ---------------------------------------------------------------------------
# 3. Rigobon-Sack identification through heteroskedasticity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSplit:
    """High/low volatility news regimes, with the rule that produced them."""

    rule: str
    window: int
    quantile: float
    threshold: float
    n_high: int
    n_low: int
    n_unassigned: int
    high_dates: tuple[pd.Timestamp, ...]
    low_dates: tuple[pd.Timestamp, ...]
    unassigned_dates: tuple[pd.Timestamp, ...]
    mean_rolling_var_high: float
    mean_rolling_var_low: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IdentificationCounting:
    """Order-condition arithmetic for the system as actually specified.

    Two countings are reported because they answer different questions and the
    brief cites only the first:

    * `brief_formula_value` is the brief's ``N^2 - N - 2K``. With N=2 it is
      ``2 - 2K``, so it is satisfied only when K=0 and fails for any exogenous
      regressor at all. It is reported verbatim so the report can say what
      happened to it rather than quietly dropping it.
    * `moments_available` / `parameters_to_estimate` is the counting that
      actually governs heteroskedasticity-based identification: S regimes each
      supply N(N+1)/2 distinct second moments, against N(N-1) simultaneity
      coefficients plus S*N shock variances.

    With N=2 and S=2 the second counting gives 6 moments against 6 parameters,
    which *looks* just-identified and is not: the three equations in
    ``Omega_H - Omega_L`` involve only four unknowns (two coefficients and two
    variance shifts) and have rank 1 in the relevant direction, leaving the
    system short by one restriction. That is why `identified_without_assumption`
    is False and `required_assumption` is populated.
    """

    n_endogenous: int
    n_regimes: int
    n_exogenous: int
    brief_formula_value: int
    brief_formula_satisfied: bool
    moments_available: int
    parameters_to_estimate: int
    degrees_of_freedom: int
    difference_equations: int
    difference_unknowns: int
    identified_without_assumption: bool
    required_assumption: str
    explanation: str


@dataclass(frozen=True)
class RankCondition:
    """Numerically verified rank/variance-shift evidence for the regime split.

    ``det(Omega_H - Omega_L) != 0`` is the condition the brief cites and it is
    computed and bootstrapped here. But note the tension, which the report must
    state rather than hide: under the Rigobon-Sack assumption that *one* shock's
    variance shifts, the population ``Omega_H - Omega_L`` is rank 1 and its
    determinant is exactly zero. A determinant indistinguishable from zero is
    therefore *consistent with* the identifying assumption, not a failure of it.

    The condition the estimator actually needs is weaker and is tested
    separately as `variance_shift_detected`: the media series' own variance must
    genuinely differ across regimes, i.e. ``(Omega_H - Omega_L)[media, media]``
    is distinguishable from zero. `passes` reports that operative condition.
    """

    omega_high: np.ndarray
    omega_low: np.ndarray
    delta_omega: np.ndarray
    determinant: float
    delta_variance_media: float
    delta_variance_gap: float
    bootstrap_draws: int
    bootstrap_determinants: np.ndarray
    determinant_ci95: tuple[float, float]
    determinant_excludes_zero: bool
    bootstrap_delta_variance_media: np.ndarray
    delta_variance_media_ci95: tuple[float, float]
    variance_shift_detected: bool
    variance_ratio: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passes(self) -> bool:
        """The operative condition: the regimes' media variances really differ.

        Deliberately *not* ``determinant_excludes_zero`` -- see the class
        docstring. Reported as a named property so no caller has to rediscover
        which of the two conditions the estimator depends on.
        """
        return self.variance_shift_detected


@dataclass
class RigobonSackResult:
    """Heteroskedasticity-IV estimate, or a reasoned refusal to produce one.

    When identification fails, `transmission_coefficient` is None and
    `caveats` says why. That is the reported result, in line with the
    pre-registered commitment in BRAINSTORMING.md section 5: an unidentified
    system is an outcome, not an error to be worked around.
    """

    media_column: str
    gap_column: str
    regime: RegimeSplit
    identification: IdentificationCounting
    rank: RankCondition
    transmission_coefficient: float | None
    std_error: float | None
    bootstrap_std_error: float | None
    ci95: tuple[float, float] | None
    p_value: float | None
    ols_coefficient: float
    ols_p_value: float
    sign_agrees_with_ols: bool | None
    first_stage_f: float | None
    n_obs: int
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    dropped_reasons: tuple[str, ...] = field(default_factory=tuple)
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    caveats: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = (
        "instrumented, CONDITIONAL on an untestable variance-shift assumption"
    )
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def identified(self) -> bool:
        return self.transmission_coefficient is not None and self.rank.passes

    def summary(self) -> str:
        if not self.identified:
            return (
                f"NOT IDENTIFIED: {self.caveats[0] if self.caveats else 'rank condition failed'}"
            )
        line = (
            f"transmission {self.media_column} -> {self.gap_column} = "
            f"{self.transmission_coefficient:.4f}"
        )
        if self.std_error is not None:
            line += f" (se {self.std_error:.4f}, p {self.p_value:.3f})"
        line += f"; OLS {self.ols_coefficient:.4f}"
        if self.underpowered:
            line += " [UNDERPOWERED]"
        return line


def split_volatility_regimes(
    series: pd.Series,
    window: int = DEFAULT_VOLATILITY_WINDOW,
    quantile: float = DEFAULT_VOLATILITY_QUANTILE,
) -> RegimeSplit:
    """Assign months to high/low news-volatility regimes by rolling variance.

    **Stated rule.** For each month t, the variance of the media series over the
    `window` months ending at **t-1** is computed. Months whose lagged rolling
    variance is strictly above the `quantile` of that distribution are regime H;
    the rest are regime L.

    **Why the window is lagged, not contemporaneous.** This is the single most
    consequential choice in the whole IV. Assigning month t by a window that
    *includes* ``m_t`` sorts months by their own realised squared deviation, so
    regime H collects the large draws by construction. The resulting
    ``Omega_H - Omega_L`` is then large and positive even when the data are
    perfectly homoskedastic, and the rank condition would "pass" on noise every
    time. Excluding ``m_t`` makes the assignment a function of strictly prior
    information, which under homoskedasticity is independent of ``m_t^2``, so a
    detected variance shift is evidence rather than an artefact of the sort.
    It also keeps the split free of look-ahead.

    Some selection bias survives if the media series is itself
    autocorrelated in squares (volatility clustering) -- but that *is*
    heteroskedasticity, which is what the method is supposed to exploit.

    The *threshold* is a full-sample quantile of the lagged rolling variances.
    The split is a sample partition for covariance estimation, not a forecasting
    rule, and nothing here predicts anything; a real-time variant would need an
    expanding quantile instead. Recorded because it is contestable.

    Months without a full `window` of prior observations have no rolling
    variance and are returned as `unassigned_dates`, never folded into either
    regime.
    """
    if window < 2:
        raise EconometricsError(f"volatility window must be >= 2, got {window}")
    if not 0.0 < quantile < 1.0:
        raise EconometricsError(f"quantile must be in (0, 1), got {quantile}")

    values = series.astype(float)
    # shift(1): the window ends at t-1, excluding month t's own observation.
    rolling = values.rolling(window=window, min_periods=window).var(ddof=1).shift(1)
    unassigned = tuple(rolling.index[rolling.isna()])
    scored = rolling.dropna()
    if scored.empty:
        raise EconometricsError(
            f"no month has {window} observations of rolling history; cannot split "
            f"regimes from {len(values)} rows"
        )

    threshold = float(scored.quantile(quantile))
    high_mask = scored > threshold
    high = tuple(scored.index[high_mask])
    low = tuple(scored.index[~high_mask])

    notes = [
        (
            f"rolling variance over the {window} months ending at t-1 (month t's "
            "own value excluded, so the sort cannot manufacture a variance gap)"
        ),
        (
            f"threshold = full-sample quantile {quantile:.2f} of that rolling "
            f"variance = {threshold:.6g}"
        ),
    ]
    if unassigned:
        notes.append(
            f"{len(unassigned)} months unassigned for lack of {window} months of "
            f"history: {', '.join(_label(s) for s in unassigned[:6])}"
            + ("" if len(unassigned) <= 6 else f" (+{len(unassigned) - 6} more)")
        )

    return RegimeSplit(
        rule=(
            f"{window}-month rolling variance of the media series ending at t-1, "
            f"split at its own {quantile:.2f} quantile"
        ),
        window=window,
        quantile=quantile,
        threshold=threshold,
        n_high=len(high),
        n_low=len(low),
        n_unassigned=len(unassigned),
        high_dates=high,
        low_dates=low,
        unassigned_dates=unassigned,
        mean_rolling_var_high=float(scored[high_mask].mean()) if high else float("nan"),
        mean_rolling_var_low=float(scored[~high_mask].mean()) if low else float("nan"),
        notes=tuple(notes),
    )


def _count_identification(
    n_endogenous: int, n_regimes: int, n_exogenous: int
) -> IdentificationCounting:
    """Do the order-condition arithmetic honestly, both ways."""
    brief_value = n_endogenous**2 - n_endogenous - 2 * n_exogenous
    moments = n_regimes * n_endogenous * (n_endogenous + 1) // 2
    parameters = n_endogenous * (n_endogenous - 1) + n_regimes * n_endogenous
    difference_equations = n_endogenous * (n_endogenous + 1) // 2
    difference_unknowns = n_endogenous * (n_endogenous - 1) + n_endogenous

    explanation = (
        f"System as specified: N={n_endogenous} endogenous variables (media "
        f"negativity, sentiment gap), S={n_regimes} volatility regimes, "
        f"K={n_exogenous} exogenous regressors.\n"
        f"Brief's order condition N^2 - N - 2K = {brief_value}"
        f" ({'satisfied' if brief_value > 0 else 'NOT satisfied'}): with N=2 this "
        "reduces to 2 - 2K and can only hold with no exogenous regressors at all, "
        "so it is uninformative for this system and is reported, not relied on.\n"
        f"Heteroskedasticity counting: {moments} distinct second moments "
        f"({n_regimes} regimes x {difference_equations} each) against {parameters} "
        f"parameters ({n_endogenous * (n_endogenous - 1)} simultaneity coefficients "
        f"+ {n_regimes * n_endogenous} regime-specific shock variances), i.e. "
        f"{moments - parameters} degrees of freedom.\n"
        f"That zero is NOT just-identification. The identifying information is "
        f"confined to Omega_H - Omega_L, which supplies {difference_equations} "
        f"equations in {difference_unknowns} unknowns (both simultaneity "
        "coefficients plus one variance shift per shock) -- short by "
        f"{difference_unknowns - difference_equations}. With two regimes the "
        "fully simultaneous system is therefore UNDER-IDENTIFIED, and one "
        "additional restriction is required."
    )

    return IdentificationCounting(
        n_endogenous=n_endogenous,
        n_regimes=n_regimes,
        n_exogenous=n_exogenous,
        brief_formula_value=int(brief_value),
        brief_formula_satisfied=bool(brief_value > 0),
        moments_available=int(moments),
        parameters_to_estimate=int(parameters),
        degrees_of_freedom=int(moments - parameters),
        difference_equations=int(difference_equations),
        difference_unknowns=int(difference_unknowns),
        identified_without_assumption=bool(
            difference_equations >= difference_unknowns
        ),
        required_assumption=RIGOBON_SACK_ASSUMPTION,
        explanation=explanation,
    )


def _covariance(frame: pd.DataFrame) -> np.ndarray:
    """Within-regime covariance matrix, each regime demeaned on its own rows.

    Demeaning within regime matters: a level shift between regimes would
    otherwise masquerade as a variance shift and the IV would be identified by
    the wrong feature of the data.
    """
    centred = frame - frame.mean()
    return np.asarray(centred.cov(ddof=1).to_numpy(dtype=float))


def _transmission_from_delta(delta: np.ndarray) -> float:
    """b = Delta_Omega[media, gap] / Delta_Omega[media, media].

    Derivation, for the system ``m = a*g + e_m``, ``g = b*m + e_g``:
    with ``A = [[1, -a], [-b, 1]]`` and ``A^-1 = (1/(1-ab)) [[1, a], [b, 1]]``,
    if only the media shock's variance shifts by ``d`` then

        Omega_H - Omega_L = d/(1-ab)^2 * [[1, b], [b, b^2]]

    so the ratio of the (media, gap) entry to the (media, media) entry is
    exactly ``b``, the transmission of media negativity into the gap. Note the
    matrix on the right is rank 1 -- see `RankCondition`.
    """
    return float(delta[0, 1] / delta[0, 0])


def rigobon_sack_iv(
    frame: pd.DataFrame,
    media_column: str = "media_tone",
    gap_column: str = "gap",
    window: int = DEFAULT_VOLATILITY_WINDOW,
    quantile: float = DEFAULT_VOLATILITY_QUANTILE,
    n_exogenous: int = 0,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    random_state: int | None = 20261015,
    allow_underpowered: bool = False,
) -> RigobonSackResult:
    """Estimate media -> gap transmission by identification through heteroskedasticity.

    Follows Rigobon (2003) and Rigobon & Sack (2003, 2004): when the variance of
    one structural shock shifts across known regimes while the structural
    coefficients do not, the change in the reduced-form covariance matrix
    identifies the transmission coefficient without requiring an excluded
    instrument. The estimator is the just-identified IV with instrument

        w_t = m_t / n_H   for t in H,   -m_t / n_L   for t in L

    applied to within-regime-demeaned data, which is numerically the moment
    estimator ``b = dOmega[m, g] / dOmega[m, m]``.

    What is verified rather than assumed:

    * the **order condition**, counted for this system both in the brief's
      ``N^2 - N - 2K`` form and in the form that actually governs
      heteroskedasticity identification (`IdentificationCounting`);
    * the **rank condition**, as ``det(Omega_H - Omega_L)`` with a bootstrap
      distribution, alongside the weaker condition the estimator truly needs
      (`RankCondition`).

    If the rank condition fails -- a homoskedastic split, or regimes that do not
    actually differ -- no coefficient is returned. `transmission_coefficient` is
    None, `caveats` explains, and that is the finding.

    Raises `EconometricsError` when either regime is too small for its own
    covariance matrix, unless ``allow_underpowered=True``.
    """
    _require_columns(frame, (media_column, gap_column))
    if media_column == gap_column:
        raise EconometricsError("media_column and gap_column must differ")
    if bootstrap_draws < 50:
        raise EconometricsError(
            f"bootstrap_draws={bootstrap_draws} is too few to describe a "
            "distribution; use at least 50"
        )

    subset = frame.loc[:, [media_column, gap_column]].astype(float)
    usable = subset.dropna()
    dropped = tuple(subset.index.difference(usable.index))
    reasons = _missing_reasons(subset, dropped)
    if dropped:
        logger.info(
            "Rigobon-Sack input dropped %d rows: %s",
            len(dropped),
            "; ".join(reasons[:6]),
        )
    if len(usable) < 2 * window:
        raise EconometricsError(
            f"{len(usable)} usable rows cannot support a rolling-{window} "
            "volatility split"
        )

    regime = split_volatility_regimes(usable[media_column], window=window, quantile=quantile)

    warnings_out: list[str] = []
    caveats: list[str] = [RIGOBON_SACK_ASSUMPTION]
    notes: list[str] = list(regime.notes)

    smaller = min(regime.n_high, regime.n_low)
    if smaller < 3:
        raise EconometricsError(
            f"regime split leaves {regime.n_high} high / {regime.n_low} low "
            "observations; a covariance matrix needs at least 3 per regime"
        )
    if smaller < MIN_REGIME_OBS:
        message = (
            f"regime split leaves {regime.n_high} high / {regime.n_low} low "
            f"observations; this module requires {MIN_REGIME_OBS} per regime "
            "before reporting a covariance difference as evidence. A 2x2 "
            "covariance matrix from a handful of months is noise with a "
            "plausible shape."
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " Pass allow_underpowered=True to report it anyway."
            )
        warnings_out.append(message)
        caveats.append(
            "SMALL-SAMPLE WARNING: this estimate comes from regimes of "
            f"{regime.n_high} and {regime.n_low} observations and must not be "
            "quoted without that fact."
        )

    high = usable.loc[list(regime.high_dates)]
    low = usable.loc[list(regime.low_dates)]
    omega_high = _covariance(high)
    omega_low = _covariance(low)
    delta = omega_high - omega_low
    determinant = float(np.linalg.det(delta))

    rng = np.random.default_rng(random_state)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    boot_dets = np.empty(bootstrap_draws, dtype=float)
    boot_delta_media = np.empty(bootstrap_draws, dtype=float)
    for draw in range(bootstrap_draws):
        # Stratified resampling: each regime is resampled within itself, so the
        # bootstrap describes uncertainty in the covariance matrices and not in
        # the regime assignment, which is the quantity the rank condition is
        # about.
        h = high_values[rng.integers(0, len(high_values), len(high_values))]
        l = low_values[rng.integers(0, len(low_values), len(low_values))]
        d = np.cov(h, rowvar=False, ddof=1) - np.cov(l, rowvar=False, ddof=1)
        boot_dets[draw] = np.linalg.det(d)
        boot_delta_media[draw] = d[0, 0]

    det_ci = (
        float(np.nanpercentile(boot_dets, 2.5)),
        float(np.nanpercentile(boot_dets, 97.5)),
    )
    media_ci = (
        float(np.nanpercentile(boot_delta_media, 2.5)),
        float(np.nanpercentile(boot_delta_media, 97.5)),
    )
    shift_detected = bool(media_ci[0] > 0.0 or media_ci[1] < 0.0)
    variance_ratio = (
        float(omega_high[0, 0] / omega_low[0, 0])
        if omega_low[0, 0] > 0
        else float("inf")
    )

    rank_notes = [
        (
            "det(Omega_H - Omega_L) is bootstrapped by stratified resampling "
            "within each regime, not asserted"
        ),
        (
            "under the Rigobon-Sack single-shifting-variance assumption the "
            "population Omega_H - Omega_L is RANK 1 and its determinant is "
            "zero, so a determinant indistinguishable from zero is consistent "
            "with the assumption rather than a failure of it"
        ),
        (
            "the operative condition is the media variance shift "
            f"(Delta={delta[0, 0]:.6g}, 95% bootstrap CI "
            f"[{media_ci[0]:.6g}, {media_ci[1]:.6g}])"
        ),
    ]
    rank = RankCondition(
        omega_high=omega_high,
        omega_low=omega_low,
        delta_omega=delta,
        determinant=determinant,
        delta_variance_media=float(delta[0, 0]),
        delta_variance_gap=float(delta[1, 1]),
        bootstrap_draws=bootstrap_draws,
        bootstrap_determinants=boot_dets,
        determinant_ci95=det_ci,
        determinant_excludes_zero=bool(det_ci[0] > 0.0 or det_ci[1] < 0.0),
        bootstrap_delta_variance_media=boot_delta_media,
        delta_variance_media_ci95=media_ci,
        variance_shift_detected=shift_detected,
        variance_ratio=variance_ratio,
        notes=tuple(rank_notes),
    )

    identification = _count_identification(
        n_endogenous=2, n_regimes=2, n_exogenous=n_exogenous
    )
    if not identification.identified_without_assumption:
        caveats.append(
            "ORDER CONDITION: the fully simultaneous 2-variable system is "
            "under-identified by one restriction with a single regime split. "
            "The estimate below is identified ONLY under the variance-shift "
            "assumption above; the reverse coefficient (gap -> media) is NOT "
            "identified here at all and is not reported."
        )

    # Plain OLS of the gap on media, for the sign comparison that Hypothesis B's
    # falsification condition is stated in terms of.
    centred = usable - usable.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(
            centred[gap_column],
            sm.add_constant(centred[[media_column]], has_constant="add"),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": min(HAC_LAGS, max(1, len(usable) // 4))})
    ols_coefficient = float(ols.params[media_column])
    ols_p = float(ols.pvalues[media_column])

    if not rank.passes:
        caveats.insert(
            0,
            "RANK CONDITION FAILED: the two regimes' media variances are not "
            f"distinguishable (Delta={delta[0, 0]:.6g}, 95% bootstrap CI "
            f"[{media_ci[0]:.6g}, {media_ci[1]:.6g}] contains 0). There is no "
            "heteroskedasticity to identify from, so NO transmission "
            "coefficient is reported. This is the result, not a failure to "
            "compute one.",
        )
        logger.warning("Rigobon-Sack rank condition failed; no estimate reported")
        return RigobonSackResult(
            media_column=media_column,
            gap_column=gap_column,
            regime=regime,
            identification=identification,
            rank=rank,
            transmission_coefficient=None,
            std_error=None,
            bootstrap_std_error=None,
            ci95=None,
            p_value=None,
            ols_coefficient=ols_coefficient,
            ols_p_value=ols_p,
            sign_agrees_with_ols=None,
            first_stage_f=None,
            n_obs=len(usable),
            dropped_dates=dropped,
            dropped_reasons=reasons,
            underpowered=True,
            power_warnings=tuple(warnings_out),
            caveats=tuple(caveats),
            notes=tuple(notes),
        )

    # The IV itself. Demean within regime so that regime-level shifts in the
    # mean cannot contribute, then build the Rigobon-Sack contrast instrument.
    pieces = []
    for block, sign in ((high, 1.0), (low, -1.0)):
        centred_block = block - block.mean()
        weight = sign / len(block)
        pieces.append((centred_block, weight))
    media_stack = np.concatenate([b[media_column].to_numpy() for b, _ in pieces])
    gap_stack = np.concatenate([b[gap_column].to_numpy() for b, _ in pieces])
    instrument = np.concatenate(
        [w * b[media_column].to_numpy() for b, w in pieces]
    )

    denominator = float(instrument @ media_stack)
    if abs(denominator) < 1e-12:  # pragma: no cover - guarded by rank condition
        raise EconometricsError(
            "instrument is orthogonal to the media series; the regime contrast "
            "carries no identifying variation"
        )
    beta = float((instrument @ gap_stack) / denominator)
    moment_beta = _transmission_from_delta(delta)
    if not np.isclose(beta, moment_beta, rtol=1e-6, atol=1e-9):
        # They are algebraically the same estimator; a mismatch would mean a
        # construction bug, so it is checked rather than trusted.
        notes.append(
            f"IV estimate {beta:.6g} and moment estimate {moment_beta:.6g} differ; "
            "investigate before quoting"
        )

    residuals = gap_stack - beta * media_stack
    n = len(media_stack)
    # Just-identified IV variance: s^2 (z'z) / (z'x)^2.
    sigma2 = float(residuals @ residuals) / max(1, n - 1)
    std_error = float(np.sqrt(sigma2 * (instrument @ instrument) / denominator**2))

    boot_betas = np.empty(bootstrap_draws, dtype=float)
    for draw in range(bootstrap_draws):
        h = high_values[rng.integers(0, len(high_values), len(high_values))]
        l = low_values[rng.integers(0, len(low_values), len(low_values))]
        d = np.cov(h, rowvar=False, ddof=1) - np.cov(l, rowvar=False, ddof=1)
        boot_betas[draw] = (
            d[0, 1] / d[0, 0] if abs(d[0, 0]) > 1e-12 else np.nan
        )
    finite = boot_betas[np.isfinite(boot_betas)]
    bootstrap_se = float(np.std(finite, ddof=1)) if len(finite) > 2 else None
    ci95 = (
        (float(np.nanpercentile(finite, 2.5)), float(np.nanpercentile(finite, 97.5)))
        if len(finite) > 2
        else None
    )
    if len(finite) < bootstrap_draws:
        warnings_out.append(
            f"{bootstrap_draws - len(finite)} of {bootstrap_draws} bootstrap draws "
            "had a near-zero variance shift and were discarded; the regime "
            "contrast is fragile"
        )

    # First-stage strength: regress the media series on the instrument.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first_stage = sm.OLS(
            media_stack, sm.add_constant(instrument, has_constant="add")
        ).fit()
    first_stage_f = float(first_stage.fvalue) if np.isfinite(first_stage.fvalue) else None
    if first_stage_f is not None and first_stage_f < 10.0:
        warnings_out.append(
            f"first-stage F={first_stage_f:.2f} is below the conventional weak-"
            "instrument threshold of 10; the IV point estimate is unreliable and "
            "its confidence interval understates uncertainty"
        )

    p_value = (
        float(2.0 * (1.0 - _normal_cdf(abs(beta) / std_error)))
        if std_error and std_error > 0
        else None
    )

    notes.append(
        "standard error is the just-identified IV sandwich assuming serially "
        "uncorrelated residuals; monthly data is autocorrelated, so the "
        "bootstrap standard error is the one to quote"
    )
    caveats.append(
        "Granger precedence and this IV answer different questions; agreement "
        "between them is corroboration, not proof of a mechanism."
    )

    return RigobonSackResult(
        media_column=media_column,
        gap_column=gap_column,
        regime=regime,
        identification=identification,
        rank=rank,
        transmission_coefficient=beta,
        std_error=std_error,
        bootstrap_std_error=bootstrap_se,
        ci95=ci95,
        p_value=p_value,
        ols_coefficient=ols_coefficient,
        ols_p_value=ols_p,
        sign_agrees_with_ols=bool(np.sign(beta) == np.sign(ols_coefficient)),
        first_stage_f=first_stage_f,
        n_obs=len(usable),
        dropped_dates=dropped,
        dropped_reasons=reasons,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        caveats=tuple(caveats),
        notes=tuple(notes),
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF, via the error function (avoids a scipy import)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------------------------------------------------------------------------
# 4. The brief's third rung: household and political controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlBlockReport:
    """What the household and political control block actually delivered.

    A control that FRED did not give is named here with its series id and the
    reason, so that a model reported as unavailable can be traced to one
    missing series instead of to "the data".
    """

    columns: tuple[str, ...]
    requested: tuple[str, ...]
    missing: tuple[str, ...]
    missing_reasons: tuple[str, ...]
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    coverage: dict[str, float]
    offline_columns: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        parts = [f"controls: {', '.join(self.columns) or 'none'}"]
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing)}")
        if self.first is not None:
            parts.append(f"{_label(self.first)}..{_label(self.last)}")
        return " | ".join(parts)


def election_proximity_features(
    index: pd.DatetimeIndex,
    election_dates: tuple[str, ...] = FEDERAL_ELECTION_DATES,
) -> pd.DataFrame:
    """Months to the next US federal general election, and a bounded index.

    Two columns:

    * ``months_to_election`` -- whole months from each observation to the next
      federal election date, counted forward only;
    * ``election_proximity`` -- ``1 / (1 + months_to_election)``, which is 1 in
      the election month and falls away smoothly. The bounded form is the one
      the model ladder uses, because a raw month count is a sawtooth and its
      coefficient is then a mixture of level and cycle.

    Calculated from the calendar, with no network and no look-ahead: the date
    of a scheduled election is known in advance, which is the whole reason it
    is usable as a control.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise EconometricsError("election proximity needs a DatetimeIndex")
    if not election_dates:
        raise EconometricsError("no election dates supplied")
    stamps = sorted(pd.Timestamp(date) for date in election_dates)
    months: list[float] = []
    for stamp in index:
        # Counted in whole MONTHS, not in days. A month-end observation falls
        # after election day inside the election month, and a day count would
        # therefore send the election month to the next election, 24 months
        # away. The analysis frequency is monthly, so the month is the unit.
        deltas = [
            (election.year - stamp.year) * 12 + (election.month - stamp.month)
            for election in stamps
        ]
        upcoming = [delta for delta in deltas if delta >= 0]
        months.append(float(min(upcoming)) if upcoming else float("nan"))
    frame = pd.DataFrame({"months_to_election": months}, index=index)
    frame["election_proximity"] = 1.0 / (1.0 + frame["months_to_election"])
    return frame


def build_household_political_controls(
    index: pd.DatetimeIndex,
    fred=None,
    series: dict[str, str] | None = None,
    election_dates: tuple[str, ...] = FEDERAL_ELECTION_DATES,
    retries: int = 1,
) -> tuple[pd.DataFrame, ControlBlockReport]:
    """Assemble the brief's household and political controls on `index`.

    Four controls, one of them offline:

    * ``policy_uncertainty`` -- FRED ``USEPUINDXD``, daily, averaged by month;
    * ``real_dpi_pc_yoy`` -- FRED ``A229RX0``, real disposable personal income
      per capita, as a year-over-year percentage change (the level has a trend
      that would absorb the gap rather than explain it; the level is kept too,
      as ``real_dpi_pc``);
    * ``u6_rate`` -- FRED ``U6RATE``, labour underutilisation;
    * ``election_proximity`` -- from the calendar, no network.

    `fred` is any object with a ``series(series_id)`` method that returns a
    month-end `pandas.Series` and an empty series on failure -- that is,
    `data_loader.FredSource`. Pass ``fred=None`` to get the offline columns
    only, which is what the tests use. **Every series that does not arrive is
    named with its id**, because the brief requires an unavailable control to
    be reported rather than quietly dropped.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise EconometricsError("the control block needs a DatetimeIndex")
    if len(index) == 0:
        raise EconometricsError("cannot build controls on an empty index")

    wanted = dict(series) if series is not None else dict(HOUSEHOLD_POLITICAL_SERIES)
    frame = pd.DataFrame(index=index)
    notes: list[str] = []
    missing: list[str] = []
    missing_reasons: list[str] = []

    if fred is None:
        for name, series_id in wanted.items():
            missing.append(name)
            missing_reasons.append(
                f"{name}: no FRED source supplied, so series {series_id} was not "
                "fetched"
            )
        notes.append(
            "offline build: the FRED controls are absent and the models that "
            "need them will report as unavailable"
        )
    else:
        for name, series_id in wanted.items():
            fetched = pd.Series(dtype=float)
            # One retry. `FredSource.series` turns any failure into an empty
            # series, and a single dropped HTTP call would otherwise be
            # reported as "FRED does not carry this control".
            for attempt in range(1 + max(retries, 0)):
                try:
                    fetched = pd.Series(fred.series(series_id))
                except Exception as exc:  # pragma: no cover - network dependent
                    notes.append(f"{name}: fetch of {series_id} raised {exc}")
                    fetched = pd.Series(dtype=float)
                if not fetched.dropna().empty:
                    if attempt:
                        notes.append(
                            f"{name}: series {series_id} arrived on retry "
                            f"{attempt}"
                        )
                    break
            fetched = pd.Series(fetched).dropna()
            if fetched.empty:
                missing.append(name)
                missing_reasons.append(
                    f"{name}: FRED series {series_id} returned no data"
                )
                continue
            fetched.index = pd.to_datetime(fetched.index) + pd.offsets.MonthEnd(0)
            monthly = fetched.groupby(level=0).mean()
            aligned = monthly.reindex(index)
            if name == "real_dpi_pc":
                frame["real_dpi_pc"] = aligned
                # 12-month change on the monthly series, which needs the twelve
                # months before `index` starts; reindex on the raw series, not
                # on the already-truncated column, or the first year is lost.
                yoy = monthly.pct_change(12) * 100.0
                frame["real_dpi_pc_yoy"] = yoy.reindex(index)
                notes.append(
                    "real_dpi_pc_yoy: 100 x (level / level 12 months earlier - 1), "
                    f"from FRED {series_id}"
                )
            else:
                frame[name] = aligned
                notes.append(f"{name}: FRED {series_id}, monthly mean")

    proximity = election_proximity_features(index, election_dates=election_dates)
    frame["months_to_election"] = proximity["months_to_election"]
    frame["election_proximity"] = proximity["election_proximity"]
    notes.append(
        "election_proximity = 1 / (1 + months to the next scheduled federal "
        "election); calculated from the calendar, so it uses no information "
        "that was unavailable at the time"
    )

    coverage = {name: column_coverage(frame, name) for name in frame.columns}
    return frame, ControlBlockReport(
        columns=tuple(frame.columns),
        requested=tuple(wanted) + ("election_proximity",),
        missing=tuple(missing),
        missing_reasons=tuple(missing_reasons),
        first=index.min(),
        last=index.max(),
        coverage=coverage,
        offline_columns=("months_to_election", "election_proximity"),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 5. Seasonality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalityReport:
    """Is there calendar seasonality in this series, and how large is it?

    The test is the joint significance of eleven monthly dummies in a
    regression of the series on a constant and those dummies, with HAC errors
    so that the autocorrelation of the series does not inflate the F-statistic.
    `seasonal` is the decision at `alpha`, and `recommendation` is what to do.
    """

    name: str
    n_obs: int
    f_statistic: float
    p_value: float
    df_num: int
    df_denom: float
    alpha: float
    seasonal: bool
    month_means: pd.Series
    amplitude: float
    hac_lags: int
    covariance: str = "HC3"
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def recommendation(self) -> str:
        if self.seasonal:
            return (
                "control monthly seasonality: pass seasonal_control='always' to "
                "compare_models, or deseasonalise before the VAR"
            )
        return (
            "no seasonal control needed: eleven monthly dummies are jointly "
            f"insignificant at alpha={self.alpha}"
        )

    def summary(self) -> str:
        verdict = "SEASONAL" if self.seasonal else "no seasonality"
        body = (
            f"{self.name}: {verdict} (F={self.f_statistic:.3f}, "
            f"p={self.p_value:.4f}, n={self.n_obs}, peak-to-trough "
            f"{self.amplitude:.3f})"
        )
        if self.underpowered:
            body += " [UNDERPOWERED]"
        return body


def seasonality_report(
    data,
    name: str | None = None,
    alpha: float = 0.05,
    covariance: str = "HC3",
    hac_lags: int = HAC_LAGS,
    allow_underpowered: bool = False,
) -> SeasonalityReport:
    """Test one monthly series for calendar seasonality and report it.

    `data` is a `pandas.Series` with a monthly `DatetimeIndex`, or a
    one-column frame. Eleven monthly dummies are tested jointly with HAC
    errors. Below `MIN_SEASONALITY_OBS` observations the dummies cannot be
    estimated with any precision and the function refuses, unless
    ``allow_underpowered=True``.

    The brief asks for seasonality to be accounted for. The honest form of
    that is a test with a reported statistic and a stated decision rule, not
    an unconditional seasonal adjustment: adjusting a series that has no
    seasonality spends eleven degrees of freedom for nothing.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise EconometricsError(
                "seasonality_report takes a Series or a one-column frame"
            )
        series = data.iloc[:, 0]
    else:
        series = pd.Series(data)
    label = name or (series.name if series.name is not None else "series")
    series = series.dropna().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise EconometricsError("seasonality_report needs a DatetimeIndex")

    n = len(series)
    warnings_out: list[str] = []
    if n < 24:
        raise EconometricsError(
            f"{label}: n={n} cannot support eleven monthly dummies; a "
            "seasonality test on this sample is arithmetic, not inference"
        )
    if n < MIN_SEASONALITY_OBS:
        message = (
            f"{label}: n={n} is below {MIN_SEASONALITY_OBS} (three years), so "
            "each month has at most two observations and the dummy estimates "
            "are single-observation means"
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " (pass allow_underpowered=True to report it anyway)"
            )
        warnings_out.append(message)

    if covariance not in {"HC3", "HAC"}:
        raise EconometricsError(
            f"covariance must be 'HC3' or 'HAC', got {covariance!r}"
        )
    dummies = month_dummies(series.index)
    design = sm.add_constant(dummies, has_constant="add")
    effective_lags = min(hac_lags, max(n // 3, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if covariance == "HAC":
            fitted = sm.OLS(series, design).fit(
                cov_type="HAC", cov_kwds={"maxlags": effective_lags}
            )
        else:
            fitted = sm.OLS(series, design).fit(cov_type="HC3")
        test = fitted.f_test(np.eye(design.shape[1])[1:])

    f_statistic = float(np.squeeze(test.fvalue))
    p_value = float(np.squeeze(test.pvalue))
    month_means = series.groupby(series.index.month).mean()
    month_means.index.name = "month"
    amplitude = float(month_means.max() - month_means.min()) if len(month_means) else 0.0

    return SeasonalityReport(
        name=str(label),
        n_obs=n,
        f_statistic=f_statistic,
        p_value=p_value,
        df_num=int(design.shape[1] - 1),
        df_denom=float(n - design.shape[1]),
        alpha=alpha,
        seasonal=bool(p_value < alpha),
        month_means=month_means,
        amplitude=amplitude,
        hac_lags=effective_lags if covariance == "HAC" else 0,
        covariance=covariance,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        notes=(
            "eleven monthly dummies, January omitted, joint F-test",
            SEASONALITY_COVARIANCE_NOTE,
        ),
    )


# ---------------------------------------------------------------------------
# 6. Rolling-origin forecast evaluation (out-of-sample accuracy)
# ---------------------------------------------------------------------------


def _ladder_specifications(
    frame: pd.DataFrame,
    model1_features: tuple[str, ...],
    model2_extra: tuple[str, ...],
    media_features: tuple[str, ...],
    household_political: tuple[str, ...],
    include_media_only: bool,
    min_coverage: float,
) -> tuple[dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]], list[str]]:
    """Resolve the ladder's specifications against the columns of `frame`.

    Shared by the rolling forecast, the stability report and the sensitivity
    grid so that all of them describe the same four models. Coverage is judged
    on `frame` alone, which the callers set to the FIRST TRAINING BLOCK: a
    specification choice made on the whole sample would be a look-ahead.
    """
    notes: list[str] = []
    blocks: dict[str, tuple[tuple[str, ...], list[str]]] = {}
    for key, requested in (
        ("econ", tuple(model1_features)),
        ("extra", tuple(model2_extra)),
        ("media", tuple(media_features)),
        ("controls", tuple(household_political)),
    ):
        present, missing, _dropped, reasons, subs = _resolve_specification(
            frame, requested, min_coverage
        )
        blocks[key] = (present, missing)
        notes.extend(reasons)
        notes.extend(subs)

    specs: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
        "Model 1": (
            blocks["econ"][0],
            tuple(model1_features),
            tuple(blocks["econ"][1]),
        ),
        "Model 2": (
            blocks["econ"][0] + blocks["extra"][0],
            tuple(model1_features) + tuple(model2_extra),
            tuple(blocks["econ"][1] + blocks["extra"][1]),
        ),
    }
    if include_media_only:
        specs["Model 2b"] = (
            blocks["media"][0],
            tuple(media_features),
            tuple(blocks["media"][1]),
        )
    specs["Model 3"] = (
        blocks["econ"][0] + blocks["extra"][0] + blocks["media"][0],
        tuple(model1_features) + tuple(model2_extra) + tuple(media_features),
        tuple(blocks["econ"][1] + blocks["extra"][1] + blocks["media"][1]),
    )
    if household_political:
        specs["Model 4"] = (
            blocks["econ"][0]
            + blocks["extra"][0]
            + blocks["media"][0]
            + blocks["controls"][0],
            tuple(model1_features)
            + tuple(model2_extra)
            + tuple(media_features)
            + tuple(household_political),
            tuple(
                blocks["econ"][1]
                + blocks["extra"][1]
                + blocks["media"][1]
                + blocks["controls"][1]
            ),
        )
    return specs, notes


@dataclass(frozen=True)
class ForecastModelResult:
    """One model's out-of-sample record over the rolling origins."""

    name: str
    features: tuple[str, ...]
    available: bool
    unavailable_reason: str | None = None
    n_folds: int = 0
    rmse: float = float("nan")
    mae: float = float("nan")
    mean_error: float = float("nan")
    rmse_ratio_to_naive: float = float("nan")
    dm_statistic: float | None = None
    dm_p_value: float | None = None
    dm_reference: str | None = None
    errors: pd.Series | None = None
    predictions: pd.Series | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def beats_naive(self) -> bool:
        return bool(self.rmse_ratio_to_naive < 1.0)


@dataclass
class RollingForecastReport:
    """Expanding-window, one-step-ahead forecasts for every model of the ladder.

    This is the brief's "chronological holdouts or rolling forecasts" in its
    stronger form. Each origin fits on the rows that precede it and predicts
    the next month only. **Nothing fitted uses the month it predicts**: the
    coefficients, the specification choice and the seasonal decision all come
    from the training block.

    A single chronological holdout gives one number from one split. A rolling
    origin gives a distribution of errors, which is what makes the
    Diebold-Mariano comparison against the economic-only model possible.
    """

    models: dict[str, ForecastModelResult]
    target: str
    initial_train: int
    step: int
    horizon: int
    n_folds: int
    first_origin: pd.Timestamp | None
    last_origin: pd.Timestamp | None
    common_sample_n: int
    reference_model: str | None
    naive_rmse: float
    naive_mae: float
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    dropped_reasons: tuple[str, ...] = field(default_factory=tuple)
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = "out-of-sample predictive accuracy (not causation)"
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(n for n, m in self.models.items() if m.available)

    @property
    def best_model(self) -> str | None:
        """The lowest out-of-sample RMSE, or None when nothing was fitted."""
        candidates = [
            (m.rmse, n) for n, m in self.models.items() if m.available and m.n_folds
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    def table(self) -> pd.DataFrame:
        rows = []
        for name, result in self.models.items():
            rows.append(
                {
                    "model": name,
                    "available": result.available,
                    "folds": result.n_folds if result.available else np.nan,
                    "k": len(result.features) + 1 if result.available else np.nan,
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "mean_error": result.mean_error,
                    "rmse_vs_naive": result.rmse_ratio_to_naive,
                    "dm_stat": result.dm_statistic,
                    "dm_p": result.dm_p_value,
                    "unavailable_reason": result.unavailable_reason or "",
                }
            )
        rows.append(
            {
                "model": "Naive (last observed)",
                "available": True,
                "folds": self.n_folds,
                "k": 0,
                "rmse": self.naive_rmse,
                "mae": self.naive_mae,
                "mean_error": np.nan,
                "rmse_vs_naive": 1.0,
                "dm_stat": None,
                "dm_p": None,
                "unavailable_reason": "",
            }
        )
        return pd.DataFrame(rows).set_index("model")

    def summary(self) -> str:
        parts = [f"{self.n_folds} one-step-ahead origins"]
        if self.first_origin is not None:
            parts.append(f"{_label(self.first_origin)}..{_label(self.last_origin)}")
        best = self.best_model
        if best is not None:
            parts.append(f"best RMSE: {best} ({self.models[best].rmse:.3f})")
        parts.append(f"naive RMSE {self.naive_rmse:.3f}")
        if self.underpowered:
            parts.append("UNDERPOWERED")
        return " | ".join(parts)


def _diebold_mariano(
    errors_a: np.ndarray, errors_b: np.ndarray
) -> tuple[float, float]:
    """Diebold-Mariano on squared-error loss, Newey-West variance.

    Positive statistic means series *b* has the smaller loss, so a positive
    value in this module means the richer model forecast better than the
    reference. Returns ``(nan, nan)`` when the differential has no variance.
    """
    differential = np.asarray(errors_a, dtype=float) ** 2 - np.asarray(
        errors_b, dtype=float
    ) ** 2
    n = differential.size
    if n < 8:
        return float("nan"), float("nan")
    mean = float(differential.mean())
    centred = differential - mean
    lag_max = max(round(n ** (1.0 / 3.0)), 1)
    variance = float((centred**2).mean())
    for lag in range(1, min(lag_max, n - 1) + 1):
        weight = 1.0 - lag / (lag_max + 1.0)
        covariance = float((centred[lag:] * centred[:-lag]).mean())
        variance += 2.0 * weight * covariance
    if not np.isfinite(variance) or variance <= 0:
        return float("nan"), float("nan")
    statistic = mean / float(np.sqrt(variance / n))
    p_value = 2.0 * (1.0 - _normal_cdf(abs(statistic)))
    return statistic, p_value


def rolling_origin_forecast(
    panel: pd.DataFrame,
    target: str = TARGET,
    model1_features: tuple[str, ...] = MODEL1_FEATURES,
    model2_extra: tuple[str, ...] = MODEL2_EXTRA_FEATURES,
    media_features: tuple[str, ...] = MEDIA_FEATURES,
    household_political: tuple[str, ...] = (),
    include_media_only: bool = False,
    initial_train: int = DEFAULT_INITIAL_TRAIN,
    step: int = 1,
    min_coverage: float = MIN_FEATURE_COVERAGE,
    reference_model: str = "Model 1",
    allow_underpowered: bool = False,
) -> RollingForecastReport:
    """Expanding-window, one-step-ahead forecasts, model by model.

    The first origin sits after `initial_train` rows. Every later origin moves
    forward by `step` months, refits each model on all rows that precede the
    origin, and predicts that one month. The result carries RMSE, MAE and mean
    error per model, the ratio to a last-observed-value benchmark, and a
    Diebold-Mariano test of each model against `reference_model`.

    **No look-ahead.** Coefficients come from the training rows only; the
    specification is resolved on the first training block only; the benchmark
    uses the last *observed* value, not the contemporaneous one.

    All models are evaluated on identical origins, because an RMSE comparison
    across different test sets is not a comparison. Models whose regressors
    are absent are reported as unavailable and are excluded from the common
    sample.

    Raises `EconometricsError` when fewer than `MIN_FORECAST_FOLDS` origins are
    available, unless ``allow_underpowered=True``.
    """
    if panel.empty:
        raise EconometricsError("cannot forecast on an empty panel")
    _require_columns(panel, (target,))
    if step < 1:
        raise EconometricsError(f"step must be at least 1, got {step}")
    if initial_train < ABSOLUTE_MIN_REGRESSION_OBS:
        raise EconometricsError(
            f"initial_train={initial_train} is below the minimum of "
            f"{ABSOLUTE_MIN_REGRESSION_OBS} rows; a model fitted on fewer rows "
            "than that does not produce a forecast worth scoring"
        )
    if len(panel) <= initial_train:
        raise EconometricsError(
            f"the panel has {len(panel)} rows and initial_train={initial_train} "
            "leaves no origin to forecast"
        )

    training_block = panel.iloc[:initial_train]
    specs, notes = _ladder_specifications(
        training_block,
        model1_features,
        model2_extra,
        media_features,
        household_political,
        include_media_only,
        min_coverage,
    )

    models: dict[str, ForecastModelResult] = {}
    fittable: dict[str, tuple[str, ...]] = {}
    for name, (features, requested, missing) in specs.items():
        if missing or not features:
            reason = (
                f"{name} not forecast: regressors absent from the panel: "
                f"{list(missing) or 'all dropped for sparse coverage'}"
            )
            models[name] = ForecastModelResult(
                name=name,
                features=features,
                available=False,
                unavailable_reason=reason,
                notes=(
                    ("Unavailable is NOT the same finding as 'this model "
                    "forecast badly'."),
                ),
            )
        else:
            fittable[name] = features

    if not fittable:
        raise EconometricsError(
            "no model can be forecast: every specification is missing regressors"
        )

    needed = sorted({target, *(c for cols in fittable.values() for c in cols)})
    subset = panel.loc[:, needed]
    usable = subset.dropna()
    dropped = tuple(subset.index.difference(usable.index))
    reasons = _missing_reasons(subset, dropped)
    if len(usable) <= initial_train:
        raise EconometricsError(
            f"after dropping {len(dropped)} incomplete rows only {len(usable)} "
            f"remain, which is not more than initial_train={initial_train}"
        )

    origins = list(range(initial_train, len(usable), step))
    warnings_out: list[str] = []
    if len(origins) < MIN_FORECAST_FOLDS:
        message = (
            f"{len(origins)} rolling origins is fewer than "
            f"{MIN_FORECAST_FOLDS}; an out-of-sample RMSE over this few "
            "forecasts is a number, not a measure of accuracy"
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " (pass allow_underpowered=True to report it anyway)"
            )
        warnings_out.append(message)

    endog = usable[target]
    records: dict[str, dict[pd.Timestamp, float]] = {n: {} for n in fittable}
    naive_errors: dict[pd.Timestamp, float] = {}
    skipped: list[str] = []

    for position in origins:
        stamp = usable.index[position]
        train = usable.iloc[:position]
        actual = float(endog.iloc[position])
        naive_errors[stamp] = actual - float(endog.iloc[position - 1])
        for name, features in fittable.items():
            if len(train) <= len(features) + 1:
                skipped.append(
                    f"{name} at {_label(stamp)}: {len(train)} training rows "
                    f"cannot identify {len(features) + 1} parameters"
                )
                continue
            design = sm.add_constant(
                train.loc[:, list(features)], has_constant="add"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = sm.OLS(train[target], design).fit()
            row = usable.iloc[[position]].loc[:, list(features)]
            row = sm.add_constant(row, has_constant="add").reindex(
                columns=design.columns, fill_value=1.0
            )
            predicted = float(np.asarray(fitted.predict(row))[0])
            records[name][stamp] = actual - predicted

    naive_series = pd.Series(naive_errors).sort_index()
    naive_rmse = float(np.sqrt(np.mean(naive_series.to_numpy() ** 2)))
    naive_mae = float(np.mean(np.abs(naive_series.to_numpy())))

    reference_errors: np.ndarray | None = None
    if records.get(reference_model):
        reference_errors = pd.Series(records[reference_model]).sort_index().to_numpy()

    for name, features in fittable.items():
        errors = pd.Series(records[name]).sort_index()
        if errors.empty:
            models[name] = ForecastModelResult(
                name=name,
                features=features,
                available=False,
                unavailable_reason=(
                    f"{name} produced no forecast: every origin had too few "
                    "training rows for its parameter count"
                ),
            )
            continue
        values = errors.to_numpy()
        statistic: float | None = None
        p_value: float | None = None
        if (
            reference_errors is not None
            and name != reference_model
            and reference_errors.size == values.size
        ):
            statistic, p_value = _diebold_mariano(reference_errors, values)
        models[name] = ForecastModelResult(
            name=name,
            features=features,
            available=True,
            n_folds=int(values.size),
            rmse=float(np.sqrt(np.mean(values**2))),
            mae=float(np.mean(np.abs(values))),
            mean_error=float(np.mean(values)),
            rmse_ratio_to_naive=(
                float(np.sqrt(np.mean(values**2)) / naive_rmse)
                if naive_rmse > 0
                else float("nan")
            ),
            dm_statistic=statistic,
            dm_p_value=p_value,
            dm_reference=reference_model if statistic is not None else None,
            errors=errors,
            predictions=(endog.reindex(errors.index) - errors).rename("prediction"),
            notes=(
                "one-step-ahead, expanding window, OLS refitted at every origin",
            ),
        )

    if skipped:
        warnings_out.append(
            f"{len(skipped)} model-origin pairs were skipped for too few "
            f"training rows; first: {skipped[0]}"
        )
    notes.append(
        "the benchmark is the last observed value of the target, which is the "
        "forecast a reader gets for free"
    )
    if reference_errors is None:
        notes.append(
            f"no Diebold-Mariano test: the reference model {reference_model!r} "
            "produced no forecasts"
        )

    ordered = {name: models[name] for name in specs if name in models}
    return RollingForecastReport(
        models=ordered,
        target=target,
        initial_train=initial_train,
        step=step,
        horizon=1,
        n_folds=len(naive_series),
        first_origin=naive_series.index.min() if len(naive_series) else None,
        last_origin=naive_series.index.max() if len(naive_series) else None,
        common_sample_n=len(usable),
        reference_model=reference_model if reference_errors is not None else None,
        naive_rmse=naive_rmse,
        naive_mae=naive_mae,
        dropped_dates=dropped,
        dropped_reasons=reasons,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 7. Coefficient stability (rolling coefficient paths with bands)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientWindow:
    """One rolling window's estimate of one coefficient."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_obs: int
    coef: float
    std_err: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class CoefficientPath:
    """The path of one coefficient across the rolling windows, and its verdict.

    `stable` requires three things at once:

    1. the sign never changes across the windows;
    2. the spread of the estimates is no more than four times their own mean
       standard error, so the movement is small against the noise;
    3. the path mean lies inside at least 90 percent of the windows'
       intervals.

    `all_intervals_overlap` -- whether one value is inside EVERY window's
    interval -- is reported as a diagnostic and is deliberately NOT part of
    the rule. With overlapping windows that criterion fails on long and
    perfectly stable paths, because a long path has many intervals and each
    one is narrow.
    """

    feature: str
    windows: tuple[CoefficientWindow, ...]
    mean: float
    std: float
    minimum: float
    maximum: float
    sign_flips: int
    share_significant: float
    all_intervals_overlap: bool
    share_covering_mean: float
    spread_over_mean_se: float
    stable: bool
    reason: str

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": self.feature,
                    "window_start": w.window_start,
                    "window_end": w.window_end,
                    "n": w.n_obs,
                    "coef": w.coef,
                    "std_err": w.std_err,
                    "p_value": w.p_value,
                    "ci_lower": w.ci_lower,
                    "ci_upper": w.ci_upper,
                }
                for w in self.windows
            ]
        )


@dataclass
class StabilityReport:
    """Rolling coefficient paths for one specification, with bands.

    The brief asks for coefficient stability as a separate item from
    predictive accuracy. A specification can forecast acceptably while its
    coefficients wander, and a wandering coefficient is evidence against
    reading any single estimate as a structural quantity.
    """

    model_name: str
    features: tuple[str, ...]
    target: str
    window: int
    step: int
    hac_lags: int
    paths: dict[str, CoefficientPath]
    n_windows: int
    first_window_end: pd.Timestamp | None
    last_window_end: pd.Timestamp | None
    dropped_features: tuple[str, ...] = field(default_factory=tuple)
    drop_reasons: tuple[str, ...] = field(default_factory=tuple)
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    dropped_reasons: tuple[str, ...] = field(default_factory=tuple)
    underpowered: bool = False
    power_warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = "coefficient stability diagnostic"
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def unstable_features(self) -> tuple[str, ...]:
        return tuple(name for name, path in self.paths.items() if not path.stable)

    def paths_frame(self) -> pd.DataFrame:
        """Long format: one row per feature per window, with the band."""
        frames = [path.frame() for path in self.paths.values()]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "window_start",
                    "window_end",
                    "n",
                    "coef",
                    "std_err",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def table(self) -> pd.DataFrame:
        """One row per feature: the summary of its path."""
        rows = []
        for name, path in self.paths.items():
            rows.append(
                {
                    "feature": name,
                    "windows": len(path.windows),
                    "mean": path.mean,
                    "std": path.std,
                    "min": path.minimum,
                    "max": path.maximum,
                    "sign_flips": path.sign_flips,
                    "share_significant": path.share_significant,
                    "intervals_overlap": path.all_intervals_overlap,
                    "share_covering_mean": path.share_covering_mean,
                    "stable": path.stable,
                    "reason": path.reason,
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def summary(self) -> str:
        parts = [f"{self.model_name}: {self.n_windows} rolling windows of {self.window}"]
        unstable = self.unstable_features
        parts.append(
            f"unstable: {', '.join(unstable)}" if unstable else "all paths stable"
        )
        if self.underpowered:
            parts.append("UNDERPOWERED")
        return " | ".join(parts)


def coefficient_stability(
    panel: pd.DataFrame,
    features: tuple[str, ...] | None = None,
    target: str = TARGET,
    model_name: str = "Model 3",
    window: int = MIN_STABILITY_WINDOW + 12,
    step: int = 3,
    hac_lags: int = HAC_LAGS,
    min_coverage: float = MIN_FEATURE_COVERAGE,
    alpha: float = 0.05,
    allow_underpowered: bool = False,
) -> StabilityReport:
    """Rolling-window coefficient paths with HAC confidence bands.

    `features` defaults to the Model 3 regressor set that the panel can
    actually supply (economic, level and burden, media). Each window of
    `window` consecutive rows is fitted with HAC errors and its coefficients
    and intervals are recorded, so the report shows whether the headline
    estimate is a stable quantity or an artefact of one sample period.

    Refuses when the window is shorter than `MIN_STABILITY_WINDOW`, when the
    window cannot identify the parameters, or when fewer than
    `MIN_STABILITY_WINDOWS` windows exist -- unless
    ``allow_underpowered=True``.
    """
    if panel.empty:
        raise EconometricsError("cannot measure stability on an empty panel")
    _require_columns(panel, (target,))
    if step < 1:
        raise EconometricsError(f"step must be at least 1, got {step}")

    requested = (
        tuple(features)
        if features is not None
        else MODEL1_FEATURES + MODEL2_EXTRA_FEATURES + MEDIA_FEATURES
    )
    present, missing, dropped_cols, drop_reasons, subs = _resolve_specification(
        panel, requested, min_coverage
    )
    notes = list(subs)
    if missing:
        notes.append(
            f"requested regressors absent from the panel and left out: {missing}"
        )
    notes.extend(drop_reasons)
    if not present:
        raise EconometricsError(
            "no requested regressor is available for a stability path "
            f"(missing={missing}, dropped={dropped_cols})"
        )

    subset = panel.loc[:, [target, *present]]
    usable = subset.dropna()
    dropped_rows = tuple(subset.index.difference(usable.index))
    row_reasons = _missing_reasons(subset, dropped_rows)

    k = len(present) + 1
    warnings_out: list[str] = []
    if window < MIN_STABILITY_WINDOW:
        message = (
            f"window={window} is below {MIN_STABILITY_WINDOW}; a rolling HAC "
            f"fit of {k} parameters on {window} rows is not inference"
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " (pass allow_underpowered=True to report it anyway)"
            )
        warnings_out.append(message)
    if window <= k:
        raise EconometricsError(
            f"window={window} cannot identify {k} parameters; this refuses "
            "rather than returning a path of arbitrary numbers"
        )
    if len(usable) < window:
        raise EconometricsError(
            f"{len(usable)} complete rows are fewer than window={window}"
        )

    starts = list(range(0, len(usable) - window + 1, step))
    if len(starts) < MIN_STABILITY_WINDOWS:
        message = (
            f"{len(starts)} rolling windows is fewer than "
            f"{MIN_STABILITY_WINDOWS}; a path this short cannot show "
            "instability or rule it out"
        )
        if not allow_underpowered:
            raise EconometricsError(
                message + " (pass allow_underpowered=True to report it anyway)"
            )
        warnings_out.append(message)

    collected: dict[str, list[CoefficientWindow]] = {name: [] for name in present}
    effective_lags = min(hac_lags, max(window // 3, 1))
    if effective_lags != hac_lags:
        notes.append(
            f"HAC maxlags reduced from {hac_lags} to {effective_lags} inside a "
            f"{window}-row window, where the longer kernel is degenerate"
        )

    for start in starts:
        block = usable.iloc[start : start + window]
        design = sm.add_constant(block.loc[:, list(present)], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.OLS(block[target], design).fit(
                cov_type="HAC", cov_kwds={"maxlags": effective_lags}
            )
        bounds = fitted.conf_int(alpha=alpha)
        for name in present:
            collected[name].append(
                CoefficientWindow(
                    window_start=block.index[0],
                    window_end=block.index[-1],
                    n_obs=len(block),
                    coef=float(fitted.params[name]),
                    std_err=float(fitted.bse[name]),
                    p_value=float(fitted.pvalues[name]),
                    ci_lower=float(bounds.loc[name].iloc[0]),
                    ci_upper=float(bounds.loc[name].iloc[1]),
                )
            )

    paths: dict[str, CoefficientPath] = {}
    for name, windows in collected.items():
        values = np.array([w.coef for w in windows], dtype=float)
        errors = np.array([w.std_err for w in windows], dtype=float)
        signs = np.sign(values)
        flips = int(np.sum(signs[1:] * signs[:-1] < 0))
        lower = max(w.ci_lower for w in windows)
        upper = min(w.ci_upper for w in windows)
        overlap = bool(lower <= upper)
        mean_se = float(np.mean(errors)) if errors.size else float("nan")
        spread = float(values.max() - values.min())
        ratio = spread / mean_se if mean_se and np.isfinite(mean_se) else float("nan")
        share = float(np.mean([w.p_value < alpha for w in windows]))
        centre = float(values.mean())
        covering = float(
            np.mean([w.ci_lower <= centre <= w.ci_upper for w in windows])
        )
        problems = []
        if flips:
            problems.append(f"{flips} sign change(s)")
        if np.isfinite(ratio) and ratio > 4.0:
            problems.append(
                f"spread is {ratio:.1f} times the mean standard error"
            )
        if covering < 0.9:
            problems.append(
                f"the path mean is inside only {covering:.0%} of the "
                "windows' confidence intervals"
            )
        paths[name] = CoefficientPath(
            feature=name,
            windows=tuple(windows),
            mean=float(values.mean()),
            std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            minimum=float(values.min()),
            maximum=float(values.max()),
            sign_flips=flips,
            share_significant=share,
            all_intervals_overlap=overlap,
            share_covering_mean=covering,
            spread_over_mean_se=ratio,
            stable=not problems,
            reason="; ".join(problems) if problems else "stable on all three checks",
        )

    notes.append(
        "a stable path is a necessary condition for reading a coefficient as a "
        "structural quantity, not a sufficient one"
    )
    return StabilityReport(
        model_name=model_name,
        features=tuple(present),
        target=target,
        window=window,
        step=step,
        hac_lags=effective_lags,
        paths=paths,
        n_windows=len(starts),
        first_window_end=usable.index[window - 1] if starts else None,
        last_window_end=usable.index[starts[-1] + window - 1] if starts else None,
        dropped_features=tuple(dropped_cols),
        drop_reasons=tuple(drop_reasons),
        dropped_dates=dropped_rows,
        dropped_reasons=row_reasons,
        underpowered=bool(warnings_out),
        power_warnings=tuple(warnings_out),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 8. Specification sensitivity (the headline result across a stated grid)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecificationCell:
    """One cell of the sensitivity grid: the choices, and what they gave."""

    label: str
    start: str
    hac_lags: int
    with_debt_service: bool
    seasonal_control: bool
    feasible: bool
    n_obs: int = 0
    coef: float = float("nan")
    std_err: float = float("nan")
    p_value: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    reason: str = ""

    @property
    def significant(self) -> bool:
        return bool(self.feasible and self.p_value < 0.05)


@dataclass
class SensitivityReport:
    """The headline coefficient recomputed across a documented grid.

    The brief asks for sensitivity to alternative specifications. The grid is
    therefore fixed in the code and reported in full, including the cells that
    could not be estimated and why: a sensitivity table that shows only the
    cells that worked is a selection of results.
    """

    focus_feature: str
    target: str
    cells: tuple[SpecificationCell, ...]
    grid_description: tuple[str, ...]
    n_cells: int
    n_feasible: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = "specification sensitivity (association, no identification)"
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    @property
    def feasible_cells(self) -> tuple[SpecificationCell, ...]:
        return tuple(cell for cell in self.cells if cell.feasible)

    @property
    def share_significant(self) -> float:
        cells = self.feasible_cells
        if not cells:
            return float("nan")
        return float(np.mean([cell.significant for cell in cells]))

    @property
    def sign_stable(self) -> bool:
        signs = {np.sign(cell.coef) for cell in self.feasible_cells}
        return len(signs) <= 1

    @property
    def coef_range(self) -> tuple[float, float]:
        cells = self.feasible_cells
        if not cells:
            return (float("nan"), float("nan"))
        values = [cell.coef for cell in cells]
        return (float(min(values)), float(max(values)))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "specification": cell.label,
                    "start": cell.start,
                    "hac_lags": cell.hac_lags,
                    "debt_service": cell.with_debt_service,
                    "seasonal": cell.seasonal_control,
                    "feasible": cell.feasible,
                    "n": cell.n_obs if cell.feasible else np.nan,
                    "coef": cell.coef,
                    "std_err": cell.std_err,
                    "p_value": cell.p_value,
                    "significant_5pct": cell.significant,
                    "reason": cell.reason,
                }
                for cell in self.cells
            ]
        ).set_index("specification")

    def verdict(self) -> str:
        cells = self.feasible_cells
        if not cells:
            return (
                f"{self.focus_feature}: no cell of the grid could be estimated; "
                "the sensitivity question is unanswered, not answered negatively"
            )
        low, high = self.coef_range
        share = self.share_significant
        body = (
            f"{self.focus_feature}: estimated in {len(cells)} of {self.n_cells} "
            f"cells; coefficient from {low:.3f} to {high:.3f}; significant at 5% "
            f"in {share:.0%} of them; sign "
            + ("stable" if self.sign_stable else "NOT stable")
        )
        if share == 1.0:
            body += ". The result does not depend on these choices."
        elif share == 0.0:
            body += ". The result is absent under every choice in the grid."
        else:
            body += ". THE RESULT DEPENDS ON THE SPECIFICATION."
        return body


def specification_sensitivity(
    panel: pd.DataFrame,
    focus_feature: str = "political_attribution",
    target: str = TARGET,
    starts: tuple[str, ...] = ("2010-01-01", "2015-01-01"),
    hac_lag_grid: tuple[int, ...] = (6, 12, 18),
    debt_service_column: str = "debt_service",
    seasonal_options: tuple[bool, ...] = (False, True),
    model1_features: tuple[str, ...] = MODEL1_FEATURES,
    model2_extra: tuple[str, ...] = MODEL2_EXTRA_FEATURES,
    media_features: tuple[str, ...] = MEDIA_FEATURES,
    household_political: tuple[str, ...] = (),
    min_coverage: float = MIN_FEATURE_COVERAGE,
    allow_underpowered: bool = True,
) -> SensitivityReport:
    """Recompute one coefficient across a documented grid of reasonable choices.

    The grid has four dimensions, each a choice a reader could reasonably have
    made differently:

    * the training window start (`starts`);
    * the HAC truncation lag (`hac_lag_grid`);
    * the debt-service control in or out (it is insignificant and its FRED
      series ends early, so leaving it out is defensible and so is putting it
      back);
    * monthly dummies in or out.

    Every cell is reported. A cell that cannot be estimated keeps its row with
    ``feasible=False`` and the reason. ``allow_underpowered`` defaults to True
    here on purpose: the point of the table is to show what the short windows
    do to the estimate, and a refusal in one cell would hide that.
    """
    if panel.empty:
        raise EconometricsError("cannot run a sensitivity grid on an empty panel")
    _require_columns(panel, (target,))
    if not starts or not hac_lag_grid or not seasonal_options:
        raise EconometricsError("the sensitivity grid must have at least one cell")

    debt_available = debt_service_column in panel.columns
    notes: list[str] = []
    if not debt_available:
        notes.append(
            f"'{debt_service_column}' is not in the panel, so the grid's "
            "debt-service dimension collapses to 'out'"
        )
    debt_options = (False, True) if debt_available else (False,)

    cells: list[SpecificationCell] = []
    for start in starts:
        for lags in hac_lag_grid:
            for with_debt in debt_options:
                for seasonal in seasonal_options:
                    label = (
                        f"start={start[:4]}, hac={lags}, "
                        f"debt={'in' if with_debt else 'out'}, "
                        f"seasonal={'yes' if seasonal else 'no'}"
                    )
                    window = panel.loc[start:]
                    extra = tuple(model2_extra)
                    if with_debt:
                        extra = extra + (debt_service_column,)
                    try:
                        comparison = compare_models(
                            window,
                            target=target,
                            model1_features=model1_features,
                            model2_extra=extra,
                            media_features=media_features,
                            household_political=household_political,
                            hac_lags=lags,
                            min_coverage=min_coverage,
                            seasonal_control="always" if seasonal else "none",
                            allow_underpowered=allow_underpowered,
                        )
                    except EconometricsError as exc:
                        cells.append(
                            SpecificationCell(
                                label=label,
                                start=start,
                                hac_lags=lags,
                                with_debt_service=with_debt,
                                seasonal_control=seasonal,
                                feasible=False,
                                reason=str(exc),
                            )
                        )
                        continue
                    frame = comparison.coefficient_table()
                    hit = frame[frame["regressor"] == focus_feature]
                    if hit.empty:
                        cells.append(
                            SpecificationCell(
                                label=label,
                                start=start,
                                hac_lags=lags,
                                with_debt_service=with_debt,
                                seasonal_control=seasonal,
                                feasible=False,
                                reason=(
                                    f"'{focus_feature}' is in no estimated model of "
                                    "this cell"
                                ),
                            )
                        )
                        continue
                    # The richest model that carries the focus regressor.
                    row = hit.iloc[-1]
                    fit = comparison.models[row["model"]]
                    cells.append(
                        SpecificationCell(
                            label=label,
                            start=start,
                            hac_lags=lags,
                            with_debt_service=with_debt,
                            seasonal_control=seasonal,
                            feasible=True,
                            n_obs=int(fit.n),
                            coef=float(row["coef"]),
                            std_err=float(row["std_err"]),
                            p_value=float(row["p_value"]),
                            ci_lower=float(row["ci_lower"]),
                            ci_upper=float(row["ci_upper"]),
                            reason=f"from {row['model']}",
                        )
                    )

    return SensitivityReport(
        focus_feature=focus_feature,
        target=target,
        cells=tuple(cells),
        grid_description=(
            f"window start: {', '.join(starts)}",
            f"HAC truncation lag: {', '.join(str(l) for l in hac_lag_grid)}",
            f"debt-service control: {'in or out' if debt_available else 'out only'}",
            "monthly dummies: "
            + ", ".join("yes" if option else "no" for option in seasonal_options),
        ),
        n_cells=len(cells),
        n_feasible=sum(1 for cell in cells if cell.feasible),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 9. Multiple testing across the family of media tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisTest:
    """One member of a test family, with everything needed to adjust it.

    `family` groups tests that belong to the same question (a coefficient
    family, a Granger family). `label` must be unique inside the family,
    because the adjusted tables are indexed by it.
    """

    label: str
    family: str
    p_value: float
    statistic: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0:
            raise EconometricsError(
                f"{self.label}: p-value {self.p_value} is not in [0, 1]"
            )


@dataclass
class MultipleTestingReport:
    """Raw and adjusted p-values over one family of tests.

    **Why this exists.** The project runs several tests across the media
    measures and both Granger directions. With a family of that size, one test
    at p < 0.05 is the expected number of false rejections under a complete
    null. Reporting the raw p-value alone would let a single hit carry the
    conclusion. Both corrections are reported:

    * Holm-Bonferroni controls the family-wise error rate -- the probability of
      ANY false rejection. It is the strict one.
    * Benjamini-Hochberg controls the false discovery rate -- the expected
      share of the rejections that are false. It is the appropriate one for a
      screening family, and it is the headline here.

    Raw p-values are kept in the table, because hiding them would make the
    correction unauditable.
    """

    tests: tuple[HypothesisTest, ...]
    alpha: float
    holm_adjusted: dict[str, float]
    bh_adjusted: dict[str, float]
    family_size: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = "multiplicity-adjusted hypothesis tests"

    @property
    def raw_significant(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.tests if t.p_value < self.alpha)

    @property
    def holm_significant(self) -> tuple[str, ...]:
        return tuple(
            t.label for t in self.tests if self.holm_adjusted[t.label] < self.alpha
        )

    @property
    def bh_significant(self) -> tuple[str, ...]:
        return tuple(
            t.label for t in self.tests if self.bh_adjusted[t.label] < self.alpha
        )

    @property
    def lost_to_correction(self) -> tuple[str, ...]:
        """Raw rejections that do not survive Benjamini-Hochberg."""
        survivors = set(self.bh_significant)
        return tuple(
            label for label in self.raw_significant if label not in survivors
        )

    def survives(self, label: str, method: str = "bh") -> bool:
        """Does one named test survive the correction at `alpha`?"""
        table = self.bh_adjusted if method == "bh" else self.holm_adjusted
        if label not in table:
            raise EconometricsError(
                f"{label!r} is not in this family: {sorted(table)}"
            )
        return bool(table[label] < self.alpha)

    def table(self) -> pd.DataFrame:
        rows = []
        for test in self.tests:
            rows.append(
                {
                    "test": test.label,
                    "family": test.family,
                    "statistic": test.statistic,
                    "p_raw": test.p_value,
                    "p_holm": self.holm_adjusted[test.label],
                    "q_bh": self.bh_adjusted[test.label],
                    "raw_significant": test.p_value < self.alpha,
                    "holm_significant": self.holm_adjusted[test.label] < self.alpha,
                    "bh_significant": self.bh_adjusted[test.label] < self.alpha,
                    "detail": test.detail,
                }
            )
        return pd.DataFrame(rows).set_index("test")

    def summary(self) -> str:
        lost = self.lost_to_correction
        parts = [
            f"family of {self.family_size} tests at alpha={self.alpha}",
            f"raw rejections: {len(self.raw_significant)}",
            f"BH survivors: {len(self.bh_significant)}",
            f"Holm survivors: {len(self.holm_significant)}",
        ]
        if lost:
            parts.append(f"does NOT survive BH: {', '.join(lost)}")
        return " | ".join(parts)


def multiple_testing_correction(
    tests, alpha: float = 0.05
) -> MultipleTestingReport:
    """Holm-Bonferroni and Benjamini-Hochberg over one family of tests.

    Both procedures are written out here rather than imported, because the
    report has to state exactly what was done: Holm compares the i-th smallest
    p-value with ``alpha / (m - i + 1)`` and enforces monotonicity upward; BH
    gives ``q_(i) = min_{j >= i} m * p_(j) / j``, also monotone. Both are
    capped at 1.

    The family must be declared by the caller, not inferred. `alpha` applies to
    the adjusted values.
    """
    members = tuple(tests)
    if not members:
        raise EconometricsError(
            "a multiple-testing correction needs at least one test; an empty "
            "family is not a result"
        )
    if not 0.0 < alpha < 1.0:
        raise EconometricsError(f"alpha must be in (0, 1), got {alpha}")
    labels = [t.label for t in members]
    if len(set(labels)) != len(labels):
        duplicates = sorted({l for l in labels if labels.count(l) > 1})
        raise EconometricsError(
            f"test labels must be unique inside a family; repeated: {duplicates}"
        )

    m = len(members)
    order = sorted(range(m), key=lambda i: members[i].p_value)
    holm: dict[str, float] = {}
    running = 0.0
    for rank, index in enumerate(order, start=1):
        value = min((m - rank + 1) * members[index].p_value, 1.0)
        running = max(running, value)
        holm[members[index].label] = running

    bh: dict[str, float] = {}
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        value = min(m * members[index].p_value / rank, 1.0)
        running = min(running, value)
        bh[members[index].label] = running

    notes = [
        f"family declared by the caller: {m} tests",
        ("Holm-Bonferroni controls the family-wise error rate; "
        "Benjamini-Hochberg controls the false discovery rate"),
        "raw p-values are retained so that the adjustment can be checked",
    ]
    if m == 1:
        notes.append(
            "a family of one leaves the p-value unchanged; this is reported so "
            "that a single-test family cannot be mistaken for a corrected one"
        )
    return MultipleTestingReport(
        tests=members,
        alpha=alpha,
        holm_adjusted=holm,
        bh_adjusted=bh,
        family_size=m,
        notes=tuple(notes),
    )


def collect_media_test_family(
    comparison: ModelComparison | None = None,
    granger_reports=(),
    media_features: tuple[str, ...] = MEDIA_FEATURES,
    model_name: str | None = None,
    extra_tests=(),
) -> tuple[HypothesisTest, ...]:
    """Build the family of media hypothesis tests from the result objects.

    The family is every test that could have supported Hypothesis B:

    * the HAC coefficient test of each media regressor, taken from one model
      only (`model_name`, by default the richest available model that carries
      media regressors) -- the same regressor in a nested model is the same
      hypothesis tested twice, and counting it twice would inflate the family;
    * both Granger directions of every `GrangerReport` supplied, because the
      reverse direction is a test the project actually ran and reports;
    * anything the caller adds through `extra_tests`.

    Declaring the family explicitly is the point. A correction over a family
    chosen after seeing the p-values is not a correction.
    """
    collected: list[HypothesisTest] = []
    if comparison is not None:
        candidates = [
            name
            for name, fit in comparison.models.items()
            if fit.available
            and fit.coefficients is not None
            and any(feature in fit.features for feature in media_features)
        ]
        chosen = model_name or (candidates[-1] if candidates else None)
        if chosen is not None:
            fit = comparison.models[chosen]
            if not fit.available or fit.coefficients is None:
                raise EconometricsError(
                    f"{chosen} carries no coefficients, so it cannot contribute "
                    "tests to the family"
                )
            for feature in media_features:
                if feature not in fit.coefficients.index:
                    continue
                row = fit.coefficients.loc[feature]
                collected.append(
                    HypothesisTest(
                        label=f"{chosen}: {feature}",
                        family="media coefficient (HAC)",
                        p_value=float(row["p_value"]),
                        statistic=float(row["t_stat"]),
                        detail=(
                            f"coefficient {float(row['coef']):.4f} on {feature} in "
                            f"{chosen}, n={fit.n}"
                        ),
                    )
                )
    for report in granger_reports:
        for test in (report.forward, report.reverse):
            if test is None:
                continue
            collected.append(
                HypothesisTest(
                    label=f"Granger: {test.causing} -> {test.caused}",
                    family="Granger precedence",
                    p_value=float(test.p_value),
                    statistic=float(test.f_statistic),
                    detail=(
                        f"VAR({test.lag_order}) F-test, "
                        f"n={report.n_obs_used}"
                    ),
                )
            )
    collected.extend(extra_tests)
    return tuple(collected)


# ---------------------------------------------------------------------------
# 10. Reverse causality, as output rather than prose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionEvidence:
    """Both Granger directions and the lead-lag correlations for one measure."""

    media_column: str
    gap_column: str
    tested: bool
    forward_p: float = float("nan")
    reverse_p: float = float("nan")
    forward_f: float = float("nan")
    reverse_f: float = float("nan")
    lag_order: int = 0
    n_obs: int = 0
    contemporaneous_correlation: float = float("nan")
    media_leads_correlation: float = float("nan")
    gap_leads_correlation: float = float("nan")
    lead_lag_balance: float = float("nan")
    reason: str = ""

    @property
    def verdict(self) -> str:
        if not self.tested:
            return f"{self.media_column}: not tested ({self.reason})"
        forward = self.forward_p < 0.05
        reverse = self.reverse_p < 0.05
        if forward and reverse:
            return (
                f"{self.media_column}: precedence in BOTH directions -- feedback, "
                "so neither direction identifies a mechanism"
            )
        if forward:
            return (
                f"{self.media_column}: coverage precedes the gap and the reverse "
                "is not rejected"
            )
        if reverse:
            return (
                f"{self.media_column}: the gap precedes coverage and not the "
                "reverse -- the reverse-causality narratives are the ones that "
                "fit"
            )
        return (
            f"{self.media_column}: no precedence in either direction; the "
            "association, where there is one, is contemporaneous"
        )


@dataclass
class ReverseCausalityReport:
    """The brief's reverse-causality discussion, computed rather than asserted.

    Section 4 of the brief names two reverse narratives and one common-cause
    narrative. This object scores all three against the evidence the project
    actually has, and -- importantly -- states where the evidence cannot
    separate them.

    **The two reverse narratives share one statistical signature.** "Pessimistic
    consumers seek negative content" and "publishers respond to audience
    demand" both predict that sentiment moves first and coverage follows. A
    bidirectional Granger test can find that signature. It cannot say which of
    the two produced it, because one is about exposure choice and the other is
    about editorial supply, and this corpus observes supply only.
    """

    gap_column: str
    evidence: dict[str, DirectionEvidence]
    narratives: dict[str, str]
    narrative_verdicts: dict[str, str]
    separable: bool
    separation_requirements: tuple[str, ...]
    max_lead: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    evidence_class: str = "predictive precedence in both directions (NOT causation)"
    interpretation: str = GRANGER_CAVEAT
    exposure_gap: str = EXPOSURE_GAP_STATEMENT

    def table(self) -> pd.DataFrame:
        rows = []
        for name, item in self.evidence.items():
            rows.append(
                {
                    "media_measure": name,
                    "tested": item.tested,
                    "n": item.n_obs if item.tested else np.nan,
                    "lag_order": item.lag_order if item.tested else np.nan,
                    "p_media_to_gap": item.forward_p,
                    "p_gap_to_media": item.reverse_p,
                    "corr_contemporaneous": item.contemporaneous_correlation,
                    "corr_media_leads": item.media_leads_correlation,
                    "corr_gap_leads": item.gap_leads_correlation,
                    "lead_lag_balance": item.lead_lag_balance,
                    "verdict": item.verdict,
                    "reason": item.reason,
                }
            )
        return pd.DataFrame(rows).set_index("media_measure")

    def summary(self) -> str:
        lines = [f"reverse causality against '{self.gap_column}':"]
        for item in self.evidence.values():
            lines.append("  " + item.verdict)
        for key, verdict in self.narrative_verdicts.items():
            lines.append(f"  {key}: {verdict}")
        lines.append(
            "  the two reverse narratives are "
            + ("separable" if self.separable else "NOT separable with these data")
        )
        return "\n".join(lines)


def reverse_causality_evidence(
    frame: pd.DataFrame,
    media_columns=MEDIA_FEATURES,
    gap_column: str = "gap",
    max_lags: int = 6,
    max_lead: int = 3,
    alpha: float = 0.05,
    allow_underpowered: bool = False,
) -> ReverseCausalityReport:
    """Score the brief's reverse-causality narratives against the evidence.

    For every media measure present in `frame`, both Granger directions are
    tested (through `granger_media_gap`, so the ADF checks, the lag selection
    and the power guards all apply) and the lead-lag cross-correlations are
    calculated out to `max_lead` months. A measure that cannot be tested keeps
    its row with the reason.

    The output then states, per narrative:

    * "pessimistic consumers seek negative content" -- needs the gap to
      precede coverage, AND exposure data to separate it from the next one;
    * "publishers respond to audience demand" -- needs the gap to precede
      coverage, and is the supply-side reading of the same signature;
    * "a common shock moves both" -- predicts contemporaneous association with
      no precedence.
    """
    _require_columns(frame, (gap_column,))
    wanted = [name for name in media_columns if name in frame.columns]
    if not wanted:
        raise EconometricsError(
            f"none of the media columns {list(media_columns)} is in the frame; "
            "the reverse-causality question cannot be put to these data"
        )
    if max_lead < 1:
        raise EconometricsError(f"max_lead must be at least 1, got {max_lead}")

    evidence: dict[str, DirectionEvidence] = {}
    notes: list[str] = []
    for name in wanted:
        pair = frame.loc[:, [name, gap_column]].dropna()
        if len(pair) < 3:
            evidence[name] = DirectionEvidence(
                media_column=name,
                gap_column=gap_column,
                tested=False,
                reason=f"only {len(pair)} months carry both series",
            )
            continue
        contemporaneous = float(pair[name].corr(pair[gap_column]))
        media_leads = [
            float(pair[name].shift(lead).corr(pair[gap_column]))
            for lead in range(1, max_lead + 1)
        ]
        gap_leads = [
            float(pair[gap_column].shift(lead).corr(pair[name]))
            for lead in range(1, max_lead + 1)
        ]
        best_media_lead = max(media_leads, key=abs) if media_leads else float("nan")
        best_gap_lead = max(gap_leads, key=abs) if gap_leads else float("nan")
        try:
            report = granger_media_gap(
                pair,
                media_column=name,
                gap_column=gap_column,
                max_lags=max_lags,
                alpha=alpha,
                allow_underpowered=allow_underpowered,
            )
        except EconometricsError as exc:
            evidence[name] = DirectionEvidence(
                media_column=name,
                gap_column=gap_column,
                tested=False,
                contemporaneous_correlation=contemporaneous,
                media_leads_correlation=best_media_lead,
                gap_leads_correlation=best_gap_lead,
                lead_lag_balance=abs(best_media_lead) - abs(best_gap_lead),
                reason=str(exc),
            )
            continue
        forward, reverse = report.forward, report.reverse
        evidence[name] = DirectionEvidence(
            media_column=name,
            gap_column=gap_column,
            tested=True,
            forward_p=float(forward.p_value) if forward else float("nan"),
            reverse_p=float(reverse.p_value) if reverse else float("nan"),
            forward_f=float(forward.f_statistic) if forward else float("nan"),
            reverse_f=float(reverse.f_statistic) if reverse else float("nan"),
            lag_order=report.lag_order,
            n_obs=report.n_obs_used,
            contemporaneous_correlation=contemporaneous,
            media_leads_correlation=best_media_lead,
            gap_leads_correlation=best_gap_lead,
            lead_lag_balance=abs(best_media_lead) - abs(best_gap_lead),
            reason="",
        )
        if report.underpowered:
            notes.append(f"{name}: the Granger test was underpowered")

    tested = [item for item in evidence.values() if item.tested]
    any_forward = any(item.forward_p < alpha for item in tested)
    any_reverse = any(item.reverse_p < alpha for item in tested)
    strong_contemporaneous = any(
        abs(item.contemporaneous_correlation) > 0.3 for item in tested
    )

    if any_reverse:
        reverse_verdict = (
            "SUPPORTED by precedence: the gap precedes at least one media "
            f"measure at alpha={alpha:.2f}"
        )
    elif tested:
        reverse_verdict = (
            "NOT supported by precedence: no media measure is preceded by the "
            "gap. This weakens the reverse narrative; it does not remove it, "
            "because a within-month response would not appear in a monthly "
            "Granger test."
        )
    else:
        reverse_verdict = "not assessed: no measure could be tested"

    verdicts = {
        "consumers_seek_negative_content": (
            reverse_verdict
            + " In either case this narrative is about EXPOSURE choice, and no "
            "exposure measure exists in this project, so it cannot be "
            "confirmed from the corpus."
        ),
        "publishers_respond_to_demand": (
            reverse_verdict
            + " This narrative is about SUPPLY, which the corpus does observe, "
            "so it is the testable one of the two reverse narratives."
        ),
        "common_shock": (
            (
                "CONSISTENT with the evidence: there is contemporaneous "
                "association with no precedence in either direction, which is "
                "the signature of a shared driver"
            )
            if strong_contemporaneous and not (any_forward or any_reverse)
            else (
                "not distinguished: the macro regressors absorb the shared "
                "component, and these tests cannot rule a common driver in or out"
            )
        ),
    }

    notes.append(
        "a monthly frequency cannot see a within-month response in either "
        "direction; both narratives stay live at a daily frequency"
    )
    return ReverseCausalityReport(
        gap_column=gap_column,
        evidence=evidence,
        narratives=dict(REVERSE_CAUSALITY_NARRATIVES),
        narrative_verdicts=verdicts,
        separable=False,
        separation_requirements=(
            ("Exposure data at the respondent or market level, to separate what "
            "consumers chose to read from what publishers chose to print."),
            ("Engagement data by story, to measure demand that publishers could "
            "respond to, distinct from supply."),
            ("A higher frequency (daily) series of both coverage and a sentiment "
            "proxy, because a monthly test cannot order events inside a month."),
            ("An instrument that moves coverage without moving sentiment, which "
            "is what the Rigobon-Sack attempt looks for and does not find."),
        ),
        max_lead=max_lead,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 11. The causal-design proposal, machine-readable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalDesignProposal:
    """One candidate identification design, stated so it can be audited.

    The brief's third option -- "observational analysis with a causal-design
    proposal" -- requires a detailed plan for stronger identification when
    exposure data are unavailable. A paragraph of prose cannot be checked
    against the project's own output, so the plan is an object: the estimand,
    the assumption each design needs, the data it needs, the threats it faces,
    what would falsify it, and its status in THIS project.
    """

    name: str
    design_class: str
    status: str
    target_estimand: str
    identifying_assumptions: tuple[str, ...]
    required_data: tuple[str, ...]
    threats: dict[str, str]
    falsification_tests: tuple[str, ...]
    feasible_now: bool
    blocking_constraint: str
    transportability: tuple[str, ...] = TRANSPORTABILITY_REQUIREMENTS
    exposure_gap: str = EXPOSURE_GAP_STATEMENT
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Flat mapping, for a table or a JSON dump."""
        return {
            "name": self.name,
            "design_class": self.design_class,
            "status": self.status,
            "target_estimand": self.target_estimand,
            "identifying_assumptions": " | ".join(self.identifying_assumptions),
            "required_data": " | ".join(self.required_data),
            "threats": " | ".join(f"{k}: {v}" for k, v in self.threats.items()),
            "falsification_tests": " | ".join(self.falsification_tests),
            "feasible_now": self.feasible_now,
            "blocking_constraint": self.blocking_constraint,
        }


def causal_design_proposals(
    identified_iv: bool = False,
) -> tuple[CausalDesignProposal, ...]:
    """The three designs of the brief, scored for this project.

    `identified_iv` says whether the heteroskedasticity IV returned an
    identified estimate. Pass ``rigobon_sack_iv(...).identified`` so that the
    observational row states the project's real position instead of a fixed
    one.

    The order is the brief's order: experiment, quasi-experiment, then the
    observational design with its plan. The third is this project's actual
    category, and its `status` says so.
    """
    experiment = CausalDesignProposal(
        name="Randomised framing experiment",
        design_class="experiment",
        status="NOT RUN: no human-subject approval and no participant sample",
        target_estimand=(
            "the average effect, on a sentiment-style response and on stated "
            "economic beliefs, of reading one truthful framing of the same "
            "economic release instead of another"
        ),
        identifying_assumptions=(
            ("random assignment of framing, so that the treatment is "
            "independent of every prior belief;"),
            ("both arms receive factually equivalent information, so that the "
            "contrast is framing and not content;"),
            "no interference between participants.",
        ),
        required_data=(
            "consenting participants with institutional approval;",
            ("a pre-registered battery of belief and sentiment items that maps "
            "to the Michigan questions;"),
            ("balanced factual summaries, with the equivalence of their factual "
            "content documented and reviewed."),
        ),
        threats={
            "external validity": (
                "a one-off exposure in a survey is not months of ambient "
                "coverage, so the measured effect is an upper bound on "
                "attention and a lower bound on accumulation"
            ),
            "demand effects": (
                "participants who see the manipulation answer the question "
                "they think is being asked"
            ),
            "transportability": (
                "an effect in the sample is not an effect on the national "
                "index; see `transportability`"
            ),
        },
        falsification_tests=(
            "no difference between arms in stated sentiment;",
            ("a difference in stated sentiment with no difference in the "
            "beliefs that the framing addressed, which would point to a survey "
            "artefact rather than belief change;"),
            "balance tests failing on pre-treatment covariates.",
        ),
        feasible_now=False,
        blocking_constraint=(
            "human-subject approval and a participant budget, neither of which "
            "this six-day project has"
        ),
    )
    quasi = CausalDesignProposal(
        name="Quasi-experiment on external variation in exposure",
        design_class="quasi-experiment",
        status="NOT RUN: no plausibly external exposure variation was found",
        target_estimand=(
            "the effect of a change in local economic-news supply on local "
            "consumer sentiment, for the markets that the change affected"
        ),
        identifying_assumptions=(
            ("the exposure change is external to local sentiment -- a station "
            "ownership change, a newspaper closure, a carriage dispute, a "
            "channel-position reshuffle;"),
            "parallel pre-trends between affected and comparison markets;",
            ("no spillover from affected markets into comparison markets, which "
            "national outlets and social platforms make hard to defend."),
        ),
        required_data=(
            ("a market-level sentiment series, which Michigan's public release "
            "does not provide at that geography;"),
            "a dated register of the exposure changes;",
            ("market-level economic controls, so that the comparison is not "
            "absorbing a local labour-market shock."),
        ),
        threats={
            "selection": (
                "outlets enter and leave markets for commercial reasons that "
                "correlate with local economic conditions"
            ),
            "spillovers": (
                "national coverage reaches every market, which compresses the "
                "contrast and biases the estimate toward zero"
            ),
            "pre-trends": (
                "a market already diverging before the change invalidates the "
                "comparison"
            ),
        },
        falsification_tests=(
            "a non-zero effect in the periods BEFORE the exposure change;",
            ("an effect of the same size in markets that the change did not "
            "touch;"),
            ("the effect appearing in a variable the design should not move, "
            "for example the local unemployment rate."),
        ),
        feasible_now=False,
        blocking_constraint=(
            "sub-national sentiment data. The Michigan public release is "
            "national, and the regional breakdowns are too coarse and too "
            "noisy for a market-level design."
        ),
    )
    observational = CausalDesignProposal(
        name="Observational association with a plan for identification",
        design_class="observational + design proposal",
        status=(
            "THIS PROJECT'S CATEGORY. Associations estimated with HAC errors, "
            "both Granger directions tested, multiplicity corrected, and the "
            "heteroskedasticity IV attempted. The IV "
            + ("returned an estimate." if identified_iv else "did NOT identify.")
        ),
        target_estimand=(
            "the effect on the Michigan index of a one-unit change in measured "
            "economic-news tone, holding the economic indicators fixed. This "
            "estimand is DEFINED here and NOT identified here: what the "
            "regressions deliver is a conditional association"
        ),
        identifying_assumptions=(
            ("for the regression coefficient to be the effect: no unobserved "
            "driver of both coverage and sentiment once the economic "
            "indicators, the household controls and the political controls are "
            "included -- which an election cycle alone makes implausible;"),
            ("for the heteroskedasticity IV: the regime split changes the "
            "variance of the media shock only, with the structural "
            "coefficients and the sentiment-shock variance constant across "
            "regimes (see `RIGOBON_SACK_ASSUMPTION`). Two covariance matrices "
            "cannot test this;"),
            ("for any of it to speak about people: that measured content supply "
            "stands in for exposure. THIS ASSUMPTION IS NOT MADE and the gap "
            "is left open (see `exposure_gap`)."),
        ),
        required_data=(
            ("exposure: panel metering, platform delivery logs, or survey recall "
            "items tied to the Michigan sample frame;"),
            ("a daily or weekly sentiment proxy, so that precedence can be "
            "ordered inside a month;"),
            ("an instrument that moves coverage without moving sentiment -- a "
            "newsroom strike, an ownership change, a platform ranking change;"),
            ("respondent-level Michigan microdata with political affiliation, to "
            "treat affiliation as the moderator and confounder the brief names."),
        ),
        threats={
            "reverse causality": (
                "pessimistic consumers may seek negative content and publishers "
                "may follow audience demand; both are tested through the "
                "reverse Granger direction and neither can be separated from "
                "the other without exposure data"
            ),
            "common shocks": (
                "an economic shock moves coverage and sentiment together. The "
                "economic regressors absorb part of this and cannot absorb the "
                "part they do not measure"
            ),
            "selection": (
                "the corpus is the coverage that GDELT indexed, which is a "
                "selection of what was published, which is a selection of what "
                "happened"
            ),
            "multiplicity": (
                "several media measures and two directions make one rejection "
                "at five percent the expected number under a complete null; "
                "this is why the family is corrected"
            ),
            "transportability": (
                "any effect measured on any sample must be carried to a "
                "weighted national survey statistic; see `transportability`"
            ),
        },
        falsification_tests=(
            "the media coefficients are insignificant in the richest model;",
            "no Granger precedence from coverage to the gap;",
            ("the raw rejections do not survive a Benjamini-Hochberg correction "
            "over the declared family;"),
            ("the heteroskedasticity IV finds no variance shift, so no "
            "instrument exists;"),
            ("the coefficient changes sign or significance across the "
            "specification grid;"),
            ("the media models do not beat the economic-only model "
            "out of sample at the rolling origins."),
        ),
        feasible_now=True,
        blocking_constraint=(
            "exposure data. Everything else on the list above can be bought or "
            "built; the exposure measure is what turns an association into an "
            "effect on people, and it does not exist in any public source."
        ),
        notes=(
            ("This row is the brief's required 'detailed plan for stronger "
            "identification if suitable exposure data are unavailable'."),
        ),
    )
    return (experiment, quasi, observational)


def causal_design_table(identified_iv: bool = False) -> pd.DataFrame:
    """The three proposals as one frame, for the report and the dashboard."""
    return pd.DataFrame(
        [proposal.to_dict() for proposal in causal_design_proposals(identified_iv)]
    ).set_index("name")
