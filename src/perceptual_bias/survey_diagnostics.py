"""Hypothesis C: is part of the Sentiment Gap an artefact of the instrument?

Hypotheses A and B are substantive explanations of the gap. Hypothesis C says
the thing being explained is partly a measurement artefact: the University of
Michigan Surveys of Consumers moved from telephone to web collection over
2024-2026, mode changes are well known to shift response distributions, and
publication lags plus a post-2020 regime shift could do the rest. If C holds
strongly, the magnitude A and B compete to explain is itself overstated, so an
honest report establishes the size of the real target before attributing it.

Three pieces of evidence, each with its own module-level entry point:

1. **Structural break tests.** `chow_test` tests coefficient stability at a
   breakpoint the analyst *specifies* (the methodology transition).
   `sup_wald_test` searches a trimmed interior window for an *unknown* break
   and returns the estimated date. These answer different questions and have
   different null distributions — see `SUP_WALD_IDENTIFICATION`.

2. **Cross-survey comparison.** `cross_survey_gap` refits the same baseline
   specification with an independently collected confidence series as the
   target and compares the two gaps after standardising, because the series are
   on unrelated scales.

3. **A decision.** `assess_hypothesis_c` applies BRAINSTORMING.md's
   pre-registered falsification condition explicitly: if an independently
   collected series shows a gap of comparable magnitude over the same months,
   C is rejected as a *primary* explanation (it may survive as a contributor).

**Power is the binding constraint and is never papered over.** The holdout is
19-22 monthly observations. A sup-Wald search over 19 points with a trimmed
window leaves segments of three or four observations; fitting anything there
produces output that looks like a result and is not. Every result object
carries `feasible`, `underpowered` and `notes`, and the statistic is `None`
rather than a confident number when the test cannot be supported at that n.

**On critical values.** A sup-Wald statistic maximised over candidate dates is
*not* chi-squared or F distributed, because the breakpoint was estimated rather
than known: comparing it to a fixed-breakpoint critical value overstates
significance badly. The p-value reported here comes from a moving-block
bootstrap under the no-break null, which also preserves the serial correlation
that monthly macro residuals always have. No asymptotic table is embedded.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from perceptual_bias.baseline_model import (
    BASELINE_FEATURES,
    TARGET,
    ModelError,
    fit_baseline,
    sentiment_gap,
)
from perceptual_bias.data_loader import FredSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Specified dates and thresholds
# ---------------------------------------------------------------------------

# The Michigan survey's transition to web collection. Interviewing moved to a
# web panel over mid-2024, with telephone interviewing discontinued from the
# July 2024 survey onward. This is the *specified* breakpoint for the Chow
# test: a date chosen from the instrument's documented history, not estimated
# from the data. Callers may override it; the test is only as good as the date.
MICHIGAN_WEB_TRANSITION = pd.Timestamp("2024-07-31")

# How close an *estimated* break has to fall to the specified transition before
# it counts as corroborating the mode-change story. Six months is generous on
# purpose: the transition was phased, not a single survey.
TRANSITION_TOLERANCE_MONTHS = 6

# Andrews-style trimming: breakpoints in the first/last 15% of the sample are
# not candidates, because segments there are too short to estimate.
SUP_WALD_TRIM = 0.15

# Soft guards. Below these the test still runs where arithmetically possible,
# but `underpowered` is set and the note says so in words.
# A segment shorter than a full seasonal cycle cannot separate a level from
# seasonality, which is the same reasoning behind the project's HAC(12) choice.
MIN_SEGMENT_OBS = 12
MIN_SUP_WALD_OBS = 30
MIN_COMPARISON_MONTHS = 6

BOOTSTRAP_DRAWS = 999
BOOTSTRAP_SEED = 20260101

# |comparison gap| / |Michigan gap|, both in training-window standard
# deviations of their own series. At or above COMPARABLE_GAP_RATIO the
# pre-registered falsification condition fires: the gap is a property of the
# economy-perception relationship, not of the Michigan instrument. At or below
# SMALL_GAP_RATIO the independent series genuinely does not show the gap.
COMPARABLE_GAP_RATIO = 0.50
SMALL_GAP_RATIO = 0.25

# Two series that *look* like independent confidence measures and are not.
# Correlation this high with Michigan in BOTH levels and first differences
# means the candidate is a rescaling of the Michigan instrument itself, so a
# comparison against it would be circular. The first-difference threshold is
# the looser of the two on purpose: USACSCICP02STSAM is literally Michigan
# rescaled, yet publication rounding drags its first-difference correlation
# over the full 1978-2026 history down to 0.976, so a 0.999 screen on both
# legs lets the impostor through. Two genuinely independent household surveys
# do not come close to 0.90 in differences — Fannie Mae's index against
# Michigan manages 0.34. Verified empirically, not assumed.
IDENTITY_LEVEL_CORRELATION = 0.99
IDENTITY_DIFF_CORRELATION = 0.90
MIN_VERIFICATION_OVERLAP = 24


@dataclass(frozen=True)
class ComparisonCandidate:
    """A candidate cross-survey series, with what is known about it."""

    series_id: str
    label: str
    independent_of_michigan: bool
    note: str


# Ordered best-first. The OECD indicator cited in BRAINSTORMING.md is kept here
# deliberately, with what verification actually found, so the report does not
# repeat the plan as though it were a fact.
COMPARISON_CANDIDATES: tuple[ComparisonCandidate, ...] = (
    ComparisonCandidate(
        series_id="FMNHSHPSIUS",
        label="Fannie Mae National Housing Survey: Home Purchase Sentiment Index",
        independent_of_michigan=True,
        note=(
            "Independently fielded consumer survey (~1,000 households/month). "
            "DISCONTINUED after 2025-09, so it overlaps only the first nine "
            "holdout months, and it is housing-weighted rather than a general "
            "confidence index."
        ),
    ),
    ComparisonCandidate(
        series_id="BSCICP02USM460S",
        label="OECD Business Tendency Surveys: manufacturing confidence, United States",
        independent_of_michigan=True,
        note=(
            "Covers the whole holdout and is collected from firms, not "
            "households, so it is independent of any consumer-survey mode "
            "change — but it measures business rather than consumer "
            "confidence, which is a different construct."
        ),
    ),
    ComparisonCandidate(
        series_id="CSCICP03USM665S",
        label="OECD composite consumer confidence, United States (amplitude adjusted)",
        independent_of_michigan=False,
        note=(
            "The series ID in BRAINSTORMING.md. It EXISTS on FRED but its last "
            "observation is 2024-01: it is stale/discontinued and covers none "
            "of the 2025-26 holdout."
        ),
    ),
    ComparisonCandidate(
        series_id="USACSCICP02STSAM",
        label="OECD consumer opinion surveys: composite consumer confidence, United States",
        independent_of_michigan=False,
        note=(
            "The live successor to CSCICP03USM665S, and a trap: it correlates "
            "1.000 with UMCSENT in both levels and first differences, because "
            "OECD's US consumer input IS the Michigan survey. Comparing the "
            "Michigan gap against it would be circular."
        ),
    ),
)

DEFAULT_COMPARISON_SERIES = COMPARISON_CANDIDATES[0].series_id

CHOW_IDENTIFICATION = (
    "The Chow test asks whether the relationship's coefficients differ either "
    "side of a date the analyst supplied. It identifies a change at that date; "
    "it cannot identify its cause. Any event clustered near the survey's mode "
    "change — a policy shock, an election, a data revision — produces the same "
    "rejection, so a significant Chow statistic is consistent with Hypothesis C "
    "but is not evidence for it over any other same-dated explanation."
)

SUP_WALD_IDENTIFICATION = (
    "The sup-Wald (Quandt-Andrews) statistic maximises the break statistic over "
    "candidate dates, so the break date is ESTIMATED and the statistic is not "
    "chi-squared or F distributed. The p-value here is a moving-block bootstrap "
    "under the no-break null; no fixed-breakpoint critical value is used or "
    "valid. What is identified is the single most likely break date under a "
    "one-break alternative, with no attribution of cause and no claim that there "
    "is exactly one break."
)

STANDARDISATION_NOTE = (
    "Gaps are divided by the standard deviation of their own target series over "
    "the TRAINING window only (never the holdout), so the two surveys' "
    "unrelated scales are removed without look-ahead. Gaps relative to each "
    "model's own in-sample RMSE are reported alongside."
)

VERDICTS = ("supported", "partially_supported", "rejected", "indeterminate")


class SurveyDiagnosticsError(Exception):
    """Raised when a diagnostic cannot be computed as specified."""


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChowTestResult:
    """A fixed-breakpoint stability test, with its own power stated."""

    breakpoint: pd.Timestamp
    n: int
    k: int
    n_pre: int
    n_post: int
    f_statistic: float | None
    p_value: float | None
    wald_statistic: float | None
    rss_restricted: float | None
    rss_unrestricted: float | None
    level_shift: float | None
    feasible: bool
    underpowered: bool
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    identification: str = CHOW_IDENTIFICATION

    def summary(self) -> str:
        if not self.feasible:
            return (
                f"Chow at {self.breakpoint:%Y-%m}: NOT COMPUTED "
                f"(n={self.n}, pre={self.n_pre}, post={self.n_post}, k={self.k})"
            )
        flag = " [UNDERPOWERED]" if self.underpowered else ""
        return (
            f"Chow at {self.breakpoint:%Y-%m}: F({self.k}, {self.n - 2 * self.k})="
            f"{self.f_statistic:.3f}, p={self.p_value:.4f}, "
            f"level shift={self.level_shift:+.3f}{flag}"
        )


@dataclass(frozen=True)
class SupWaldResult:
    """An unknown-breakpoint test: estimated date plus a bootstrap p-value."""

    break_date: pd.Timestamp | None
    f_statistic: float | None
    wald_statistic: float | None
    p_value: float | None
    n: int
    k: int
    trim: float
    n_candidates: int
    first_candidate: pd.Timestamp | None
    last_candidate: pd.Timestamp | None
    n_bootstrap: int
    block_length: int | None
    seed: int
    statistic_path: pd.Series
    feasible: bool
    underpowered: bool
    dropped_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    identification: str = SUP_WALD_IDENTIFICATION

    def summary(self) -> str:
        if not self.feasible:
            return (
                f"sup-Wald: NOT COMPUTED (n={self.n}, k={self.k}, "
                f"{self.n_candidates} admissible breakpoints)"
            )
        flag = " [UNDERPOWERED]" if self.underpowered else ""
        return (
            f"sup-Wald: break at {self.break_date:%Y-%m}, supF={self.f_statistic:.3f}, "
            f"bootstrap p={self.p_value:.4f} ({self.n_bootstrap} draws, "
            f"block={self.block_length}){flag}"
        )

    def near(self, date: pd.Timestamp, tolerance_months: int) -> bool | None:
        """Is the estimated break within `tolerance_months` of `date`?

        Returns None when nothing was estimated — NaN is not False, and an
        infeasible test must not read as "the break is elsewhere".
        """
        if not self.feasible or self.break_date is None:
            return None
        months = abs(
            (self.break_date.year - date.year) * 12 + (self.break_date.month - date.month)
        )
        return months <= tolerance_months


@dataclass(frozen=True)
class CrossSurveyGap:
    """The same gap, measured against an independently collected survey."""

    series_id: str
    label: str
    n_months: int
    months: tuple[pd.Timestamp, ...]
    michigan_mean_gap: float | None
    comparison_mean_gap: float | None
    michigan_mean_gap_z: float | None
    comparison_mean_gap_z: float | None
    michigan_train_sd: float | None
    comparison_train_sd: float | None
    michigan_rmse: float | None
    comparison_rmse: float | None
    gap_ratio: float | None
    comparable: bool | None
    standardisation: str = STANDARDISATION_NOTE
    excluded_months: tuple[str, ...] = field(default_factory=tuple)
    feasible: bool = True
    underpowered: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if not self.feasible:
            return f"cross-survey vs {self.series_id}: NOT COMPUTED ({self.n_months} common months)"
        flag = " [UNDERPOWERED]" if self.underpowered else ""
        return (
            f"cross-survey vs {self.series_id}: {self.n_months} common months, "
            f"Michigan z-gap={self.michigan_mean_gap_z:+.2f}, "
            f"comparison z-gap={self.comparison_mean_gap_z:+.2f}, "
            f"ratio={self.gap_ratio:.2f}{flag}"
        )


@dataclass(frozen=True)
class HypothesisCVerdict:
    """Where the evidence leaves Hypothesis C, and why."""

    verdict: str
    break_at_transition: bool | None
    break_date: pd.Timestamp | None
    chow_p_value: float | None
    sup_wald_p_value: float | None
    gap_ratio: float | None
    falsification_fired: bool
    level_to_variation: float | None
    underpowered: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        flag = " [UNDERPOWERED]" if self.underpowered else ""
        head = f"Hypothesis C: {self.verdict.upper()}{flag}"
        return head + "".join(f"\n  - {reason}" for reason in self.reasons)


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------


def _rss(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual sum of squares of ``y`` on ``x``, vectorised over columns of y.

    ``y`` may be (n,) or (n, B). Using the QR factor rather than ``lstsq`` lets
    one factorisation serve all B bootstrap replications, which is what makes
    the bootstrap affordable at 999 draws times ~150 candidate breakpoints.
    """
    y2d = y.reshape(-1, 1) if y.ndim == 1 else y
    q, _ = np.linalg.qr(x)
    fitted_energy = (q.T @ y2d) ** 2
    out = (y2d**2).sum(axis=0) - fitted_energy.sum(axis=0)
    # Floating-point cancellation can push a near-perfect fit slightly negative.
    return np.maximum(out, 0.0)


def _align(y: pd.Series, x: pd.DataFrame | None) -> tuple[pd.Series, pd.DataFrame, tuple[pd.Timestamp, ...]]:
    """Align target and design, returning the rows dropped for missingness.

    With ``x=None`` the design is a constant only, so the test becomes a test
    of a *level shift* in ``y`` — which is exactly what one wants when ``y`` is
    a vector of baseline residuals.
    """
    if not isinstance(y, pd.Series):
        raise SurveyDiagnosticsError("y must be a pandas Series indexed by date")
    if y.empty:
        raise SurveyDiagnosticsError("y is empty; nothing to test")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise SurveyDiagnosticsError("y must have a DatetimeIndex")
    if not y.index.is_monotonic_increasing:
        raise SurveyDiagnosticsError("y must be sorted by date before testing for a break")

    if x is None:
        design = pd.DataFrame({"const": 1.0}, index=y.index)
    else:
        if not isinstance(x, pd.DataFrame):
            raise SurveyDiagnosticsError("x must be a pandas DataFrame or None")
        design = x.reindex(y.index)
        if "const" not in design.columns:
            design = design.assign(const=1.0)

    frame = pd.concat([y.rename("__y"), design], axis=1)
    usable = frame.dropna()
    dropped = tuple(frame.index.difference(usable.index))
    if dropped:
        logger.info(
            "break test dropped %d incomplete rows: %s",
            len(dropped),
            ", ".join(pd.Timestamp(stamp).strftime("%Y-%m") for stamp in dropped[:6]),
        )
    if usable.empty:
        raise SurveyDiagnosticsError("no complete rows available for a break test")
    return usable["__y"], usable.drop(columns="__y"), dropped


def _dropped_note(dropped: tuple[pd.Timestamp, ...]) -> list[str]:
    if not dropped:
        return []
    shown = ", ".join(pd.Timestamp(stamp).strftime("%Y-%m") for stamp in dropped[:6])
    suffix = "" if len(dropped) <= 6 else f" (+{len(dropped) - 6} more)"
    return [f"dropped {len(dropped)} rows with missing values: {shown}{suffix}"]


# ---------------------------------------------------------------------------
# 1. Structural break tests
# ---------------------------------------------------------------------------


def chow_test(
    y: pd.Series,
    x: pd.DataFrame | None = None,
    *,
    breakpoint: pd.Timestamp | str = MICHIGAN_WEB_TRANSITION,
    min_segment_obs: int = MIN_SEGMENT_OBS,
) -> ChowTestResult:
    """Chow test for a break at a **specified** date.

    The breakpoint belongs to the *pre* segment: observations at or before it
    are "before the methodology change". The statistic is the standard

        F = ((RSS_r - RSS_u) / k) / (RSS_u / (n - 2k))

    which is valid F(k, n-2k) only because the date was fixed in advance from
    the instrument's documented history rather than chosen by looking at the
    data. `sup_wald_test` is the right tool when the date is not known, and the
    two must not be confused — that substitution is the usual way break tests
    get reported as far more significant than they are.

    `level_shift` is the change in the intercept across the break, in the units
    of ``y``: for baseline residuals it is the estimated mode-change level
    effect, which is the quantity Hypothesis C is actually about.
    """
    y_aligned, design, dropped = _align(y, x)
    stamp = pd.Timestamp(breakpoint)
    notes = _dropped_note(dropped)

    values = y_aligned.to_numpy(dtype=float)
    matrix = design.to_numpy(dtype=float)
    n, k = matrix.shape
    pre_mask = y_aligned.index <= stamp
    n_pre, n_post = int(pre_mask.sum()), int((~pre_mask).sum())

    if stamp < y_aligned.index.min() or stamp >= y_aligned.index.max():
        raise SurveyDiagnosticsError(
            f"breakpoint {stamp:%Y-%m} lies outside the sample "
            f"({y_aligned.index.min():%Y-%m}..{y_aligned.index.max():%Y-%m})"
        )

    if n_pre <= k or n_post <= k or n - 2 * k <= 0:
        notes.append(
            f"NOT COMPUTED: a Chow test at {stamp:%Y-%m} needs more than k={k} "
            f"observations each side and n>2k; this sample gives pre={n_pre}, "
            f"post={n_post}, n={n}. No statistic is reported rather than a "
            "number that cannot be supported."
        )
        return ChowTestResult(
            breakpoint=stamp,
            n=n,
            k=k,
            n_pre=n_pre,
            n_post=n_post,
            f_statistic=None,
            p_value=None,
            wald_statistic=None,
            rss_restricted=None,
            rss_unrestricted=None,
            level_shift=None,
            feasible=False,
            underpowered=True,
            dropped_dates=dropped,
            notes=tuple(notes),
        )

    rss_r = float(_rss(values, matrix)[0])
    rss_u = float(
        _rss(values[pre_mask], matrix[pre_mask])[0] + _rss(values[~pre_mask], matrix[~pre_mask])[0]
    )
    if rss_u <= 0:
        raise SurveyDiagnosticsError(
            "unrestricted residual sum of squares is zero; the segment designs "
            "fit perfectly, so no stability test is defined"
        )

    f_statistic = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
    p_value = float(stats.f.sf(f_statistic, k, n - 2 * k))
    level_shift = float(values[~pre_mask].mean() - values[pre_mask].mean())

    underpowered = min(n_pre, n_post) < min_segment_obs
    if underpowered:
        notes.append(
            f"UNDERPOWERED: the shorter segment has {min(n_pre, n_post)} "
            f"observations (< {min_segment_obs}). The F statistic is computable "
            "but its sampling distribution is only approximate at this n, and "
            "the level shift is estimated from very few months."
        )
    notes.append(
        f"breakpoint {stamp:%Y-%m} was SPECIFIED, not estimated "
        "(Michigan web-collection transition); the F distribution is only valid "
        "under that fixed-date reading."
    )

    return ChowTestResult(
        breakpoint=stamp,
        n=n,
        k=k,
        n_pre=n_pre,
        n_post=n_post,
        f_statistic=float(f_statistic),
        p_value=p_value,
        wald_statistic=float(k * f_statistic),
        rss_restricted=rss_r,
        rss_unrestricted=rss_u,
        level_shift=level_shift,
        feasible=True,
        underpowered=underpowered,
        dropped_dates=dropped,
        notes=tuple(notes),
    )


def _moving_block_resample(
    residuals: np.ndarray, rng: np.random.Generator, draws: int, block: int
) -> np.ndarray:
    """Resample residuals in overlapping blocks, preserving serial correlation.

    An i.i.d. bootstrap would destroy the autocorrelation that monthly macro
    residuals always carry, and an autocorrelated series produces large break
    statistics by accident — so an i.i.d. null would make the test
    anti-conservative in exactly the direction that favours Hypothesis C.
    """
    n = residuals.size
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]
    return residuals[index].T  # (n, draws)


def sup_wald_test(
    y: pd.Series,
    x: pd.DataFrame | None = None,
    *,
    trim: float = SUP_WALD_TRIM,
    n_bootstrap: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    block_length: int | None = None,
    min_obs: int = MIN_SUP_WALD_OBS,
) -> SupWaldResult:
    """Supremum-Wald (Quandt-Andrews) test for a break at an **unknown** date.

    The break statistic is computed at every candidate date in the trimmed
    interior ``[trim, 1-trim]`` of the sample and maximised. Because the date is
    chosen by that maximisation, the resulting statistic is stochastically much
    larger than an F(k, n-2k) draw: comparing it to an F or chi-squared critical
    value is a real error, not a conservative approximation.

    The p-value is instead obtained by a **moving-block bootstrap under the
    null of no break**: residuals from the restricted (single-regime) fit are
    resampled in blocks, the restricted fitted values are added back, and the
    whole sup-F search is rerun on each artificial sample. The p-value is the
    share of bootstrap suprema at least as large as the observed one, with the
    standard ``(1 + count) / (1 + draws)`` correction so it can never be zero.

    `statistic_path` is the whole F sequence by candidate date, which is what a
    reader should look at before believing a single argmax: a flat path with a
    marginal peak is a different object from a sharp spike.
    """
    if not 0.0 < trim < 0.5:
        raise SurveyDiagnosticsError(f"trim must lie in (0, 0.5); got {trim}")
    if n_bootstrap < 1:
        raise SurveyDiagnosticsError(f"n_bootstrap must be positive; got {n_bootstrap}")

    y_aligned, design, dropped = _align(y, x)
    notes = _dropped_note(dropped)

    values = y_aligned.to_numpy(dtype=float)
    matrix = design.to_numpy(dtype=float)
    n, k = matrix.shape
    if np.linalg.matrix_rank(matrix) < k:
        raise SurveyDiagnosticsError(
            "design matrix is rank deficient; a break test is not identified"
        )

    lower = max(math.ceil(trim * n), k + 1)
    upper = min(math.floor((1 - trim) * n), n - k - 1)
    candidate_positions = list(range(lower, upper + 1))
    empty_path = pd.Series(dtype=float, name="sup_wald_f")

    if n - 2 * k <= 0 or not candidate_positions:
        notes.append(
            f"NOT COMPUTED: with n={n} observations, k={k} parameters and "
            f"trim={trim:.2f} there are no admissible breakpoints leaving more "
            "than k observations on each side. A sup-Wald search cannot be "
            "supported at this sample size; this is the expected outcome on the "
            "19-22 month holdout and must be reported, not worked around."
        )
        return SupWaldResult(
            break_date=None,
            f_statistic=None,
            wald_statistic=None,
            p_value=None,
            n=n,
            k=k,
            trim=trim,
            n_candidates=0,
            first_candidate=None,
            last_candidate=None,
            n_bootstrap=0,
            block_length=None,
            seed=seed,
            statistic_path=empty_path,
            feasible=False,
            underpowered=True,
            dropped_dates=dropped,
            notes=tuple(notes),
        )

    def sup_f(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """sup-F and its argmax position for each column of `targets`."""
        columns = targets.reshape(-1, 1) if targets.ndim == 1 else targets
        rss_r = _rss(columns, matrix)
        best = np.full(columns.shape[1], -np.inf)
        arg = np.zeros(columns.shape[1], dtype=int)
        for position in candidate_positions:
            rss_u = _rss(columns[:position], matrix[:position]) + _rss(
                columns[position:], matrix[position:]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
            statistic = np.where(np.isfinite(statistic), statistic, -np.inf)
            improved = statistic > best
            arg = np.where(improved, position, arg)
            best = np.where(improved, statistic, best)
        return best, arg

    observed, observed_arg = sup_f(values)
    observed_statistic = float(observed[0])
    if not np.isfinite(observed_statistic):
        raise SurveyDiagnosticsError(
            "sup-F statistic is not finite at any candidate break; the segment "
            "fits are degenerate"
        )
    # The break is dated by the last observation of the pre-break segment, so
    # the shift is read as occurring between that month and the next.
    break_position = int(observed_arg[0])
    break_date = y_aligned.index[break_position - 1]

    path = pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([y_aligned.index[position - 1] for position in candidate_positions]),
        name="sup_wald_f",
    )
    rss_restricted = float(_rss(values, matrix)[0])
    for slot, position in enumerate(candidate_positions):
        rss_u = float(
            _rss(values[:position], matrix[:position])[0]
            + _rss(values[position:], matrix[position:])[0]
        )
        path.iloc[slot] = (
            ((rss_restricted - rss_u) / k) / (rss_u / (n - 2 * k)) if rss_u > 0 else np.nan
        )

    # Bootstrap under the null: restricted fit plus block-resampled residuals.
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    fitted = matrix @ coefficients
    residuals = values - fitted
    block = block_length if block_length is not None else max(1, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    resampled = _moving_block_resample(residuals, rng, n_bootstrap, block)
    bootstrap_statistics, _ = sup_f(fitted[:, None] + resampled)
    exceedances = int(np.sum(bootstrap_statistics >= observed_statistic))
    p_value = (1.0 + exceedances) / (1.0 + n_bootstrap)

    underpowered = n < min_obs or min(break_position, n - break_position) < MIN_SEGMENT_OBS
    if underpowered:
        notes.append(
            f"UNDERPOWERED: n={n} observations (guard {min_obs}) and the shorter "
            f"segment at the estimated break has {min(break_position, n - break_position)}. "
            "The bootstrap p-value is still valid by construction, but the "
            "estimated break DATE has very wide uncertainty at this n — treat "
            "it as a region, not a month."
        )
    notes.append(
        f"p-value from a moving-block bootstrap (block={block}, {n_bootstrap} draws) "
        "under the no-break null; no F or chi-squared critical value is used, "
        "because the breakpoint was estimated."
    )

    return SupWaldResult(
        break_date=break_date,
        f_statistic=observed_statistic,
        wald_statistic=float(k * observed_statistic),
        p_value=float(p_value),
        n=n,
        k=k,
        trim=trim,
        n_candidates=len(candidate_positions),
        first_candidate=y_aligned.index[candidate_positions[0] - 1],
        last_candidate=y_aligned.index[candidate_positions[-1] - 1],
        n_bootstrap=n_bootstrap,
        block_length=block,
        seed=seed,
        statistic_path=path,
        feasible=True,
        underpowered=underpowered,
        dropped_dates=dropped,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 2. Cross-survey comparison
# ---------------------------------------------------------------------------


def load_comparison_series(
    series_id: str = DEFAULT_COMPARISON_SERIES,
    fred: FredSource | None = None,
) -> tuple[pd.Series, tuple[str, ...]]:
    """Fetch a comparison confidence series from FRED.

    Reuses `FredSource` rather than writing another client, and loads ``.env``
    the way `scripts/probe_twitterapi.py` does so a configured key is picked up
    without the caller having to export it.

    **Network.** This is the only function in the module that touches the
    network; tests stub `FredSource` instead of calling it.
    """
    notes: list[str] = []
    if fred is None:
        if not os.environ.get("FRED_API_KEY"):
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:  # pragma: no cover - dotenv is a declared dep
                notes.append("python-dotenv unavailable; relying on the ambient environment")
        fred = FredSource()

    candidate = next((c for c in COMPARISON_CANDIDATES if c.series_id == series_id), None)
    if candidate is not None:
        notes.append(f"{series_id}: {candidate.note}")
        if not candidate.independent_of_michigan:
            notes.append(
                f"WARNING: {series_id} is recorded in COMPARISON_CANDIDATES as NOT "
                "independent of the Michigan instrument; a gap comparison against "
                "it is circular."
            )

    if not fred.configured:
        raise SurveyDiagnosticsError(
            "FRED is not configured (no usable FRED_API_KEY), so no independent "
            "survey can be fetched. Hypothesis C's cross-survey leg is "
            "untestable without it and must be reported as untested."
        )

    series = fred.series(series_id)
    if series.empty:
        raise SurveyDiagnosticsError(
            f"FRED series {series_id} returned no data. Verify the ID before "
            "relying on it — BRAINSTORMING.md's CSCICP03USM665S exists but ends "
            "in 2024-01, which is a different failure from not existing at all."
        )
    notes.append(
        f"{series_id}: {len(series)} observations, "
        f"{series.index.min():%Y-%m}..{series.index.max():%Y-%m}"
    )
    return series.rename(series_id), tuple(notes)


def verify_comparison_series(
    comparison: pd.Series,
    michigan: pd.Series,
    *,
    required_through: pd.Timestamp | str | None = None,
    min_overlap: int = MIN_VERIFICATION_OVERLAP,
) -> tuple[bool, tuple[str, ...]]:
    """Check a candidate really is an independent, usable comparison series.

    Three things go wrong in practice and all three are checked rather than
    assumed:

    * **The series does not reach the holdout.** `CSCICP03USM665S` exists and
      ends 2024-01, which would silently produce a comparison over zero months.
    * **The series is the Michigan survey wearing a different scale.** OECD's
      US composite consumer confidence correlates 0.999 with `UMCSENT` in levels
      (1.000 post-2010) because the Michigan survey is its US input.
      Comparing the Michigan gap against it would be circular, and the
      correlation screen catches it without needing to know the provenance.
    * **Too little overlap to compare at all.**
    """
    notes: list[str] = []
    if comparison.empty:
        return False, ("comparison series is empty",)

    joined = pd.concat(
        [comparison.rename("comparison"), michigan.rename("michigan")], axis=1, join="inner"
    ).dropna()
    if len(joined) < min_overlap:
        notes.append(
            f"only {len(joined)} overlapping months with the Michigan series "
            f"(need {min_overlap}) — independence cannot be assessed"
        )
        return False, tuple(notes)

    level_correlation = float(joined["comparison"].corr(joined["michigan"]))
    diff_correlation = float(
        joined["comparison"].diff().corr(joined["michigan"].diff())
    )
    notes.append(
        f"correlation with Michigan: {level_correlation:.4f} in levels, "
        f"{diff_correlation:.4f} in first differences ({len(joined)} months)"
    )
    if (
        abs(level_correlation) > IDENTITY_LEVEL_CORRELATION
        and abs(diff_correlation) > IDENTITY_DIFF_CORRELATION
    ):
        notes.append(
            "REJECTED as a comparison series: it is a rescaling of the Michigan "
            "instrument, not an independent measurement, so any gap comparison "
            "against it is circular."
        )
        return False, tuple(notes)

    if required_through is not None:
        needed = pd.Timestamp(required_through)
        if comparison.dropna().index.max() < needed:
            notes.append(
                f"coverage ends {comparison.dropna().index.max():%Y-%m}, before the "
                f"required {needed:%Y-%m}: the comparison would cover only part of "
                "the holdout (or none of it)"
            )
            return False, tuple(notes)

    return True, tuple(notes)


def cross_survey_gap(
    panel: pd.DataFrame,
    comparison: pd.Series,
    *,
    series_id: str | None = None,
    label: str | None = None,
    features: tuple[str, ...] = BASELINE_FEATURES,
    michigan_target: str = TARGET,
    train_start: str = "2010-01-01",
    train_end: str = "2024-12-31",
    holdout_start: str = "2025-01-01",
    min_months: int = MIN_COMPARISON_MONTHS,
    comparable_ratio: float = COMPARABLE_GAP_RATIO,
) -> CrossSurveyGap:
    """Measure the same gap against an independently collected survey.

    The comparison is deliberately *like for like*: the same regressors, the
    same training window, the same out-of-sample window, with only the target
    swapped. Anything else — comparing the Michigan gap against the raw level
    of another index, say — would confound the instrument with the model.

    **Scales.** The two surveys are on unrelated scales (Michigan is a
    1966=100 index; a diffusion balance runs roughly -100..+100), so raw gaps
    are not comparable. Each gap is divided by the standard deviation of its own
    target over the **training window only**, giving a gap in that survey's own
    historical standard deviations. Training-window moments keep the comparison
    free of look-ahead, and each model's in-sample RMSE is reported alongside so
    a reader can see the gap in units of the model's own ordinary error too.

    **Months.** Only months where *both* gaps exist are compared, and the
    Michigan figure is recomputed on exactly those months rather than quoted
    from the full holdout — the comparison series may be discontinued mid-window
    (Fannie Mae's index ends 2025-09), and comparing nine months of one survey
    against nineteen of the other would flatter whichever had the kinder
    sub-period. Excluded months are named.
    """
    if michigan_target not in panel.columns:
        raise SurveyDiagnosticsError(f"panel has no '{michigan_target}' column")
    if comparison.empty:
        raise SurveyDiagnosticsError("comparison series is empty")
    if not isinstance(comparison.index, pd.DatetimeIndex):
        raise SurveyDiagnosticsError("comparison series must have a DatetimeIndex")

    series_id = series_id or (comparison.name if comparison.name else "comparison")
    known = next((c for c in COMPARISON_CANDIDATES if c.series_id == series_id), None)
    label = label or (known.label if known else str(series_id))
    notes: list[str] = []
    if known is not None and not known.independent_of_michigan:
        notes.append(
            f"WARNING: {series_id} is not independent of the Michigan instrument; "
            "this comparison is circular and must not be used to decide Hypothesis C."
        )

    column = "__comparison"
    work = panel.copy()
    work[column] = comparison.reindex(work.index)

    train = work.loc[train_start:train_end]
    holdout = work.loc[holdout_start:]
    if holdout.empty:
        raise SurveyDiagnosticsError(f"no holdout rows from {holdout_start}")

    def gaps_for(target: str) -> tuple[pd.Series, object]:
        fit = fit_baseline(train, features=features, target=target)
        frame, _ = sentiment_gap(fit, holdout)
        return frame["gap"], fit

    try:
        michigan_gaps, michigan_fit = gaps_for(michigan_target)
        comparison_gaps, comparison_fit = gaps_for(column)
    except ModelError as exc:
        raise SurveyDiagnosticsError(f"could not refit the baseline per survey: {exc}") from exc

    both = pd.concat(
        [michigan_gaps.rename("michigan"), comparison_gaps.rename("comparison")], axis=1
    )
    usable = both.dropna()
    excluded = []
    for stamp in both.index.difference(usable.index):
        missing = [name for name in both.columns if pd.isna(both.loc[stamp, name])]
        excluded.append(f"{pd.Timestamp(stamp):%Y-%m}: no gap for {', '.join(missing)}")
    if comparison.dropna().index.max() < holdout.index.max():
        notes.append(
            f"{series_id} ends {comparison.dropna().index.max():%Y-%m} while the "
            f"holdout runs to {holdout.index.max():%Y-%m}; the comparison covers "
            "only the overlapping months and both sides are measured on those."
        )

    michigan_sd = float(train[michigan_target].dropna().std(ddof=1))
    comparison_sd = float(train[column].dropna().std(ddof=1))

    if usable.empty or not np.isfinite(michigan_sd) or not np.isfinite(comparison_sd):
        notes.append(
            "NOT COMPUTED: no holdout month has a gap for both surveys (or a "
            "training standard deviation is undefined), so the cross-survey leg "
            "of Hypothesis C is untested, not passed."
        )
        return CrossSurveyGap(
            series_id=str(series_id),
            label=label,
            n_months=0,
            months=(),
            michigan_mean_gap=None,
            comparison_mean_gap=None,
            michigan_mean_gap_z=None,
            comparison_mean_gap_z=None,
            michigan_train_sd=michigan_sd if np.isfinite(michigan_sd) else None,
            comparison_train_sd=comparison_sd if np.isfinite(comparison_sd) else None,
            michigan_rmse=None,
            comparison_rmse=None,
            gap_ratio=None,
            comparable=None,
            excluded_months=tuple(excluded),
            feasible=False,
            underpowered=True,
            notes=tuple(notes),
        )
    if comparison_sd == 0 or michigan_sd == 0:
        raise SurveyDiagnosticsError(
            "a target series has zero variance over the training window; it "
            "cannot be standardised or modelled"
        )

    michigan_mean = float(usable["michigan"].mean())
    comparison_mean = float(usable["comparison"].mean())
    michigan_z = michigan_mean / michigan_sd
    comparison_z = comparison_mean / comparison_sd
    ratio = abs(comparison_z) / abs(michigan_z) if michigan_z != 0 else float("inf")

    underpowered = len(usable) < min_months
    if underpowered:
        notes.append(
            f"UNDERPOWERED: only {len(usable)} common months (guard {min_months}). "
            "The ratio is a point description of those months, not an estimate "
            "with usable precision."
        )
    notes.append(
        f"gap ratio {ratio:.2f} against a comparability threshold of {comparable_ratio:.2f}"
    )

    return CrossSurveyGap(
        series_id=str(series_id),
        label=label,
        n_months=len(usable),
        months=tuple(usable.index),
        michigan_mean_gap=michigan_mean,
        comparison_mean_gap=comparison_mean,
        michigan_mean_gap_z=float(michigan_z),
        comparison_mean_gap_z=float(comparison_z),
        michigan_train_sd=michigan_sd,
        comparison_train_sd=comparison_sd,
        michigan_rmse=float(michigan_fit.rmse_in_sample),
        comparison_rmse=float(comparison_fit.rmse_in_sample),
        gap_ratio=float(ratio),
        comparable=bool(ratio >= comparable_ratio),
        excluded_months=tuple(excluded),
        feasible=True,
        underpowered=underpowered,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 3. The decision
# ---------------------------------------------------------------------------


def assess_hypothesis_c(
    *,
    chow: ChowTestResult | None = None,
    sup_wald: SupWaldResult | None = None,
    cross_survey: CrossSurveyGap | None = None,
    gap: pd.Series | None = None,
    transition: pd.Timestamp | str = MICHIGAN_WEB_TRANSITION,
    tolerance_months: int = TRANSITION_TOLERANCE_MONTHS,
    comparable_ratio: float = COMPARABLE_GAP_RATIO,
    small_ratio: float = SMALL_GAP_RATIO,
    alpha: float = 0.05,
) -> HypothesisCVerdict:
    """Apply BRAINSTORMING.md's pre-registered conditions and return a verdict.

    The decision rule is fixed in advance, in this order, so the conclusion
    cannot be steered after seeing the numbers:

    1. **Falsification first.** If an independently collected series shows a gap
       of comparable magnitude over the same months (ratio >= `comparable_ratio`
       in training-window standard deviations), the gap is a property of the
       economy-perception relationship, not of the Michigan instrument, and C is
       **rejected as a primary explanation** — whatever the break tests say. A
       break can still be noted as a contributor, but it cannot rescue the
       hypothesis once the falsification condition has fired.
    2. **Support.** If the independent series shows a much smaller gap
       (ratio <= `small_ratio`) *and* a break is located at or near the
       specified transition and is significant, C is **supported**.
    3. Anything in between — including a break with no usable comparison, or a
       small independent gap with no break — is **partially supported**.
    4. With no usable evidence at all the verdict is **indeterminate**. That is
       not a hedge: at 19-22 observations "we could not test this" is the
       honest answer and must be distinguished from "we tested and found
       nothing".

    `level_to_variation` serves implication 3: |mean gap| / sd(gap). A gap that
    is almost all level with little month-to-month movement is what a mode
    effect looks like; a gap that moves with events points away from C.
    """
    transition = pd.Timestamp(transition)
    reasons: list[str] = []
    notes: list[str] = []
    underpowered = False

    chow_p = chow.p_value if chow is not None and chow.feasible else None
    sup_p = sup_wald.p_value if sup_wald is not None and sup_wald.feasible else None
    break_date = sup_wald.break_date if sup_wald is not None and sup_wald.feasible else None
    near = sup_wald.near(transition, tolerance_months) if sup_wald is not None else None

    if chow is not None:
        underpowered = underpowered or chow.underpowered
        notes.extend(chow.notes)
        if chow.feasible:
            verdict_word = "rejects" if chow_p is not None and chow_p < alpha else "does not reject"
            reasons.append(
                f"Chow test at the specified transition {chow.breakpoint:%Y-%m} "
                f"{verdict_word} stability (p={chow_p:.4f}, level shift "
                f"{chow.level_shift:+.2f})."
            )
        else:
            reasons.append(
                f"Chow test at {chow.breakpoint:%Y-%m} could not be computed at this "
                "sample size; no stability evidence from it."
            )

    if sup_wald is not None:
        underpowered = underpowered or sup_wald.underpowered
        notes.extend(sup_wald.notes)
        if sup_wald.feasible:
            significant = sup_p is not None and sup_p < alpha
            reasons.append(
                f"sup-Wald locates its most likely break at {break_date:%Y-%m} "
                f"(bootstrap p={sup_p:.4f}, {'significant' if significant else 'not significant'}); "
                f"that is {'within' if near else 'outside'} {tolerance_months} months of the "
                f"specified transition {transition:%Y-%m}."
            )
            searchable = (
                sup_wald.first_candidate is not None
                and sup_wald.first_candidate <= transition <= sup_wald.last_candidate
            )
            if not searchable:
                reasons.append(
                    f"NOTE: the specified transition {transition:%Y-%m} lies outside the "
                    f"trimmed search window ({sup_wald.first_candidate:%Y-%m}.."
                    f"{sup_wald.last_candidate:%Y-%m}), so the sup-Wald search could not "
                    "have selected it. A break was not found there because it was never "
                    "a candidate — that is not evidence against Hypothesis C. Re-run with "
                    "a smaller trim, or rely on the specified-date Chow test, which can "
                    "test a date the search cannot reach."
                )
        else:
            reasons.append(
                "sup-Wald search was not supportable at this sample size, so no "
                "break date is estimated."
            )

    if gap is not None and gap.notna().sum() >= 3:
        gap_clean = gap.dropna()
        spread = float(gap_clean.std(ddof=1))
        level_to_variation = abs(float(gap_clean.mean())) / spread if spread > 0 else None
        if level_to_variation is not None:
            reasons.append(
                f"The gap is {level_to_variation:.1f}x more level than month-to-month "
                "variation (|mean| / sd), which "
                + (
                    "is the shape a collection-mode shift would produce."
                    if level_to_variation >= 2.0
                    else "is as much movement as level, pointing away from a pure mode effect."
                )
            )
    else:
        level_to_variation = None
        if gap is not None:
            notes.append("gap series had fewer than 3 observations; level-vs-variation not computed")

    falsification_fired = False
    verdict = "indeterminate"

    if cross_survey is not None and cross_survey.feasible and cross_survey.gap_ratio is not None:
        underpowered = underpowered or cross_survey.underpowered
        notes.extend(cross_survey.notes)
        ratio = cross_survey.gap_ratio
        reasons.append(
            f"{cross_survey.label} ({cross_survey.series_id}) shows a mean gap of "
            f"{cross_survey.comparison_mean_gap_z:+.2f} of its own training-window "
            f"standard deviations over the same {cross_survey.n_months} months, against "
            f"{cross_survey.michigan_mean_gap_z:+.2f} for Michigan — a ratio of {ratio:.2f}."
        )
        if ratio >= comparable_ratio:
            falsification_fired = True
            verdict = "rejected"
            reasons.append(
                f"FALSIFICATION CONDITION FIRED: ratio {ratio:.2f} >= {comparable_ratio:.2f}. "
                "An independently collected series shows a gap of comparable magnitude "
                "over the same months, so the gap belongs to the economy-perception "
                "relationship rather than to the Michigan instrument. Hypothesis C is "
                "REJECTED as a primary explanation; it may survive as a partial "
                "contributor if a break is also present."
            )
        elif ratio <= small_ratio and bool(near) and sup_p is not None and sup_p < alpha:
            verdict = "supported"
            reasons.append(
                f"Ratio {ratio:.2f} <= {small_ratio:.2f} and a significant break sits near "
                "the specified transition: both pre-registered implications hold."
            )
        else:
            verdict = "partially_supported"
            reasons.append(
                f"Ratio {ratio:.2f} falls between {small_ratio:.2f} and {comparable_ratio:.2f}, "
                "or the break evidence does not corroborate it: Hypothesis C is a "
                "partial contributor on this evidence, not the primary explanation."
            )
    else:
        if cross_survey is not None:
            underpowered = True
            notes.extend(cross_survey.notes)
        reasons.append(
            "No usable independently collected series, so the decisive "
            "falsification condition could not be applied."
        )
        if sup_p is not None and sup_p < alpha and bool(near):
            verdict = "partially_supported"
            reasons.append(
                "A significant break near the transition is consistent with C, but "
                "without a cross-survey comparison it cannot be separated from any "
                "other event at the same date, so support stays partial."
            )
        elif chow_p is not None and chow_p < alpha:
            verdict = "partially_supported"
            reasons.append(
                "The specified-date Chow test rejects stability, which is consistent "
                "with C but equally consistent with any same-dated shock."
            )

    if underpowered:
        notes.append(
            "At least one input was flagged underpowered. The verdict is a reading "
            "of the available evidence, not an inference with usable precision."
        )

    return HypothesisCVerdict(
        verdict=verdict,
        break_at_transition=near,
        break_date=break_date,
        chow_p_value=chow_p,
        sup_wald_p_value=sup_p,
        gap_ratio=cross_survey.gap_ratio if cross_survey is not None else None,
        falsification_fired=falsification_fired,
        level_to_variation=level_to_variation,
        underpowered=underpowered,
        reasons=tuple(reasons),
        notes=tuple(notes),
    )
