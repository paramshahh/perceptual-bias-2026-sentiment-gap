"""Tests for the Hypothesis C diagnostics.

All of these run offline on synthetic data. The one function that touches the
network (`load_comparison_series`) is exercised through a `_StubFred` stand-in
following the pattern in `test_data_loader.py` — no test may make a FRED call.

Several tests pin the specific errors this module exists to avoid: comparing a
sup-Wald statistic to a fixed-breakpoint critical value, letting a
small-sample test return a confident number, and comparing a Michigan gap
against a series that is secretly the Michigan survey rescaled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.data_loader import FredSource
from perceptual_bias.survey_diagnostics import (
    COMPARABLE_GAP_RATIO,
    MICHIGAN_WEB_TRANSITION,
    ComparisonCandidate,
    SurveyDiagnosticsError,
    assess_hypothesis_c,
    chow_test,
    cross_survey_gap,
    load_comparison_series,
    sup_wald_test,
    verify_comparison_series,
)

BOOT = 199  # keeps the bootstrap honest but the suite fast
SEED = 7


class _StubFred(FredSource):
    """FRED stub: claims to be configured, serves only what it was handed.

    Same shape as the data-loader tests' stub, so a test can model "key
    present, series missing" as well as a successful fetch.
    """

    def __init__(self, *, configured: bool = True, payload: dict[str, pd.Series] | None = None):
        self._configured = configured
        self._payload = payload or {}

    @property
    def configured(self) -> bool:  # type: ignore[override]
        return self._configured

    def series(self, series_id: str) -> pd.Series:  # type: ignore[override]
        return self._payload.get(series_id, pd.Series(dtype=float, name=series_id))


def _months(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _series_with_break(
    n: int = 200,
    break_at: str = "2024-07-31",
    shift: float = 12.0,
    noise: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    """A flat series with a clean level shift after `break_at`."""
    index = _months(n)
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, noise, size=n)
    values[index > pd.Timestamp(break_at)] += shift
    return pd.Series(values, index=index, name="residual")


def _series_without_break(n: int = 200, noise: float = 1.0, seed: int = 1) -> pd.Series:
    index = _months(n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, noise, size=n), index=index, name="residual")


class TestChowTest:
    def test_finds_a_specified_break_and_estimates_its_size(self) -> None:
        result = chow_test(_series_with_break(shift=12.0), breakpoint="2024-07-31")
        assert result.feasible
        assert result.p_value < 1e-6
        assert result.level_shift == pytest.approx(12.0, abs=0.5)
        assert result.k == 1  # constant-only design: a level-shift test

    def test_does_not_fire_without_a_break(self) -> None:
        result = chow_test(_series_without_break(), breakpoint="2024-07-31")
        assert result.feasible
        assert result.p_value > 0.05
        assert abs(result.level_shift) < 1.0

    def test_regression_design_tests_coefficient_stability(self) -> None:
        """With regressors supplied, a slope break is detected, not just a level one."""
        index = _months(200)
        rng = np.random.default_rng(3)
        x = pd.DataFrame({"unemployment": rng.normal(5.0, 1.0, size=200)}, index=index)
        slope = np.where(index > pd.Timestamp("2024-07-31"), -12.0, -4.0)
        y = pd.Series(
            100.0 + slope * x["unemployment"] + rng.normal(0, 1.0, size=200),
            index=index,
            name="sentiment",
        )
        result = chow_test(y, x, breakpoint="2024-07-31")
        assert result.k == 2  # unemployment + constant
        assert result.feasible
        assert result.p_value < 0.01

    def test_small_sample_guard_fires_loudly(self) -> None:
        """A 19-month holdout trips the soft guard but still reports a number."""
        short = _series_with_break(n=19, break_at="2010-10-31", shift=10.0)
        result = chow_test(short, breakpoint="2010-10-31")
        assert result.feasible
        assert result.underpowered
        assert any("UNDERPOWERED" in note for note in result.notes)

    def test_infeasible_sample_returns_none_not_a_number(self) -> None:
        """k=2 with two observations each side: no statistic, loud note."""
        index = _months(4)
        y = pd.Series([1.0, 1.1, 9.0, 9.1], index=index)
        x = pd.DataFrame({"z": [1.0, 2.0, 3.0, 4.0]}, index=index)
        result = chow_test(y, x, breakpoint=index[1])
        assert result.feasible is False
        assert result.f_statistic is None
        assert result.p_value is None
        assert any("NOT COMPUTED" in note for note in result.notes)

    def test_breakpoint_outside_the_sample_raises(self) -> None:
        with pytest.raises(SurveyDiagnosticsError, match="outside the sample"):
            chow_test(_series_without_break(n=36), breakpoint="2030-01-31")

    def test_missing_rows_are_named_never_silently_dropped(self) -> None:
        y = _series_with_break()
        y.iloc[5] = np.nan
        y.iloc[6] = np.nan
        result = chow_test(y, breakpoint="2024-07-31")
        assert len(result.dropped_dates) == 2
        assert result.n == len(y) - 2
        assert any("dropped 2 rows" in note for note in result.notes)

    def test_unsorted_index_raises_rather_than_mis_splitting(self) -> None:
        y = _series_with_break(n=36)
        with pytest.raises(SurveyDiagnosticsError, match="sorted"):
            chow_test(y.iloc[::-1], breakpoint=y.index[18])


class TestSupWaldTest:
    def test_locates_a_known_break_near_the_true_date(self) -> None:
        result = sup_wald_test(
            _series_with_break(break_at="2020-06-30", shift=12.0),
            n_bootstrap=BOOT,
            seed=SEED,
        )
        assert result.feasible
        offset = abs(
            (result.break_date.year - 2020) * 12 + (result.break_date.month - 6)
        )
        assert offset <= 2, f"break located at {result.break_date}"
        assert result.p_value <= 0.01

    def test_does_not_fire_when_there_is_no_break(self) -> None:
        result = sup_wald_test(_series_without_break(), n_bootstrap=BOOT, seed=SEED)
        assert result.feasible
        assert result.p_value > 0.05
        # The statistic is large even under the null — which is exactly why it
        # must not be read against a fixed-breakpoint critical value.
        assert result.f_statistic > 0.0

    def test_bootstrap_is_not_a_fixed_breakpoint_test(self) -> None:
        """Pins the error the module exists to avoid.

        Under the null the sup-F statistic routinely exceeds the 5% F(1, n-2)
        critical value (~3.9), so a fixed-breakpoint comparison would 'find' a
        break in white noise. The bootstrap p-value must not.
        """
        from scipy import stats

        fired_naive = 0
        fired_bootstrap = 0
        for seed in range(6):
            result = sup_wald_test(
                _series_without_break(n=120, seed=100 + seed), n_bootstrap=99, seed=seed
            )
            naive_p = float(stats.f.sf(result.f_statistic, result.k, result.n - 2 * result.k))
            fired_naive += naive_p < 0.05
            fired_bootstrap += result.p_value < 0.05
        assert fired_naive > fired_bootstrap
        assert fired_bootstrap == 0

    def test_p_value_can_never_be_zero(self) -> None:
        result = sup_wald_test(_series_with_break(shift=50.0), n_bootstrap=49, seed=SEED)
        assert result.p_value == pytest.approx(1.0 / 50.0)

    def test_is_reproducible_under_a_seed(self) -> None:
        kwargs = {"n_bootstrap": 99, "seed": 11}
        first = sup_wald_test(_series_with_break(), **kwargs)
        second = sup_wald_test(_series_with_break(), **kwargs)
        assert first.p_value == second.p_value
        assert first.break_date == second.break_date

    def test_statistic_path_covers_the_trimmed_window_only(self) -> None:
        result = sup_wald_test(_series_with_break(n=100), n_bootstrap=49, seed=SEED)
        assert len(result.statistic_path) == result.n_candidates
        assert result.statistic_path.index.min() == result.first_candidate
        assert result.statistic_path.index.max() == result.last_candidate
        assert result.n_candidates < result.n  # trimming actually happened

    def test_small_n_guard_fires_on_a_holdout_length_sample(self) -> None:
        result = sup_wald_test(_series_with_break(n=20, break_at="2010-10-31", shift=10.0),
                               n_bootstrap=99, seed=SEED)
        assert result.underpowered
        assert any("UNDERPOWERED" in note for note in result.notes)
        assert any("region, not a month" in note for note in result.notes)

    def test_no_admissible_breakpoints_returns_no_statistic(self) -> None:
        """19 months with 4 regressors: nothing is estimable, and it says so."""
        index = _months(19)
        rng = np.random.default_rng(5)
        x = pd.DataFrame(rng.normal(size=(19, 8)), index=index)
        y = pd.Series(rng.normal(size=19), index=index)
        result = sup_wald_test(y, x, n_bootstrap=99, seed=SEED)
        assert result.feasible is False
        assert result.f_statistic is None
        assert result.p_value is None
        assert result.break_date is None
        assert any("NOT COMPUTED" in note for note in result.notes)

    def test_near_returns_none_when_nothing_was_estimated(self) -> None:
        index = _months(10)
        rng = np.random.default_rng(6)
        x = pd.DataFrame(rng.normal(size=(10, 4)), index=index)
        result = sup_wald_test(pd.Series(rng.normal(size=10), index=index), x, n_bootstrap=9)
        assert result.near(MICHIGAN_WEB_TRANSITION, 6) is None  # not False

    def test_invalid_trim_raises(self) -> None:
        with pytest.raises(SurveyDiagnosticsError, match="trim"):
            sup_wald_test(_series_with_break(), trim=0.75)


class TestVerifyComparisonSeries:
    def test_rejects_a_rescaled_copy_of_the_michigan_series(self) -> None:
        """Pins the real trap: OECD's US CCI is the Michigan survey rescaled."""
        index = _months(120)
        rng = np.random.default_rng(9)
        michigan = pd.Series(70 + rng.normal(0, 8, size=120).cumsum() / 4, index=index)
        impostor = michigan * 1.07 + 3.0  # an affine rescaling, nothing more
        ok, notes = verify_comparison_series(impostor, michigan)
        assert ok is False
        assert any("rescaling of the Michigan" in note for note in notes)

    def test_rejects_a_rescaled_copy_that_rounding_has_blurred(self) -> None:
        """Pins the threshold bug found against live data.

        `USACSCICP02STSAM` IS the Michigan survey rescaled, but published
        rounding pulls its first-difference correlation down to ~0.976 over the
        full 1978-2026 history. A 0.999-on-both-legs screen let it through, and
        it then produced a gap ratio of exactly 1.00 — i.e. it would have
        "falsified" Hypothesis C using Michigan's own numbers.
        """
        index = _months(300)
        rng = np.random.default_rng(12)
        michigan = pd.Series(70 + rng.normal(0, 6, size=300).cumsum() / 5, index=index)
        blurred = (michigan * 1.07 + 3.0).round(0) + pd.Series(
            rng.normal(0, 0.25, size=300), index=index
        )
        level = float(blurred.corr(michigan))
        diff = float(blurred.diff().corr(michigan.diff()))
        assert level > 0.99 and 0.90 < diff < 0.999, (level, diff)
        ok, notes = verify_comparison_series(blurred, michigan)
        assert ok is False
        assert any("rescaling of the Michigan" in note for note in notes)

    def test_accepts_a_genuinely_independent_series(self) -> None:
        index = _months(120)
        rng = np.random.default_rng(10)
        michigan = pd.Series(70 + rng.normal(0, 8, size=120).cumsum() / 4, index=index)
        independent = michigan * 0.3 + pd.Series(rng.normal(0, 8, size=120), index=index)
        ok, notes = verify_comparison_series(independent, michigan)
        assert ok is True
        assert any("correlation with Michigan" in note for note in notes)

    def test_rejects_a_series_that_stops_before_the_holdout(self) -> None:
        """CSCICP03USM665S exists but ends 2024-01; that must not pass."""
        index = _months(120)
        rng = np.random.default_rng(11)
        michigan = pd.Series(rng.normal(size=120), index=index)
        stale = pd.Series(rng.normal(size=60), index=index[:60])
        ok, notes = verify_comparison_series(stale, michigan, required_through="2026-08-31")
        assert ok is False
        assert any("coverage ends" in note for note in notes)

    def test_rejects_too_little_overlap(self) -> None:
        index = _months(120)
        michigan = pd.Series(np.arange(120.0), index=index)
        short = pd.Series(np.arange(5.0), index=index[:5])
        ok, notes = verify_comparison_series(short, michigan)
        assert ok is False
        assert any("overlapping months" in note for note in notes)


class TestLoadComparisonSeries:
    def test_unconfigured_fred_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(SurveyDiagnosticsError, match="not configured"):
            load_comparison_series(fred=_StubFred(configured=False))

    def test_empty_series_raises_with_the_verification_warning(self) -> None:
        with pytest.raises(SurveyDiagnosticsError, match="returned no data"):
            load_comparison_series("NOPE", fred=_StubFred())

    def test_successful_fetch_reports_coverage(self) -> None:
        index = _months(24, start="2024-01-31")
        payload = {"FMNHSHPSIUS": pd.Series(np.arange(24.0), index=index, name="FMNHSHPSIUS")}
        series, notes = load_comparison_series("FMNHSHPSIUS", fred=_StubFred(payload=payload))
        assert len(series) == 24
        assert any("24 observations" in note for note in notes)
        assert any("DISCONTINUED" in note for note in notes)

    def test_known_non_independent_candidate_is_flagged(self) -> None:
        index = _months(24, start="2024-01-31")
        payload = {
            "USACSCICP02STSAM": pd.Series(np.arange(24.0), index=index, name="USACSCICP02STSAM")
        }
        _, notes = load_comparison_series("USACSCICP02STSAM", fred=_StubFred(payload=payload))
        assert any("circular" in note for note in notes)


# ---------------------------------------------------------------------------
# Cross-survey comparison
# ---------------------------------------------------------------------------


def _synthetic_panel(
    n: int = 190,
    gap_from: str = "2025-01-31",
    michigan_shift: float = -24.0,
    seed: int = 21,
) -> pd.DataFrame:
    """A panel whose Michigan sentiment collapses out of sample by construction.

    The macro regressors stay well behaved, so the whole holdout gap is the
    injected shift: that makes the cross-survey arithmetic checkable by hand.
    """
    index = _months(n)
    rng = np.random.default_rng(seed)
    unemployment = 5.0 + rng.normal(0, 0.4, size=n)
    pce = 2.0 + rng.normal(0, 0.3, size=n)
    gdp = 2.0 + rng.normal(0, 1.0, size=n)
    sentiment = 120 - 4.0 * unemployment - 9.0 * pce - 0.2 * gdp + rng.normal(0, 3.0, size=n)
    sentiment = pd.Series(sentiment, index=index)
    sentiment.loc[index >= pd.Timestamp(gap_from)] += michigan_shift
    return pd.DataFrame(
        {
            "unemployment": unemployment,
            "pce_yoy": pce,
            "real_gdp_qoq": gdp,
            "sentiment": sentiment,
        },
        index=index,
    )


def _comparison_from_panel(
    panel: pd.DataFrame,
    *,
    scale: float,
    shift: float,
    gap_from: str = "2025-01-31",
    seed: int = 22,
    name: str = "STUBSURVEY",
) -> pd.Series:
    """An independent survey on a different scale, with a chosen holdout shift.

    `scale` makes the series' units unrelated to Michigan's, which is what the
    standardisation has to cope with; `shift` is the gap injected out of sample
    in that series' *own* units.
    """
    index = panel.index
    rng = np.random.default_rng(seed)
    base = scale * (
        2.0
        - 0.4 * panel["unemployment"]
        - 0.9 * panel["pce_yoy"]
        + rng.normal(0, 0.3, size=len(index))
    )
    base.loc[index >= pd.Timestamp(gap_from)] += shift
    return base.rename(name)


class TestCrossSurveyGap:
    def test_comparable_gap_on_a_different_scale_is_detected(self) -> None:
        """A stub survey with a proportionally equal gap must read as comparable."""
        panel = _synthetic_panel()
        train = panel.loc[:"2024-12-31", "sentiment"]
        michigan_sd = float(train.std(ddof=1))
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0)
        comparison_sd = float(comparison.loc[:"2024-12-31"].std(ddof=1))
        # Inject a shift worth the same number of own-series SDs as Michigan's.
        comparison = _comparison_from_panel(
            panel, scale=40.0, shift=-24.0 / michigan_sd * comparison_sd
        )
        result = cross_survey_gap(panel, comparison)
        assert result.feasible
        assert result.michigan_train_sd != pytest.approx(result.comparison_train_sd, rel=0.2)
        assert result.gap_ratio == pytest.approx(1.0, abs=0.25)
        assert result.comparable is True

    def test_flat_comparison_survey_yields_a_small_ratio(self) -> None:
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0)
        result = cross_survey_gap(panel, comparison)
        assert result.feasible
        assert result.gap_ratio < 0.25
        assert result.comparable is False

    def test_only_overlapping_months_are_compared_and_the_rest_are_named(self) -> None:
        """A discontinued comparison series must not silently shrink the window."""
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0).loc[:"2025-06-30"]
        result = cross_survey_gap(panel, comparison)
        assert result.n_months == 6
        assert result.excluded_months  # the uncovered holdout months are listed
        assert any("ends 2025-06" in note for note in result.notes)

    def test_too_few_common_months_flags_underpowered(self) -> None:
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0).loc[:"2025-03-31"]
        result = cross_survey_gap(panel, comparison)
        assert result.n_months == 3
        assert result.underpowered
        assert any("UNDERPOWERED" in note for note in result.notes)

    def test_no_overlap_is_not_computed_rather_than_passed(self) -> None:
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0).loc[:"2024-12-31"]
        result = cross_survey_gap(panel, comparison)
        assert result.feasible is False
        assert result.gap_ratio is None
        assert result.comparable is None  # NaN is not False
        assert any("NOT COMPUTED" in note for note in result.notes)

    def test_standardisation_uses_training_moments_only(self) -> None:
        """No look-ahead: holdout values must not move the training SD."""
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(panel, scale=40.0, shift=0.0)
        baseline = cross_survey_gap(panel, comparison)

        perturbed = panel.copy()
        perturbed.loc["2025-01-31":, "sentiment"] -= 50.0
        shifted = cross_survey_gap(perturbed, comparison)
        assert shifted.michigan_train_sd == pytest.approx(baseline.michigan_train_sd)

    def test_missing_target_column_raises(self) -> None:
        panel = _synthetic_panel().drop(columns="sentiment")
        comparison = _comparison_from_panel(_synthetic_panel(), scale=40.0, shift=0.0)
        with pytest.raises(SurveyDiagnosticsError, match="no 'sentiment' column"):
            cross_survey_gap(panel, comparison)

    def test_non_independent_candidate_is_flagged_in_the_result(self) -> None:
        panel = _synthetic_panel()
        comparison = _comparison_from_panel(
            panel, scale=40.0, shift=0.0, name="USACSCICP02STSAM"
        )
        result = cross_survey_gap(panel, comparison)
        assert any("circular" in note for note in result.notes)


# ---------------------------------------------------------------------------
# The decision helper
# ---------------------------------------------------------------------------


def _comparable_cross_survey(panel: pd.DataFrame) -> object:
    train_sd = float(panel.loc[:"2024-12-31", "sentiment"].std(ddof=1))
    flat = _comparison_from_panel(panel, scale=40.0, shift=0.0)
    comparison_sd = float(flat.loc[:"2024-12-31"].std(ddof=1))
    matched = _comparison_from_panel(panel, scale=40.0, shift=-24.0 / train_sd * comparison_sd)
    return cross_survey_gap(panel, matched)


class TestAssessHypothesisC:
    def test_comparable_independent_gap_rejects_hypothesis_c(self) -> None:
        """The pre-registered falsification condition, applied explicitly."""
        panel = _synthetic_panel()
        cross = _comparable_cross_survey(panel)
        assert cross.gap_ratio >= COMPARABLE_GAP_RATIO
        verdict = assess_hypothesis_c(cross_survey=cross)
        assert verdict.verdict == "rejected"
        assert verdict.falsification_fired is True
        assert any("FALSIFICATION CONDITION FIRED" in reason for reason in verdict.reasons)

    def test_a_break_cannot_rescue_c_once_falsification_fires(self) -> None:
        panel = _synthetic_panel()
        cross = _comparable_cross_survey(panel)
        chow = chow_test(_series_with_break(shift=12.0), breakpoint="2024-07-31")
        sup = sup_wald_test(_series_with_break(shift=12.0), n_bootstrap=BOOT, seed=SEED)
        verdict = assess_hypothesis_c(chow=chow, sup_wald=sup, cross_survey=cross)
        assert verdict.verdict == "rejected"
        assert verdict.sup_wald_p_value < 0.05  # the break is real and still loses

    def test_small_independent_gap_plus_a_break_at_the_transition_supports_c(self) -> None:
        panel = _synthetic_panel()
        flat = _comparison_from_panel(panel, scale=40.0, shift=0.0)
        cross = cross_survey_gap(panel, flat)
        residuals = _series_with_break(break_at="2024-07-31", shift=12.0)
        # 2024-07 sits late in a 2010-2026 sample, so the standard 15% trim
        # would not even offer it as a candidate; trim=0.10 is passed
        # deliberately and the module says so when the transition is
        # unsearchable.
        verdict = assess_hypothesis_c(
            chow=chow_test(residuals, breakpoint="2024-07-31"),
            sup_wald=sup_wald_test(residuals, trim=0.10, n_bootstrap=BOOT, seed=SEED),
            cross_survey=cross,
        )
        assert verdict.verdict == "supported"
        assert verdict.break_at_transition is True
        assert verdict.falsification_fired is False

    def test_small_gap_without_a_break_is_only_partial(self) -> None:
        panel = _synthetic_panel()
        cross = cross_survey_gap(panel, _comparison_from_panel(panel, scale=40.0, shift=0.0))
        flat_residuals = _series_without_break()
        verdict = assess_hypothesis_c(
            chow=chow_test(flat_residuals, breakpoint="2024-07-31"),
            sup_wald=sup_wald_test(flat_residuals, n_bootstrap=BOOT, seed=SEED),
            cross_survey=cross,
        )
        assert verdict.verdict == "partially_supported"

    def test_no_evidence_at_all_is_indeterminate_not_rejected(self) -> None:
        verdict = assess_hypothesis_c()
        assert verdict.verdict == "indeterminate"
        assert any("could not be applied" in reason for reason in verdict.reasons)

    def test_break_without_a_comparison_series_stays_partial(self) -> None:
        residuals = _series_with_break(break_at="2024-07-31", shift=12.0)
        verdict = assess_hypothesis_c(
            sup_wald=sup_wald_test(residuals, trim=0.10, n_bootstrap=BOOT, seed=SEED),
            transition="2024-07-31",
        )
        assert verdict.verdict == "partially_supported"
        assert verdict.gap_ratio is None

    def test_unsearchable_transition_is_called_out(self) -> None:
        """A transition outside the trimmed window is not evidence against C."""
        residuals = _series_with_break(break_at="2020-06-30", shift=12.0)
        verdict = assess_hypothesis_c(
            sup_wald=sup_wald_test(residuals, n_bootstrap=99, seed=SEED),
            transition="2026-06-30",
        )
        assert any("outside the trimmed search window" in reason for reason in verdict.reasons)

    def test_underpowered_inputs_propagate_to_the_verdict(self) -> None:
        short = _series_with_break(n=20, break_at="2010-10-31", shift=10.0)
        verdict = assess_hypothesis_c(
            sup_wald=sup_wald_test(short, n_bootstrap=99, seed=SEED),
            transition="2010-10-31",
        )
        assert verdict.underpowered is True
        assert any("not an inference with usable precision" in note for note in verdict.notes)

    def test_level_dominated_gap_is_described_as_mode_shaped(self) -> None:
        gap = pd.Series(
            -25.0 + np.random.default_rng(4).normal(0, 1.0, size=19), index=_months(19, "2025-01-31")
        )
        verdict = assess_hypothesis_c(gap=gap)
        assert verdict.level_to_variation > 2.0
        assert any("collection-mode shift" in reason for reason in verdict.reasons)

    def test_event_driven_gap_points_away_from_c(self) -> None:
        gap = pd.Series(
            np.random.default_rng(8).normal(0, 12.0, size=19), index=_months(19, "2025-01-31")
        )
        verdict = assess_hypothesis_c(gap=gap)
        assert verdict.level_to_variation < 2.0
        assert any("pointing away" in reason for reason in verdict.reasons)

    def test_verdict_summary_is_quotable(self) -> None:
        panel = _synthetic_panel()
        verdict = assess_hypothesis_c(cross_survey=_comparable_cross_survey(panel))
        text = verdict.summary()
        assert "REJECTED" in text
        assert text.count("\n  - ") == len(verdict.reasons)


class TestComparisonCandidateRegistry:
    def test_the_brainstorming_series_is_recorded_as_verified_and_stale(self) -> None:
        """BRAINSTORMING.md flags CSCICP03USM665S as unverified; this is the answer.

        The ID exists on FRED but its last observation is 2024-01, so it covers
        none of the holdout. Recording that here keeps the plan from being
        quoted back as a fact.
        """
        from perceptual_bias.survey_diagnostics import COMPARISON_CANDIDATES

        by_id = {c.series_id: c for c in COMPARISON_CANDIDATES}
        stale = by_id["CSCICP03USM665S"]
        assert isinstance(stale, ComparisonCandidate)
        assert stale.independent_of_michigan is False
        assert "2024-01" in stale.note
        assert by_id["USACSCICP02STSAM"].independent_of_michigan is False
        assert by_id["FMNHSHPSIUS"].independent_of_michigan is True
