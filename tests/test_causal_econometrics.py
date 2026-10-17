"""Tests for the comparative regressions, Granger precedence and the HSK IV.

Everything here is synthetic and offline. The point of the synthetic data is
that the *answer is known*, so these tests check the procedures recover a truth
rather than merely that they return floats:

* a series pair built with a one-directional lead/lag must be detected in that
  direction and *not* the reverse;
* a system with a known variance shift must pass the rank condition, and a
  homoskedastic one must fail it loudly;
* Model 3 must degrade to "unavailable" rather than crash when the media
  columns do not exist yet;
* the small-n guards must fire, because 19 monthly observations is the
  project's binding constraint and the module's job is to refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.causal_econometrics import (
    EXPOSURE_GAP_STATEMENT,
    GRANGER_CAVEAT,
    HOUSEHOLD_POLITICAL_FEATURES,
    HOUSEHOLD_POLITICAL_SERIES,
    MEDIA_FEATURES,
    PREDICTED_SIGNS,
    CausalDesignProposal,
    EconometricsError,
    HypothesisTest,
    build_household_political_controls,
    causal_design_proposals,
    causal_design_table,
    coefficient_stability,
    collect_media_test_family,
    column_coverage,
    compare_model_ladder,
    compare_models,
    election_proximity_features,
    granger_media_gap,
    month_dummies,
    multiple_testing_correction,
    predicted_sign_check,
    resolve_price_level_column,
    reverse_causality_evidence,
    rigobon_sack_iv,
    rolling_origin_forecast,
    seasonality_report,
    specification_sensitivity,
    split_volatility_regimes,
)

RNG_SEED = 20261015


def _index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2008-01-31", periods=periods, freq="ME")


def macro_panel(periods: int = 180, media: bool = False, seed: int = RNG_SEED) -> pd.DataFrame:
    """Panel with a known linear truth for sentiment, optionally with media."""
    rng = np.random.default_rng(seed)
    index = _index(periods)
    steps = np.arange(periods)
    unemployment = 5.0 + 1.5 * np.sin(steps / 12)
    pce_yoy = 2.0 + 0.8 * np.cos(steps / 18)
    real_gdp_qoq = 2.5 + rng.normal(0, 0.4, periods)
    pce_level = 100.0 + np.cumsum(pce_yoy / 12.0)
    mortgage_30y = 4.5 + 0.9 * np.sin(steps / 20)
    debt_service = 10.0 + 0.3 * np.cos(steps / 24)

    frame = pd.DataFrame(
        {
            "unemployment": unemployment,
            "pce_yoy": pce_yoy,
            "real_gdp_qoq": real_gdp_qoq,
            "pce_level": pce_level,
            "mortgage_30y": mortgage_30y,
            "debt_service": debt_service,
        },
        index=index,
    )
    frame.index.name = "date"

    if media:
        frame["media_tone"] = -0.1 + 0.2 * np.sin(steps / 7) + rng.normal(0, 0.05, periods)
        frame["topic_salience"] = 0.3 + 0.1 * np.cos(steps / 9) + rng.normal(0, 0.02, periods)
        frame["political_attribution"] = 0.2 + rng.normal(0, 0.03, periods)
        # The framing dimension (price-LEVEL framing against inflation-RATE
        # framing) is the fifth media regressor; `MEDIA_FEATURES` carries it,
        # so a panel without it would make Model 3 unavailable.
        frame["framing"] = 0.6 + 0.2 * np.sin(steps / 11) + rng.normal(0, 0.04, periods)

    sentiment = (
        120.0
        - 4.0 * unemployment
        - 9.0 * pce_yoy
        - 0.2 * real_gdp_qoq
        - 0.25 * (pce_level - 100.0)
        + rng.normal(0, 1.0, periods)
    )
    if media:
        sentiment = sentiment + 12.0 * frame["media_tone"].to_numpy()
    frame["sentiment"] = sentiment
    return frame


def lead_lag_pair(
    periods: int = 300, strength: float = 0.9, seed: int = RNG_SEED
) -> pd.DataFrame:
    """Media leads the gap by one month, with no feedback the other way.

    ``media_tone`` is an AR(1) driven only by its own shocks, so nothing about
    the gap can help predict it. The gap loads on *lagged* media. The truth is
    therefore strictly one-directional, which is what the Granger test should
    find -- and the reverse test should not reject.
    """
    rng = np.random.default_rng(seed)
    media = np.zeros(periods)
    gap = np.zeros(periods)
    for t in range(1, periods):
        media[t] = 0.5 * media[t - 1] + rng.normal(0, 1.0)
        gap[t] = 0.3 * gap[t - 1] + strength * media[t - 1] + rng.normal(0, 1.0)
    frame = pd.DataFrame({"media_tone": media, "gap": gap}, index=_index(periods))
    frame.index.name = "date"
    return frame


def heteroskedastic_system(
    periods: int = 360,
    b: float = 0.5,
    a: float = 0.6,
    block: int = 30,
    high_sd: float = 2.0,
    low_sd: float = 0.8,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Simultaneous 2-variable system where only the media shock variance shifts.

    ``m = a*g + e_m``, ``g = b*m + e_g``, with ``sd(e_m)`` alternating between
    `high_sd` and `low_sd` in blocks and ``sd(e_g)`` constant. This is exactly
    the data-generating process Rigobon-Sack identification assumes, so
    ``b`` must be recoverable.
    """
    rng = np.random.default_rng(seed)
    regime_high = (np.arange(periods) // block) % 2 == 0
    sd_m = np.where(regime_high, high_sd, low_sd)
    e_m = rng.normal(0, 1.0, periods) * sd_m
    e_g = rng.normal(0, 1.0, periods)
    scale = 1.0 / (1.0 - a * b)
    media = scale * (e_m + a * e_g)
    gap = scale * (b * e_m + e_g)
    frame = pd.DataFrame({"media_tone": media, "gap": gap}, index=_index(periods))
    frame.index.name = "date"
    return frame


def homoskedastic_system(
    periods: int = 360, b: float = 0.5, a: float = 0.6, seed: int = RNG_SEED
) -> pd.DataFrame:
    """The same system with a constant media-shock variance: unidentified."""
    return heteroskedastic_system(
        periods=periods, b=b, a=a, high_sd=1.0, low_sd=1.0, seed=seed
    )


# ---------------------------------------------------------------------------
# 1. Comparative regression framework
# ---------------------------------------------------------------------------


def test_all_three_models_fit_when_media_present():
    comparison = compare_models(macro_panel(media=True))
    assert comparison.available_models == ("Model 1", "Model 2", "Model 3")
    table = comparison.table()
    assert list(table.index) == ["Model 1", "Model 2", "Model 3"]
    # Nesting means R2 cannot fall as regressors are added.
    assert table.loc["Model 1", "r_squared"] <= table.loc["Model 2", "r_squared"]
    assert table.loc["Model 2", "r_squared"] <= table.loc["Model 3", "r_squared"]
    # The data were generated with media tone in the truth, so Model 3 should win.
    assert table.loc["Model 3", "aic"] < table.loc["Model 1", "aic"]
    assert table.loc["Model 3", "rmse"] < table.loc["Model 1", "rmse"]


def test_model3_degrades_cleanly_when_media_columns_absent():
    """A missing corpus is a project state, not an exception."""
    comparison = compare_models(macro_panel(media=False))
    assert comparison.available_models == ("Model 1", "Model 2")
    model3 = comparison.models["Model 3"]
    assert model3.available is False
    assert set(model3.missing_features) == set(MEDIA_FEATURES)
    assert "absent" in (model3.unavailable_reason or "")
    assert model3.coefficients is None
    # It keeps its row in Table 3 so the report shows it was attempted.
    assert "Model 3" in comparison.table().index
    assert np.isnan(comparison.table().loc["Model 3", "r_squared"])
    # And it must not be confusable with "media did not help".
    assert any("NOT the same finding" in note for note in model3.notes)


def test_partially_present_media_columns_still_degrade():
    panel = macro_panel(media=True).drop(columns=["political_attribution"])
    comparison = compare_models(panel)
    model3 = comparison.models["Model 3"]
    assert model3.available is False
    assert model3.missing_features == ("political_attribution",)


def test_common_sample_is_shared_and_dropped_rows_are_named():
    panel = macro_panel(media=True)
    # mortgage_30y stands in for the Model 2 extras here: debt_service was
    # dropped from MODEL2_EXTRA_FEATURES (insignificant, and TDSP ends early),
    # so a gap in it no longer affects the common sample.
    panel.loc[panel.index[10], "mortgage_30y"] = np.nan
    panel.loc[panel.index[20], "media_tone"] = np.nan
    comparison = compare_models(panel)
    counts = {fit.n for fit in comparison.models.values() if fit.available}
    assert len(counts) == 1, "models must be estimated on identical rows"
    assert comparison.common_sample_n == len(panel) - 2
    assert len(comparison.dropped_dates) == 2
    joined = " ".join(comparison.dropped_reasons)
    assert "mortgage_30y" in joined and "media_tone" in joined
    assert panel.index[10].strftime("%Y-%m") in joined


def test_coefficients_carry_hac_inference():
    comparison = compare_models(macro_panel(media=True))
    coefficients = comparison.models["Model 1"].coefficients
    assert list(coefficients.columns) == [
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    # The truth has unemployment at -4.0 and pce_yoy at -9.0; with 180 points
    # and low noise the signs and significance must come through.
    assert coefficients.loc["unemployment", "coef"] < 0
    assert coefficients.loc["pce_yoy", "coef"] < 0
    assert "unemployment" in comparison.models["Model 1"].significant_features()
    assert comparison.hac_lags == 12
    assert any("HAC" in note for note in comparison.models["Model 1"].notes)


def test_coefficient_table_is_long_and_labelled_by_model():
    comparison = compare_models(macro_panel(media=True))
    long = comparison.coefficient_table()
    assert set(long["model"]) == {"Model 1", "Model 2", "Model 3"}
    assert {"regressor", "coef", "p_value"} <= set(long.columns)
    media_rows = long[long["regressor"] == "media_tone"]
    assert len(media_rows) == 1 and media_rows.iloc[0]["model"] == "Model 3"


def test_price_level_approximation_is_substituted_and_flagged():
    panel = macro_panel(media=False).rename(columns={"pce_level": "pce_level_approx"})
    assert resolve_price_level_column(panel) == "pce_level_approx"
    comparison = compare_models(panel)
    assert comparison.models["Model 2"].available
    assert "pce_level_approx" in comparison.models["Model 2"].features
    assert any("approximation" in note for note in comparison.notes)


def test_small_sample_regression_refuses_by_default():
    """19 months is the real out-of-sample length; HAC(12) on it is not inference."""
    panel = macro_panel(media=True).iloc[-19:]
    with pytest.raises(EconometricsError, match="below the minimum"):
        compare_models(panel)


def test_small_sample_regression_reports_loudly_when_forced():
    panel = macro_panel(media=True).iloc[-19:]
    comparison = compare_models(panel, allow_underpowered=True)
    assert comparison.underpowered
    model1 = comparison.models["Model 1"]
    assert model1.available and model1.underpowered
    assert model1.power_warnings, "a forced small-n fit must carry warnings"
    assert any("per parameter" in w or "below the minimum" in w for w in model1.power_warnings)
    # Model 3 estimates 10 parameters from 19 rows -- fittable in the linear-
    # algebra sense, which is exactly why the flag has to be loud.
    model3 = comparison.models["Model 3"]
    assert model3.available and model3.underpowered
    assert any("per parameter" in w for w in model3.power_warnings)


def test_compare_models_rejects_empty_and_missing_target():
    with pytest.raises(EconometricsError, match="empty panel"):
        compare_models(pd.DataFrame())
    with pytest.raises(EconometricsError, match="missing required columns"):
        compare_models(macro_panel().drop(columns=["sentiment"]))


def test_compare_models_raises_when_no_model_is_fittable():
    panel = macro_panel(media=False).drop(columns=["unemployment", "pce_yoy"])
    with pytest.raises(EconometricsError, match="no model can be estimated"):
        compare_models(panel)


# ---------------------------------------------------------------------------
# 2. VAR + Granger causality
# ---------------------------------------------------------------------------


def test_granger_detects_known_direction_and_rejects_the_reverse():
    report = granger_media_gap(lead_lag_pair(), max_lags=6)
    assert report.forward is not None and report.reverse is not None
    assert report.forward.direction == "media_tone -> gap"
    assert report.forward.rejects(0.05), report.forward
    assert not report.reverse.rejects(0.05), report.reverse
    assert "media_tone precedes gap" in report.verdict()


def test_granger_detects_the_reverse_when_the_truth_is_reversed():
    """Guards against a transposed `caused`/`causing` argument."""
    frame = lead_lag_pair().rename(columns={"media_tone": "gap", "gap": "media_tone"})
    report = granger_media_gap(frame, max_lags=6)
    assert report.reverse is not None and report.reverse.rejects(0.05)
    assert report.forward is not None and not report.forward.rejects(0.05)
    assert "falsification condition for Hypothesis B" in report.verdict()


def test_granger_result_states_it_is_not_causation():
    report = granger_media_gap(lead_lag_pair(), max_lags=4)
    assert report.interpretation == GRANGER_CAVEAT
    assert "NOT CAUSATION" in report.interpretation
    assert "NOT causation" in report.evidence_class


def test_granger_reports_both_lag_criteria_and_the_stated_maximum():
    report = granger_media_gap(lead_lag_pair(), max_lags=8, lag_criterion="bic")
    assert report.max_lags_considered == 8
    assert report.lag_criterion == "bic"
    assert report.lag_order == report.lag_order_bic or report.lag_order == 1
    assert report.lag_order_aic is not None
    assert 1 <= report.lag_order <= 8


def test_granger_differences_a_unit_root_series_and_says_so():
    rng = np.random.default_rng(7)
    periods = 300
    media = np.cumsum(rng.normal(0, 1.0, periods))  # random walk: I(1)
    gap = np.zeros(periods)
    for t in range(1, periods):
        gap[t] = 0.3 * gap[t - 1] + 0.8 * (media[t - 1] - media[t - 2 if t > 1 else 0])
        gap[t] += rng.normal(0, 1.0)
    frame = pd.DataFrame({"media_tone": media, "gap": gap}, index=_index(periods))

    report = granger_media_gap(frame, max_lags=4)
    check = report.stationarity["media_tone"]
    assert check.stationary is False
    assert check.differenced is True
    assert check.stationary_after is True
    assert "media_tone" in report.differenced_series
    assert "FIRST-DIFFERENCED" in check.note
    assert any("first-differenced" in note for note in report.notes)
    # Differencing costs an observation, and that must show in the counts.
    assert report.n_obs_used < report.n_obs_input


def test_granger_names_dropped_rows():
    frame = lead_lag_pair(periods=200)
    frame.loc[frame.index[50], "gap"] = np.nan
    report = granger_media_gap(frame, max_lags=4)
    assert len(report.dropped_dates) == 1
    assert "missing gap" in report.dropped_reasons[0]
    assert frame.index[50].strftime("%Y-%m") in report.dropped_reasons[0]


def test_granger_refuses_a_19_month_sample_at_six_lags():
    frame = lead_lag_pair(periods=200).iloc[-19:]
    with pytest.raises(EconometricsError, match="cannot support a VAR search"):
        granger_media_gap(frame, max_lags=6)


def test_granger_forced_on_a_19_month_sample_is_flagged():
    frame = lead_lag_pair(periods=200).iloc[-19:]
    report = granger_media_gap(frame, max_lags=6, allow_underpowered=True)
    assert report.underpowered
    assert report.power_warnings
    assert "UNDERPOWERED" in report.verdict()


def test_granger_validates_its_arguments():
    frame = lead_lag_pair(periods=120)
    with pytest.raises(EconometricsError, match="lag_criterion"):
        granger_media_gap(frame, lag_criterion="hqic")
    with pytest.raises(EconometricsError, match="max_lags"):
        granger_media_gap(frame, max_lags=0)
    with pytest.raises(EconometricsError, match="must be different"):
        granger_media_gap(frame, media_column="gap", gap_column="gap")
    with pytest.raises(EconometricsError, match="missing required columns"):
        granger_media_gap(frame, media_column="not_a_column")


# ---------------------------------------------------------------------------
# 3. Rigobon-Sack heteroskedasticity IV
# ---------------------------------------------------------------------------


def test_regime_split_states_its_rule_and_never_folds_in_unassigned_months():
    frame = heteroskedastic_system()
    split = split_volatility_regimes(frame["media_tone"], window=6, quantile=0.5)
    assert "ending at t-1" in split.rule
    assert split.n_unassigned == 6, "window months of history plus the t-1 lag"
    assert split.n_high + split.n_low + split.n_unassigned == len(frame)
    assert set(split.high_dates).isdisjoint(split.low_dates)
    assert set(split.unassigned_dates).isdisjoint(split.high_dates)
    assert split.mean_rolling_var_high > split.mean_rolling_var_low


def test_regime_split_validates_arguments():
    series = heteroskedastic_system(periods=60)["media_tone"]
    with pytest.raises(EconometricsError, match="window"):
        split_volatility_regimes(series, window=1)
    with pytest.raises(EconometricsError, match="quantile"):
        split_volatility_regimes(series, quantile=1.0)


def test_rank_condition_passes_under_known_heteroskedasticity():
    result = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=300)
    assert result.rank.passes
    assert result.rank.variance_shift_detected
    assert result.rank.delta_variance_media > 0
    low, _high = result.rank.delta_variance_media_ci95
    assert low > 0, "the bootstrap CI for the variance shift must exclude zero"
    # Diluted relative to the true 6.25x shock-variance ratio: the lagged
    # window misclassifies months at every block boundary, and the gap shock
    # contributes to the media variance in both regimes. The dilution is the
    # price of a split that cannot manufacture its own variance gap.
    assert result.rank.variance_ratio > 1.5
    # The determinant and its bootstrap distribution are reported either way.
    assert result.rank.bootstrap_determinants.shape == (300,)
    assert np.isfinite(result.rank.determinant)
    assert result.rank.determinant_ci95[0] < result.rank.determinant_ci95[1]


def test_rank_condition_fails_loudly_under_homoskedasticity():
    """No variance shift means no identification, and that is the finding."""
    result = rigobon_sack_iv(homoskedastic_system(), bootstrap_draws=300)
    assert not result.rank.passes
    assert not result.rank.variance_shift_detected
    assert result.transmission_coefficient is None
    assert result.std_error is None and result.ci95 is None
    assert result.identified is False
    assert result.caveats[0].startswith("RANK CONDITION FAILED")
    assert "This is the result, not a failure to compute one" in result.caveats[0]
    assert result.underpowered
    assert "NOT IDENTIFIED" in result.summary()
    # The OLS comparison is still reported, so the report can say what the
    # naive association was even when the IV cannot be formed.
    assert np.isfinite(result.ols_coefficient)


def test_transmission_coefficient_recovers_the_known_structural_parameter():
    truth = 0.5
    result = rigobon_sack_iv(
        heteroskedastic_system(periods=600, b=truth), bootstrap_draws=400
    )
    assert result.identified
    assert result.transmission_coefficient == pytest.approx(truth, abs=0.12)
    # Plain OLS is biased by the simultaneity; the IV should be closer to truth.
    assert abs(result.transmission_coefficient - truth) < abs(
        result.ols_coefficient - truth
    )
    assert result.std_error is not None and result.std_error > 0
    assert result.bootstrap_std_error is not None and result.bootstrap_std_error > 0
    assert result.ci95 is not None and result.ci95[0] < truth < result.ci95[1]
    assert result.p_value is not None and 0.0 <= result.p_value <= 1.0
    assert result.sign_agrees_with_ols is True
    assert result.first_stage_f is not None


def test_order_condition_is_counted_honestly_for_the_real_system():
    result = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=100)
    counting = result.identification
    assert counting.n_endogenous == 2 and counting.n_regimes == 2
    # The brief's N^2 - N - 2K with N=2, K=0 gives 2.
    assert counting.brief_formula_value == 2
    assert counting.brief_formula_satisfied is True
    # ... but it fails for any exogenous regressor at all, which must be visible.
    with_exog = rigobon_sack_iv(
        heteroskedastic_system(), n_exogenous=3, bootstrap_draws=100
    )
    assert with_exog.identification.brief_formula_value == -4
    assert with_exog.identification.brief_formula_satisfied is False
    # The counting that actually governs: 6 moments, 6 parameters, but the
    # identifying difference supplies 3 equations in 4 unknowns.
    assert counting.moments_available == 6
    assert counting.parameters_to_estimate == 6
    assert counting.degrees_of_freedom == 0
    assert counting.difference_equations == 3
    assert counting.difference_unknowns == 4
    assert counting.identified_without_assumption is False
    assert "UNDER-IDENTIFIED" in counting.explanation
    assert "variance" in counting.required_assumption


def test_identification_assumption_travels_with_the_estimate():
    result = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=100)
    joined = " ".join(result.caveats)
    assert "Rigobon-Sack assumption" in joined
    assert "ORDER CONDITION" in joined
    assert "NOT identified here at all" in joined
    assert "CONDITIONAL" in result.evidence_class
    assert any("rank 1" in note.lower() for note in result.rank.notes)


def test_rigobon_sack_refuses_a_19_month_sample():
    frame = heteroskedastic_system(periods=400).iloc[-19:]
    with pytest.raises(EconometricsError, match="per regime"):
        rigobon_sack_iv(frame, bootstrap_draws=100)


def test_rigobon_sack_forced_on_a_19_month_sample_is_flagged():
    frame = heteroskedastic_system(periods=400).iloc[-19:]
    result = rigobon_sack_iv(frame, bootstrap_draws=200, allow_underpowered=True)
    assert result.underpowered
    assert any("SMALL-SAMPLE WARNING" in caveat for caveat in result.caveats)
    assert any("per regime" in warning for warning in result.power_warnings)


def test_rigobon_sack_names_dropped_rows_and_validates_arguments():
    frame = heteroskedastic_system(periods=200)
    frame.loc[frame.index[30], "media_tone"] = np.nan
    result = rigobon_sack_iv(frame, bootstrap_draws=100)
    assert len(result.dropped_dates) == 1
    assert "missing media_tone" in result.dropped_reasons[0]

    with pytest.raises(EconometricsError, match="too few"):
        rigobon_sack_iv(frame, bootstrap_draws=10)
    with pytest.raises(EconometricsError, match="must differ"):
        rigobon_sack_iv(frame, media_column="gap", gap_column="gap")
    with pytest.raises(EconometricsError, match="missing required columns"):
        rigobon_sack_iv(frame, gap_column="nope")
    with pytest.raises(EconometricsError, match="rolling"):
        rigobon_sack_iv(frame.iloc[:8], bootstrap_draws=100)


def test_rigobon_sack_is_deterministic_for_a_fixed_seed():
    first = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=150)
    second = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=150)
    assert first.transmission_coefficient == second.transmission_coefficient
    assert first.ci95 == second.ci95


# ---------------------------------------------------------------------------
# 4. The brief's three-rung ladder and its household/political controls
# ---------------------------------------------------------------------------


class _StubFred:
    """Offline stand-in for `data_loader.FredSource`.

    `series` returns a monthly series for the ids in `payload` and an empty
    series for anything else, which is exactly how the real source signals a
    failed fetch. Nothing here touches the network.
    """

    def __init__(self, payload: dict[str, pd.Series], configured: bool = True):
        self.payload = payload
        self.configured = configured
        self.calls: list[str] = []

    def series(self, series_id: str) -> pd.Series:
        self.calls.append(series_id)
        return self.payload.get(series_id, pd.Series(dtype=float))


def control_payload(periods: int = 200, seed: int = RNG_SEED) -> dict[str, pd.Series]:
    """Synthetic FRED payload for the three household/political series."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2005-01-31", periods=periods, freq="ME")
    steps = np.arange(periods)
    return {
        "USEPUINDXD": pd.Series(
            120.0 + 30.0 * np.sin(steps / 10) + rng.normal(0, 5, periods), index=index
        ),
        "A229RX0": pd.Series(
            40000.0 * np.exp(steps / 600.0) + rng.normal(0, 50, periods), index=index
        ),
        "U6RATE": pd.Series(
            8.0 + 1.5 * np.cos(steps / 14) + rng.normal(0, 0.1, periods), index=index
        ),
    }


def ladder_panel(periods: int = 180, seed: int = RNG_SEED) -> pd.DataFrame:
    """`macro_panel` with media columns and the household/political controls."""
    frame = macro_panel(periods=periods, media=True, seed=seed)
    controls, _ = build_household_political_controls(
        frame.index, fred=_StubFred(control_payload(seed=seed))
    )
    joined = frame.join(controls, how="left")
    return joined.dropna(subset=list(HOUSEHOLD_POLITICAL_FEATURES))


def test_the_ladder_has_all_three_rungs_of_the_brief():
    comparison = compare_model_ladder(ladder_panel())
    assert comparison.brief_ladder == {
        "1. economic indicators only": "Model 2",
        "2. economic indicators plus media": "Model 3",
        "3. economic, media, household or political controls": "Model 4",
    }
    table = comparison.table()
    assert list(table.index) == ["Model 1", "Model 2", "Model 2b", "Model 3", "Model 4"]
    # Nested specifications cannot lose fit as regressors are added.
    assert table.loc["Model 2", "r_squared"] <= table.loc["Model 3", "r_squared"]
    assert table.loc["Model 3", "r_squared"] <= table.loc["Model 4", "r_squared"]
    # Every model is estimated on the same rows or the statistics do not compare.
    assert len({fit.n for fit in comparison.models.values() if fit.available}) == 1


def test_rung_three_is_unavailable_and_named_when_the_controls_are_absent():
    comparison = compare_model_ladder(macro_panel(media=True))
    assert comparison.brief_ladder[
        "3. economic, media, household or political controls"
    ] is None
    model4 = comparison.models["Model 4"]
    assert model4.available is False
    assert set(model4.missing_features) == set(HOUSEHOLD_POLITICAL_FEATURES)
    assert any("rung 3 is not estimated" in note for note in comparison.ladder_notes)
    # Rungs 1 and 2 are unaffected: an absent control must not shorten them.
    assert comparison.brief_ladder["2. economic indicators plus media"] == "Model 3"


def test_the_media_only_model_is_available_separately():
    comparison = compare_models(macro_panel(media=True), include_media_only=True)
    media_only = comparison.models["Model 2b"]
    assert media_only.available
    assert set(media_only.features) == set(MEDIA_FEATURES)
    assert "unemployment" not in media_only.features
    # Defaults keep the old table: no Model 2b, no Model 4.
    plain = compare_models(macro_panel(media=True))
    assert list(plain.models) == ["Model 1", "Model 2", "Model 3"]


def test_election_proximity_is_calendar_only_and_peaks_in_the_election_month():
    index = pd.date_range("2024-01-31", periods=24, freq="ME")
    frame = election_proximity_features(index)
    november_2024 = frame.loc["2024-11-30"]
    assert november_2024["months_to_election"] == 0.0
    assert november_2024["election_proximity"] == pytest.approx(1.0)
    # 2025-11 is twelve months before the 2026 election.
    assert frame.loc["2025-11-30", "months_to_election"] == 12.0
    assert frame["months_to_election"].min() >= 0
    assert frame["election_proximity"].between(0, 1).all()
    # The value of a month cannot depend on which other months are present:
    # that is what makes this control free of look-ahead.
    subset = election_proximity_features(index[5:9])
    pd.testing.assert_frame_equal(subset, frame.iloc[5:9])


def test_control_block_names_every_series_it_could_not_fetch():
    index = pd.date_range("2015-01-31", periods=60, freq="ME")
    frame, report = build_household_political_controls(index, fred=None)
    assert not report.complete
    assert set(report.missing) == set(HOUSEHOLD_POLITICAL_SERIES)
    joined = " ".join(report.missing_reasons)
    for series_id in HOUSEHOLD_POLITICAL_SERIES.values():
        assert series_id in joined
    # The offline control still arrives: a dead FRED key does not remove it.
    assert "election_proximity" in frame.columns
    assert frame["election_proximity"].notna().all()


def test_control_block_builds_the_income_change_from_the_level():
    index = pd.date_range("2015-01-31", periods=60, freq="ME")
    stub = _StubFred(control_payload())
    frame, report = build_household_political_controls(index, fred=stub)
    assert report.complete
    for name in ("policy_uncertainty", "real_dpi_pc", "real_dpi_pc_yoy", "u6_rate"):
        assert name in frame.columns
    # The YoY column must be computed on the raw series, so the first year of
    # the requested index is not lost.
    assert frame["real_dpi_pc_yoy"].notna().all()
    assert set(stub.calls) == set(HOUSEHOLD_POLITICAL_SERIES.values())
    assert report.coverage["policy_uncertainty"] == pytest.approx(1.0)


def test_control_block_validates_its_index():
    with pytest.raises(EconometricsError, match="DatetimeIndex"):
        build_household_political_controls(pd.Index([1, 2, 3]))
    with pytest.raises(EconometricsError, match="empty index"):
        build_household_political_controls(pd.DatetimeIndex([]))
    with pytest.raises(EconometricsError, match="election dates"):
        election_proximity_features(
            pd.date_range("2020-01-31", periods=3, freq="ME"), election_dates=()
        )


def test_an_internally_sparse_regressor_is_dropped_by_name_not_by_deletion():
    """Quarterly GDP that is correctly NOT interpolated must not eat the sample."""
    panel = macro_panel(media=True)
    sparse = panel["real_gdp_qoq"].copy()
    sparse[~sparse.index.month.isin((3, 6, 9, 12))] = np.nan
    panel["real_gdp_qoq"] = sparse
    comparison = compare_models(panel)
    model1 = comparison.models["Model 1"]
    assert model1.available
    assert "real_gdp_qoq" not in model1.features
    assert "real_gdp_qoq" in model1.dropped_features
    assert any("not interpolated" in reason for reason in model1.drop_reasons)
    # The sample survives: the column was dropped, not the rows.
    assert comparison.common_sample_n > len(panel) - 10


def test_a_series_that_merely_starts_late_is_kept():
    panel = macro_panel(media=True)
    for name in MEDIA_FEATURES:
        panel.loc[panel.index[:120], name] = np.nan
    assert column_coverage(panel, "media_tone") < 0.4
    assert column_coverage(panel, "media_tone", within_span=True) == pytest.approx(1.0)
    comparison = compare_models(panel)
    model3 = comparison.models["Model 3"]
    assert model3.available
    assert set(MEDIA_FEATURES) <= set(model3.features)
    assert model3.dropped_features == ()
    # The cost is a shorter common sample, and it is reported.
    assert comparison.common_sample_n == 60
    assert len(comparison.dropped_dates) == 120


def test_framing_carries_a_sign_predicted_in_advance():
    assert PREDICTED_SIGNS["framing"] == "negative"
    panel = macro_panel(media=True)
    # Build the truth with a POSITIVE framing effect, the opposite of the
    # anchoring prediction, and check the table says the sign disagrees.
    panel["sentiment"] = panel["sentiment"] + 30.0 * panel["framing"]
    comparison = compare_models(panel)
    table = predicted_sign_check(comparison)
    row = table.loc["framing"]
    assert row["predicted_sign"] == "negative"
    assert row["coef"] > 0
    assert not row["sign_agrees"]
    assert row["distinguishable_from_zero"]
    assert "anchoring" in row["note"]


# ---------------------------------------------------------------------------
# 5. Seasonality
# ---------------------------------------------------------------------------


def test_seasonality_is_detected_when_it_is_there():
    index = pd.date_range("2010-01-31", periods=144, freq="ME")
    rng = np.random.default_rng(RNG_SEED)
    seasonal = np.where(index.month == 12, 6.0, 0.0) + rng.normal(0, 0.5, len(index))
    report = seasonality_report(pd.Series(seasonal, index=index, name="gap"))
    assert report.seasonal
    assert report.p_value < 0.01
    assert report.amplitude > 4.0
    assert "control monthly seasonality" in report.recommendation
    assert "SEASONAL" in report.summary()


def test_no_seasonality_is_reported_as_no_seasonality():
    index = pd.date_range("2010-01-31", periods=144, freq="ME")
    rng = np.random.default_rng(RNG_SEED)
    report = seasonality_report(
        pd.Series(rng.normal(0, 1, len(index)), index=index), name="noise"
    )
    assert not report.seasonal
    assert "no seasonal control needed" in report.recommendation


def test_seasonality_refuses_a_short_sample_and_flags_a_forced_one():
    index = pd.date_range("2024-01-31", periods=30, freq="ME")
    series = pd.Series(np.arange(30, dtype=float), index=index)
    with pytest.raises(EconometricsError, match="below 36"):
        seasonality_report(series)
    forced = seasonality_report(series, allow_underpowered=True)
    assert forced.underpowered and forced.power_warnings
    with pytest.raises(EconometricsError, match="cannot support eleven"):
        seasonality_report(series.iloc[:18])
    with pytest.raises(EconometricsError, match="DatetimeIndex"):
        seasonality_report(pd.Series([1.0, 2.0, 3.0] * 10))


def test_month_dummies_omit_one_category():
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    dummies = month_dummies(index)
    assert len(dummies.columns) == 11
    assert "month_01" not in dummies.columns
    assert dummies.loc["2020-07-31", "month_07"] == 1.0
    assert dummies.loc["2020-01-31"].sum() == 0.0


def test_seasonal_control_is_reported_and_never_silent():
    panel = macro_panel(media=True)
    always = compare_models(panel, seasonal_control="always")
    assert len(always.seasonal_terms) == 11
    assert any("monthly dummies" in note for note in always.notes)
    assert "month_12" in always.models["Model 1"].features
    auto = compare_models(panel, seasonal_control="auto")
    assert any("seasonality test" in note for note in auto.notes)
    assert compare_models(panel).seasonal_terms == ()
    with pytest.raises(EconometricsError, match="seasonal_control must be"):
        compare_models(panel, seasonal_control="maybe")


# ---------------------------------------------------------------------------
# 6. Rolling-origin forecasts
# ---------------------------------------------------------------------------


def test_rolling_forecast_scores_every_model_on_the_same_origins():
    report = rolling_origin_forecast(
        ladder_panel(), household_political=HOUSEHOLD_POLITICAL_FEATURES,
        include_media_only=True, initial_train=60,
    )
    assert report.n_folds > MIN_FOLDS_FOR_TEST
    folds = {m.n_folds for m in report.models.values() if m.available}
    assert len(folds) == 1, "an RMSE comparison needs identical origins"
    table = report.table()
    assert "Naive (last observed)" in table.index
    assert table.loc["Model 3", "rmse"] > 0
    # Media is in the truth of `macro_panel`, so Model 3 must beat Model 1.
    assert report.models["Model 3"].rmse < report.models["Model 1"].rmse
    assert report.models["Model 3"].dm_statistic > 0
    assert report.models["Model 3"].dm_p_value < 0.1
    assert report.reference_model == "Model 1"
    assert report.best_model in {"Model 3", "Model 4"}


def test_rolling_forecast_fits_on_training_rows_only():
    """The fold's coefficients must come from the rows before the origin."""
    panel = macro_panel(periods=120, media=True)
    report = rolling_origin_forecast(
        panel, media_features=(), initial_train=60, model2_extra=()
    )
    model1 = report.models["Model 1"]
    first_origin = model1.errors.index[0]
    position = panel.index.get_loc(first_origin)
    assert position == 60
    train = panel.iloc[:60]
    import statsmodels.api as sm

    design = sm.add_constant(train.loc[:, list(model1.features)], has_constant="add")
    manual = sm.OLS(train["sentiment"], design).fit()
    row = sm.add_constant(
        panel.iloc[[60]].loc[:, list(model1.features)], has_constant="add"
    ).reindex(columns=design.columns, fill_value=1.0)
    predicted = float(np.asarray(manual.predict(row))[0])
    expected = float(panel["sentiment"].iloc[60]) - predicted
    assert model1.errors.iloc[0] == pytest.approx(expected)


def test_rolling_forecast_refuses_too_few_origins():
    panel = macro_panel(media=True).iloc[:66]
    with pytest.raises(EconometricsError, match="fewer than"):
        rolling_origin_forecast(panel, initial_train=60)
    forced = rolling_origin_forecast(panel, initial_train=60, allow_underpowered=True)
    assert forced.underpowered and forced.power_warnings
    assert forced.n_folds == 6


def test_rolling_forecast_degrades_when_media_is_absent():
    report = rolling_origin_forecast(macro_panel(media=False), initial_train=60)
    assert report.models["Model 3"].available is False
    assert "absent" in (report.models["Model 3"].unavailable_reason or "")
    assert any("NOT the same finding" in n for n in report.models["Model 3"].notes)
    assert report.models["Model 1"].available
    assert report.models["Model 2"].n_folds > 0


def test_rolling_forecast_validates_its_arguments():
    panel = macro_panel(media=True)
    with pytest.raises(EconometricsError, match="empty panel"):
        rolling_origin_forecast(pd.DataFrame())
    with pytest.raises(EconometricsError, match="step must be"):
        rolling_origin_forecast(panel, step=0)
    with pytest.raises(EconometricsError, match="below the minimum"):
        rolling_origin_forecast(panel, initial_train=12)
    with pytest.raises(EconometricsError, match="no origin"):
        rolling_origin_forecast(panel.iloc[:60], initial_train=60)
    with pytest.raises(EconometricsError, match="missing required columns"):
        rolling_origin_forecast(panel.drop(columns=["sentiment"]))


MIN_FOLDS_FOR_TEST = 12


# ---------------------------------------------------------------------------
# 7. Coefficient stability
# ---------------------------------------------------------------------------


def test_stability_reports_a_constant_coefficient_as_stable():
    index = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(RNG_SEED)
    driver = rng.normal(0, 1, len(index))
    frame = pd.DataFrame({"media_tone": driver}, index=index)
    frame["sentiment"] = 80.0 + 5.0 * driver + rng.normal(0, 0.3, len(index))
    report = coefficient_stability(frame, features=("media_tone",), window=60, step=6)
    path = report.paths["media_tone"]
    assert path.sign_flips == 0
    assert path.stable
    assert path.share_covering_mean >= 0.9
    assert path.spread_over_mean_se <= 4.0
    assert path.share_significant == 1.0
    assert report.unstable_features == ()
    assert path.mean == pytest.approx(5.0, abs=0.3)


def test_stability_detects_a_coefficient_that_changes_sign_midway():
    index = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(RNG_SEED)
    driver = rng.normal(0, 1, len(index))
    effect = np.where(np.arange(len(index)) < 120, 6.0, -6.0)
    frame = pd.DataFrame({"media_tone": driver}, index=index)
    frame["sentiment"] = 80.0 + effect * driver + rng.normal(0, 0.3, len(index))
    report = coefficient_stability(frame, features=("media_tone",), window=60, step=6)
    path = report.paths["media_tone"]
    assert not path.stable
    assert path.sign_flips >= 1
    assert "sign change" in path.reason
    assert path.share_covering_mean < 0.9
    assert report.unstable_features == ("media_tone",)
    long = report.paths_frame()
    assert {"ci_lower", "ci_upper", "window_end"} <= set(long.columns)
    assert len(long) == report.n_windows
    assert "reason" in report.table().columns


def test_stability_refuses_windows_that_cannot_carry_inference():
    panel = macro_panel(media=True)
    with pytest.raises(EconometricsError, match="below 36"):
        coefficient_stability(panel, window=24)
    forced = coefficient_stability(panel, window=24, step=12, allow_underpowered=True)
    assert forced.underpowered
    with pytest.raises(EconometricsError, match="cannot identify"):
        coefficient_stability(
            panel, features=tuple(MEDIA_FEATURES), window=4, allow_underpowered=True
        )
    with pytest.raises(EconometricsError, match="are fewer than window"):
        coefficient_stability(panel.iloc[:50], window=60, allow_underpowered=True)
    with pytest.raises(EconometricsError, match="fewer than"):
        coefficient_stability(panel.iloc[:62], window=60, step=1)
    with pytest.raises(EconometricsError, match="no requested regressor"):
        coefficient_stability(panel, features=("nothing_here",))


# ---------------------------------------------------------------------------
# 8. Specification sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_reports_every_cell_of_a_stated_grid():
    panel = macro_panel(media=True)
    report = specification_sensitivity(
        panel,
        focus_feature="media_tone",
        starts=("2008-01-01", "2012-01-01"),
        hac_lag_grid=(6, 12),
    )
    # 2 starts x 2 lags x (debt in/out) x (seasonal yes/no)
    assert report.n_cells == 16
    assert report.n_feasible == 16
    assert set(report.table().columns) >= {"coef", "p_value", "significant_5pct"}
    # The truth has media tone at +12.0, so every cell must find it.
    assert report.share_significant == 1.0
    assert report.sign_stable
    low, high = report.coef_range
    assert low > 0 and high > 0
    assert "does not depend on these choices" in report.verdict()
    assert any("debt-service" in line for line in report.grid_description)


def test_sensitivity_keeps_infeasible_cells_with_a_reason():
    panel = macro_panel(media=False)
    report = specification_sensitivity(
        panel, focus_feature="media_tone", starts=("2008-01-01",), hac_lag_grid=(12,)
    )
    assert report.n_feasible == 0
    assert all(cell.feasible is False for cell in report.cells)
    assert all("in no estimated model" in cell.reason for cell in report.cells)
    assert "unanswered" in report.verdict()


def test_sensitivity_collapses_the_debt_dimension_when_the_column_is_absent():
    panel = macro_panel(media=True).drop(columns=["debt_service"])
    report = specification_sensitivity(
        panel, focus_feature="media_tone", starts=("2008-01-01",), hac_lag_grid=(12,)
    )
    assert report.n_cells == 2  # seasonal yes/no only
    assert any("debt-service dimension collapses" in note for note in report.notes)
    with pytest.raises(EconometricsError, match="at least one cell"):
        specification_sensitivity(panel, starts=())


def test_sensitivity_says_so_when_the_result_depends_on_the_specification():
    """A result present only in the long window must not read as robust."""
    panel = macro_panel(media=True)
    rng = np.random.default_rng(RNG_SEED)
    # Give `political_attribution` an effect in the first half only.
    half = len(panel) // 2
    effect = np.concatenate([np.full(half, 60.0), np.zeros(len(panel) - half)])
    panel["sentiment"] = panel["sentiment"] + effect * panel["political_attribution"]
    panel["sentiment"] += rng.normal(0, 0.1, len(panel))
    report = specification_sensitivity(
        panel,
        focus_feature="political_attribution",
        starts=("2008-01-01", "2016-01-01"),
        hac_lag_grid=(12,),
        seasonal_options=(False,),
    )
    shares = report.share_significant
    assert 0.0 < shares < 1.0
    assert "DEPENDS ON THE SPECIFICATION" in report.verdict()


# ---------------------------------------------------------------------------
# 9. Multiple testing
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_and_holm_match_the_hand_calculation():
    tests = [
        HypothesisTest(label="a", family="f", p_value=0.010),
        HypothesisTest(label="b", family="f", p_value=0.040),
        HypothesisTest(label="c", family="f", p_value=0.300),
        HypothesisTest(label="d", family="f", p_value=0.500),
    ]
    report = multiple_testing_correction(tests)
    # Holm: 4*0.01=0.04, 3*0.04=0.12, 2*0.30=0.60, 1*0.50 -> 0.60 by monotonicity.
    assert report.holm_adjusted["a"] == pytest.approx(0.04)
    assert report.holm_adjusted["b"] == pytest.approx(0.12)
    assert report.holm_adjusted["c"] == pytest.approx(0.60)
    assert report.holm_adjusted["d"] == pytest.approx(0.60)
    # BH: 4*0.01/1=0.04, 4*0.04/2=0.08, 4*0.30/3=0.40, 4*0.50/4=0.50.
    assert report.bh_adjusted["a"] == pytest.approx(0.04)
    assert report.bh_adjusted["b"] == pytest.approx(0.08)
    assert report.bh_adjusted["c"] == pytest.approx(0.40)
    assert report.bh_adjusted["d"] == pytest.approx(0.50)
    assert report.raw_significant == ("a", "b")
    assert report.bh_significant == ("a",)
    assert report.holm_significant == ("a",)
    assert report.lost_to_correction == ("b",)
    assert report.survives("a") and not report.survives("b")


def test_adjusted_values_are_monotone_and_capped_at_one():
    rng = np.random.default_rng(RNG_SEED)
    values = rng.uniform(0, 1, 30)
    tests = [
        HypothesisTest(label=f"t{i}", family="f", p_value=float(p))
        for i, p in enumerate(values)
    ]
    report = multiple_testing_correction(tests)
    frame = report.table().sort_values("p_raw")
    assert frame["p_holm"].is_monotonic_increasing
    assert frame["q_bh"].is_monotonic_increasing
    assert frame["p_holm"].max() <= 1.0 and frame["q_bh"].max() <= 1.0
    # BH is never stricter than Holm.
    assert (frame["q_bh"] <= frame["p_holm"] + 1e-12).all()
    # The raw values stay in the table so the adjustment can be audited.
    assert frame["p_raw"].min() == pytest.approx(values.min())


def test_a_family_of_one_is_labelled_as_such():
    report = multiple_testing_correction(
        [HypothesisTest(label="only", family="f", p_value=0.02)]
    )
    assert report.bh_adjusted["only"] == pytest.approx(0.02)
    assert report.family_size == 1
    assert any("family of one" in note for note in report.notes)


def test_multiple_testing_validates_its_input():
    with pytest.raises(EconometricsError, match="at least one test"):
        multiple_testing_correction([])
    with pytest.raises(EconometricsError, match="unique"):
        multiple_testing_correction(
            [
                HypothesisTest(label="x", family="f", p_value=0.1),
                HypothesisTest(label="x", family="f", p_value=0.2),
            ]
        )
    with pytest.raises(EconometricsError, match="alpha must be"):
        multiple_testing_correction(
            [HypothesisTest(label="x", family="f", p_value=0.1)], alpha=1.5
        )
    with pytest.raises(EconometricsError, match="not in"):
        HypothesisTest(label="bad", family="f", p_value=1.4)
    with pytest.raises(EconometricsError, match="not in"):
        HypothesisTest(label="nan", family="f", p_value=float("nan"))
    report = multiple_testing_correction(
        [HypothesisTest(label="x", family="f", p_value=0.1)]
    )
    with pytest.raises(EconometricsError, match="not in this family"):
        report.survives("nope")


def test_the_family_is_collected_from_the_result_objects():
    comparison = compare_models(macro_panel(media=True), include_media_only=True)
    granger = granger_media_gap(lead_lag_pair(), max_lags=4)
    family = collect_media_test_family(comparison, granger_reports=[granger])
    labels = [test.label for test in family]
    # One model only: the same regressor in a nested model is the same
    # hypothesis, and counting it twice would inflate the family.
    models_used = {label.split(":")[0] for label in labels if label.startswith("Model")}
    assert models_used == {"Model 3"}
    assert len([l for l in labels if l.startswith("Model 3")]) == len(MEDIA_FEATURES)
    # Both Granger directions are family members, because both were run.
    assert sum(1 for l in labels if l.startswith("Granger")) == 2
    assert {test.family for test in family} == {
        "media coefficient (HAC)",
        "Granger precedence",
    }
    chosen = collect_media_test_family(comparison, model_name="Model 2b")
    assert all(test.label.startswith("Model 2b") for test in chosen)


def test_one_raw_hit_in_a_family_of_twelve_does_not_survive():
    """The project's own pattern: p = 0.013 among twelve declared tests."""
    tests = [HypothesisTest(label="hit", family="coefficient", p_value=0.0128)]
    tests += [
        HypothesisTest(label=f"null{i}", family="other", p_value=p)
        for i, p in enumerate(
            [0.514, 0.137, 0.005, 0.632, 0.194, 0.062, 0.167, 0.590, 0.131, 0.887, 0.717]
        )
    ]
    report = multiple_testing_correction(tests)
    assert report.family_size == 12
    assert "hit" in report.raw_significant
    assert not report.survives("hit")
    assert not report.survives("hit", method="holm")
    assert "hit" in report.lost_to_correction
    assert "does NOT survive BH" in report.summary()


# ---------------------------------------------------------------------------
# 10. Reverse causality as output
# ---------------------------------------------------------------------------


def test_reverse_causality_finds_the_direction_that_is_in_the_data():
    frame = lead_lag_pair(periods=300)
    report = reverse_causality_evidence(
        frame, media_columns=("media_tone",), gap_column="gap", max_lags=4
    )
    item = report.evidence["media_tone"]
    assert item.tested
    assert item.forward_p < 0.05
    assert item.reverse_p > 0.05
    assert "coverage precedes the gap" in item.verdict
    table = report.table()
    assert table.loc["media_tone", "p_media_to_gap"] == pytest.approx(item.forward_p)
    assert np.isfinite(table.loc["media_tone", "corr_contemporaneous"])


def test_reverse_causality_separates_supply_from_exposure():
    report = reverse_causality_evidence(
        lead_lag_pair(periods=300),
        media_columns=("media_tone",),
        gap_column="gap",
        max_lags=4,
    )
    assert report.separable is False
    assert len(report.separation_requirements) >= 3
    assert any("exposure" in item.lower() for item in report.separation_requirements)
    verdicts = report.narrative_verdicts
    assert "EXPOSURE" in verdicts["consumers_seek_negative_content"]
    assert "SUPPLY" in verdicts["publishers_respond_to_demand"]
    assert "common_shock" in verdicts
    assert set(report.narratives) == set(verdicts)
    assert "NOT separable" in report.summary()


def test_reverse_causality_keeps_a_row_for_a_measure_it_cannot_test():
    frame = lead_lag_pair(periods=300)
    frame["topic_salience"] = np.nan
    frame.loc[frame.index[:2], "topic_salience"] = [0.1, 0.2]
    report = reverse_causality_evidence(
        frame,
        media_columns=("media_tone", "topic_salience"),
        gap_column="gap",
        max_lags=4,
    )
    item = report.evidence["topic_salience"]
    assert item.tested is False
    assert "2 months carry both" in item.reason
    assert "not tested" in item.verdict


def test_reverse_causality_refuses_when_no_media_measure_exists():
    frame = lead_lag_pair(periods=120).drop(columns=["media_tone"])
    with pytest.raises(EconometricsError, match="none of the media columns"):
        reverse_causality_evidence(frame, gap_column="gap")
    with pytest.raises(EconometricsError, match="missing required columns"):
        reverse_causality_evidence(lead_lag_pair(), gap_column="absent")
    with pytest.raises(EconometricsError, match="max_lead"):
        reverse_causality_evidence(
            lead_lag_pair(), media_columns=("media_tone",), max_lead=0
        )


def test_an_underpowered_granger_test_is_recorded_not_hidden():
    frame = lead_lag_pair(periods=300).iloc[-19:]
    report = reverse_causality_evidence(
        frame, media_columns=("media_tone",), gap_column="gap", max_lags=6
    )
    item = report.evidence["media_tone"]
    assert item.tested is False
    assert "cannot support" in item.reason
    # The correlations are still reported: they do not need the VAR.
    assert np.isfinite(item.contemporaneous_correlation)


# ---------------------------------------------------------------------------
# 11. The causal-design proposal
# ---------------------------------------------------------------------------


def test_three_designs_are_proposed_and_one_is_this_project():
    proposals = causal_design_proposals()
    assert len(proposals) == 3
    assert all(isinstance(item, CausalDesignProposal) for item in proposals)
    classes = [item.design_class for item in proposals]
    assert classes == ["experiment", "quasi-experiment", "observational + design proposal"]
    observational = proposals[-1]
    assert observational.feasible_now is True
    assert "THIS PROJECT'S CATEGORY" in observational.status
    assert "exposure data" in observational.blocking_constraint
    # The others are infeasible, and each says what blocks it.
    for item in proposals[:2]:
        assert item.feasible_now is False
        assert item.blocking_constraint


def test_every_design_names_its_estimand_assumptions_threats_and_falsifiers():
    for proposal in causal_design_proposals():
        assert proposal.target_estimand
        assert proposal.identifying_assumptions
        assert proposal.required_data
        assert proposal.threats
        assert proposal.falsification_tests
        assert proposal.transportability
        assert "SUPPLY, NOT EXPOSURE" in proposal.exposure_gap
    observational = causal_design_proposals()[-1]
    assert "reverse causality" in observational.threats
    assert "common shocks" in observational.threats
    assert "selection" in observational.threats
    assert "transportability" in observational.threats
    assert "multiplicity" in observational.threats


def test_the_proposal_states_whether_the_iv_identified():
    without = causal_design_proposals(identified_iv=False)[-1]
    with_iv = causal_design_proposals(identified_iv=True)[-1]
    assert "did NOT identify" in without.status
    assert "returned an estimate" in with_iv.status
    frame = causal_design_table()
    assert len(frame) == 3
    assert {"target_estimand", "threats", "falsification_tests"} <= set(frame.columns)
    assert frame.loc["Observational association with a plan for identification", "feasible_now"]


# ---------------------------------------------------------------------------
# 12. Supply is not exposure: the statement travels on every result
# ---------------------------------------------------------------------------


def test_every_result_object_carries_the_exposure_gap_statement():
    panel = ladder_panel()
    comparison = compare_model_ladder(panel)
    granger = granger_media_gap(lead_lag_pair(), max_lags=4)
    iv = rigobon_sack_iv(heteroskedastic_system(), bootstrap_draws=100)
    forecast = rolling_origin_forecast(panel, initial_train=60)
    stability = coefficient_stability(panel, window=60, step=6)
    sensitivity = specification_sensitivity(
        panel, focus_feature="media_tone", starts=("2008-01-01",), hac_lag_grid=(12,)
    )
    reverse = reverse_causality_evidence(
        lead_lag_pair(), media_columns=("media_tone",), gap_column="gap", max_lags=4
    )
    for result in (
        comparison,
        granger,
        iv,
        forecast,
        stability,
        sensitivity,
        reverse,
    ):
        assert result.exposure_gap == EXPOSURE_GAP_STATEMENT
        assert "NOT MADE" in result.exposure_gap
