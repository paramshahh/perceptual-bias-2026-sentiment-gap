"""Tests for the baseline fit, prediction reporting and gap arithmetic.

The NaN-handling tests pin review findings where the code returned a
confident-looking number from data that had never been tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.baseline_model import (
    BASELINE_FEATURES,
    ModelError,
    fit_baseline,
    gap_summary,
    predict,
    sentiment_gap,
)
from perceptual_bias.data_loader import build_monthly_panel, holdout_slice, training_slice

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "Copy_of_Perceptual_Bias_Data.csv"

RNG = np.random.default_rng(20261014)


def synthetic_panel(periods: int = 180, noise: float = 1.0) -> pd.DataFrame:
    """A panel with a known linear truth, so coefficients can be checked."""
    index = pd.date_range("2010-01-31", periods=periods, freq="ME")
    unemployment = 5.0 + np.sin(np.arange(periods) / 12) * 1.5
    pce_yoy = 2.0 + np.cos(np.arange(periods) / 18) * 0.8
    real_gdp = 2.5 + RNG.normal(0, 0.4, periods)
    sentiment = (
        120.0 - 4.0 * unemployment - 9.0 * pce_yoy - 0.2 * real_gdp
        + RNG.normal(0, noise, periods)
    )
    frame = pd.DataFrame(
        {
            "unemployment": unemployment,
            "pce_yoy": pce_yoy,
            "real_gdp_qoq": real_gdp,
            "sentiment": sentiment,
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


class TestFitBaseline:
    def test_recovers_known_coefficients(self) -> None:
        fit = fit_baseline(synthetic_panel(noise=0.2))
        assert fit.params["unemployment"] == pytest.approx(-4.0, abs=0.15)
        assert fit.params["pce_yoy"] == pytest.approx(-9.0, abs=0.3)
        assert fit.r_squared > 0.95

    def test_missing_feature_raises_modelerror_not_keyerror(self) -> None:
        """Pins the review's LOW finding: callers catch ModelError only."""
        frame = synthetic_panel().drop(columns=["pce_yoy"])
        with pytest.raises(ModelError, match="missing required columns"):
            fit_baseline(frame)

    def test_all_rows_incomplete_raises(self) -> None:
        frame = synthetic_panel()
        frame["pce_yoy"] = np.nan
        with pytest.raises(ModelError, match="no complete rows"):
            fit_baseline(frame)

    def test_incomplete_rows_are_dropped_and_reported(self) -> None:
        frame = synthetic_panel()
        frame.loc[frame.index[5], "unemployment"] = np.nan
        frame.loc[frame.index[9], "sentiment"] = np.nan
        fit = fit_baseline(frame)
        assert fit.n_train == len(frame) - 2
        assert len(fit.dropped_train_dates) == 2
        assert any("dropped 2 incomplete training rows" in note for note in fit.notes)

    def test_ridge_cv_fallback_is_declared_when_it_cannot_run(self) -> None:
        """Pins the review's LOW finding: the note must not claim CV ran."""
        fit = fit_baseline(synthetic_panel(periods=5), cv_splits=5)
        assert fit.ridge_cv_ran is False
        assert fit.ridge_alpha == pytest.approx(1.0)
        assert any("fallback default" in note for note in fit.notes)

    def test_hac_errors_are_used(self) -> None:
        fit = fit_baseline(synthetic_panel())
        assert "HAC" in str(fit.ols_result.cov_type).upper()
        assert any("Newey-West" in note for note in fit.notes)

    def test_equation_renders_every_feature(self) -> None:
        fit = fit_baseline(synthetic_panel())
        equation = fit.equation()
        for name in BASELINE_FEATURES:
            assert name in equation


class TestPredict:
    def test_dropped_rows_are_named_with_reasons(self) -> None:
        """Pins the review's MEDIUM finding about silent row drops."""
        train = synthetic_panel()
        future = synthetic_panel(periods=6).copy()
        future.index = pd.date_range("2025-01-31", periods=6, freq="ME")
        future.loc[future.index[2], "pce_yoy"] = np.nan
        _, report = predict(fit_baseline(train), future)
        assert report.predicted_rows == 5
        assert len(report.dropped_dates) == 1
        assert "missing pce_yoy" in report.dropped_reasons[0]
        assert "dropped" in report.summary()

    def test_prediction_interval_brackets_the_point_estimate(self) -> None:
        fit = fit_baseline(synthetic_panel())
        frame, _ = predict(fit, synthetic_panel(periods=12))
        assert (frame["pi_lower"] < frame["predicted"]).all()
        assert (frame["predicted"] < frame["pi_upper"]).all()

    def test_no_usable_rows_raises(self) -> None:
        fit = fit_baseline(synthetic_panel())
        frame = synthetic_panel(periods=4)
        frame["unemployment"] = np.nan
        with pytest.raises(ModelError, match="no complete rows"):
            predict(fit, frame)


class TestSentimentGap:
    def test_gap_is_observed_minus_predicted(self) -> None:
        fit = fit_baseline(synthetic_panel())
        gaps, _ = sentiment_gap(fit, synthetic_panel(periods=12))
        assert gaps["gap"].equals(gaps["observed"] - gaps["predicted"])

    def test_missing_observation_is_not_scored_as_inside_the_interval(self) -> None:
        """Pins the review's MEDIUM finding.

        A NaN observation compared against both interval bounds yields False on
        each side, which previously read as "inside the interval" — i.e. the
        model appeared to explain a month that was never tested.
        """
        fit = fit_baseline(synthetic_panel())
        future = synthetic_panel(periods=6)
        future["sentiment"] = np.nan
        gaps, _ = sentiment_gap(fit, future)
        assert gaps["outside_pi"].isna().all()
        with pytest.raises(ModelError, match="every gap is missing"):
            gap_summary(gaps)

    def test_partially_missing_observations_are_counted_honestly(self) -> None:
        fit = fit_baseline(synthetic_panel())
        future = synthetic_panel(periods=6)
        future.loc[future.index[:2], "sentiment"] = np.nan
        gaps, _ = sentiment_gap(fit, future)
        summary = gap_summary(gaps)
        assert summary["n_months"] == 6.0
        assert summary["n_tested"] == 4.0


class TestGapSummary:
    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ModelError, match="empty gap frame"):
            gap_summary(pd.DataFrame())

    def test_missing_columns_raise_modelerror(self) -> None:
        with pytest.raises(ModelError, match="missing required columns"):
            gap_summary(pd.DataFrame({"gap": [1.0]}))

    def test_non_datetime_index_does_not_crash(self) -> None:
        """Pins the review's MEDIUM finding about `worst.strftime`."""
        frame = pd.DataFrame(
            {
                "observed": [50.0, 40.0],
                "predicted": [70.0, 75.0],
                "gap": [-20.0, -35.0],
                "outside_pi": pd.array([True, True], dtype="boolean"),
            },
            index=["a", "b"],
        )
        summary = gap_summary(frame)
        assert summary["worst_month"] == "b"
        assert summary["worst_gap"] == pytest.approx(-35.0)

    def test_identifies_the_worst_month(self) -> None:
        fit = fit_baseline(synthetic_panel())
        future = synthetic_panel(periods=6)
        future.loc[future.index[3], "sentiment"] -= 30.0
        gaps, _ = sentiment_gap(fit, future)
        summary = gap_summary(gaps)
        assert summary["worst_month"] == future.index[3].strftime("%Y-%m")


@pytest.mark.skipif(not REAL_CSV.exists(), reason="provided CSV not present")
class TestAgainstProvidedData:
    """Regression tests tying the pipeline to the spec's published benchmark."""

    def test_reproduces_the_may_2026_sentiment_gap(self) -> None:
        panel, _ = build_monthly_panel(REAL_CSV)
        fit = fit_baseline(training_slice(panel))
        gaps, _ = sentiment_gap(fit, holdout_slice(panel))
        row = gaps.loc["2026-05-31"]
        assert row["observed"] == pytest.approx(44.8)
        # The brief's own benchmark is a predicted ~77.8 and a -33.0 gap, which
        # it computed on a contemporaneous panel. Under the real-time alignment
        # the brief requires, the same month gives 78.44 and -33.64: the month
        # the brief names, in the direction it names, about 0.6 points further
        # out. The tolerance is 1.0 so that this test pins the real-time number
        # and still demonstrates agreement with the published benchmark.
        assert row["predicted"] == pytest.approx(78.44, abs=0.05)
        assert row["gap"] == pytest.approx(-33.64, abs=0.05)
        assert row["predicted"] == pytest.approx(77.8, abs=1.0)

    def test_training_window_is_the_specified_180_months(self) -> None:
        panel, _ = build_monthly_panel(REAL_CSV)
        fit = fit_baseline(training_slice(panel))
        assert fit.n_train == 180
        assert fit.train_start.strftime("%Y-%m") == "2010-01"
        assert fit.train_end.strftime("%Y-%m") == "2024-12"

    def test_almost_every_holdout_month_falls_outside_the_interval(self) -> None:
        """The defensible form of "the macro model broke down"."""
        panel, _ = build_monthly_panel(REAL_CSV)
        fit = fit_baseline(training_slice(panel))
        gaps, _ = sentiment_gap(fit, holdout_slice(panel))
        summary = gap_summary(gaps)
        assert summary["share_outside_pi"] > 0.9
        assert summary["mean_gap"] < -20.0
