"""Tests for CSV parsing, timing alignment and the price-level fallback.

Several of these pin bugs found in review rather than merely exercising the
happy path — the quarterly alignment and the price-level sawtooth in
particular, both of which produced plausible-looking numbers while being wrong.

The timing tests pin the two rules of the brief that the first panel broke:

* No row may hold an invented intra-quarter GDP observation.
* No row may hold a value that was published after that row's own month.

Every test runs offline. The FRED stub never makes a call.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.data_loader import (
    PUBLICATION_LAGS,
    AlignmentError,
    DataError,
    FredSource,
    PanelReport,
    PublicationLag,
    _parse_csv_date,
    align_real_time,
    approximate_price_level,
    build_monthly_panel,
    expand_quarterly,
    load_provided_csv,
    quarterly_gdp_vintages,
    quarterly_yoy_growth,
)

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "Copy_of_Perceptual_Bias_Data.csv"

# Minimal stand-in for the provided file: same block offsets, newest-first,
# blank spacer columns, monthly series plus one quarterly series.
SYNTHETIC_CSV = (
    "Date,Unemployment,,Date,Consumer Sentiment,,,Date,PCE YoY,,Date,Real GDP QoQ Annualized\n"
    "3/31/26,4.0,,3/31/26,50.0,,,3/31/26,3.0,,3/31/26,2.0\n"
    "2/28/26,4.1,,2/28/26,51.0,,,2/28/26,3.1,,12/31/25,1.0\n"
    "1/31/26,4.2,,1/31/26,52.0,,,1/31/26,3.2,,,\n"
)


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.csv"
    path.write_text(SYNTHETIC_CSV, encoding="utf-8")
    return path


class TestParseCsvDate:
    def test_two_digit_years_resolve_either_side_of_the_cutoff(self) -> None:
        assert _parse_csv_date("9/30/26") == date(2026, 9, 30)
        assert _parse_csv_date("1/31/78") == date(1978, 1, 31)

    def test_blank_and_malformed_values_return_none(self) -> None:
        assert _parse_csv_date("") is None
        assert _parse_csv_date("   ") is None
        assert _parse_csv_date("not-a-date") is None
        assert _parse_csv_date("13/45/26") is None  # impossible month/day

    def test_four_digit_years_are_taken_literally(self) -> None:
        assert _parse_csv_date("6/15/2051") == date(2051, 6, 15)


class TestLoadProvidedCsv:
    def test_parses_all_four_blocks_to_month_end(self, synthetic_csv: Path) -> None:
        frame = load_provided_csv(synthetic_csv)
        assert list(frame.columns) == ["unemployment", "sentiment", "pce_yoy", "real_gdp_qoq"]
        assert frame.loc["2026-01-31", "unemployment"] == pytest.approx(4.2)
        assert frame.loc["2026-03-31", "sentiment"] == pytest.approx(50.0)
        # The quarterly block keeps its native sparsity at this stage.
        assert frame["real_gdp_qoq"].notna().sum() == 2

    def test_missing_file_raises_dataerror(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            load_provided_csv(tmp_path / "absent.csv")

    def test_header_only_file_raises_dataerror(self, tmp_path: Path) -> None:
        path = tmp_path / "header_only.csv"
        path.write_text(SYNTHETIC_CSV.splitlines()[0] + "\n", encoding="utf-8")
        with pytest.raises(DataError, match="no data rows"):
            load_provided_csv(path)

    def test_duplicate_month_keeps_the_newest_vintage(self, tmp_path: Path, caplog) -> None:
        """Blocks are newest-first, so the first row seen for a month must win."""
        path = tmp_path / "dupes.csv"
        rows = SYNTHETIC_CSV.splitlines()
        # Insert a stale mid-month vintage for March 2026 *after* the 3/31 row.
        rows.insert(2, "3/15/26,9.9,,3/15/26,99.0,,,3/15/26,9.9,,,")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            frame = load_provided_csv(path)
        assert frame.loc["2026-03-31", "unemployment"] == pytest.approx(4.0)
        assert "multiple observations" in caplog.text

    def test_non_numeric_value_is_skipped_with_a_warning(self, tmp_path: Path, caplog) -> None:
        path = tmp_path / "junk.csv"
        text = SYNTHETIC_CSV.replace("1/31/26,4.2", "1/31/26,n/a")
        path.write_text(text, encoding="utf-8")
        with caplog.at_level("WARNING"):
            frame = load_provided_csv(path)
        assert pd.isna(frame.loc["2026-01-31", "unemployment"])
        assert "non-numeric" in caplog.text

    def test_bom_prefixed_file_still_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.csv"
        path.write_text("﻿" + SYNTHETIC_CSV, encoding="utf-8")
        frame = load_provided_csv(path)
        assert frame.loc["2026-01-31", "unemployment"] == pytest.approx(4.2)


class TestExpandQuarterly:
    """The non-compliant legacy function.

    `expand_quarterly` stays in the module because `BRAINSTORMING.md` records it
    and the report explains the change away from it. These tests keep its old
    behaviour fixed, so the comparison in the report stays reproducible. They do
    not say that the function is correct for the brief: it is not, and
    `TestRealTimeAlignment` pins the replacement.
    """

    def test_quarter_end_value_covers_its_own_three_months(self) -> None:
        """Pins the review's HIGH finding: 3/31 is Q1, so it owns Jan-Mar.

        A naive ffill would have put it on Mar-May, leaving January and February
        carrying the previous quarter.
        """
        values = pd.Series(
            {pd.Timestamp("2010-03-31"): 1.7, pd.Timestamp("2010-06-30"): 3.9}
        )
        index = pd.date_range("2010-01-31", "2010-07-31", freq="ME")
        expanded = expand_quarterly(values, index)
        assert list(expanded.loc["2010-01-31":"2010-03-31"]) == [1.7, 1.7, 1.7]
        assert list(expanded.loc["2010-04-30":"2010-06-30"]) == [3.9, 3.9, 3.9]
        assert pd.isna(expanded.loc["2010-07-31"])

    def test_a_missing_quarter_stays_missing(self) -> None:
        values = pd.Series({pd.Timestamp("2010-03-31"): 1.7})
        index = pd.date_range("2010-01-31", "2010-09-30", freq="ME")
        expanded = expand_quarterly(values, index)
        assert expanded.notna().sum() == 3
        assert expanded.loc["2010-04-30":"2010-09-30"].isna().all()

    def test_empty_input_yields_all_missing(self) -> None:
        index = pd.date_range("2010-01-31", "2010-06-30", freq="ME")
        expanded = expand_quarterly(pd.Series(dtype=float), index)
        assert expanded.isna().all()


class TestPublicationLagTable:
    """The lag table is the documented assumption, so it is tested as one."""

    def test_the_documented_lags_are_the_applied_lags(self) -> None:
        """A grader reads these numbers in the module; the panel must use them."""
        assert PUBLICATION_LAGS["unemployment"].lag_days == 7
        assert PUBLICATION_LAGS["pce_yoy"].lag_days == 28
        assert PUBLICATION_LAGS["real_gdp_qoq"].lag_days == 30
        assert PUBLICATION_LAGS["sentiment"].lag_days == 0

    def test_every_lag_entry_names_its_source(self) -> None:
        for name, lag in PUBLICATION_LAGS.items():
            assert lag.column == name
            assert lag.source.strip()
            assert lag.note.strip()
            assert lag.lag_days >= 0
            assert lag.max_stale_days >= 0

    def test_survey_values_never_carry_forward(self) -> None:
        """A survey answer for month t is only valid for month t."""
        assert PUBLICATION_LAGS["sentiment"].max_stale_days == 0
        assert PUBLICATION_LAGS["inflation_expect_1y"].max_stale_days == 0

    def test_publication_date_is_the_period_end_plus_the_lag(self) -> None:
        lag = PUBLICATION_LAGS["real_gdp_qoq"]
        assert lag.publication_date(pd.Timestamp("2026-03-31")) == pd.Timestamp("2026-04-30")


class TestRealTimeAlignment:
    """`align_real_time` is the compliant replacement for `expand_quarterly`."""

    @staticmethod
    def _gdp_lag() -> PublicationLag:
        return PUBLICATION_LAGS["real_gdp_qoq"]

    def test_quarterly_gdp_becomes_a_step_function_of_published_estimates(self) -> None:
        """Q1 is published about 30 days after Q1 ends, so it owns April to June."""
        values = pd.Series(
            {
                pd.Timestamp("2010-03-31"): 1.7,
                pd.Timestamp("2010-06-30"): 3.9,
            }
        )
        index = pd.date_range("2010-03-31", "2010-09-30", freq="ME")
        aligned, _ = align_real_time(values, index, self._gdp_lag())
        # March itself cannot know Q1: the advance estimate is late April.
        assert pd.isna(aligned.loc["2010-03-31"])
        assert list(aligned.loc["2010-04-30":"2010-06-30"]) == [1.7, 1.7, 1.7]
        assert list(aligned.loc["2010-07-31":"2010-09-30"]) == [3.9, 3.9, 3.9]

    def test_no_month_holds_an_invented_gdp_value(self) -> None:
        """Every monthly GDP entry must equal one published quarterly estimate."""
        values = pd.Series(
            {
                pd.Timestamp("2010-03-31"): 1.7,
                pd.Timestamp("2010-06-30"): 3.9,
                pd.Timestamp("2010-09-30"): 2.4,
            }
        )
        index = pd.date_range("2010-01-31", "2011-03-31", freq="ME")
        aligned, _ = align_real_time(values, index, self._gdp_lag())
        published = set(values.to_numpy(dtype=float))
        assert set(aligned.dropna().to_numpy(dtype=float)) <= published

    def test_a_value_published_after_the_month_never_reaches_that_month(self) -> None:
        """The core look-ahead test, run over every month of the index."""
        values = pd.Series(
            {
                pd.Timestamp("2010-03-31"): 1.7,
                pd.Timestamp("2010-06-30"): 3.9,
                pd.Timestamp("2010-09-30"): 2.4,
            }
        )
        index = pd.date_range("2010-01-31", "2011-03-31", freq="ME")
        lag = self._gdp_lag()
        aligned, provenance = align_real_time(values, index, lag)
        for stamp, value in aligned.dropna().items():
            period_end = provenance.loc[stamp, "observation_period_end"]
            published = provenance.loc[stamp, "publication_date"]
            assert published <= stamp
            assert published == lag.publication_date(period_end)
            assert value == pytest.approx(values.loc[period_end])

    def test_monthly_series_lag_by_one_month(self) -> None:
        """Month t cannot see its own unemployment rate: BLS publishes in t+1."""
        values = pd.Series(
            {
                pd.Timestamp("2010-01-31"): 9.8,
                pd.Timestamp("2010-02-28"): 9.8,
                pd.Timestamp("2010-03-31"): 9.9,
            }
        )
        index = pd.date_range("2010-01-31", "2010-04-30", freq="ME")
        aligned, _ = align_real_time(values, index, PUBLICATION_LAGS["unemployment"])
        assert pd.isna(aligned.loc["2010-01-31"])
        assert aligned.loc["2010-02-28"] == pytest.approx(9.8)
        assert aligned.loc["2010-04-30"] == pytest.approx(9.9)

    def test_a_stale_value_is_dropped_instead_of_carried_forward(self) -> None:
        """A missing release must stay missing, not become an old number."""
        values = pd.Series({pd.Timestamp("2010-01-31"): 9.8})
        index = pd.date_range("2010-02-28", "2010-06-30", freq="ME")
        aligned, provenance = align_real_time(values, index, PUBLICATION_LAGS["unemployment"])
        assert aligned.loc["2010-02-28"] == pytest.approx(9.8)
        assert aligned.loc["2010-03-31":"2010-06-30"].isna().all()
        assert provenance.loc["2010-03-31":"2010-06-30", "publication_date"].isna().all()

    def test_provenance_records_the_age_of_each_value(self) -> None:
        values = pd.Series({pd.Timestamp("2010-03-31"): 1.7})
        index = pd.date_range("2010-04-30", "2010-05-31", freq="ME")
        _, provenance = align_real_time(values, index, self._gdp_lag())
        assert provenance.loc["2010-04-30", "age_days"] == pytest.approx(30.0)
        assert provenance.loc["2010-05-31", "age_days"] == pytest.approx(61.0)
        assert set(provenance["column"]) == {"real_gdp_qoq"}

    def test_empty_input_yields_an_empty_column(self) -> None:
        index = pd.date_range("2010-01-31", "2010-06-30", freq="ME")
        aligned, provenance = align_real_time(
            pd.Series(dtype=float), index, self._gdp_lag()
        )
        assert aligned.isna().all()
        assert provenance["publication_date"].isna().all()


class TestQuarterlyGrowthHelpers:
    def test_year_over_year_compounds_four_published_quarters(self) -> None:
        index = pd.date_range("2010-03-31", periods=4, freq="QE")
        qoq = pd.Series(4.0, index=index)
        yoy = quarterly_yoy_growth(qoq)
        assert yoy.iloc[:3].isna().all()
        expected = ((1.0 + 4.0 / 400.0) ** 4 - 1.0) * 100.0
        assert yoy.iloc[3] == pytest.approx(expected)

    def test_year_over_year_is_empty_without_data(self) -> None:
        assert quarterly_yoy_growth(pd.Series(dtype=float)).empty

    def test_vintage_frame_states_the_observation_quarter_and_release(
        self, synthetic_csv: Path
    ) -> None:
        frame = load_provided_csv(synthetic_csv)
        vintages = quarterly_gdp_vintages(frame)
        row = vintages.loc["2026-03-31"]
        assert row["observation_quarter"] == "2026Q1"
        assert row["publication_date"] == pd.Timestamp("2026-04-30")
        assert row["real_gdp_qoq_annualised"] == pytest.approx(2.0)

    def test_vintage_frame_needs_the_gdp_column(self) -> None:
        with pytest.raises(DataError, match="real_gdp_qoq"):
            quarterly_gdp_vintages(pd.DataFrame({"sentiment": [1.0]}))


class TestApproximatePriceLevel:
    def _yoy(self, start: str = "2018-01-31", periods: int = 120, rate: float = 3.0) -> pd.Series:
        index = pd.date_range(start, periods=periods, freq="ME")
        return pd.Series(rate, index=index, name="pce_yoy")

    def test_rebased_so_base_year_mean_is_100(self) -> None:
        level = approximate_price_level(self._yoy())
        base_mean = level[level.index.year == 2020].mean()
        assert base_mean == pytest.approx(100.0)

    def test_no_january_sawtooth(self) -> None:
        """Pins the review's HIGH finding about seeding twelve base-year months.

        Under a constant YoY rate, month-over-month growth must be constant.
        The old twelve-chain construction put a whole year's increase into every
        January, producing a repeating annual spike that would masquerade as
        signal in a monthly regression.
        """
        level = approximate_price_level(self._yoy(rate=6.0))
        mom = level.pct_change().dropna() * 100
        januaries = mom[mom.index.month == 1]
        others = mom[mom.index.month != 1]
        assert januaries.std() < 1e-9
        assert januaries.mean() == pytest.approx(others.mean(), rel=1e-6)

    def test_monotone_under_positive_inflation(self) -> None:
        level = approximate_price_level(self._yoy())
        assert (level.diff().dropna() > 0).all()

    def test_base_year_absent_raises(self) -> None:
        yoy = self._yoy(start="2021-01-31", periods=24)
        with pytest.raises(DataError, match="base year"):
            approximate_price_level(yoy, base_year=2020)

    def test_empty_series_returns_empty(self) -> None:
        assert approximate_price_level(pd.Series(dtype=float)).empty


class _StubFred(FredSource):
    """FRED stub: claims to be configured but returns nothing.

    Models the realistic failure the review flagged — expired key, offline
    machine, renamed series — which must still leave a usable panel behind.
    """

    def __init__(self, *, configured: bool = True, payload: dict[str, pd.Series] | None = None):
        self._configured = configured
        self._payload = payload or {}

    @property
    def configured(self) -> bool:  # type: ignore[override]
        return self._configured

    def series(self, series_id: str) -> pd.Series:  # type: ignore[override]
        return self._payload.get(series_id, pd.Series(dtype=float, name=series_id))


class TestBuildMonthlyPanel:
    def test_without_fred_falls_back_to_approximate_level(self, synthetic_csv: Path) -> None:
        panel, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(configured=False), base_year=2026
        )
        assert report.fred_available is False
        assert report.price_level_column == "pce_level_approx"
        assert "pce_level_approx" in panel.columns

    def test_configured_but_failing_fred_still_yields_a_price_level(
        self, synthetic_csv: Path
    ) -> None:
        """Pins the review's MEDIUM finding: no silent gap, no false claim."""
        panel, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(configured=True), base_year=2026
        )
        assert report.fred_available is False  # nothing was actually retrieved
        assert report.price_level_column == "pce_level_approx"
        assert "pce_level_approx" in panel.columns
        assert any("no series were retrieved" in note for note in report.notes)
        assert "FRED: unavailable" in report.summary()

    def test_successful_fred_series_are_used_and_reported(self, synthetic_csv: Path) -> None:
        index = pd.date_range("2026-01-31", "2026-03-31", freq="ME")
        payload = {"MORTGAGE30US": pd.Series([7.0, 7.1, 7.2], index=index, name="MORTGAGE30US")}
        panel, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(payload=payload), base_year=2026
        )
        assert report.fred_available is True
        assert "mortgage_30y" in report.fred_columns
        assert panel.loc["2026-02-28", "mortgage_30y"] == pytest.approx(7.1)
        # PCEPI was not supplied, so the approximation must still stand in.
        assert report.price_level_column == "pce_level_approx"

    def test_index_is_contiguous_month_end(self, synthetic_csv: Path) -> None:
        panel, _ = build_monthly_panel(
            synthetic_csv, fred=_StubFred(configured=False), base_year=2026
        )
        assert panel.index.freqstr == "ME"
        assert panel.index.is_monotonic_increasing

    def test_an_unknown_alignment_is_refused(self, synthetic_csv: Path) -> None:
        with pytest.raises(AlignmentError, match="unknown alignment"):
            build_monthly_panel(
                synthetic_csv,
                fred=_StubFred(configured=False),
                base_year=2026,
                alignment="whatever",  # type: ignore[arg-type]
            )

    def test_a_column_without_a_documented_lag_is_refused(
        self, synthetic_csv: Path, monkeypatch
    ) -> None:
        """A real-time panel cannot carry a column whose release date is unknown."""
        monkeypatch.delitem(PUBLICATION_LAGS, "pce_yoy")
        with pytest.raises(AlignmentError, match="no publication lag is documented"):
            build_monthly_panel(
                synthetic_csv, fred=_StubFred(configured=False), base_year=2026
            )


class TestPanelAlignment:
    """The panel-level form of the two timing rules in the brief."""

    def _panel(self, path: Path, alignment: str = "real_time"):
        return build_monthly_panel(
            path,
            fred=_StubFred(configured=False),
            base_year=2026,
            alignment=alignment,  # type: ignore[arg-type]
        )

    def test_real_time_is_the_default(self, synthetic_csv: Path) -> None:
        _, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(configured=False), base_year=2026
        )
        assert report.alignment == "real_time"
        assert report.real_time is True

    def test_the_report_carries_the_applied_lag_table(self, synthetic_csv: Path) -> None:
        _, report = self._panel(synthetic_csv)
        table = report.lag_table().set_index("column")
        assert table.loc["unemployment", "lag_days"] == 7
        assert table.loc["pce_yoy", "lag_days"] == 28
        assert table.loc["real_gdp_qoq", "lag_days"] == 30
        assert table.loc["sentiment", "lag_days"] == 0

    def test_no_month_carries_an_invented_gdp_observation(
        self, synthetic_csv: Path
    ) -> None:
        """The first rule of the brief, measured on the panel itself."""
        frame = load_provided_csv(synthetic_csv)
        panel, _ = self._panel(synthetic_csv)
        published = set(frame["real_gdp_qoq"].dropna().to_numpy(dtype=float))
        monthly = set(panel["real_gdp_qoq"].dropna().to_numpy(dtype=float))
        assert monthly <= published

    def test_no_regressor_carries_a_value_published_later(
        self, synthetic_csv: Path
    ) -> None:
        """The second rule of the brief, checked on every row of every column."""
        _, report = self._panel(synthetic_csv)
        provenance = report.provenance
        assert provenance is not None
        used = provenance.dropna(subset=["publication_date"])
        assert not used.empty
        assert (used["publication_date"] <= used["date"]).all()

    def test_each_row_uses_the_month_the_lag_table_implies(
        self, synthetic_csv: Path
    ) -> None:
        """February 2026 must hold January's unemployment rate, not its own."""
        frame = load_provided_csv(synthetic_csv)
        panel, _ = self._panel(synthetic_csv)
        assert panel.loc["2026-02-28", "unemployment"] == pytest.approx(
            frame.loc["2026-01-31", "unemployment"]
        )
        assert panel.loc["2026-03-31", "pce_yoy"] == pytest.approx(
            frame.loc["2026-02-28", "pce_yoy"]
        )
        # The target keeps its own month. Lagging it would move the question.
        assert panel.loc["2026-03-31", "sentiment"] == pytest.approx(
            frame.loc["2026-03-31", "sentiment"]
        )

    def test_the_real_time_panel_differs_from_the_contemporaneous_one(
        self, synthetic_csv: Path
    ) -> None:
        real_time, rt_report = self._panel(synthetic_csv)
        legacy, legacy_report = self._panel(synthetic_csv, alignment="contemporaneous")
        assert legacy_report.real_time is False
        assert any("NOT compliant" in note for note in legacy_report.notes)
        assert any("REAL TIME" in note for note in rt_report.notes)
        # The legacy panel puts Q1 2026 on January; the compliant one cannot.
        assert legacy.loc["2026-01-31", "real_gdp_qoq"] == pytest.approx(2.0)
        assert real_time.loc["2026-01-31", "real_gdp_qoq"] == pytest.approx(1.0)

    def test_the_contemporaneous_panel_keeps_its_old_shape(
        self, synthetic_csv: Path
    ) -> None:
        """So the report can quote both panels side by side."""
        frame = load_provided_csv(synthetic_csv)
        legacy, report = self._panel(synthetic_csv, alignment="contemporaneous")
        assert report.publication_lags == ()
        assert report.provenance is None
        assert legacy.loc["2026-02-28", "unemployment"] == pytest.approx(
            frame.loc["2026-02-28", "unemployment"]
        )

    def test_year_over_year_growth_is_reported_beside_the_quarterly_rate(
        self, synthetic_csv: Path
    ) -> None:
        panel, report = self._panel(synthetic_csv)
        assert "real_gdp_yoy" in panel.columns
        assert any("real_gdp_yoy" in note for note in report.notes)


class TestMichiganSubIndices:
    """The brief asks for the headline index and both sub-indices."""

    def test_absent_sub_indices_are_stated_not_omitted(
        self, synthetic_csv: Path
    ) -> None:
        _, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(configured=False), base_year=2026
        )
        note = next(
            (n for n in report.notes if "Michigan sub-indices" in n), ""
        )
        assert "Current Economic Conditions" in note
        assert "Consumer Expectations" in note
        assert "not on FRED" in note
        assert report.michigan_subindex_columns == ()

    def test_available_michigan_series_are_used_and_named(
        self, synthetic_csv: Path
    ) -> None:
        index = pd.date_range("2026-01-31", "2026-03-31", freq="ME")
        payload = {
            "UMCSENT": pd.Series([56.4, 56.6, 53.3], index=index, name="UMCSENT"),
            "MICH": pd.Series([4.0, 4.1, 4.2], index=index, name="MICH"),
        }
        panel, report = build_monthly_panel(
            synthetic_csv, fred=_StubFred(payload=payload), base_year=2026
        )
        assert report.michigan_subindex_columns == (
            "sentiment_umcsent",
            "inflation_expect_1y",
        )
        assert panel.loc["2026-02-28", "sentiment_umcsent"] == pytest.approx(56.6)
        assert panel.loc["2026-03-31", "inflation_expect_1y"] == pytest.approx(4.2)
        # Both carry lag 0, so each month holds its own survey round.
        assert any("Michigan sub-indices" in n for n in report.notes)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="provided CSV not present")
class TestRealProvidedData:
    """Regression tests against the actual issued file."""

    def test_panel_shape_and_coverage(self) -> None:
        panel, report = build_monthly_panel(REAL_CSV)
        assert report.rows == 586
        assert str(report.first) == "1978-01-31"
        assert panel.loc["2026-05-31", "sentiment"] == pytest.approx(44.8)

    def test_gdp_is_a_step_function_of_published_quarters(self) -> None:
        """April to June 2010 must all hold the published Q1 2010 estimate."""
        frame = load_provided_csv(REAL_CSV)
        panel, _ = panel_and_report()
        q1 = frame.loc["2010-03-31", "real_gdp_qoq"]
        quarter = panel.loc["2010-04-30":"2010-06-30", "real_gdp_qoq"]
        assert quarter.to_numpy(dtype=float) == pytest.approx([q1, q1, q1])

    def test_every_monthly_gdp_value_is_a_published_quarterly_value(self) -> None:
        """No invented intra-quarter observation survives in the issued data."""
        frame = load_provided_csv(REAL_CSV)
        panel, _ = panel_and_report()
        published = set(np.round(frame["real_gdp_qoq"].dropna().to_numpy(dtype=float), 6))
        monthly = set(np.round(panel["real_gdp_qoq"].dropna().to_numpy(dtype=float), 6))
        assert monthly <= published

    def test_no_column_uses_data_published_after_its_own_month(self) -> None:
        """The brief's real-time rule, over the whole issued panel."""
        _, report = panel_and_report()
        provenance = report.provenance
        assert provenance is not None
        used = provenance.dropna(subset=["publication_date"])
        assert (used["publication_date"] <= used["date"]).all()
        assert set(used["column"]) >= {
            "sentiment",
            "unemployment",
            "pce_yoy",
            "real_gdp_qoq",
        }

    def test_cumulative_price_level_exceeds_2020_by_over_twenty_percent(self) -> None:
        """Hypothesis A's premise, measured rather than assumed."""
        panel, report = panel_and_report()
        level = panel[report.price_level_column].dropna()
        base = panel.loc["2020-01-31":"2020-12-31", report.price_level_column].mean()
        assert (level.iloc[-1] / base - 1) * 100 > 20.0


def panel_and_report() -> tuple[pd.DataFrame, PanelReport]:
    return build_monthly_panel(REAL_CSV)
