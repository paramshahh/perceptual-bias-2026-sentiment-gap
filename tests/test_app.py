"""Tests for the dashboard logic.

These tests run offline and without a screen. They do not start a Streamlit
server. They import `app` and call the pure functions.

The most important tests are the degradation tests. A panel without data must
report `available=False`, it must name each absent input, and the reason must
say that the corpus is in collection. A chart must never appear in that state.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # `app.py` is at the repository root. `pyproject.toml` puts only `src` on
    # the path, so the test adds the root.
    sys.path.insert(0, str(REPO_ROOT))

import app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gap_frame() -> pd.DataFrame:
    """A small gap frame with the shape of `baseline_model.sentiment_gap`."""
    index = pd.date_range("2024-10-31", periods=6, freq="ME", name="date")
    observed = [70.0, 71.0, 69.0, 55.0, 50.0, 44.8]
    predicted = [72.0, 72.5, 71.0, 76.0, 77.0, 77.93]
    frame = pd.DataFrame(
        {
            "observed": observed,
            "predicted": predicted,
            "pi_lower": [p - 12.0 for p in predicted],
            "pi_upper": [p + 12.0 for p in predicted],
            "predicted_ridge": predicted,
        },
        index=index,
    )
    frame["gap"] = frame["observed"] - frame["predicted"]
    frame["gap_ridge"] = frame["gap"]
    outside = (frame["observed"] < frame["pi_lower"]) | (frame["observed"] > frame["pi_upper"])
    frame["outside_pi"] = outside.astype("boolean")
    return frame


class _FakeFit:
    def __init__(self, name: str, available: bool, **kwargs) -> None:
        self.name = name
        self.available = available
        self.missing_features = kwargs.get("missing_features", ())
        self.unavailable_reason = kwargs.get("unavailable_reason")
        self.n = kwargs.get("n", 0)
        self.r_squared = kwargs.get("r_squared", float("nan"))
        self.adj_r_squared = kwargs.get("adj_r_squared", float("nan"))
        self.rmse = kwargs.get("rmse", float("nan"))
        self.aic = kwargs.get("aic", float("nan"))


class _FakeComparison:
    def __init__(self, models) -> None:
        self.models = models


@pytest.fixture
def comparison() -> _FakeComparison:
    """Models 1 and 2 are fitted. Model 3 waits for the media corpus."""
    return _FakeComparison(
        {
            "Model 1": _FakeFit(
                "Model 1", True, n=180, r_squared=0.766, adj_r_squared=0.762, rmse=6.20, aic=1175.5
            ),
            "Model 2": _FakeFit(
                "Model 2", True, n=180, r_squared=0.837, adj_r_squared=0.833, rmse=5.40, aic=1116.2
            ),
            "Model 3": _FakeFit(
                "Model 3",
                False,
                missing_features=("media_tone", "topic_salience", "political_attribution"),
                unavailable_reason=(
                    "Model 3 not estimated: columns absent from the panel: "
                    "['media_tone', 'topic_salience', 'political_attribution']"
                ),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


def test_both_palettes_are_selected_not_inverted():
    light = app.get_palette("light")
    dark = app.get_palette("dark")
    assert light.surface == "#fcfcfb"
    assert dark.surface == "#1a1a19"
    # A selected dark palette gives its own series steps, not the light values.
    assert dark.series_1 != light.series_1
    assert dark.series_2 != light.series_2


def test_unknown_palette_name_gives_the_light_palette():
    assert app.get_palette("solarized").name == "light"
    assert app.get_palette("").name == "light"


def test_diverging_scale_has_a_gray_midpoint():
    for name in ("light", "dark"):
        palette = app.get_palette(name)
        scale = app.diverging_colorscale(palette)
        assert [stop[0] for stop in scale] == [0.0, 0.5, 1.0]
        assert scale[1][1] == palette.diverging_mid
        # The midpoint is gray: its three channels are within a small range.
        channels = [int(palette.diverging_mid[i : i + 2], 16) for i in (1, 3, 5)]
        assert max(channels) - min(channels) <= 12


# ---------------------------------------------------------------------------
# Panel availability — honest degradation
# ---------------------------------------------------------------------------


def test_panels_with_data_are_available(gap_frame):
    statuses = app.panel_availability(gap_frame.columns)
    assert statuses["timeline"].available
    assert statuses["gap"].available
    assert statuses["models"].available


def test_media_panels_are_unavailable_without_the_artifacts(gap_frame):
    """A panel is unavailable when its derived file is absent.

    Availability depends on the artifact the panel reads, not on a count of
    raw files from any pull. An earlier version counted the day files of the
    headline pull, which this project abandoned, so the dashboard reported
    "still collecting" while the finished corpus sat beside it.
    """
    statuses = app.panel_availability(gap_frame.columns, artifacts={})
    for key in ("scree", "media_tone", "granger"):
        status = statuses[key]
        assert not status.available, key
        assert status.missing, key
        assert status.reason


def test_unavailable_reason_names_every_absent_input(gap_frame):
    statuses = app.panel_availability(gap_frame.columns)
    assert "media_tone" in statuses["media_tone"].missing
    assert "media_monthly" in statuses["media_tone"].missing
    assert "svd_spectrum" in statuses["scree"].missing
    # Panel 6 needs both series; the gap exists, so the gap is not named.
    assert "media_tone" in statuses["granger"].missing
    assert "media_tone" in statuses["granger"].reason
    assert "gap" not in statuses["granger"].missing


def test_granger_names_both_inputs_when_both_are_absent():
    statuses = app.panel_availability(["observed", "predicted"], artifacts={})
    assert "media_tone" in statuses["granger"].missing
    assert "gap" in statuses["granger"].missing
    assert statuses["gap"].available is False


def test_media_panels_become_available_when_the_inputs_arrive(gap_frame):
    columns = list(gap_frame.columns) + ["media_tone"]
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    statuses = app.panel_availability(columns, artifacts=present)
    for key in ("scree", "media_tone", "granger"):
        assert statuses[key].available, key
        assert statuses[key].missing == ()


def test_status_summary_states_the_state(gap_frame):
    statuses = app.panel_availability(gap_frame.columns)
    summary = statuses["media_tone"].summary()
    assert summary.startswith("5. Media tone over time: not available")
    assert "media_tone" in summary
    assert statuses["gap"].summary().endswith("available")


def test_every_panel_in_the_brief_has_a_status(gap_frame):
    """Each display of paragraph 5 of the brief is one panel with one status.

    The brief asks for the sentiment and its prediction with uncertainty, the
    gap, four media measures, the releases and events, the source composition,
    the missingness and the model performance, and a split by income, age,
    geography and affiliation.
    """
    statuses = app.panel_availability(gap_frame.columns)
    assert sorted(statuses) == sorted(
        [
            "timeline",
            "gap",
            "models",
            "scree",
            "media_tone",
            "granger",
            "framing",
            "calibration",
            "events",
            "composition",
            "coverage",
            "quality",
            "demographics",
        ]
    )
    numbers = sorted(status.number for status in statuses.values())
    assert numbers == list(range(1, 14))


def test_the_panels_of_the_brief_cover_each_required_display(gap_frame):
    statuses = app.panel_availability(gap_frame.columns)
    # Tone and salience are panel 5; framing and calibration are their own
    # panels, because the brief asks for four media measures.
    assert statuses["framing"].title.startswith("Media framing")
    assert "calibration" in statuses["calibration"].title.lower()
    assert "events" in statuses["events"].title.lower()
    assert "composition" in statuses["composition"].title.lower()
    assert "performance" in statuses["quality"].title.lower()
    assert "income" in statuses["demographics"].title.lower()


# ---------------------------------------------------------------------------
# Corpus progress
# ---------------------------------------------------------------------------


def test_artifact_status_reports_a_present_frame(tmp_path: Path):
    """Readiness comes from the derived file, not from a count of raw files."""
    specs = (("media_monthly", "The monthly media series", tmp_path / "m.parquet", "cmd"),)
    frames = {"media_monthly": pd.DataFrame({"media_tone": [0.1, 0.2]})}
    status = app.artifact_status(frames, specs)["media_monthly"]
    assert status.present
    assert status.name == "The monthly media series"


def test_artifact_status_reports_an_absent_frame(tmp_path: Path):
    specs = (("svd_spectrum", "The SVD spectrum", tmp_path / "absent.parquet", "cmd"),)
    status = app.artifact_status({"svd_spectrum": None}, specs)["svd_spectrum"]
    assert not status.present
    assert status.command


# ---------------------------------------------------------------------------
# FRED degradation
# ---------------------------------------------------------------------------


class _Report:
    def __init__(self, fred_columns) -> None:
        self.fred_columns = tuple(fred_columns)


def test_fred_status_reports_the_series_it_received():
    status = app.fred_status(_Report(("pce_level", "mortgage_30y")), key_present=True)
    assert status.available
    assert status.reduced_panels == ()
    assert "pce_level" in status.message


def test_fred_status_names_the_reduced_panel_without_a_key():
    status = app.fred_status(_Report(()), key_present=False)
    assert not status.available
    assert status.reduced_panels == ("3. Model comparison, Models 1 to 3",)
    assert "No FRED key" in status.message
    # Without FRED, Model 2 loses the mortgage rate. The message must say that
    # Model 2 shows n/a, not only that the price level is approximate.
    assert "Model 2 shows n/a" in status.message
    assert "approximation" in status.message
    assert "Panel 1 and panel 2 do not change" in status.message


def test_a_configured_key_that_gives_nothing_is_not_available():
    status = app.fred_status(_Report(()), key_present=True)
    assert not status.available
    assert "gave no series" in status.message
    assert status.reduced_panels == ("3. Model comparison, Models 1 to 3",)


def test_fred_key_present_reads_the_environment_and_returns_a_boolean():
    assert app.fred_key_present({"FRED_API_KEY": "abc"}) is True
    assert app.fred_key_present({"FRED_API_KEY": "   "}) is False
    assert app.fred_key_present({}) is False


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def test_prepare_timeline_marks_the_out_of_sample_period(gap_frame):
    timeline = app.prepare_timeline(gap_frame, holdout_start="2025-01-01")
    assert list(timeline.columns[:5]) == ["date", "observed", "predicted", "pi_lower", "pi_upper"]
    assert timeline["out_of_sample"].tolist() == [False, False, False, True, True, True]
    assert timeline.loc[0, "sample"] == "in sample"
    assert timeline.loc[5, "sample"] == "out of sample"
    assert timeline.loc[5, "month"] == "2025-03"


def test_prepare_timeline_does_not_change_the_input(gap_frame):
    before = gap_frame.copy(deep=True)
    app.prepare_timeline(gap_frame)
    pd.testing.assert_frame_equal(gap_frame, before)


def test_prepare_timeline_raises_for_an_absent_column(gap_frame):
    with pytest.raises(KeyError, match="pi_upper"):
        app.prepare_timeline(gap_frame.drop(columns=["pi_upper"]))


def test_prepare_gap_months_keeps_only_months_with_a_gap(gap_frame):
    frame = gap_frame.copy()
    frame.loc[frame.index[-1], "gap"] = np.nan
    months = app.prepare_gap_months(frame, holdout_start="2025-01-01")
    # 2025-01 and 2025-02 stay. 2025-03 has no gap value, so it goes.
    assert months["month"].tolist() == ["2025-01", "2025-02"]
    assert "2025-03" not in months["month"].tolist()


def test_prepare_gap_months_starts_at_the_holdout(gap_frame):
    months = app.prepare_gap_months(gap_frame, holdout_start="2025-01-01")
    assert months["month"].tolist() == ["2025-01", "2025-02", "2025-03"]
    assert months[app.GAP_COLUMN].max() < 0


def test_prepare_gap_months_raises_without_a_gap_column(gap_frame):
    with pytest.raises(KeyError, match="gap"):
        app.prepare_gap_months(gap_frame.drop(columns=["gap"]))


def test_diverging_domain_holds_zero_at_the_middle():
    low, high = app.diverging_zero_domain([-33.13, -25.0, -11.0])
    assert low == pytest.approx(-33.13)
    assert high == pytest.approx(33.13)
    assert low + high == pytest.approx(0.0)


def test_diverging_domain_survives_empty_and_zero_input():
    assert app.diverging_zero_domain([]) == (-1.0, 1.0)
    assert app.diverging_zero_domain([0.0, 0.0]) == (-1.0, 1.0)
    assert app.diverging_zero_domain([np.nan, np.nan]) == (-1.0, 1.0)


def test_worst_month_finds_the_largest_negative_gap(gap_frame):
    months = app.prepare_gap_months(gap_frame, holdout_start="2025-01-01")
    month, value = app.worst_month(months)
    assert month == "2025-03"
    assert value == pytest.approx(-33.13)


def test_worst_month_returns_none_for_an_empty_frame():
    assert app.worst_month(pd.DataFrame(columns=["month", "gap"])) is None


# ---------------------------------------------------------------------------
# Model comparison — reuse of table_generator
# ---------------------------------------------------------------------------


def test_comparison_to_table3_maps_the_metric_names(comparison):
    metrics, unavailable = app.comparison_to_table3(comparison)
    assert set(metrics) == {"Model 1", "Model 2"}
    assert metrics["Model 1"]["r2"] == pytest.approx(0.766)
    assert metrics["Model 1"]["adj_r2"] == pytest.approx(0.762)
    assert metrics["Model 2"]["aic"] == pytest.approx(1116.2)
    assert metrics["Model 1"]["n"] == 180
    assert "Model 3" in unavailable


def test_comparison_to_table3_carries_the_mean_gap(comparison):
    metrics, _ = app.comparison_to_table3(comparison, mean_gaps={"Model 1": -25.24})
    assert metrics["Model 1"]["mean_gap"] == pytest.approx(-25.24)
    assert "mean_gap" not in metrics["Model 2"]


def test_comparison_to_table3_rejects_a_wrong_object():
    with pytest.raises(TypeError):
        app.comparison_to_table3(object())


def test_model_comparison_markdown_keeps_model_3_with_a_cause(comparison):
    text = app.model_comparison_markdown(comparison)
    assert "Model 1" in text and "Model 2" in text
    # Model 3 keeps its column, marked n/a, with the cause in a footnote.
    assert "Model 3 (n/a)" in text
    assert "Model 3 is not available" in text
    assert "0.766" in text  # table_generator rounds; the dashboard does not
    # The mean-gap row stays out: an "n/a" there would read as "no gap".
    assert "Mean out-of-sample gap" not in text


def test_model_comparison_markdown_states_a_corpus_cause_without_a_reason():
    comparison = _FakeComparison(
        {
            "Model 1": _FakeFit("Model 1", True, n=10, r_squared=0.5, adj_r_squared=0.4, rmse=1.0, aic=2.0),
            "Model 3": _FakeFit("Model 3", False, missing_features=("media_tone",)),
        }
    )
    text = app.model_comparison_markdown(comparison)
    assert "media_tone" in text


# ---------------------------------------------------------------------------
# SVD and Granger helpers — they exist, and they refuse empty input
# ---------------------------------------------------------------------------


class _FakeBasis:
    def __init__(self, values) -> None:
        self.singular_values = np.asarray(values, dtype=float)
        total = float((self.singular_values**2).sum())
        self.energy_share = (self.singular_values**2) / total
        self.cumulative_energy = np.cumsum(self.energy_share)


def test_scree_frame_numbers_the_components():
    frame = app.scree_frame(_FakeBasis([3.0, 2.0, 1.0]))
    assert frame["component"].tolist() == [1, 2, 3]
    assert frame["singular_value"].tolist() == [3.0, 2.0, 1.0]
    assert frame["cumulative_energy"].iloc[-1] == pytest.approx(1.0)


def test_scree_frame_refuses_an_empty_basis():
    with pytest.raises(ValueError, match="no singular values"):
        app.scree_frame(_FakeBasis([]))


class _FakeTest:
    def __init__(self, direction, f_statistic, p_value, lag_order) -> None:
        self.direction = direction
        self.f_statistic = f_statistic
        self.p_value = p_value
        self.lag_order = lag_order


class _FakeGranger:
    def __init__(self, forward, reverse) -> None:
        self.forward = forward
        self.reverse = reverse
        self.n_obs_used = 100


def test_granger_records_keep_both_directions():
    report = _FakeGranger(
        _FakeTest("media_tone -> gap", 3.1, 0.02, 2),
        _FakeTest("gap -> media_tone", 1.2, 0.31, 2),
    )
    records = app.granger_records(report)
    assert [record["direction"] for record in records] == [
        "media_tone -> gap",
        "gap -> media_tone",
    ]
    assert records[0]["n"] == 100


def test_granger_records_are_empty_without_a_test():
    assert app.granger_records(_FakeGranger(None, None)) == []


# ---------------------------------------------------------------------------
# Titles state the insight
# ---------------------------------------------------------------------------


def test_titles_state_the_finding():
    summary = pd.Series(
        {
            "mean_gap": -25.24,
            "worst_month": "2026-05",
            "worst_gap": -33.13,
            "months_outside_pi": 18.0,
            "n_tested": 19.0,
        }
    )
    assert app.timeline_title(summary) == (
        "Observed sentiment fell below the model's 95% prediction interval in "
        "18 of 19 out-of-sample months"
    )
    title = app.gap_title(summary)
    assert "-25.24" in title and "2026-05" in title and "-33.13" in title


# ---------------------------------------------------------------------------
# Figures — no dual axis, a tooltip on every chart, the palette is applied
# ---------------------------------------------------------------------------


def _axis_count(figure) -> int:
    return len([key for key in figure.layout.to_plotly_json() if key.startswith("yaxis")])


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_timeline_figure_has_one_axis_a_band_and_a_tooltip(gap_frame, mode):
    palette = app.get_palette(mode)
    timeline = app.prepare_timeline(gap_frame)
    figure = app.figure_observed_vs_predicted(timeline, palette, "A title")
    assert _axis_count(figure) == 1, "a dual axis is never allowed"
    assert not any(trace.to_plotly_json().get("yaxis") not in (None, "y") for trace in figure.data)
    names = [trace.name for trace in figure.data]
    assert "95% prediction interval" in names
    assert "Observed sentiment" in names
    assert any(getattr(trace, "hovertemplate", None) for trace in figure.data)
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.plot_bgcolor == palette.surface
    assert figure.layout.xaxis.title.text == "Month"
    assert figure.layout.yaxis.title.text == "Consumer sentiment index"
    assert figure.layout.title.text.startswith("A title")


def test_timeline_figure_marks_the_out_of_sample_period(gap_frame):
    timeline = app.prepare_timeline(gap_frame)
    figure = app.figure_observed_vs_predicted(timeline, app.get_palette("light"), "A title")
    texts = [annotation.text for annotation in figure.layout.annotations]
    assert any("never saw 2025" in text for text in texts)
    assert len(figure.layout.shapes) >= 2  # the wash and the hairline


def test_timeline_figure_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        app.figure_observed_vs_predicted(
            pd.DataFrame(columns=["date", "observed", "predicted", "pi_lower", "pi_upper", "sample"]),
            app.get_palette("light"),
            "A title",
        )


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_gap_figure_uses_a_diverging_scale_anchored_at_zero(gap_frame, mode):
    palette = app.get_palette(mode)
    months = app.prepare_gap_months(gap_frame, holdout_start="2025-01-01")
    figure = app.figure_gap_by_month(months, palette, "A title")
    marker = figure.data[0].marker
    assert marker.cmid == 0.0
    assert marker.cmin == pytest.approx(-marker.cmax)
    stops = [stop[1] for stop in marker.colorscale]
    assert stops == [palette.diverging_low, palette.diverging_mid, palette.diverging_high]
    assert figure.data[0].hovertemplate
    assert _axis_count(figure) == 1


def test_gap_figure_labels_the_worst_month(gap_frame):
    months = app.prepare_gap_months(gap_frame, holdout_start="2025-01-01")
    figure = app.figure_gap_by_month(months, app.get_palette("light"), "A title")
    texts = [annotation.text for annotation in figure.layout.annotations]
    assert any("2025-03" in text and "-33.13" in text for text in texts)


def test_gap_figure_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        app.figure_gap_by_month(pd.DataFrame(columns=["month", "gap"]), app.get_palette("light"), "T")


def test_scree_and_media_figures_are_single_series_with_tooltips():
    scree = app.scree_frame(_FakeBasis([3.0, 2.0, 1.0]))
    figure = app.figure_scree(scree, app.get_palette("dark"), "A title")
    assert len(figure.data) == 1
    assert figure.data[0].hovertemplate
    assert figure.data[0].showlegend is False
    assert _axis_count(figure) == 1

    tone = pd.DataFrame(
        {"date": pd.date_range("2025-01-31", periods=3, freq="ME"), "media_tone": [0.1, -0.2, 0.0]}
    )
    tone_figure = app.figure_media_tone(tone, app.get_palette("light"), "A title")
    assert len(tone_figure.data) == 1
    assert tone_figure.data[0].hovertemplate
    assert _axis_count(tone_figure) == 1


def test_no_figure_uses_a_dashed_gridline(gap_frame):
    timeline = app.prepare_timeline(gap_frame)
    figure = app.figure_observed_vs_predicted(timeline, app.get_palette("light"), "A title")
    assert figure.layout.xaxis.griddash == "solid"
    assert figure.layout.yaxis.griddash == "solid"


# ---------------------------------------------------------------------------
# The module must import without heavy work
# ---------------------------------------------------------------------------


def test_the_module_does_not_fit_a_model_at_import_time():
    # `main` is the only entry point that touches the screen or the network.
    assert callable(app.main)
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "\nimport streamlit" not in source, "Streamlit imports inside functions only"
    assert source.count("st.set_page_config") == 1


def test_the_constants_match_the_project_brief():
    assert app.HOLDOUT_START == "2025-01-01"
    assert app.TRAIN_START == "2010-01-01"
    assert app.TRAIN_END == "2024-12-31"


# ---------------------------------------------------------------------------
# The render path must agree with the availability decision
# ---------------------------------------------------------------------------


PANEL_RENDERERS = (
    "_render_timeline",
    "_render_gap",
    "_render_models",
    "_render_scree",
    "_render_media",
    "_render_granger",
    "_render_framing",
    "_render_calibration",
    "_render_events",
    "_render_composition",
    "_render_coverage",
    "_render_quality",
    "_render_demographics",
)


def test_every_panel_has_a_renderer_that_main_can_call():
    """`main` once called renderers that did not exist.

    An agent was interrupted after it wrote the calls in `main` and before it
    wrote the functions. The file still imported, because Python resolves a
    name at call time, so only a person opening the tab would have seen the
    failure.
    """
    for name in PANEL_RENDERERS:
        assert hasattr(app, name), f"{name} is called by main but is not defined"
        assert callable(getattr(app, name))


def test_no_renderer_degrades_unconditionally():
    """A renderer must ask whether its panel is available.

    This pins the defect that produced a false screen: `main` called
    `render_unavailable` for panels 4, 5 and 6 with no condition, so the page
    said "the corpus is still being collected" while the finished corpus sat
    on the disk. `panel_availability` said one thing and the screen said
    another.
    """
    import ast
    import inspect

    for name in PANEL_RENDERERS:
        source = inspect.getsource(getattr(app, name))
        tree = ast.parse(textwrap.dedent(source))
        function = tree.body[0]
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "render_unavailable"):
                continue
            # The call must sit inside an `if`, never straight in the body.
            in_guard = any(
                isinstance(parent, ast.If) and node in list(ast.walk(parent))
                for parent in ast.walk(function)
                if isinstance(parent, ast.If)
            )
            assert in_guard, f"{name} degrades without testing status.available"


def test_a_panel_marked_available_is_not_described_as_waiting():
    """No fixed text may contradict the availability decision.

    The footer once read "Panels 4, 5 and 6 wait for the media corpus" as
    fixed text. When the corpus arrived the sentence became false, and nothing
    in the code could notice.
    """
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    banned = (
        "Panels 4, 5 and 6 wait",
        "The media corpus is still in collection",
        "day files of",
        # A result of the study must come from the object that holds it. Each
        # number below moved when the panel changed its alignment, and a
        # sentence that holds a number cannot move with it.
        "p = 0.570",
        "0.679",
        "-34.30",
        "25.24",
        "33.13",
        "77.93",
        # A verdict about one publisher must come from the measured share.
        # The corpus can change with a new pull, and a fixed sentence cannot.
        "dominates the headline",
        "iheart.com gives",
    )
    for phrase in banned:
        assert phrase not in source, f"fixed text that can contradict the status: {phrase!r}"


# ---------------------------------------------------------------------------
# Panel 7 and panel 8 — framing and factual calibration
# ---------------------------------------------------------------------------


@pytest.fixture
def media_panel() -> pd.DataFrame:
    """A panel frame with the media columns of the monthly file."""
    index = pd.date_range("2024-10-31", periods=6, freq="ME", name="date")
    return pd.DataFrame(
        {
            "sentiment": [70.0, 71.0, 69.0, 55.0, 50.0, 44.8],
            "media_tone": [-0.05, -0.02, 0.01, -0.10, -0.06, 0.03],
            "topic_salience": [0.17, 0.18, 0.16, 0.19, 0.20, 0.18],
            "political_attribution": [0.04, 0.05, 0.03, 0.06, 0.05, 0.04],
        },
        index=index,
    )


def test_framing_panel_is_unavailable_without_a_framing_column(media_panel):
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(media_panel.columns, present)["framing"]
    assert not status.available
    assert status.missing == ("framing",), "one canonical name, not seven"
    # The reason must list every name that the panel accepts, so a reader of
    # the screen knows what to write into the monthly file.
    for name in app.FRAMING_ALTERNATIVES:
        assert name in status.reason


@pytest.mark.parametrize(
    "column", ["framing", "level_vs_rate", "framing_household_vs_macro"]
)
def test_framing_panel_accepts_any_name_of_the_group(media_panel, column):
    frame = media_panel.copy()
    frame[column] = 0.4
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(frame.columns, present)["framing"]
    assert status.available
    assert status.missing == ()


def test_calibration_panel_is_unavailable_without_its_column(media_panel):
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(media_panel.columns, present)["calibration"]
    assert not status.available
    assert status.missing == ("factual_calibration",)
    assert "calibration_gap" in status.reason


def test_the_framing_panel_never_substitutes_the_tone_series(media_panel):
    """Tone is not framing. A panel without framing stays unavailable."""
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    statuses = app.panel_availability(media_panel.columns, present)
    assert statuses["media_tone"].available
    assert not statuses["framing"].available


def test_framing_series_prefers_the_three_contrasts():
    columns = [
        "framing",
        "level_vs_rate",
        "household_vs_macro",
        "current_vs_forecast",
        "media_tone",
    ]
    selected = app.framing_series(columns)
    assert selected == ("level_vs_rate", "household_vs_macro", "current_vs_forecast")
    assert "framing" not in selected, "a composite column would repeat a contrast"
    assert len(selected) <= 3


def test_framing_series_falls_back_to_the_single_column():
    assert app.framing_series(["framing", "media_tone"]) == ("framing",)
    assert app.framing_series(["media_tone"]) == ()


def test_prepare_measure_series_keeps_only_months_with_a_value(media_panel):
    frame = media_panel.copy()
    frame["framing"] = [np.nan, np.nan, 0.2, 0.3, np.nan, 0.1]
    prepared = app.prepare_measure_series(frame, ["framing"])
    assert prepared["month"].tolist() == ["2024-12", "2025-01", "2025-03"]
    # A month before the measure exists is not a month with a value of zero.
    assert prepared["framing"].notna().all()


def test_prepare_measure_series_does_not_change_the_input(media_panel):
    frame = media_panel.copy()
    frame["framing"] = 0.5
    before = frame.copy(deep=True)
    app.prepare_measure_series(frame, ["framing"])
    pd.testing.assert_frame_equal(frame, before)


def test_prepare_measure_series_raises_when_no_column_is_present(media_panel):
    with pytest.raises(KeyError):
        app.prepare_measure_series(media_panel, ["framing"])


def _framing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-31", periods=4, freq="ME"),
            "month": ["2025-01", "2025-02", "2025-03", "2025-04"],
            "level_vs_rate": [0.5, 0.6, 0.4, 0.55],
            "household_vs_macro": [0.8, 0.7, 0.85, 0.82],
            "current_vs_forecast": [-0.3, -0.2, -0.35, -0.28],
        }
    )


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_framing_figure_has_one_axis_three_series_and_a_legend(mode):
    palette = app.get_palette(mode)
    frame = _framing_frame()
    columns = ("level_vs_rate", "household_vs_macro", "current_vs_forecast")
    figure = app.figure_framing(frame, palette, "A title", columns)
    assert _axis_count(figure) == 1, "a dual axis is never allowed"
    assert len(figure.data) == 3
    colors = [trace.line.color for trace in figure.data]
    assert colors == [palette.series_1, palette.series_2, palette.series_3]
    assert len(set(colors)) == 3, "no color repeats between series"
    assert all(trace.showlegend for trace in figure.data)
    assert all(trace.hovertemplate for trace in figure.data)
    # series_3 is low contrast on the light surface, so each series is also
    # labelled directly.
    texts = [annotation.text for annotation in figure.layout.annotations]
    for name in columns:
        assert name in texts


def test_framing_figure_refuses_a_fourth_series():
    frame = _framing_frame()
    frame["framing"] = 0.5
    with pytest.raises(ValueError, match="series at most"):
        app.figure_framing(
            frame,
            app.get_palette("light"),
            "A title",
            (
                "level_vs_rate",
                "household_vs_macro",
                "current_vs_forecast",
                "framing",
            ),
        )


def test_framing_figure_refuses_an_absent_column():
    with pytest.raises(KeyError):
        app.figure_framing(_framing_frame(), app.get_palette("light"), "T", ("absent",))


def test_framing_title_states_the_mean_of_each_contrast():
    frame = _framing_frame()
    title = app.framing_title(frame, ("level_vs_rate", "current_vs_forecast"))
    for name in ("level_vs_rate", "current_vs_forecast"):
        mean = float(frame[name].mean())
        assert f"{name} has a mean of {mean:.3f}" in title
    assert "4 months" in title
    # The sign of the mean is the finding, so the title must keep it.
    assert "-" in title.split("current_vs_forecast")[1][:20]


def test_calibration_title_states_the_range_and_the_mean():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-31", periods=3, freq="ME"),
            "factual_calibration": [0.1, -0.2, 0.0],
        }
    )
    title = app.calibration_title(frame, "factual_calibration")
    assert "-0.200" in title and "0.100" in title and "3 months" in title


# ---------------------------------------------------------------------------
# Panel 9 — the economic releases and the major events
# ---------------------------------------------------------------------------


def test_the_events_panel_needs_no_file_and_no_column(gap_frame):
    """The event list is a constant of the module, so panel 9 always draws."""
    statuses = app.panel_availability(gap_frame.columns, artifacts={})
    assert statuses["events"].available
    assert statuses["quality"].available


def test_every_event_has_a_date_and_a_source():
    assert app.ECONOMIC_EVENTS
    for event in app.ECONOMIC_EVENTS:
        assert pd.Timestamp(event.date) is not pd.NaT
        assert event.source.strip(), f"{event.label} has no source"
        assert event.kind.strip()
        assert event.label.strip()


def test_the_event_list_holds_the_dates_that_the_brief_names():
    months = {event.month for event in app.ECONOMIC_EVENTS}
    kinds = {event.kind for event in app.ECONOMIC_EVENTS}
    # The two election dates, the survey transition and the policy decisions.
    assert "2024-11" in months
    assert "2026-11" in months
    assert "2024-07" in months
    assert "election" in kinds
    assert "survey method" in kinds
    assert "policy decision" in kinds


def test_each_election_date_obeys_the_statute():
    """The election is the Tuesday after the first Monday of November."""
    for event in app.ECONOMIC_EVENTS:
        if event.kind != "election":
            continue
        stamp = event.timestamp
        assert stamp.month == 11
        assert stamp.weekday() == 1, f"{event.date} is not a Tuesday"
        assert 2 <= stamp.day <= 8, f"{event.date} is not after the first Monday"


def test_each_policy_decision_is_a_meeting_day():
    """An FOMC decision is on the second day of a two-day meeting."""
    for event in app.ECONOMIC_EVENTS:
        if event.kind != "policy decision":
            continue
        assert event.timestamp.weekday() in (2, 3), f"{event.date} is not a Wednesday"


def test_the_survey_transition_agrees_with_the_diagnostics_module():
    from perceptual_bias.survey_diagnostics import MICHIGAN_WEB_TRANSITION

    transition = [
        event for event in app.ECONOMIC_EVENTS if event.kind == "survey method"
    ]
    assert transition
    assert transition[0].month == pd.Timestamp(MICHIGAN_WEB_TRANSITION).strftime("%Y-%m")


def test_marked_events_are_a_small_subset():
    marked = app.marked_events()
    assert marked
    assert all(event.mark for event in marked)
    assert len(marked) < len(app.ECONOMIC_EVENTS)
    assert len(marked) <= 6, "a chart with many lines carries no information"


def test_events_frame_is_sorted_and_holds_the_source():
    frame = app.events_frame()
    assert list(frame.columns) == [
        "date",
        "month",
        "event",
        "kind",
        "note",
        "source",
        "marked",
    ]
    assert frame["date"].is_monotonic_increasing
    assert frame["source"].str.len().gt(0).all()


def test_events_frame_keeps_one_window():
    frame = app.events_frame(start="2026-01-01", end="2026-12-31")
    assert not frame.empty
    assert frame["month"].str.startswith("2026").all()
    assert app.events_frame(start="2100-01-01").empty


def test_the_event_list_states_which_dates_were_checked():
    assert app.FOMC_VERIFIED_THROUGH == "2025-12"
    later = [
        event
        for event in app.ECONOMIC_EVENTS
        if event.kind == "policy decision" and event.month > app.FOMC_VERIFIED_THROUGH
    ]
    # A date that this project did not check against the calendar of its year
    # carries that remark, so a reader checks it before a quotation.
    assert later
    for event in later:
        assert "Check this date" in event.note


def test_the_release_schedule_names_a_publisher_and_a_source():
    series = {spec.series for spec in app.RELEASE_SCHEDULE}
    assert any("CPI" in name for name in series)
    assert any("PCE" in name for name in series)
    assert any("GDP" in name for name in series)
    assert any("unemployment" in name for name in series)
    for spec in app.RELEASE_SCHEDULE:
        assert spec.publisher.strip()
        assert spec.source.strip()
        assert spec.lag_days >= 0


def test_release_calendar_puts_the_publication_after_the_reference_month():
    frame = app.release_calendar(["2026-05-31"])
    assert len(frame) == len(app.RELEASE_SCHEDULE)
    assert (frame["reference month"] == "2026-05").all()
    pce = frame.loc[frame["series"].str.contains("PCE")].iloc[0]
    # The PCE of one month publishes in the next month. A model at 2026-05
    # does not hold the PCE of 2026-05.
    assert pce["publication month"] == "2026-06"


def test_event_marks_are_chrome_and_not_a_series():
    # The window holds the 2024 election and the survey transition.
    frame = _framing_frame()
    frame["date"] = pd.date_range("2024-07-31", periods=len(frame), freq="ME")
    figure = app.figure_framing(
        frame,
        app.get_palette("light"),
        "A title",
        ("level_vs_rate",),
        events=app.marked_events(),
    )
    # No event becomes a trace: an event is not a measured series.
    assert len(figure.data) == 1
    palette = app.get_palette("light")
    event_lines = [
        shape
        for shape in figure.layout.shapes
        if shape.line is not None and shape.line.dash == "dot"
    ]
    assert event_lines, "an event in the window gets a dotted hairline"
    assert all(shape.line.color == palette.ink_muted for shape in event_lines)


def test_event_marks_stay_inside_the_window_of_the_chart():
    figure = go.Figure()
    palette = app.get_palette("light")
    app.add_event_marks(
        figure,
        palette,
        app.ECONOMIC_EVENTS,
        x_min="2024-01-01",
        x_max="2024-12-31",
    )
    inside = [
        event
        for event in app.ECONOMIC_EVENTS
        if pd.Timestamp("2024-01-01") <= event.timestamp <= pd.Timestamp("2024-12-31")
    ]
    assert len(figure.layout.shapes) == len(inside)
    assert len(inside) < len(app.ECONOMIC_EVENTS)


def test_event_marks_do_nothing_without_an_event():
    figure = go.Figure()
    app.add_event_marks(figure, app.get_palette("dark"), ())
    assert figure.layout.shapes == ()


def test_the_timeline_figure_marks_the_events(gap_frame):
    timeline = app.prepare_timeline(gap_frame)
    plain = app.figure_observed_vs_predicted(timeline, app.get_palette("light"), "T")
    marked = app.figure_observed_vs_predicted(
        timeline, app.get_palette("light"), "T", events=app.marked_events()
    )
    assert len(marked.layout.shapes) > len(plain.layout.shapes)
    assert len(marked.data) == len(plain.data)


# ---------------------------------------------------------------------------
# Panel 10 — the composition of the corpus by source
# ---------------------------------------------------------------------------


@pytest.fixture
def headlines() -> pd.DataFrame:
    """A headline frame where one publisher uses many hosts."""
    urls = [f"https://station{index}.network.com/a/{index}" for index in range(12)]
    urls += ["http://www.big.com/x"] * 6
    urls += ["https://small.co.uk/y"] * 2
    urls += ["https://news.small.co.uk/z"]
    urls += ["not a url"]
    return pd.DataFrame({"url": urls})


def test_url_domain_reads_the_host_only():
    assert app.url_domain("https://www.Example.com/path?a=b") == "example.com"
    assert app.url_domain("http://sub.example.com") == "sub.example.com"
    assert app.url_domain("not a url") == "not a url"
    assert app.url_domain("") == ""


def test_registrable_domain_folds_a_network_of_hosts():
    assert app.registrable_domain("960weli.iheart.com") == "iheart.com"
    assert app.registrable_domain("finance.yahoo.com") == "yahoo.com"
    assert app.registrable_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert app.registrable_domain("msn.com") == "msn.com"


def test_domain_composition_counts_the_hosts_of_one_publisher_together(headlines):
    composition = app.domain_composition(headlines, top_n=3)
    first = composition.iloc[0]
    assert first["domain"] == "network.com"
    assert first["n_headlines"] == 12
    assert first["n_hosts"] == 12, "a network of hosts is one publisher"
    assert composition["n_headlines"].sum() == len(headlines)
    other = composition.loc[composition["domain"] == app.OTHER_DOMAIN_LABEL]
    assert not other.empty
    assert int(other.iloc[0]["n_domains"]) >= 1


def test_domain_composition_can_count_by_host(headlines):
    composition = app.domain_composition(headlines, top_n=3, fold_hosts=False)
    assert composition.iloc[0]["n_headlines"] == 6, "big.com is the largest host"


def test_domain_composition_refuses_a_frame_without_a_url():
    with pytest.raises(KeyError, match="url"):
        app.domain_composition(pd.DataFrame({"title": ["a"]}))


def test_composition_report_states_the_measured_share(headlines):
    report = app.composition_report(headlines)
    assert report.n_headlines == len(headlines)
    assert report.top_domain == "network.com"
    assert report.top_share == pytest.approx(12 / len(headlines))
    assert report.top_hosts == 12
    assert report.concentrated  # 55% is above the limit
    assert f"{app.CONCENTRATION_LIMIT:.0%}" in report.note()
    assert "network.com" in app.composition_title(report)


def test_composition_report_has_no_fixed_verdict():
    """A spread corpus and a concentrated corpus give different sentences."""
    spread = pd.DataFrame(
        {"url": [f"https://site{index}.com/a" for index in range(50)]}
    )
    report = app.composition_report(spread)
    assert not report.concentrated
    assert "No domain reaches the limit" in report.note()
    assert "measures that publisher more" not in report.note()


def test_composition_report_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="no URL"):
        app.composition_report(pd.DataFrame({"url": []}))


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_composition_figure_is_one_series_with_one_axis(headlines, mode):
    palette = app.get_palette(mode)
    composition = app.domain_composition(headlines, top_n=3)
    figure = app.figure_domain_composition(composition, palette, "A title")
    assert _axis_count(figure) == 1
    assert len(figure.data) == 1
    assert figure.data[0].orientation == "h"
    assert figure.data[0].marker.color == palette.series_1
    assert figure.data[0].hovertemplate
    assert figure.data[0].showlegend is False


def test_composition_figure_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        app.figure_domain_composition(
            pd.DataFrame(columns=["domain", "share", "n_headlines"]),
            app.get_palette("light"),
            "T",
        )


def test_the_composition_panel_needs_the_headline_corpus(gap_frame):
    absent = {
        key: app.ArtifactStatus(
            key=key, name=name, path=path, present=False, command=command
        )
        for key, name, path, command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(gap_frame.columns, absent)["composition"]
    assert not status.available
    assert "headlines" in status.missing
    # The reason names the file and the script that writes it, so a reader of
    # the screen can act.
    assert "headlines.parquet" in status.reason
    assert "pull_gdelt.py" in status.reason


# ---------------------------------------------------------------------------
# Panel 11 — the months with thin coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def media_monthly() -> pd.DataFrame:
    index = pd.date_range("2025-01-31", periods=5, freq="ME", name="date")
    return pd.DataFrame(
        {
            "media_tone": [-0.05, -0.02, 0.01, -0.10, -0.06],
            "n_articles": [2000, 1800, 1600, 300, 1400],
        },
        index=index,
    )


def test_coverage_report_uses_a_share_of_the_median(media_monthly):
    report = app.coverage_report(media_monthly)
    assert report.median_articles == pytest.approx(1600.0)
    assert report.threshold == pytest.approx(800.0)
    assert report.thin_months == ("2025-04",)
    assert report.share_thin == pytest.approx(0.2)
    assert report.min_month == "2025-04"
    assert report.min_articles == 300
    assert f"{app.THIN_COVERAGE_SHARE:.0%}" in report.note()


def test_coverage_report_refuses_a_frame_without_a_count(media_monthly):
    with pytest.raises(KeyError, match="n_articles"):
        app.coverage_report(media_monthly.drop(columns=["n_articles"]))


def test_coverage_report_refuses_an_empty_count():
    frame = pd.DataFrame({"n_articles": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="no count"):
        app.coverage_report(frame)


def test_prepare_coverage_gives_a_month_and_a_count(media_monthly):
    frame = app.prepare_coverage(media_monthly)
    assert frame["month"].tolist() == [
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
    ]
    assert frame["n_articles"].iloc[3] == pytest.approx(300.0)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_coverage_figure_colors_the_decision_and_keeps_one_axis(
    media_monthly, mode
):
    palette = app.get_palette(mode)
    report = app.coverage_report(media_monthly)
    frame = app.prepare_coverage(media_monthly)
    figure = app.figure_monthly_coverage(
        frame, palette, "A title", threshold=report.threshold
    )
    assert _axis_count(figure) == 1
    names = [trace.name for trace in figure.data]
    assert names == ["At or above the limit", "Below the limit"]
    colors = [trace.marker.color for trace in figure.data]
    assert colors == [palette.series_1, palette.ink_muted]
    # The threshold is the only dashed line of the figure.
    dashed = [
        shape
        for shape in figure.layout.shapes
        if shape.line is not None and shape.line.dash == "dash"
    ]
    assert len(dashed) == 1


def test_coverage_figure_without_a_threshold_is_one_series(media_monthly):
    frame = app.prepare_coverage(media_monthly)
    figure = app.figure_monthly_coverage(frame, app.get_palette("light"), "T")
    assert len(figure.data) == 1
    assert figure.data[0].showlegend is False


def test_coverage_figure_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        app.figure_monthly_coverage(
            pd.DataFrame(columns=["date", "n_articles"]), app.get_palette("light"), "T"
        )


def test_coverage_title_states_the_count_of_thin_months(media_monthly):
    report = app.coverage_report(media_monthly)
    title = app.coverage_title(report)
    assert "1 of 5 months" in title
    assert "800" in title


# ---------------------------------------------------------------------------
# Panel 12 — the missingness and the performance of the model
# ---------------------------------------------------------------------------


class _FitStub:
    def __init__(self, n_train: int = 180, rmse: float = 6.20) -> None:
        self.n_train = n_train
        self.rmse_in_sample = rmse


def test_model_performance_counts_only_the_months_with_a_test(gap_frame):
    frame = gap_frame.copy()
    frame.loc[frame.index[-1], "outside_pi"] = pd.NA
    performance = app.model_performance(_FitStub(), frame, holdout_start="2025-01-01")
    # Three out-of-sample months; the last has no test, so two are tested.
    assert performance.n_tested == 2
    assert performance.months_inside_pi == 0
    assert performance.share_inside_pi == pytest.approx(0.0)
    assert performance.rmse_in_sample == pytest.approx(6.20)
    assert performance.n_train == 180
    # The mean absolute gap uses every month with a gap value.
    assert performance.mean_abs_gap == pytest.approx(
        abs(frame.loc["2025-01-31":, "gap"]).mean()
    )


def test_model_performance_counts_a_month_inside_the_interval(gap_frame):
    frame = gap_frame.copy()
    frame["outside_pi"] = pd.array([False] * len(frame), dtype="boolean")
    performance = app.model_performance(_FitStub(), frame, holdout_start="2025-01-01")
    assert performance.months_inside_pi == 3
    assert performance.share_inside_pi == pytest.approx(1.0)
    assert "95%" in performance.note()


def test_model_performance_refuses_a_frame_without_the_test_column(gap_frame):
    with pytest.raises(KeyError, match="outside_pi"):
        app.model_performance(_FitStub(), gap_frame.drop(columns=["outside_pi"]))


def test_performance_tiles_have_one_lead_figure(gap_frame):
    performance = app.model_performance(_FitStub(), gap_frame, holdout_start="2025-01-01")
    tiles = app.performance_tiles(performance)
    assert len([tile for tile in tiles if tile.lead]) == 1
    html = app.stat_tiles_html(tiles)
    assert "pb-tile--lead" in html
    assert "RMSE, in sample" in html


def test_missing_month_table_names_the_absent_input():
    index = pd.date_range("2025-01-31", periods=3, freq="ME", name="date")
    panel = pd.DataFrame(
        {
            "unemployment": [4.0, np.nan, 4.2],
            "pce_yoy": [2.5, 2.4, np.nan],
            "real_gdp_qoq": [2.0, 2.0, 2.0],
        },
        index=index,
    )
    gaps = pd.DataFrame({"gap": [-20.0, np.nan, np.nan]}, index=index)
    table = app.missing_month_table(
        panel,
        gaps,
        features=("unemployment", "pce_yoy", "real_gdp_qoq"),
        holdout_start="2025-01-01",
    )
    assert table["month"].tolist() == ["2025-02", "2025-03"]
    assert table.loc[0, "absent inputs"] == "unemployment"
    assert table.loc[1, "absent inputs"] == "pce_yoy"
    assert (table["gap value"] == "absent").all()
    # A month with a gap value and every input is not in the table. A zero row
    # there would read as "this month has a gap of zero".
    assert "2025-01" not in table["month"].tolist()


def test_missing_month_table_is_empty_when_nothing_is_missing():
    index = pd.date_range("2025-01-31", periods=2, freq="ME", name="date")
    panel = pd.DataFrame({"unemployment": [4.0, 4.1]}, index=index)
    gaps = pd.DataFrame({"gap": [-20.0, -21.0]}, index=index)
    table = app.missing_month_table(
        panel, gaps, features=("unemployment",), holdout_start="2025-01-01"
    )
    assert table.empty
    assert list(table.columns) == ["month", "gap value", "absent inputs"]


def test_panel_missingness_gives_the_first_and_the_last_month():
    index = pd.date_range("2024-11-30", periods=4, freq="ME", name="date")
    panel = pd.DataFrame(
        {
            "sentiment": [70.0, 69.0, 55.0, 50.0],
            "media_tone": [np.nan, np.nan, 0.1, 0.2],
            "absent": [np.nan] * 4,
        },
        index=index,
    )
    table = app.panel_missingness(panel).set_index("column")
    assert table.loc["sentiment", "months with a value"] == 4
    assert table.loc["media_tone", "months without a value"] == 2
    assert table.loc["media_tone", "first month"] == "2025-01"
    assert table.loc["sentiment", "last month"] == "2025-02"
    # A column with no value says so. It never reports a month.
    assert table.loc["absent", "first month"] == "none"


def test_quality_title_states_both_error_measures(gap_frame):
    performance = app.model_performance(_FitStub(), gap_frame, holdout_start="2025-01-01")
    missing = pd.DataFrame({"month": ["2026-10"]})
    title = app.quality_title(performance, missing)
    assert "in-sample RMSE" in title
    assert "out-of-sample mean absolute gap" in title
    assert "1 months hold no gap value" in title


# ---------------------------------------------------------------------------
# Panel 13 — the split that the data do not permit
# ---------------------------------------------------------------------------


def test_the_demographic_panel_is_unavailable_and_names_the_columns(gap_frame):
    statuses = app.panel_availability(gap_frame.columns)
    status = statuses["demographics"]
    assert not status.available
    for name in app.DEMOGRAPHIC_COLUMNS:
        assert name in status.missing
    assert status.reason


def test_the_demographic_note_names_the_microdata_and_invents_nothing():
    note = app.DEMOGRAPHIC_NOTE
    assert "microdata" in note
    assert "do not permit" in note
    assert "invention" in note or "invent" in note
    plan = app.DEMOGRAPHIC_PLAN
    assert "weights" in plan
    assert "party" in plan
    # Michigan documents a difference by party. The panel must say that the
    # affiliation is a moderator and a confounder, not a measured result.
    assert "confounder" in plan


def test_no_demographic_figure_builder_exists():
    """A panel without data must have no figure path at all."""
    builders = [name for name in dir(app) if name.startswith("figure_")]
    for name in builders:
        assert "demograph" in name.lower() is False or "demograph" not in name.lower()
    assert not any("income" in name or "party" in name for name in builders)


# ---------------------------------------------------------------------------
# The render path of the new panels
# ---------------------------------------------------------------------------


def test_main_renders_every_panel_that_the_requirement_table_holds():
    """A panel in the table must reach the screen.

    A new PanelSpec with no call in `main` would give a footer line about a
    panel that no tab shows.
    """
    import inspect

    source = inspect.getsource(app.main)
    for spec in app.PANEL_REQUIREMENTS:
        assert f'statuses["{spec.key}"]' in source, f"main never renders {spec.key}"


def test_every_tab_of_main_has_a_label():
    import inspect

    source = inspect.getsource(app.main)
    assert "7-8. Framing and calibration" in source
    assert "9. Events and releases" in source
    assert "10-12. Data quality" in source
    assert "13. Demographic splits" in source


def test_the_dashboard_uses_the_current_streamlit_width_argument():
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in source, "deprecated in streamlit 1.66"
    assert 'width="stretch"' in source


# ---------------------------------------------------------------------------
# The Granger verdict and the framing validity note come from the data
# ---------------------------------------------------------------------------


def test_granger_verdict_states_the_failure_with_its_numbers():
    class _NoReject(_FakeTest):
        def rejects(self) -> bool:
            return False

    report = _FakeGranger(
        _NoReject("media_tone -> gap", 0.441, 0.779, 2),
        _NoReject("gap -> media_tone", 0.501, 0.735, 2),
    )
    text = app.granger_verdict(report)
    assert "fails its own falsification condition" in text
    assert "p = 0.779" in text
    assert "p = 0.735" in text
    assert "F = 0.441" in text


def test_granger_verdict_changes_when_a_direction_rejects():
    class _Reject(_FakeTest):
        def rejects(self) -> bool:
            return True

    report = _FakeGranger(
        _Reject("media_tone -> gap", 6.2, 0.002, 2),
        _Reject("gap -> media_tone", 5.1, 0.007, 2),
    )
    text = app.granger_verdict(report)
    assert "rejects its null" in text
    assert "fails its own falsification condition" not in text
    assert "p = 0.002" in text


def test_granger_verdict_says_nothing_without_a_test():
    assert "no result" in app.granger_verdict(_FakeGranger(None, None))


def test_framing_agreement_measures_the_two_text_sources():
    index = pd.date_range("2025-01-31", periods=5, freq="ME", name="date")
    frame = pd.DataFrame(
        {
            "framing_headline": [0.1, 0.4, 0.5, 0.9, 0.2],
            "framing_slug": [0.2, 0.3, 0.6, 0.8, 0.1],
        },
        index=index,
    )
    agreement = app.framing_agreement(frame)
    assert agreement is not None
    assert agreement.n_months == 5
    expected = float(frame["framing_headline"].corr(frame["framing_slug"]))
    assert agreement.correlation == pytest.approx(expected)
    assert f"r = {expected:.3f}" in agreement.note()
    assert "validity" in agreement.note()


def test_framing_agreement_is_none_with_one_source_only():
    index = pd.date_range("2025-01-31", periods=5, freq="ME", name="date")
    frame = pd.DataFrame({"framing_headline": [0.1, 0.4, 0.5, 0.9, 0.2]}, index=index)
    # None means "the dashboard says nothing". A correlation of zero would
    # read as "the two sources disagree".
    assert app.framing_agreement(frame) is None


def test_framing_agreement_is_none_with_too_few_months():
    index = pd.date_range("2025-01-31", periods=2, freq="ME", name="date")
    frame = pd.DataFrame(
        {"framing_headline": [0.1, 0.4], "framing_slug": [0.2, 0.3]}, index=index
    )
    assert app.framing_agreement(frame) is None


def test_the_media_loader_keeps_the_framing_and_calibration_columns(tmp_path: Path):
    index = pd.date_range("2025-01-31", periods=3, freq="ME", name="date")
    frame = pd.DataFrame(
        {
            "media_tone": [0.1, 0.2, 0.3],
            "topic_salience": [0.1, 0.2, 0.3],
            "political_attribution": [0.1, 0.2, 0.3],
            "n_articles": [100, 200, 300],
            "level_vs_rate": [0.5, 0.6, 0.4],
            "household_vs_macro": [0.8, 0.7, 0.9],
            "current_vs_forecast": [-0.3, -0.2, -0.4],
            "framing": [0.5, 0.6, 0.4],
            "framing_headline": [0.5, 0.6, 0.4],
            "framing_slug": [0.4, 0.5, 0.3],
            "mean_raw_tone": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    path = tmp_path / "media_monthly.parquet"
    frame.to_parquet(path)
    loaded = app.load_media_series(path)
    assert loaded is not None
    for name in (
        "media_tone",
        "n_articles",
        "framing",
        "level_vs_rate",
        "household_vs_macro",
        "current_vs_forecast",
        "framing_headline",
        "framing_slug",
    ):
        assert name in loaded.columns
    # A column that no panel reads stays out of the dashboard.
    assert "mean_raw_tone" not in loaded.columns


def test_the_headline_loader_keeps_the_url_only(tmp_path: Path):
    path = tmp_path / "headlines.parquet"
    pd.DataFrame(
        {
            "url": ["https://a.com/x", "https://b.com/y"],
            "title": ["a", "b"],
            "month": ["2025-01", "2025-01"],
        }
    ).to_parquet(path)
    loaded = app.load_headlines(path)
    assert loaded is not None
    assert list(loaded.columns) == ["url", "month"]


def test_the_headline_loader_returns_none_for_an_absent_file(tmp_path: Path):
    assert app.load_headlines(tmp_path / "absent.parquet") is None


def test_the_headline_loader_refuses_a_file_without_a_url(tmp_path: Path):
    path = tmp_path / "headlines.parquet"
    pd.DataFrame({"title": ["a"]}).to_parquet(path)
    assert app.load_headlines(path) is None


# ---------------------------------------------------------------------------
# The heading of an unavailable panel states what is absent now
# ---------------------------------------------------------------------------


def test_a_present_file_with_an_absent_column_is_not_called_an_absent_file(
    media_panel,
):
    """The heading once said that a file was absent when the file was present.

    Panel 8 reads the monthly media file, and that file is on the disk. The
    file carries no calibration column yet. A heading that said "the derived
    file is not on the disk" contradicted the status.
    """
    present = {
        key: app.ArtifactStatus(key=key, name=name, path=path, present=True)
        for key, name, path, _command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(media_panel.columns, present)["calibration"]
    assert not status.available
    assert status.needs_artifact, "the panel does read a derived file"
    assert not status.artifact_absent, "but no file of the panel is absent"
    assert status.missing_artifacts == ()


def test_an_absent_file_is_reported_as_an_absent_file(gap_frame):
    absent = {
        key: app.ArtifactStatus(
            key=key, name=name, path=path, present=False, command=command
        )
        for key, name, path, command in app.ARTIFACT_SPECS
    }
    status = app.panel_availability(gap_frame.columns, absent)["scree"]
    assert status.artifact_absent
    assert status.missing_artifacts == ("svd_spectrum",)


def test_the_render_path_chooses_the_heading_from_the_status():
    """`render_unavailable` must read `artifact_absent`, not `needs_artifact`."""
    import inspect

    source = inspect.getsource(app.render_unavailable)
    assert "status.artifact_absent" in source
    assert "status.needs_artifact" not in source


# ---------------------------------------------------------------------------
# The entry point must be the last block of the file
# ---------------------------------------------------------------------------


def test_main_runs_after_every_renderer_is_defined():
    """Streamlit runs this file as the main module.

    `main()` once stood in the middle of the file, so it called
    `_render_scree` before Python defined it. The module imported, the tests
    passed, and tab 4 failed with a NameError on the screen.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "app.py").read_text(encoding="utf-8"))
    guards = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and "__main__" in ast.dump(node.test)
    ]
    assert len(guards) == 1, "one entry point only"
    after = tree.body[guards[0] + 1 :]
    offenders = [
        node.name
        for node in after
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not offenders, f"these names are defined after main() runs: {offenders}"


def test_build_composition_returns_the_report_and_the_table(headlines):
    report, composition = app.build_composition(headlines, len(headlines))
    assert report.top_domain == "network.com"
    assert not composition.empty
    # The function must not change the frame that the cache holds.
    before = headlines.copy(deep=True)
    app.build_composition(headlines, len(headlines))
    pd.testing.assert_frame_equal(headlines, before)


def test_the_cached_work_uses_the_correct_streamlit_cache():
    """Frames use `cache_data`; a fitted model uses `cache_resource`."""
    import inspect

    source = inspect.getsource(app.main)
    for name in ("build_panel", "build_gaps", "load_artifacts", "build_composition"):
        assert f'st.cache_data(ttl="6h", show_spinner=False)({name})' in source
    for name in ("build_fit", "build_comparison", "build_granger"):
        assert f"st.cache_resource(show_spinner=False)({name})" in source
