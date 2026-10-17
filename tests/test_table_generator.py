"""Tests for table rendering.

Most of these are golden-ish: a known input must produce an exact string, so a
change in rounding or column order is a test failure rather than something a
reader notices in the PDF. The LaTeX escaping tests pin the module's most
likely bug — every variable name in this project is snake_case, and a raw
underscore breaks a LaTeX build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.baseline_model import fit_baseline, gap_summary, sentiment_gap
from perceptual_bias.table_generator import (
    HAC_NOTE,
    NA,
    SIGNIFICANCE_NOTE,
    Panel,
    Row,
    Table,
    TableError,
    escape_latex,
    format_coefficient,
    format_integer,
    format_number,
    format_p_value,
    save_table,
    significance_stars,
    table1_baseline_and_gap,
    table2_svd_topics,
    table3_model_comparison,
    table4_granger_and_iv,
    to_latex,
    to_markdown,
)

RNG = np.random.default_rng(20261015)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def synthetic_panel(periods: int = 120) -> pd.DataFrame:
    index = pd.date_range("2010-01-31", periods=periods, freq="ME")
    unemployment = 5.0 + np.sin(np.arange(periods) / 12) * 1.5
    pce_yoy = 2.0 + np.cos(np.arange(periods) / 18) * 0.8
    real_gdp_qoq = 2.5 + RNG.normal(0, 0.4, periods)
    sentiment = (
        120.0
        - 4.0 * unemployment
        - 9.0 * pce_yoy
        - 0.2 * real_gdp_qoq
        + RNG.normal(0, 1.0, periods)
    )
    return pd.DataFrame(
        {
            "unemployment": unemployment,
            "pce_yoy": pce_yoy,
            "real_gdp_qoq": real_gdp_qoq,
            "sentiment": sentiment,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def fitted():
    panel = synthetic_panel()
    return fit_baseline(panel.iloc[:100]), panel.iloc[100:]


@pytest.fixture
def toy_table() -> Table:
    return Table(
        number=9,
        title="Toy table for pce_yoy & friends",
        panels=(
            Panel(
                columns=("Regressor", "Coefficient", "p"),
                rows=(
                    Row.data("pce_yoy", "-9.771***", "<0.001"),
                    Row.section("Subhead"),
                    Row.data("real_gdp_qoq", "-0.159*", "0.039"),
                ),
            ),
        ),
        footnotes=("50% of labels contain an underscore_", SIGNIFICANCE_NOTE),
    )


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


def test_number_formatting_never_prints_full_float_precision():
    assert format_number(0.7663245801792537, 3) == "0.766"
    assert format_number(1175.4832112, 1) == "1175.5"
    assert format_number(-4.2839991, 3) == "-4.284"
    assert format_number(6.2049, 2) == "6.20"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "", "   "])
def test_missing_values_render_as_na(missing):
    assert format_number(missing, 3) == NA
    assert format_integer(missing) == NA
    assert format_p_value(missing) == NA


def test_infinity_is_shown_not_formatted_away():
    assert format_number(float("inf"), 2) == "inf"
    assert format_number(float("-inf"), 2) == "-inf"


def test_p_value_floor_and_rounding():
    assert format_p_value(2.3e-9) == "<0.001"
    assert format_p_value(0.0009) == "<0.001"
    assert format_p_value(0.001) == "0.001"
    assert format_p_value(0.03851) == "0.039"
    assert format_p_value(0.9999) == "1.000"


@pytest.mark.parametrize(
    ("p", "stars"),
    [
        (0.0001, "***"),
        (0.009, "***"),
        (0.01, "**"),
        (0.039, "**"),
        (0.05, "*"),
        (0.09, "*"),
        (0.10, ""),
        (0.256, ""),
    ],
)
def test_significance_stars_follow_the_stated_convention(p, stars):
    assert significance_stars(p) == stars


def test_missing_p_value_gets_no_stars_rather_than_being_scored_insignificant():
    # NaN is not "insignificant" — it is untested, and must make no claim.
    assert significance_stars(None) == ""
    assert significance_stars(float("nan")) == ""
    assert format_coefficient(-1.604, None) == "-1.604"


def test_coefficient_carries_its_stars():
    assert format_coefficient(-9.7712, 1e-12) == "-9.771***"
    assert format_coefficient(-0.1593, 0.039) == "-0.159**"
    assert format_coefficient(None, 0.01) == NA


# ---------------------------------------------------------------------------
# LaTeX escaping — the module's most likely bug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pce_yoy", r"pce\_yoy"),
        ("real_gdp_qoq", r"real\_gdp\_qoq"),
        ("mortgage_30y", r"mortgage\_30y"),
        ("50%", r"50\%"),
        ("A & B", r"A \& B"),
        ("#1", r"\#1"),
        ("$5", r"\$5"),
        ("x^2", r"x\textasciicircum{}2"),
        ("a~b", r"a\textasciitilde{}b"),
        ("{x}", r"\{x\}"),
        ("<0.001", r"$<$0.001"),
    ],
)
def test_escape_latex_handles_every_special_character(raw, expected):
    assert escape_latex(raw) == expected


def test_math_labels_render_as_math_not_as_escaped_carets():
    # A published table saying "R\textasciicircum{}2" reads as a typo; these
    # whole-cell labels are mapped to math mode instead.
    assert escape_latex("R^2") == r"$R^2$"
    assert escape_latex("Adjusted R^2") == r"Adjusted $R^2$"
    assert escape_latex("Loading (U_k)") == r"Loading ($U_k$)"
    # Markdown keeps the plain form, so the two renderers stay readable.
    assert "R^2" in table3_model_comparison(COMPARISON).to_markdown()
    assert "$R^2$" in table3_model_comparison(COMPARISON).to_latex()


def test_escape_latex_does_not_double_escape_its_own_backslashes():
    # Backslash must be replaced first, or "_" -> "\_" would itself be escaped.
    assert escape_latex("a\\b") == r"a\textbackslash{}b"
    assert escape_latex("_") == r"\_"


def test_latex_output_contains_no_raw_underscore(toy_table):
    text = to_latex(toy_table)
    assert "pce\\_yoy" in text
    assert "real\\_gdp\\_qoq" in text
    assert "50\\%" in text
    for position, char in enumerate(text):
        if char == "_":
            assert text[position - 1] == "\\", f"unescaped underscore at {position}"


def test_latex_structure_is_a_booktabs_float(toy_table):
    text = to_latex(toy_table)
    assert text.startswith(r"\begin{table}[htbp]")
    assert text.rstrip().endswith(r"\end{table}")
    assert r"\toprule" in text and r"\midrule" in text and r"\bottomrule" in text
    assert r"\multicolumn{3}{l}{\textit{Subhead}}" in text
    assert r"\caption{Table 9: Toy table for pce\_yoy \& friends}" in text


def test_markdown_escapes_pipes_rather_than_breaking_the_table():
    table = Table(
        number=9,
        title="t",
        panels=(Panel(columns=("a", "b"), rows=(Row.data("x|y", "1.00"),)),),
    )
    line = next(row for row in to_markdown(table).splitlines() if "x" in row)
    assert line == r"| x\|y | 1.00 |"


# ---------------------------------------------------------------------------
# Renderer consistency
# ---------------------------------------------------------------------------


def _markdown_header_cells(text: str) -> list[list[str]]:
    headers = []
    lines = text.splitlines()
    for position, line in enumerate(lines):
        next_line = lines[position + 1] if position + 1 < len(lines) else ""
        is_rule = next_line.startswith("|") and set(next_line) <= set("|:- ")
        if line.startswith("|") and is_rule:
            headers.append([cell.strip() for cell in line.strip("|").split("|")])
    return headers


def _latex_header_cells(text: str) -> list[list[str]]:
    headers = []
    lines = text.splitlines()
    for position, line in enumerate(lines):
        if line == r"\toprule":
            raw = lines[position + 1].removesuffix(r" \\")
            headers.append(
                [
                    cell.strip().removeprefix(r"\textbf{").removesuffix("}")
                    for cell in raw.split(" & ")
                ]
            )
    return headers


def _all_tables(fitted) -> list[Table]:
    fit, holdout = fitted
    gaps, _ = sentiment_gap(fit, holdout)
    return [
        table1_baseline_and_gap(fit, gaps, gap_summary(gaps)),
        table2_svd_topics({"Topic 1 (inflation)": {"price_level": 0.41, "grocery": 0.22}}),
        table3_model_comparison({"Model 1": {"r2": 0.766, "aic": 1175.48}}),
        table4_granger_and_iv(
            [{"direction": "media -> gap", "lags": 3, "f_stat": 4.1, "p_value": 0.007}]
        ),
    ]


def test_markdown_and_latex_agree_on_column_order(fitted):
    for table in _all_tables(fitted):
        expected = [[escape_latex(name) for name in panel.columns] for panel in table.panels]
        assert _latex_header_cells(table.to_latex()) == expected
        assert _markdown_header_cells(table.to_markdown()) == [
            list(panel.columns) for panel in table.panels
        ]


def test_both_renderers_quote_the_same_numbers(fitted):
    fit, _ = fitted
    table = table1_baseline_and_gap(fit)
    latex, markdown = table.to_latex(), table.to_markdown()
    for cell in (format_number(fit.r_squared, 3), format_number(fit.aic, 1)):
        assert cell in latex
        assert cell in markdown


def test_every_table_states_its_significance_and_se_conventions(fitted):
    for table in _all_tables(fitted):
        if table.number == 2:
            continue  # Table 2 reports loadings, not inference — no stars, no SEs.
        assert SIGNIFICANCE_NOTE in table.footnotes
        assert HAC_NOTE in table.footnotes
        assert "maxlags=12" in table.to_markdown()


def test_panel_rejects_rows_of_the_wrong_width():
    with pytest.raises(TableError, match="cells"):
        Panel(columns=("a", "b"), rows=(Row.data("only one"),))


def test_table_needs_at_least_one_panel():
    with pytest.raises(TableError):
        Table(number=1, title="t", panels=())


# ---------------------------------------------------------------------------
# Table 1 — against the real BaselineFit / gap API
# ---------------------------------------------------------------------------


def test_table1_lists_regressors_in_declared_order_with_intercept_first(fitted):
    fit, _ = fitted
    rows = table1_baseline_and_gap(fit).panels[0].rows
    assert [row.cells[0] for row in rows] == ["const", *fit.features]


def test_table1_gap_panel_reports_untested_months_as_na_not_no(fitted):
    fit, holdout = fitted
    holdout = holdout.copy()
    holdout.loc[holdout.index[1], "sentiment"] = np.nan
    gaps, _ = sentiment_gap(fit, holdout)
    gap_rows = table1_baseline_and_gap(fit, gaps).panels[2].rows
    flags = [row.cells[-1] for row in gap_rows if row.kind == "data"]
    assert NA in flags
    assert set(flags) <= {"yes", "no", NA}


def test_table1_names_dropped_months_in_a_footnote(fitted):
    fit, holdout = fitted
    gaps, _ = sentiment_gap(fit, holdout)
    table = table1_baseline_and_gap(fit, gaps, dropped=("2026-09: missing pce_yoy",))
    assert any("2026-09" in note for note in table.footnotes)
    assert "2026-09" in table.to_markdown()
    assert "2026-09" in table.to_latex()


def test_table1_truncates_long_gap_panels_and_says_so(fitted):
    fit, holdout = fitted
    gaps, _ = sentiment_gap(fit, holdout)
    table = table1_baseline_and_gap(fit, gaps, max_gap_rows=3)
    sections = [row.cells[0] for row in table.panels[2].rows if row.kind == "section"]
    assert any("further months omitted" in text for text in sections)


def test_table1_renders_without_a_holdout(fitted):
    fit, _ = fitted
    table = table1_baseline_and_gap(fit)
    assert len(table.panels) == 2
    assert table.to_latex().count(r"\begin{tabular}") == 2
    assert "Panel C" not in table.to_markdown()


def test_table1_rejects_a_gap_frame_missing_the_gap_column(fitted):
    fit, holdout = fitted
    gaps, _ = sentiment_gap(fit, holdout)
    with pytest.raises(TableError, match="gap"):
        table1_baseline_and_gap(fit, gaps.drop(columns=["gap"]))


def test_table1_summary_block_rounds_to_two_decimals(fitted):
    fit, holdout = fitted
    gaps, _ = sentiment_gap(fit, holdout)
    summary = gap_summary(gaps)
    text = table1_baseline_and_gap(fit, gaps, summary).to_markdown()
    assert format_number(summary["mean_gap"], 2) in text
    assert str(summary["worst_month"]) in text


# ---------------------------------------------------------------------------
# Table 2 — SVD contract
# ---------------------------------------------------------------------------

TOPIC_FRAME = pd.DataFrame(
    {
        "topic": ["Topic 1", "Topic 1", "Topic 2"],
        "term": ["price_level", "grocery", "layoff"],
        "loading": [0.41238, 0.2211, -0.3099],
    }
)


def test_table2_accepts_a_frame_and_rounds_loadings_to_three_dp():
    table = table2_svd_topics(TOPIC_FRAME)
    rows = table.panels[0].rows
    assert [row.cells for row in rows] == [
        ("Topic 1", "price_level", "0.412"),
        ("", "grocery", "0.221"),
        ("Topic 2", "layoff", "-0.310"),
    ]
    assert "price\\_level" in table.to_latex()


def test_table2_accepts_a_mapping_of_topics_to_terms():
    mapping = {"Topic 1": [("price_level", 0.41238), ("grocery", 0.2211)]}
    assert table2_svd_topics(mapping).panels[0].rows[0].cells == (
        "Topic 1",
        "price_level",
        "0.412",
    )


def test_table2_top_n_truncates_terms():
    table = table2_svd_topics(TOPIC_FRAME, top_n=1)
    assert [row.cells[1] for row in table.panels[0].rows] == ["price_level", "layoff"]


def test_table2_extras_add_columns_only_when_supplied():
    plain = table2_svd_topics(TOPIC_FRAME)
    rich = table2_svd_topics(
        TOPIC_FRAME, extras={"Topic 1": {"singular_value": 3.21, "energy_share": 0.1842}}
    )
    assert plain.panels[0].columns == ("Topic", "Term", "Loading (U_k)")
    assert rich.panels[0].columns[:3] == ("Topic", "Singular value", "Energy share")
    assert rich.panels[0].rows[0].cells == ("Topic 1", "3.210", "0.184", "price_level", "0.412")
    # Topic 2 has no extras supplied and must not invent them.
    assert rich.panels[0].rows[2].cells[1:3] == (NA, NA)


def test_table2_renders_a_marked_placeholder_when_the_corpus_is_absent():
    table = table2_svd_topics(None, unavailable_reason="GDELT pull not run")
    text = table.to_markdown()
    assert NA in text
    assert "GDELT pull not run" in text
    assert table.panels[0].columns == ("Topic", "Term", "Loading (U_k)")


def test_table2_rejects_a_frame_that_breaks_the_contract():
    with pytest.raises(TableError, match="topic, term, loading"):
        table2_svd_topics(pd.DataFrame({"topic": ["a"], "weight": [1.0]}))


# ---------------------------------------------------------------------------
# Table 3 — model comparison and missing-model degradation
# ---------------------------------------------------------------------------

COMPARISON = {
    "Model 1": {"r2": 0.76634, "adj_r2": 0.76234, "rmse": 6.2049, "aic": 1175.4832, "n": 180},
    "Model 2": {"r2": 0.83712, "adj_r2": 0.83112, "rmse": 5.3351, "aic": 1116.2411, "n": 180},
}


def test_table3_rounds_each_statistic_to_its_own_precision():
    table = table3_model_comparison(COMPARISON)
    rows = {row.cells[0]: row.cells[1:] for row in table.panels[0].rows}
    assert rows["R^2"] == ("0.766", "0.837")
    assert rows["Adjusted R^2"] == ("0.762", "0.831")
    assert rows["RMSE"] == ("6.20", "5.34")
    assert rows["AIC"] == ("1175.5", "1116.2")
    assert rows["Observations"] == ("180", "180")


def test_table3_marks_an_unavailable_model_instead_of_dropping_it():
    table = table3_model_comparison(
        COMPARISON,
        unavailable={"Model 3": "media corpus not collected"},
        model_order=("Model 1", "Model 2", "Model 3"),
    )
    assert table.panels[0].columns == (
        "Statistic",
        "Model 1",
        "Model 2",
        f"Model 3 ({NA})",
    )
    for row in table.panels[0].rows:
        assert row.cells[-1] == NA
    assert any("media corpus not collected" in note for note in table.footnotes)
    for text in (table.to_markdown(), table.to_latex()):
        assert "Model 3" in text
        assert "media corpus not collected" in text


def test_table3_accepts_a_dataframe_with_models_as_columns():
    frame = pd.DataFrame(COMPARISON)
    assert table3_model_comparison(frame).panels[0].columns == (
        "Statistic",
        "Model 1",
        "Model 2",
    )


def test_table3_absent_metrics_render_as_na_not_zero():
    table = table3_model_comparison({"Model 1": {"r2": 0.5}})
    rows = {row.cells[0]: row.cells[1] for row in table.panels[0].rows}
    assert rows["R^2"] == "0.500"
    assert rows["AIC"] == NA
    assert rows["Mean out-of-sample gap"] == NA


def test_table3_needs_at_least_one_model():
    with pytest.raises(TableError, match="no models"):
        table3_model_comparison({})


# ---------------------------------------------------------------------------
# Table 4 — Granger / IV contract
# ---------------------------------------------------------------------------

GRANGER = [
    {"direction": "media_tone -> gap", "lags": 3, "f_stat": 4.1123, "p_value": 0.0068, "n": 96},
    {"direction": "gap -> media_tone", "lags": 3, "f_stat": 0.8821, "p_value": 0.4531, "n": 96},
]
IV = {"parameter": "media_tone -> gap", "coefficient": 2.3344, "std_err": 0.9111, "p_value": 0.0121, "n": 96}


def test_table4_formats_f_tests_with_stars_and_p_floor():
    table = table4_granger_and_iv(GRANGER, IV)
    rows = table.panels[0].rows
    assert rows[0].cells == ("media_tone -> gap", "3", "4.11***", "0.007", "96")
    assert rows[1].cells == ("gap -> media_tone", "3", "0.88", "0.453", "96")
    assert table.panels[1].rows[0].cells == ("media_tone -> gap", "2.334**", "0.911", "0.012", "96")
    assert "media\\_tone" in table.to_latex()


def test_table4_marks_the_iv_panel_unavailable_without_crashing():
    table = table4_granger_and_iv(GRANGER, None, iv_unavailable_reason="rank condition failed")
    assert len(table.panels) == 2
    assert NA in table.panels[1].rows[0].cells[0]
    assert "rank condition failed" in table.to_markdown()
    assert "rank condition failed" in table.to_latex()


def test_table4_marks_both_panels_unavailable():
    table = table4_granger_and_iv(None, None)
    assert all(NA in panel.rows[0].cells[0] for panel in table.panels)


def test_table4_accepts_a_dataframe_and_labels_precedence_not_causation():
    table = table4_granger_and_iv(pd.DataFrame(GRANGER), IV)
    assert table.panels[0].rows[0].cells[2].startswith("4.11")
    assert any("predictive precedence" in note for note in table.footnotes)


def test_table4_reports_the_regime_split_when_supplied():
    table = table4_granger_and_iv(GRANGER, {**IV, "regime_split": "2020-03 volatility break"})
    sections = [row for row in table.panels[1].rows if row.kind == "section"]
    assert sections and "2020-03" in sections[0].cells[0]


def test_table4_rejects_results_that_break_the_contract():
    with pytest.raises(TableError, match="missing required keys"):
        table4_granger_and_iv([{"direction": "a -> b", "lags": 2}])
    with pytest.raises(TableError, match="missing required keys"):
        table4_granger_and_iv(GRANGER, {"parameter": "x", "coefficient": 1.0})


# ---------------------------------------------------------------------------
# save_table
# ---------------------------------------------------------------------------


def test_save_table_round_trip(tmp_path, toy_table):
    text = to_latex(toy_table)
    written = save_table(text, "table9.tex", directory=tmp_path)
    assert written == tmp_path / "table9.tex"
    assert written.read_text(encoding="utf-8") == text + "\n"


def test_save_table_creates_missing_directories_and_accepts_absolute_paths(tmp_path):
    target = tmp_path / "nested" / "deeper" / "t.md"
    written = save_table("| a |\n", target)
    assert written == target
    assert written.read_text(encoding="utf-8") == "| a |\n"


def test_save_table_defaults_to_outputs_tables(tmp_path, toy_table):
    from perceptual_bias import table_generator

    assert table_generator.OUTPUT_DIR.name == "tables"
    assert table_generator.OUTPUT_DIR.parent.name == "outputs"
    # Write under a temp directory so the test does not litter the repo.
    written = save_table(to_markdown(toy_table), Path("roundtrip.md"), directory=tmp_path)
    assert written.parent == tmp_path


def test_save_table_refuses_an_unrendered_object(tmp_path, toy_table):
    with pytest.raises(TableError, match="rendered table text"):
        save_table(toy_table, tmp_path / "oops.tex")  # type: ignore[arg-type]
