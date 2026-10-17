"""Publication-quality table rendering for the Sentiment Gap report.

Every table in the report is built once as a :class:`Table` — an ordered,
already-rounded grid of strings — and then rendered twice, by :func:`to_latex`
for the PDF and :func:`to_markdown` for the notebook and Streamlit app. The
two renderers read the *same* object, which is the reason they cannot drift:
column order, row order, rounding and significance stars are decided when the
``Table`` is built, not when it is printed.

Rounding is fixed centrally (see :data:`DP`) because a report that quotes
``0.7663245801792537`` has told the reader the author never looked at it.

Design constraint: this module imports **no** analysis module. Tables 2-4 are
produced by modules written in parallel, so each table function accepts plain
data (a ``DataFrame``, a mapping, or a sequence of mappings) against the
minimal contract documented below. Integration is then mechanical: whoever owns
the producing module supplies those keys and nothing else changes here.

Input contracts
---------------

**Table 1 — baseline OLS + out-of-sample gap.** Built against the real API:
``fit`` is a ``baseline_model.BaselineFit`` and ``gaps`` the frame returned by
``baseline_model.sentiment_gap`` (columns ``observed``, ``predicted``,
``pi_lower``, ``pi_upper``, ``gap``, ``outside_pi``). ``summary`` is the
``pd.Series`` from ``gap_summary``; if omitted the headline block is skipped.

**Table 2 — SVD latent topics and term loadings.** ``topics`` is either

* a ``DataFrame`` with columns ``topic``, ``term``, ``loading`` (one row per
  term per topic; ``loading`` is the term's entry in that topic's column of
  ``U_k``), or
* a mapping ``{topic_label: [(term, loading), ...]}`` / ``{topic_label:
  {term: loading}}``.

Optional per-topic scalars may be supplied as ``extras``, a mapping
``{topic_label: {"singular_value": float, "energy_share": float}}``. Row order
within a topic is taken as given (producers are expected to sort by
``|loading|`` descending); ``top_n`` truncates it.

**Table 3 — model comparison matrix.** ``comparison`` is either a ``DataFrame``
whose **columns are models** and whose index holds metric keys, or a mapping
``{model_label: {metric: value}}``. Recognised metric keys, in render order:
``r2``, ``adj_r2``, ``rmse``, ``aic``, ``n``, ``mean_gap``. Unknown keys are
ignored; absent keys render as ``n/a``. Pass ``unavailable={"Model 3": "no
media corpus"}`` to emit a model as an explicitly-marked not-available column
rather than dropping it silently — a dropped column reads as a model that was
never specified, which is a different claim.

**Table 4 — Granger F-tests and Rigobon-Sack IV.** ``granger`` is a
``DataFrame`` or sequence of mappings with keys ``direction`` (e.g.
``"media tone -> gap"``), ``lags``, ``f_stat``, ``p_value``; optional ``n``.
``iv`` is a mapping, or sequence of mappings, with keys ``parameter``,
``coefficient``, ``std_err``, ``p_value``; optional ``n`` and
``regime_split``. ``iv=None`` renders the IV panel as not available, with the
reason given by ``iv_unavailable_reason``.

Both renderers escape nothing the caller has to think about: labels pass
through :func:`escape_latex`, so ``pce_yoy`` and ``R^2`` survive LaTeX intact.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conventions, stated once and reused by every table.
# ---------------------------------------------------------------------------

#: Significance thresholds, loosest first. Reported in every footnote so the
#: convention is never left to the reader to guess.
STAR_LEVELS: tuple[tuple[float, str], ...] = ((0.01, "***"), (0.05, "**"), (0.10, "*"))

SIGNIFICANCE_NOTE = "Significance: * p<0.10, ** p<0.05, *** p<0.01."
HAC_NOTE = "Standard errors are Newey-West HAC with maxlags=12."

#: Smallest p-value printed as a number; below this the report says ``<0.001``
#: rather than implying precision three more decimals would not support.
P_FLOOR = 0.001

#: Decimal places by quantity. Coefficients get 3 (2 would hide the GDP
#: coefficient's magnitude, which is -0.159); AIC gets 1 because its integer
#: part is four digits wide and its third decimal carries no information.
DP: dict[str, int] = {
    "coefficient": 3,
    "std_err": 3,
    "t_stat": 2,
    "p_value": 3,
    "r2": 3,
    "rmse": 2,
    "aic": 1,
    "sentiment": 2,
    "loading": 3,
    "f_stat": 2,
    "share": 3,
}

NA = "n/a"

#: Where :func:`save_table` puts a relative path. The directory is committed.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "tables"

_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"$<$",
    ">": r"$>$",
}


#: Cells whose LaTeX form is mathematical rather than literal. Escaping would
#: give ``R\textasciicircum{}2``, which compiles but reads as a typo in a
#: published table. Matched on the whole cell only, so a label that merely
#: contains "R^2" is still escaped character by character.
_LATEX_LITERALS: dict[str, str] = {
    "R^2": r"$R^2$",
    "Adjusted R^2": r"Adjusted $R^2$",
    "Loading (U_k)": r"Loading ($U_k$)",
}


class TableError(Exception):
    """Raised when a table cannot be built or rendered as specified."""


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


def escape_latex(text: Any) -> str:
    """Make `text` safe to paste into a LaTeX cell.

    This is the module's highest-risk function. Variable names in this project
    are snake_case (``pce_yoy``, ``real_gdp_qoq``, ``mortgage_30y``) and a bare
    underscore in text mode is a LaTeX *error*, not a typo — the report would
    fail to compile, or worse, compile with the rest of the label subscripted.
    ``<`` and ``>`` are mapped into math mode because in text mode the default
    fonts render them as inverted punctuation.
    """
    raw = "" if text is None else str(text)
    if raw in _LATEX_LITERALS:
        return _LATEX_LITERALS[raw]
    # Single pass over characters: a sequential chain of `str.replace` calls
    # would re-escape the braces and backslashes the earlier replacements had
    # just introduced (``\\`` -> ``\textbackslash{}`` -> ``\textbackslash\{\}``).
    return "".join(_LATEX_ESCAPES.get(char, char) for char in raw)


def _is_missing(value: Any) -> bool:
    """True for None, NaN, pandas NA and empty strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        if value is pd.NA or pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def format_number(value: Any, dp: int = 2, *, na: str = NA) -> str:
    """Round to `dp` decimals, or return `na` for anything unusable.

    Infinities are returned as ``inf``/``-inf`` rather than formatted: an AIC of
    ``inf`` is a fitting failure the reader must see, not a large number.
    """
    if _is_missing(value):
        return na
    try:
        number = float(value)
    except (TypeError, ValueError):
        return na
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if dp <= 0:
        return f"{number:.0f}"
    return f"{number:.{dp}f}"


def format_integer(value: Any, *, na: str = NA) -> str:
    if _is_missing(value):
        return na
    try:
        return f"{round(float(value))}"
    except (TypeError, ValueError):
        return na


def format_p_value(value: Any, *, na: str = NA) -> str:
    """``<0.001`` below the floor, three decimals above it.

    A p-value of 2.3e-9 printed as ``0.000`` claims the wrong thing (that it
    rounds to zero); ``<0.001`` is the convention and is what journals expect.
    """
    if _is_missing(value):
        return na
    try:
        p = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(p):
        return na
    if p < P_FLOOR:
        return f"<{P_FLOOR:.3f}"
    return f"{p:.{DP['p_value']}f}"


def significance_stars(p_value: Any) -> str:
    """Stars per :data:`STAR_LEVELS`; empty string when p is missing.

    Missing is *not* insignificant. A regressor whose p-value could not be
    computed gets no stars and no claim either way, which is why this returns
    ``""`` rather than raising or guessing.
    """
    if _is_missing(p_value):
        return ""
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(p):
        return ""
    for threshold, stars in STAR_LEVELS:
        if p < threshold:
            return stars
    return ""


def format_coefficient(value: Any, p_value: Any = None, dp: int | None = None) -> str:
    """Coefficient with significance stars appended, e.g. ``-4.284***``."""
    rendered = format_number(value, DP["coefficient"] if dp is None else dp)
    if rendered == NA:
        return rendered
    return rendered + significance_stars(p_value)


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One rendered row. `kind` of ``"section"`` makes it a spanning subhead."""

    cells: tuple[str, ...]
    kind: str = "data"

    @classmethod
    def data(cls, *cells: Any) -> Row:
        return cls(tuple("" if c is None else str(c) for c in cells))

    @classmethod
    def section(cls, label: str) -> Row:
        return cls((label,), kind="section")


@dataclass(frozen=True)
class Panel:
    """A block of rows sharing one header.

    Table 1 needs two panels (coefficients, then the month-by-month gap) whose
    columns genuinely differ. Forcing them into one header would mean inventing
    a column name that is true of neither.
    """

    columns: tuple[str, ...]
    rows: tuple[Row, ...]
    label: str | None = None
    align: str | None = None

    def __post_init__(self) -> None:
        if not self.columns:
            raise TableError("a panel needs at least one column")
        width = len(self.columns)
        for row in self.rows:
            if row.kind == "data" and len(row.cells) != width:
                raise TableError(
                    f"row {row.cells!r} has {len(row.cells)} cells; "
                    f"panel header has {width}"
                )

    @property
    def column_spec(self) -> str:
        if self.align:
            if len(self.align) != len(self.columns):
                raise TableError("align string must have one letter per column")
            return self.align
        return "l" + "r" * (len(self.columns) - 1)


@dataclass(frozen=True)
class Table:
    """A rendered-once, printed-twice table."""

    number: int
    title: str
    panels: tuple[Panel, ...]
    footnotes: tuple[str, ...] = field(default_factory=tuple)
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.panels:
            raise TableError(f"Table {self.number} has no panels")

    @property
    def caption(self) -> str:
        return f"Table {self.number}: {self.title}"

    @property
    def tex_label(self) -> str:
        return self.label or f"tab:table{self.number}"

    def columns(self) -> tuple[tuple[str, ...], ...]:
        """Column headers per panel — the order both renderers must honour."""
        return tuple(panel.columns for panel in self.panels)

    def to_latex(self) -> str:
        return to_latex(self)

    def to_markdown(self) -> str:
        return to_markdown(self)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def to_latex(table: Table) -> str:
    """Render as a ``table`` float with ``booktabs`` rules.

    Requires ``\\usepackage{booktabs}``. Notes are set in a ``minipage`` so they
    wrap to the table width instead of running off the page.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(table.caption)}}}",
        rf"\label{{{table.tex_label}}}",
    ]
    for index, panel in enumerate(table.panels):
        if index:
            lines.append(r"\vspace{0.75em}")
        if panel.label:
            lines.append(rf"\textit{{{escape_latex(panel.label)}}}\par\vspace{{0.25em}}")
        width = len(panel.columns)
        lines.append(rf"\begin{{tabular}}{{{panel.column_spec}}}")
        lines.append(r"\toprule")
        lines.append(
            " & ".join(rf"\textbf{{{escape_latex(name)}}}" for name in panel.columns)
            + r" \\"
        )
        lines.append(r"\midrule")
        for row in panel.rows:
            if row.kind == "section":
                lines.append(
                    rf"\multicolumn{{{width}}}{{l}}{{\textit{{{escape_latex(row.cells[0])}}}}} \\"
                )
            else:
                lines.append(
                    " & ".join(escape_latex(cell) for cell in row.cells) + r" \\"
                )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    if table.footnotes:
        lines.append(r"\vspace{0.4em}")
        lines.append(r"\begin{minipage}{\linewidth}")
        lines.append(r"\footnotesize")
        for position, note in enumerate(table.footnotes):
            prefix = r"\textit{Notes:} " if position == 0 else ""
            lines.append(prefix + escape_latex(note) + r"\\")
        lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_markdown(table: Table) -> str:
    """Render as GitHub-flavoured Markdown, same columns in the same order."""
    lines = [f"**{table.caption}**", ""]
    for index, panel in enumerate(table.panels):
        if index:
            lines.append("")
        if panel.label:
            lines.append(f"*{panel.label}*")
            lines.append("")
        width = len(panel.columns)
        lines.append("| " + " | ".join(_md_cell(c) for c in panel.columns) + " |")
        lines.append("|" + "|".join(_md_rule(letter) for letter in panel.column_spec) + "|")
        for row in panel.rows:
            if row.kind == "section":
                cells = [f"**{_md_cell(row.cells[0])}**"] + [""] * (width - 1)
            else:
                cells = [_md_cell(cell) for cell in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
    if table.footnotes:
        lines.append("")
        for position, note in enumerate(table.footnotes):
            prefix = "*Notes:* " if position == 0 else ""
            lines.append(f"{prefix}{note}  ")
    return "\n".join(lines)


def _md_cell(text: Any) -> str:
    """Markdown has one fatal character in a table cell: the pipe."""
    return ("" if text is None else str(text)).replace("|", r"\|")


def _md_rule(letter: str) -> str:
    return {"r": "---:", "c": ":---:"}.get(letter, ":---")


def save_table(text: str, path: str | Path, *, directory: Path | None = None) -> Path:
    """Write `text` to `path`, resolving relative paths under ``outputs/tables``.

    Returns the path actually written so callers can log or assert on it.
    """
    if not isinstance(text, str):
        raise TableError("save_table expects rendered table text, not an object")
    target = Path(path)
    if not target.is_absolute():
        target = (directory or OUTPUT_DIR) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.endswith("\n") else text + "\n"
    target.write_text(body, encoding="utf-8")
    logger.info("wrote table to %s", target)
    return target


# ---------------------------------------------------------------------------
# Normalisation helpers for the contract-based tables
# ---------------------------------------------------------------------------


def _records(data: Any, *, what: str) -> list[dict[str, Any]]:
    """Accept a DataFrame or a sequence/single mapping; return plain records."""
    if data is None:
        raise TableError(f"{what} is required")
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        records = []
        for item in data:
            if not isinstance(item, Mapping):
                raise TableError(f"{what} rows must be mappings, got {type(item)!r}")
            records.append(dict(item))
        return records
    raise TableError(f"{what} must be a DataFrame, mapping or sequence of mappings")


def _require_keys(
    records: Sequence[Mapping[str, Any]], keys: Sequence[str], what: str
) -> None:
    """Fail with `TableError` naming every missing contract key at once.

    Raised up front rather than letting a ``KeyError`` surface from inside a
    render loop, where the message would name a cell instead of the contract.
    """
    if not records:
        raise TableError(f"{what} is empty; nothing to tabulate")
    missing = sorted({key for key in keys if key not in records[0]})
    if missing:
        raise TableError(
            f"{what} is missing required keys {missing}; "
            f"got {sorted(records[0])}. See the module docstring's input contract."
        )


def _first(record: Mapping[str, Any], *names: str) -> Any:
    """First present key among `names` — lets producers use either spelling."""
    for name in names:
        if name in record:
            return record[name]
    return None


# ---------------------------------------------------------------------------
# Table 1 — baseline OLS and the out-of-sample gap
# ---------------------------------------------------------------------------

TABLE1_COEF_COLUMNS = ("Regressor", "Coefficient", "HAC SE", "t", "p")
TABLE1_FIT_COLUMNS = ("Statistic", "Value")
TABLE1_GAP_COLUMNS = ("Month", "Observed", "Predicted", "Gap", "Outside 95% PI")


def table1_baseline_and_gap(
    fit: Any,
    gaps: pd.DataFrame | None = None,
    summary: pd.Series | Mapping[str, Any] | None = None,
    *,
    max_gap_rows: int | None = 24,
    dropped: Sequence[str] = (),
) -> Table:
    """Table 1: the 2010-2024 baseline and the 2025-26 gap breakdown.

    `fit` is a ``BaselineFit`` (duck-typed: anything exposing ``ols_result``,
    ``features``, ``target`` and the fit-statistic properties). `gaps` is the
    frame from ``sentiment_gap``; `summary` the series from ``gap_summary``.
    Both gap arguments are optional so the baseline panel can be rendered
    before the holdout exists.

    `dropped` carries the months with no computable gap. They are printed in a
    footnote because a table headlined "2025-2026" that stops in August without
    saying so is misleading — the house rule is that dropped rows are named.
    """
    panels: list[Panel] = [
        Panel(
            columns=TABLE1_COEF_COLUMNS,
            rows=tuple(_table1_coefficient_rows(fit)),
            label="Panel A. Baseline OLS, monthly, HAC(12) standard errors",
        ),
        Panel(
            columns=TABLE1_FIT_COLUMNS,
            rows=tuple(_table1_fit_rows(fit)),
            label="Panel B. Fit statistics",
        ),
    ]

    footnotes = [
        (f"Dependent variable: {getattr(fit, 'target', 'sentiment')}. "
        f"n = {format_integer(getattr(fit, 'n_train', None))} training months."),
        HAC_NOTE,
        SIGNIFICANCE_NOTE,
    ]

    if gaps is not None:
        panels.append(
            Panel(
                columns=TABLE1_GAP_COLUMNS,
                rows=tuple(_table1_gap_rows(gaps, summary, max_gap_rows)),
                label="Panel C. Out-of-sample Sentiment Gap (observed - predicted)",
            )
        )
        footnotes.append(
            "Gap is evaluated out of sample; the prediction interval is a 95% "
            "interval for an individual month, so it includes residual variance."
        )
    if dropped:
        footnotes.append("Months with no computable gap: " + "; ".join(dropped) + ".")

    return Table(
        number=1,
        title="Historical macro OLS baseline (2010-2024) and the 2025-2026 Sentiment Gap",
        panels=tuple(panels),
        footnotes=tuple(footnotes),
        label="tab:baseline-gap",
    )


def _table1_coefficient_rows(fit: Any) -> list[Row]:
    result = getattr(fit, "ols_result", None)
    if result is None:
        raise TableError("fit has no ols_result; cannot tabulate coefficients")
    params = result.params
    std_errs = result.bse
    t_stats = result.tvalues
    p_values = result.pvalues
    names = [name for name in params.index]
    # Put the intercept first, then the features in their declared order.
    ordered = [name for name in names if name == "const"]
    ordered += [name for name in getattr(fit, "features", ()) if name in names]
    ordered += [name for name in names if name not in ordered]

    rows: list[Row] = []
    for name in ordered:
        rows.append(
            Row.data(
                name,
                format_coefficient(params[name], p_values.get(name)),
                format_number(std_errs.get(name), DP["std_err"]),
                format_number(t_stats.get(name), DP["t_stat"]),
                format_p_value(p_values.get(name)),
            )
        )
    return rows


def _table1_fit_rows(fit: Any) -> list[Row]:
    pairs = (
        ("R^2", getattr(fit, "r_squared", None), DP["r2"]),
        ("Adjusted R^2", getattr(fit, "adj_r_squared", None), DP["r2"]),
        ("RMSE (in sample)", getattr(fit, "rmse_in_sample", None), DP["rmse"]),
        ("AIC", getattr(fit, "aic", None), DP["aic"]),
    )
    rows = [Row.data(label, format_number(value, dp)) for label, value, dp in pairs]
    start, end = getattr(fit, "train_start", None), getattr(fit, "train_end", None)
    if start is not None and end is not None:
        rows.append(Row.data("Training window", f"{_month(start)} to {_month(end)}"))
    rows.append(Row.data("Observations", format_integer(getattr(fit, "n_train", None))))
    return rows


def _table1_gap_rows(
    gaps: pd.DataFrame,
    summary: pd.Series | Mapping[str, Any] | None,
    max_rows: int | None,
) -> list[Row]:
    for column in ("observed", "predicted", "gap"):
        if column not in gaps.columns:
            raise TableError(f"gap frame is missing required column {column!r}")

    rows: list[Row] = []
    frame = gaps if max_rows is None else gaps.iloc[:max_rows]
    for stamp, record in frame.iterrows():
        rows.append(
            Row.data(
                _month(stamp),
                format_number(record.get("observed"), DP["sentiment"]),
                format_number(record.get("predicted"), DP["sentiment"]),
                format_number(record.get("gap"), DP["sentiment"]),
                _yes_no(record.get("outside_pi")),
            )
        )
    if max_rows is not None and len(gaps) > max_rows:
        rows.append(Row.section(f"({len(gaps) - max_rows} further months omitted)"))

    if summary is not None:
        stats = dict(summary)
        rows.append(Row.section("Summary"))
        rows.append(
            Row.data(
                "Mean gap",
                "",
                "",
                format_number(stats.get("mean_gap"), DP["sentiment"]),
                "",
            )
        )
        rows.append(
            Row.data(
                "Median gap",
                "",
                "",
                format_number(stats.get("median_gap"), DP["sentiment"]),
                "",
            )
        )
        outside = stats.get("months_outside_pi")
        tested = stats.get("n_tested")
        rows.append(
            Row.data(
                f"Worst month ({stats.get('worst_month', NA)})",
                format_number(stats.get("worst_observed"), DP["sentiment"]),
                format_number(stats.get("worst_predicted"), DP["sentiment"]),
                format_number(stats.get("worst_gap"), DP["sentiment"]),
                "",
            )
        )
        rows.append(
            Row.data(
                "Months outside 95% PI",
                "",
                "",
                f"{format_integer(outside)} of {format_integer(tested)}",
                _share(stats.get("share_outside_pi")),
            )
        )
    return rows


def _month(stamp: Any) -> str:
    try:
        return pd.Timestamp(stamp).strftime("%Y-%m")
    except (TypeError, ValueError):
        return str(stamp)


def _yes_no(value: Any) -> str:
    """``yes``/``no``/``n/a`` — NaN must never render as ``no``.

    A month whose sentiment is missing was never tested against the interval;
    printing ``no`` would read as "the model explained this month fine".
    """
    if _is_missing(value):
        return NA
    return "yes" if bool(value) else "no"


def _share(value: Any) -> str:
    if _is_missing(value):
        return NA
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return NA


# ---------------------------------------------------------------------------
# Table 2 — SVD latent topics and term loadings
# ---------------------------------------------------------------------------

TABLE2_COLUMNS = ("Topic", "Term", "Loading (U_k)")
TABLE2_COLUMNS_WITH_EXTRAS = ("Topic", "Singular value", "Energy share", "Term", "Loading (U_k)")


def table2_svd_topics(
    topics: Any,
    *,
    extras: Mapping[str, Mapping[str, Any]] | None = None,
    top_n: int | None = 8,
    unavailable_reason: str | None = None,
) -> Table:
    """Table 2: latent economic topics and their term loadings in ``U_k``.

    See the module docstring for the accepted shapes. ``extras`` adds the
    per-topic singular value and energy share; the columns only appear when
    extras are supplied, so the table never carries an empty column.

    Pass ``unavailable_reason`` (with ``topics=None``) when the corpus has not
    been collected: the table renders a marked not-available row instead of
    being omitted, so a reader can see the analysis was specified and why it is
    absent.
    """
    if topics is None or (isinstance(topics, pd.DataFrame) and topics.empty):
        reason = unavailable_reason or "SVD topics unavailable (no media corpus)"
        return Table(
            number=2,
            title="SVD latent economic topics and term loadings",
            panels=(
                Panel(
                    columns=TABLE2_COLUMNS,
                    rows=(Row.data(NA, f"{NA} - {reason}", NA),),
                ),
            ),
            footnotes=(
                ("Loadings are entries of the left singular vectors U_k of the "
                "TF-IDF term-document matrix, which is not mean-centred."),
            ),
            label="tab:svd-topics",
        )

    grouped = _normalise_topics(topics)
    with_extras = bool(extras)
    columns = TABLE2_COLUMNS_WITH_EXTRAS if with_extras else TABLE2_COLUMNS
    align = "lrrlr" if with_extras else "llr"

    rows: list[Row] = []
    for topic, terms in grouped.items():
        shown = terms if top_n is None else terms[:top_n]
        for position, (term, loading) in enumerate(shown):
            label = topic if position == 0 else ""
            if with_extras:
                topic_extras = dict((extras or {}).get(topic, {}))
                rows.append(
                    Row.data(
                        label,
                        format_number(topic_extras.get("singular_value"), DP["coefficient"])
                        if position == 0
                        else "",
                        format_number(topic_extras.get("energy_share"), DP["share"])
                        if position == 0
                        else "",
                        term,
                        format_number(loading, DP["loading"]),
                    )
                )
            else:
                rows.append(Row.data(label, term, format_number(loading, DP["loading"])))

    footnotes = [
        ("Loadings are entries of the left singular vectors U_k of the TF-IDF "
        "term-document matrix, which is not mean-centred: the origin means "
        "'no dictionary terms present', not 'average document'."),
    ]
    if top_n is not None:
        footnotes.append(f"Top {top_n} terms per topic by absolute loading.")

    return Table(
        number=2,
        title="SVD latent economic topics and term loadings",
        panels=(Panel(columns=columns, rows=tuple(rows), align=align),),
        footnotes=tuple(footnotes),
        label="tab:svd-topics",
    )


def _normalise_topics(topics: Any) -> dict[str, list[tuple[str, Any]]]:
    grouped: dict[str, list[tuple[str, Any]]] = {}
    if isinstance(topics, pd.DataFrame):
        missing = [c for c in ("topic", "term", "loading") if c not in topics.columns]
        if missing:
            raise TableError(
                f"topic frame is missing required columns {missing}; "
                "contract is topic, term, loading"
            )
        for record in topics.to_dict(orient="records"):
            grouped.setdefault(str(record["topic"]), []).append(
                (str(record["term"]), record["loading"])
            )
        return grouped
    if isinstance(topics, Mapping):
        for topic, terms in topics.items():
            if isinstance(terms, Mapping):
                items = list(terms.items())
            elif isinstance(terms, Iterable) and not isinstance(terms, (str, bytes)):
                items = [tuple(pair) for pair in terms]  # type: ignore[misc]
            else:
                raise TableError(f"topic {topic!r} must map to terms, got {type(terms)!r}")
            grouped[str(topic)] = [(str(term), loading) for term, loading in items]
        return grouped
    raise TableError("topics must be a DataFrame or a mapping of topic -> terms")


# ---------------------------------------------------------------------------
# Table 3 — model comparison matrix
# ---------------------------------------------------------------------------

#: Metric rows in render order, with their label and decimal places.
TABLE3_METRICS: tuple[tuple[str, str, int | None], ...] = (
    ("r2", "R^2", DP["r2"]),
    ("adj_r2", "Adjusted R^2", DP["r2"]),
    ("rmse", "RMSE", DP["rmse"]),
    ("aic", "AIC", DP["aic"]),
    ("n", "Observations", None),
    ("mean_gap", "Mean out-of-sample gap", DP["sentiment"]),
)


def table3_model_comparison(
    comparison: Any,
    *,
    unavailable: Mapping[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> Table:
    """Table 3: R^2, adjusted R^2, RMSE and AIC across Models 1-3.

    Models are **columns** and metrics rows, so a model that could not be
    estimated appears as a column of ``n/a`` with its reason in a footnote.
    Dropping the column instead would misrepresent the research design: Model 3
    was specified, it simply has no media corpus yet.

    `comparison` is a DataFrame with models as columns and metric keys in the
    index, or a mapping ``{model: {metric: value}}``.
    """
    table_data = _normalise_comparison(comparison)
    unavailable = dict(unavailable or {})

    labels = list(model_order) if model_order else list(table_data)
    for name in table_data:
        if name not in labels:
            labels.append(name)
    for name in unavailable:
        if name not in labels:
            labels.append(name)
    if not labels:
        raise TableError("model comparison has no models to render")

    wanted = [key for key in (metrics or [key for key, _, _ in TABLE3_METRICS])]
    metric_specs = {key: (label, dp) for key, label, dp in TABLE3_METRICS}

    headers = tuple(
        f"{name} ({NA})" if name in unavailable else name for name in labels
    )
    rows: list[Row] = []
    for key in wanted:
        label, dp = metric_specs.get(key, (key, DP["coefficient"]))
        cells: list[str] = [label]
        for name in labels:
            if name in unavailable:
                cells.append(NA)
                continue
            value = table_data.get(name, {}).get(key)
            cells.append(format_integer(value) if dp is None else format_number(value, dp))
        rows.append(Row.data(*cells))

    footnotes = [HAC_NOTE, SIGNIFICANCE_NOTE]
    for name, reason in unavailable.items():
        footnotes.append(f"{name} is not available: {reason}.")

    return Table(
        number=3,
        title="Model comparison matrix",
        panels=(Panel(columns=("Statistic",) + headers, rows=tuple(rows)),),
        footnotes=tuple(footnotes),
        label="tab:model-comparison",
    )


def _normalise_comparison(comparison: Any) -> dict[str, dict[str, Any]]:
    if comparison is None:
        return {}
    if isinstance(comparison, pd.DataFrame):
        return {
            str(column): {str(k): v for k, v in comparison[column].items()}
            for column in comparison.columns
        }
    if isinstance(comparison, Mapping):
        out: dict[str, dict[str, Any]] = {}
        for model, metrics in comparison.items():
            if not isinstance(metrics, Mapping):
                raise TableError(
                    f"model {model!r} must map to a metric mapping, got {type(metrics)!r}"
                )
            out[str(model)] = {str(k): v for k, v in metrics.items()}
        return out
    raise TableError("comparison must be a DataFrame or a mapping of model -> metrics")


# ---------------------------------------------------------------------------
# Table 4 — Granger causality and Rigobon-Sack IV
# ---------------------------------------------------------------------------

TABLE4_GRANGER_COLUMNS = ("Direction", "Lags", "F", "p", "n")
TABLE4_IV_COLUMNS = ("Parameter", "Estimate", "SE", "p", "n")


def table4_granger_and_iv(
    granger: Any,
    iv: Any = None,
    *,
    iv_unavailable_reason: str | None = None,
    granger_unavailable_reason: str | None = None,
) -> Table:
    """Table 4: Granger F-tests and Rigobon-Sack IV transmission estimates.

    Either panel may be unavailable; the panel then renders a marked
    not-available row rather than disappearing. The footnote states that
    Granger results are *predictive precedence*, not causation — the project
    commits to labelling every causal claim by the strength of its
    identification, and the table is where a reader looks first.
    """
    panels: list[Panel] = []

    if granger is None:
        reason = granger_unavailable_reason or "no media series to test"
        panels.append(
            Panel(
                columns=TABLE4_GRANGER_COLUMNS,
                rows=(Row.data(f"{NA} - {reason}", NA, NA, NA, NA),),
                label="Panel A. Granger causality F-tests",
            )
        )
    else:
        records = _records(granger, what="granger results")
        _require_keys(records, ("direction", "lags", "f_stat", "p_value"), "granger results")
        rows = [
            Row.data(
                str(record["direction"]),
                format_integer(record.get("lags")),
                format_number(record.get("f_stat"), DP["f_stat"])
                + significance_stars(record.get("p_value")),
                format_p_value(record.get("p_value")),
                format_integer(record.get("n")),
            )
            for record in records
        ]
        panels.append(
            Panel(
                columns=TABLE4_GRANGER_COLUMNS,
                rows=tuple(rows),
                label="Panel A. Granger causality F-tests",
            )
        )

    if iv is None:
        reason = iv_unavailable_reason or "regime covariance rank condition not met"
        panels.append(
            Panel(
                columns=TABLE4_IV_COLUMNS,
                rows=(Row.data(f"{NA} - {reason}", NA, NA, NA, NA),),
                label="Panel B. Rigobon-Sack heteroskedasticity IV",
            )
        )
    else:
        records = _records(iv, what="IV results")
        _require_keys(records, ("parameter", "coefficient", "std_err", "p_value"), "IV results")
        rows = []
        for record in records:
            rows.append(
                Row.data(
                    str(record["parameter"]),
                    format_coefficient(record.get("coefficient"), record.get("p_value")),
                    format_number(record.get("std_err"), DP["std_err"]),
                    format_p_value(record.get("p_value")),
                    format_integer(record.get("n")),
                )
            )
            split = _first(record, "regime_split")
            if not _is_missing(split):
                rows.append(Row.section(f"Regime split: {split}"))
        panels.append(
            Panel(
                columns=TABLE4_IV_COLUMNS,
                rows=tuple(rows),
                label="Panel B. Rigobon-Sack heteroskedasticity IV",
            )
        )

    footnotes = (
        ("Granger tests report predictive precedence, not causation: a common "
        "driver of both coverage and sentiment can produce them with no causal "
        "channel. The IV is attempted for exactly that reason."),
        HAC_NOTE,
        SIGNIFICANCE_NOTE,
    )
    return Table(
        number=4,
        title="Granger causality tests and Rigobon-Sack IV transmission estimates",
        panels=tuple(panels),
        footnotes=footnotes,
        label="tab:granger-iv",
    )
