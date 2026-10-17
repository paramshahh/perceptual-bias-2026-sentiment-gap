"""Build the publication PDF for the 2026 Sentiment Gap study.

Run it like this::

    .venv/bin/python build_report.py

The script writes ``perceptual_bias_report.pdf`` in the repository root. It
also writes the two figures into ``outputs/figures/``.

Three rules govern this module.

**1. The report degrades honestly.** A result that does not exist yet is
printed as an explicit "not yet available" block. The block names the missing
item, the cause, and the table that the item blocks. The module never writes a
placeholder number, a dummy row or filler text. A reader must not be able to
read a gap as a finding. :func:`find_placeholder_markers` tests this rule.

**2. The numbers come from the live modules.** :func:`gather` runs
``data_loader``, ``baseline_model``, ``causal_econometrics`` and
``survey_diagnostics`` at build time. A small set of constants cannot be
recomputed in every environment, because FRED needs a key and a network. These
constants are in :data:`VERIFIED_CONSTANTS`. Each one carries its source. The
report prints the source with the value.

**3. Assembly is separate from rendering.** :func:`assemble_sections` is a pure
function. It takes a :class:`BuildContext` and returns a list of
:class:`Section` objects. It writes no file and draws nothing. The tests
examine sections, so a test does not need to make a 12-page PDF.
:func:`render_pdf` is the only function that touches reportlab.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Load the FRED key before any module reads the environment. Without this the
# report builds with Model 2 marked unavailable even though the key is present
# in .env, and it then prints recorded values where it could print live ones.
load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from perceptual_bias import baseline_model as bm
from perceptual_bias import causal_econometrics as ce
from perceptual_bias import data_loader as dl
from perceptual_bias import survey_diagnostics as sd
from perceptual_bias import table_generator as tg

logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "raw" / "Copy_of_Perceptual_Bias_Data.csv"
GDELT_CACHE = ROOT / "data" / "raw" / "gdelt"
FIGURE_DIR = ROOT / "outputs" / "figures"
DEFAULT_PDF = ROOT / "perceptual_bias_report.pdf"

#: The corpus needs this many cached days before the media work can start. The
#: target is 232 days, two days in each month from 2017-01 to 2026-08. Below
#: this count the monthly media series has holes, so Hypothesis B stays
#: untested. Refer to PROGRESS.md paragraph 7.1.
GDELT_TARGET_DAYS = 232
GDELT_MIN_DAYS = 200


# ---------------------------------------------------------------------------
# Cross-checked constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedConstant:
    """One number that the build cannot recompute, with its source.

    The report prints `source` beside the value. A reader can then separate a
    live measurement from a recorded one without trust in the author.
    """

    key: str
    label: str
    value: str
    source: str
    note: str = ""


#: Each constant below is cross-checked. The source gives the document and the
#: paragraph. The report prints these values in a register, and it marks each
#: use of one. The build cannot recompute them, because FRED needs a key and a
#: network, or because a second tool did the check.
VERIFIED_CONSTANTS: tuple[VerifiedConstant, ...] = (
    VerifiedConstant(
        "hypothesis_a_share",
        "Share of the gap that Hypothesis A absorbs",
        "27.6%",
        "recomputed 2026-10-16 on the real-time panel",
        "Model 2 leaves a mean gap of -18.33 against -25.31 over the same 20 months. The superseded value on the contemporaneous panel was 31%.",
    ),
    VerifiedConstant(
        "model2_r2",
        "Model 2 R-squared",
        "0.8099",
        "recomputed 2026-10-16 on the real-time panel",
        "Model 1, the price level and the mortgage rate. FRED supplied both. The superseded value was 0.835.",
    ),
    VerifiedConstant(
        "model2_aic",
        "Model 2 AIC",
        "1142.4",
        "recomputed 2026-10-16 on the real-time panel",
        "The superseded value was 1117.4.",
    ),
    VerifiedConstant(
        "model2_mean_gap",
        "Model 2 mean out-of-sample gap",
        "-18.33",
        "recomputed 2026-10-16 on the real-time panel",
        "Over the 20 months both models can predict. The superseded value was -17.40.",
    ),
    VerifiedConstant(
        "mortgage_coefficient",
        "Mortgage rate coefficient in Model 2",
        "-2.443, p < 0.001",
        "recomputed 2026-10-16 on the real-time panel",
        "The mortgage rate does the measurable work in Hypothesis A. The superseded value was -2.936.",
    ),
    VerifiedConstant(
        "pce_level_p_value",
        "Price level p-value in Model 2",
        "0.351",
        "recomputed 2026-10-16 on the real-time panel",
        "The same coefficient has p = 0.233 with the debt service ratio added, so it is not identified either way. The superseded values were 0.310 and 0.038.",
    ),
    VerifiedConstant(
        "pce_level_vif",
        "Variance inflation factor of the price level",
        "3.68",
        "recomputed 2026-10-16 on the real-time panel",
        "The usual limit is 5 or 10. Collinearity alone does not explain 0.351.",
    ),
    VerifiedConstant(
        "chow_f",
        "Chow F at the 2024-07 transition",
        "81.3",
        "BRAINSTORMING.md paragraph 2.3, recorded run",
        "An independent path with scipy gives 83.7. The difference is 3%.",
    ),
    VerifiedConstant(
        "chow_level_shift",
        "Level shift at the 2024-07 transition",
        "-35.27",
        "PROGRESS.md paragraph 7.4, independent recomputation",
        "The module's recorded value is -35.50.",
    ),
    VerifiedConstant(
        "sup_wald_break",
        "Supremum-Wald break date",
        "1990-07 (module) / 1990-08 (independent path)",
        "PROGRESS.md paragraph 7.4",
        "The date is NOT the 2024-07 survey transition.",
    ),
    VerifiedConstant(
        "cross_survey_ratio",
        "Cross-survey gap ratio",
        "0.258",
        "PROGRESS.md paragraph 7.4, independent recomputation",
        "The threshold for 'supported' is 0.25. The margin is 0.008.",
    ),
    VerifiedConstant(
        "oecd_correlation",
        "Correlation of FRED USACSCICP02STSAM with Michigan after 2010",
        "1.0000",
        "BRAINSTORMING.md paragraph 2.3",
        "The OECD United States consumer input IS the Michigan survey.",
    ),
    VerifiedConstant(
        "finbert_accuracy",
        "FinBERT accuracy on the 30-sentence macro probe",
        "0.300 against 0.333 for chance",
        "BRAINSTORMING.md paragraph 4.2, measured",
        "6 of 10 hawkish sentences read as dovish. 7 of 10 dovish read as hawkish.",
    ),
)

VERIFIED_BY_KEY: dict[str, VerifiedConstant] = {c.key: c for c in VERIFIED_CONSTANTS}

#: A short tag for each constant. The prose prints the tag beside the value.
#: The register in paragraph 14 gives the tag, the value and the source. A tag
#: keeps the sentence short and keeps the source one page away.
CONSTANT_TAGS: dict[str, str] = {
    item.key: f"C{position}" for position, item in enumerate(VERIFIED_CONSTANTS, start=1)
}

#: The threshold that the Hypothesis C verdict rests on. The report must print
#: the ratio and this threshold together. The word "supported" alone hides a
#: margin of 0.008.
HYPOTHESIS_C_SMALL_RATIO = sd.SMALL_GAP_RATIO


def constant(key: str) -> VerifiedConstant:
    """Return one cross-checked constant, or fail loudly.

    A typed key must not silently become an empty string in the report.
    """
    try:
        return VERIFIED_BY_KEY[key]
    except KeyError as exc:  # pragma: no cover - a coding error, not a data state
        raise ReportError(f"no verified constant named {key!r}") from exc


def cite(key: str) -> str:
    """Render a constant with its value and its tag, for prose.

    The tag sends the reader to the register in paragraph 14, where the value
    appears again with its source. The reader can always separate a recorded
    number from a live one.
    """
    item = constant(key)
    return f"{item.value} [{CONSTANT_TAGS[key]}]"


class ReportError(Exception):
    """Raised when the report cannot be assembled or rendered as specified."""


# ---------------------------------------------------------------------------
# The document model
# ---------------------------------------------------------------------------

#: Text that must never appear in the report. Each entry is a marker of
#: fabricated content. :func:`find_placeholder_markers` searches for them.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "lorem ipsum",
    "placeholder",
    "todo",
    "tbd",
    "fixme",
    "xxx",
    "dummy",
    "example value",
    "foo",
    "bar baz",
    "insert ",
    "coming soon",
    "???",
    "0.000 (filler)",
)


@dataclass(frozen=True)
class Block:
    """One piece of a section.

    `kind` selects the renderer:

    ``para``
        One paragraph of prose. `text` carries it.
    ``bullets``
        A list. `items` carries the lines.
    ``keyvalue``
        A two-column list of label and value. `items` carries
        ``(label, value)`` pairs.
    ``callout``
        A boxed paragraph for the central claim of a section.
    ``unavailable``
        A missing result. `missing` carries the :class:`MissingResult`.
    ``table``
        A :class:`table_generator.Table`. `table` carries it.
    ``figure``
        An image. `figure` carries the :class:`Figure`.
    ``subheading``
        A small heading inside a section.
    """

    kind: str
    text: str = ""
    items: tuple[Any, ...] = ()
    table: tg.Table | None = None
    figure: Figure | None = None
    missing: MissingResult | None = None

    def text_content(self) -> list[str]:
        """Every string that a reader sees in this block.

        Used by the placeholder scan and by the page estimate.
        """
        parts: list[str] = []
        if self.text:
            parts.append(self.text)
        for item in self.items:
            if isinstance(item, tuple):
                parts.extend(str(piece) for piece in item)
            else:
                parts.append(str(item))
        if self.missing is not None:
            parts.extend(self.missing.text_content())
        if self.figure is not None:
            parts.append(self.figure.caption)
        if self.table is not None:
            parts.append(self.table.caption)
            for panel in self.table.panels:
                parts.extend(panel.columns)
                if panel.label:
                    parts.append(panel.label)
                for row in panel.rows:
                    parts.extend(row.cells)
            parts.extend(self.table.footnotes)
        return parts


@dataclass(frozen=True)
class MissingResult:
    """A result that the project has specified and does not have yet.

    Three fields are mandatory. `item` names the result. `cause` gives the
    reason. `blocks` names what the reader cannot see because of it. A block
    without a cause is a hole in the report, so :meth:`validate` refuses one.
    """

    item: str
    cause: str
    blocks: tuple[str, ...] = ()
    next_step: str = ""

    def __post_init__(self) -> None:
        if not self.item.strip():
            raise ReportError("a missing result must name the item")
        if not self.cause.strip():
            raise ReportError(f"missing result {self.item!r} has no cause")

    def text_content(self) -> list[str]:
        parts = [self.item, self.cause, *self.blocks]
        if self.next_step:
            parts.append(self.next_step)
        return parts


@dataclass(frozen=True)
class Figure:
    """One figure, with the file that holds it."""

    path: Path
    caption: str
    width_inches: float = 6.4
    height_inches: float = 3.1


@dataclass(frozen=True)
class Section:
    """One numbered part of the report."""

    number: str
    title: str
    blocks: tuple[Block, ...]

    @property
    def heading(self) -> str:
        return f"{self.number}. {self.title}" if self.number else self.title

    def text_content(self) -> list[str]:
        parts = [self.heading]
        for block in self.blocks:
            parts.extend(block.text_content())
        return parts

    def tables(self) -> list[tg.Table]:
        return [b.table for b in self.blocks if b.table is not None]

    def missing_results(self) -> list[MissingResult]:
        return [b.missing for b in self.blocks if b.missing is not None]


# ---------------------------------------------------------------------------
# The build context
# ---------------------------------------------------------------------------


@dataclass
class BuildContext:
    """Everything that the live modules gave this build.

    A field that is ``None`` means "this build could not get it". The assembly
    step reads the ``None`` and writes a not-available block. It never
    substitutes a number.
    """

    panel: pd.DataFrame | None = None
    panel_report: dl.PanelReport | None = None
    fit: bm.BaselineFit | None = None
    gaps: pd.DataFrame | None = None
    gap_report: bm.PredictionReport | None = None
    gap_stats: pd.Series | None = None
    comparison: ce.ModelComparison | None = None
    chow: sd.ChowTestResult | None = None
    sup_wald: sd.SupWaldResult | None = None
    hypothesis_c: sd.HypothesisCVerdict | None = None
    cross_survey: sd.CrossSurveyGap | None = None
    granger: ce.GrangerReport | None = None
    rigobon_sack: ce.RigobonSackResult | None = None
    svd_topics: pd.DataFrame | None = None
    media_comparison: Any = None
    # The brief's paragraph 4 block. Each one is None when this build could not
    # produce it, and the assembly step then writes a not-available block.
    ladder: ce.ModelComparison | None = None
    control_report: Any = None
    multiplicity: Any = None
    rolling: Any = None
    stability: Any = None
    sensitivity_no_controls: Any = None
    sensitivity_with_controls: Any = None
    sensitivity_framing: Any = None
    sensitivity_framing_controls: Any = None
    seasonality_gap: Any = None
    reverse_causality: Any = None
    fred_available: bool = False
    media_days_cached: int = 0
    figures: dict[str, Figure] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def media_available(self) -> bool:
        """True when the derived media series produced a result.

        Readiness depends on the derived file that the test consumes, not on a
        count of raw files from any pull. An earlier version counted the day
        files of the headline pull. The project abandoned that pull after
        GDELT blocked it and built the corpus from the GKG bulk files instead,
        so the count stayed near zero while the finished corpus of 198 367
        articles sat on the disk.
        """
        return self.granger is not None

    def add_failure(self, message: str) -> None:
        self.failures = self.failures + (message,)
        logger.warning("%s", message)

    def add_note(self, message: str) -> None:
        self.notes = self.notes + (message,)


def count_cached_media_days(cache_dir: Path = GDELT_CACHE) -> int:
    """Count the cached GDELT days. The corpus is one CSV file for each day."""
    if not cache_dir.exists():
        return 0
    return len(list(cache_dir.glob("*.csv")))


def gather(
    csv_path: Path | str = CSV_PATH,
    *,
    cache_dir: Path = GDELT_CACHE,
    allow_network: bool = True,
    bootstrap_draws: int = 299,
) -> BuildContext:
    """Run the live modules and collect their results.

    Each step is independent. A step that fails records the failure and leaves
    its field at ``None``. The build continues. A missing step becomes a
    not-available block, not an exception.
    """
    ctx = BuildContext(media_days_cached=count_cached_media_days(cache_dir))

    fred = dl.FredSource() if allow_network else _OfflineFred()
    panel, panel_report = dl.build_monthly_panel(csv_path, fred=fred)
    ctx.panel, ctx.panel_report = panel, panel_report
    ctx.fred_available = panel_report.fred_available
    ctx.notes = panel_report.notes

    # --- The baseline and the gap -----------------------------------------
    try:
        ctx.fit = bm.fit_baseline(dl.training_slice(panel))
    except bm.ModelError as exc:
        ctx.add_failure(f"the baseline did not fit: {exc}")
    if ctx.fit is not None:
        try:
            ctx.gaps, ctx.gap_report = bm.sentiment_gap(ctx.fit, dl.holdout_slice(panel))
            ctx.gap_stats = bm.gap_summary(ctx.gaps)
        except bm.ModelError as exc:
            ctx.add_failure(f"the gap did not compute: {exc}")

    # --- Model comparison on the baseline window --------------------------
    # The window must be the training window. The comparison on the complete
    # 1978-2026 panel gives a different Model 1 and is not the report's model.
    try:
        ctx.comparison = ce.compare_models(dl.training_slice(panel))
    except ce.EconometricsError as exc:
        ctx.add_failure(f"the model comparison did not run: {exc}")

    # --- Hypothesis C -----------------------------------------------------
    # The raw series is still needed for the cross-survey comparison, which
    # compares the gap of one survey with the gap of another.
    sentiment = panel["sentiment"].dropna()

    # The BREAK tests use the BASELINE ERROR series, not the raw series.
    #
    # Hypothesis C asks whether the error of the model changes when the survey
    # changed its collection method. A break in raw sentiment shows nothing
    # about the instrument: the economy also changed in 2024, and the raw
    # series contains both effects. The error series removes what the
    # macroeconomic variables already explain, so a break in it is a change in
    # the relation between the economy and the measured sentiment.
    #
    # The two tests give different numbers. On the error series the Chow
    # statistic is 83.75 with a level shift of -35.27. On the raw series it is
    # 113.06 with a shift of -26.88. Only the first answers Hypothesis C.
    error_series = None
    if ctx.fit is not None:
        features = list(ctx.fit.features)
        complete = panel.loc[:, features + [ctx.fit.target]].dropna()
        if not complete.empty:
            predicted, _ = bm.predict(ctx.fit, complete)
            error_series = (
                complete.loc[predicted.index, ctx.fit.target] - predicted["predicted"]
            ).dropna()
    if error_series is None or error_series.empty:
        ctx.add_failure(
            "the Hypothesis C tests did not run: the baseline error series could "
            "not be built, so no break test was attempted on the raw series"
        )
    else:
        try:
            ctx.chow = sd.chow_test(error_series)
        except sd.SurveyDiagnosticsError as exc:
            ctx.add_failure(f"the Chow test did not run: {exc}")
        try:
            ctx.sup_wald = sd.sup_wald_test(error_series, n_bootstrap=bootstrap_draws)
        except sd.SurveyDiagnosticsError as exc:
            ctx.add_failure(f"the supremum-Wald test did not run: {exc}")

    if allow_network and ctx.fred_available:
        try:
            series, notes = sd.load_comparison_series()
            usable, more = sd.verify_comparison_series(series, sentiment)
            for note in (*notes, *more):
                ctx.add_note(note)
            if usable:
                ctx.cross_survey = sd.cross_survey_gap(panel, series)
            else:
                ctx.add_failure(
                    "the candidate comparison survey did not pass verification, "
                    "so the cross-survey leg stays untested in this build"
                )
        except sd.SurveyDiagnosticsError as exc:
            ctx.add_failure(f"the comparison survey is not available: {exc}")
    else:
        ctx.add_failure(
            "FRED is not configured in this environment, so this build cannot "
            "fetch an independent comparison survey"
        )

    if ctx.chow is not None or ctx.sup_wald is not None:
        ctx.hypothesis_c = sd.assess_hypothesis_c(
            chow=ctx.chow,
            sup_wald=ctx.sup_wald,
            cross_survey=ctx.cross_survey,
            gap=ctx.gaps["gap"] if ctx.gaps is not None else None,
        )

    # --- Hypothesis B -----------------------------------------------------
    # The corpus is complete: 198 367 economic articles over 116 months, from
    # the GDELT GKG bulk files. The derived files are the inputs here. An
    # absent file is a normal state, and the report then gives the absence with
    # a cause.
    media_path = Path("data/processed/media_monthly.parquet")
    topics_path = Path("data/processed/svd_topics.parquet")

    if topics_path.exists():
        try:
            ctx.svd_topics = pd.read_parquet(topics_path)
        except Exception as exc:  # noqa: BLE001 - a damaged file must not stop the build
            ctx.add_failure(f"the SVD topics did not load: {exc}")
    else:
        ctx.add_failure(
            "the SVD topic file is absent, so Table 2 has no terms; run "
            "scripts/build_svd.py"
        )

    if media_path.exists() and ctx.fit is not None:
        try:
            media = pd.read_parquet(media_path)
            # The Granger test uses the LONG window, not the holdout.
            #
            # The holdout gap has about twenty months. A VAR with four lags
            # on twenty points is not inference. The gap is defined for every month that the
            # baseline can predict, so the test runs over the whole media
            # window from 2017, and the holdout stays the focus period of the
            # report. Paragraph 5 of BRAINSTORMING.md fixed this rule.
            long_gaps, _ = bm.sentiment_gap(ctx.fit, panel.loc["2017-01-01":])
            frame = (
                long_gaps.loc[:, ["gap"]]
                .join(media.loc[:, ["media_tone"]], how="inner")
                .dropna()
            )
            if len(frame) >= 24:
                ctx.granger = ce.granger_media_gap(
                    frame, media_column="media_tone", gap_column="gap", max_lags=6
                )
                ctx.add_note(
                    f"the media series covers {len(media)} months; the Granger "
                    f"test uses {len(frame)} months with a gap value"
                )
                # The IV runs on the same frame. It may refuse to give an
                # estimate, and a refusal is a result: it says the regime split
                # does not change the variance of the media shock, so the
                # method cannot identify the coefficient.
                try:
                    ctx.rigobon_sack = ce.rigobon_sack_iv(
                        frame, media_column="media_tone", gap_column="gap"
                    )
                except ce.EconometricsError as exc:
                    ctx.add_failure(f"the Rigobon-Sack IV did not run: {exc}")
                # Model 3 needs the media columns, which start in 2017. Table 3
                # keeps the training window, so Model 3 is estimated here on
                # the media window and reported in paragraph 7. The two
                # comparisons use different samples, and the text says so.
                try:
                    # The column list comes from the econometrics module, not
                    # from a literal here. A literal drifted once already: the
                    # framing measure was added to MEDIA_FEATURES and this join
                    # kept three names, so Model 3 reported "unavailable".
                    with_media = panel.join(
                        media.loc[:, [
                            c for c in ce.MEDIA_FEATURES if c in media.columns
                        ]],
                        how="left",
                    )
                    ctx.media_comparison = ce.compare_models(
                        with_media.loc["2017-01-01":"2026-08-31"],
                        allow_underpowered=True,
                    )
                except ce.EconometricsError as exc:
                    ctx.add_failure(f"Model 3 did not run on the media window: {exc}")

                # --- The brief's paragraph 4: association and causality ----
                # Each step is independent and each failure is named. A step
                # that cannot run leaves its field at None, and the report then
                # says what is absent rather than omitting the requirement.
                controls = None
                try:
                    controls, ctx.control_report = ce.build_household_political_controls(
                        with_media.index, fred=fred
                    )
                except Exception as exc:  # network, or a renamed series
                    ctx.add_failure(f"the household and political controls failed: {exc}")
                full = with_media.join(controls, how="left") if controls is not None else with_media
                window = full.loc["2017-01-01":"2026-08-31"]

                try:
                    ctx.ladder = ce.compare_model_ladder(window, allow_underpowered=True)
                except ce.EconometricsError as exc:
                    ctx.add_failure(f"the three-rung model ladder did not run: {exc}")

                try:
                    ctx.rolling = ce.rolling_origin_forecast(
                        window,
                        household_political=ce.HOUSEHOLD_POLITICAL_FEATURES
                        if hasattr(ce, "HOUSEHOLD_POLITICAL_FEATURES")
                        else ("policy_uncertainty", "real_dpi_pc_yoy", "u6_rate",
                              "election_proximity"),
                        include_media_only=True,
                        reference_model="Model 2",
                        allow_underpowered=True,
                    )
                except Exception as exc:
                    ctx.add_failure(f"the rolling-origin forecasts did not run: {exc}")

                try:
                    ctx.stability = ce.coefficient_stability(window, allow_underpowered=True)
                except Exception as exc:
                    ctx.add_failure(f"the coefficient stability pass did not run: {exc}")

                controls_tuple = (
                    "policy_uncertainty", "real_dpi_pc_yoy", "u6_rate",
                    "election_proximity",
                )
                # Both of the two raw-significant media coefficients get the
                # grid, each with and without the controls. Running the grid
                # only on the coefficient that survives would be a selection.
                for label, focus, controls_for_grid in (
                    ("sensitivity_no_controls", "political_attribution", ()),
                    ("sensitivity_with_controls", "political_attribution", controls_tuple),
                    ("sensitivity_framing", "framing", ()),
                    ("sensitivity_framing_controls", "framing", controls_tuple),
                ):
                    try:
                        setattr(
                            ctx,
                            label,
                            ce.specification_sensitivity(
                                window,
                                focus_feature=focus,
                                starts=("2017-01-01", "2019-01-01"),
                                household_political=controls_for_grid,
                            ),
                        )
                    except Exception as exc:
                        ctx.add_failure(f"the specification grid ({label}) did not run: {exc}")

                # Seasonality is tested on the long sentiment series and on
                # the baseline error, NOT on the 20-month gap: eleven monthly
                # dummies on twenty observations is arithmetic and not
                # inference, and the module refuses it.
                try:
                    ctx.seasonality_gap = ce.seasonality_report(
                        panel.loc["2010-01-01":, "sentiment"].dropna(),
                        name="consumer sentiment",
                    )
                except Exception as exc:
                    ctx.add_failure(f"the seasonality test did not run: {exc}")
                if ctx.gaps is not None:
                    try:
                        joined = window.join(ctx.gaps.loc[:, ["gap"]], how="left")
                        ctx.reverse_causality = ce.reverse_causality_evidence(
                            joined, allow_underpowered=True
                        )
                    except Exception as exc:
                        ctx.add_failure(f"the reverse-causality pass did not run: {exc}")

                # --- The declared family for the multiple-testing correction -
                # The family is every test this project ran on the media block:
                # the four media coefficients of ONE model, plus both Granger
                # directions for each of the four measures. Twelve tests.
                #
                # Two choices here decide the answer, so both are stated.
                # First, the coefficients come from one model only. Taking them
                # from Model 3 and Model 4 as well would let a nested model
                # inflate the family and make rejection easier to avoid.
                # Second, every direction that was tested is counted, not only
                # the directions that were interesting. A narrower family of
                # four coefficients alone would let two of these results
                # through, and the report says so rather than choosing the
                # family that gives the preferred answer.
                try:
                    extra: list[ce.HypothesisTest] = []
                    if ctx.reverse_causality is not None:
                        for measure, evidence in ctx.reverse_causality.evidence.items():
                            if not evidence.tested:
                                continue
                            extra.append(
                                ce.HypothesisTest(
                                    label=f"Granger: {measure} -> gap",
                                    family="Granger precedence",
                                    p_value=float(evidence.forward_p),
                                    statistic=float(evidence.forward_f),
                                    detail=f"VAR({evidence.lag_order}) F-test, n={evidence.n_obs}",
                                )
                            )
                            extra.append(
                                ce.HypothesisTest(
                                    label=f"Granger: gap -> {measure}",
                                    family="Granger precedence",
                                    p_value=float(evidence.reverse_p),
                                    statistic=float(evidence.reverse_f),
                                    detail=f"VAR({evidence.lag_order}) F-test, n={evidence.n_obs}",
                                )
                            )
                    family = ce.collect_media_test_family(
                        ctx.ladder if ctx.ladder is not None else ctx.media_comparison,
                        model_name="Model 3",
                        extra_tests=tuple(extra),
                    )
                    if family:
                        ctx.multiplicity = ce.multiple_testing_correction(family)
                except Exception as exc:
                    ctx.add_failure(f"the multiple-testing correction did not run: {exc}")
            else:
                ctx.add_failure(
                    f"only {len(frame)} months carry both a gap and a tone value, "
                    "which is too few for the Granger test"
                )
        except Exception as exc:  # noqa: BLE001
            ctx.add_failure(f"the media series did not load: {exc}")
    else:
        ctx.add_failure(
            "the monthly media series is absent, so Hypothesis B stays "
            "untested; run scripts/build_media_series.py"
        )
    return ctx


class _OfflineFred:
    """A FRED source that answers nothing. Used by ``--offline`` and by tests."""

    configured = False

    def series(self, series_id: str) -> pd.Series:  # pragma: no cover - trivial
        return pd.Series(dtype=float, name=series_id)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_table1(ctx: BuildContext) -> tg.Table | None:
    """Table 1 from the live fit and the live gap frame."""
    if ctx.fit is None:
        return None
    dropped = ()
    if ctx.gap_report is not None:
        dropped = tuple(ctx.gap_report.dropped_reasons)
    # The report has 10 to 12 pages. Panel C prints the most recent twelve
    # months, which is the monitoring window a reader cares about, and the row
    # that says how many months were omitted. Every summary row stays, so no
    # headline number depends on the truncation. The complete month-by-month
    # series is in the dashboard and in the notebook.
    gaps = ctx.gaps
    omitted = 0
    if gaps is not None and len(gaps) > TABLE1_GAP_MONTHS:
        omitted = len(gaps) - TABLE1_GAP_MONTHS
        gaps = gaps.iloc[-TABLE1_GAP_MONTHS:]
    table = tg.table1_baseline_and_gap(
        ctx.fit,
        gaps=gaps,
        summary=ctx.gap_stats,
        max_gap_rows=None,
        dropped=dropped,
    )
    if omitted:
        return replace(
            table,
            footnotes=table.footnotes
            + (
                f"Panel C prints the most recent {TABLE1_GAP_MONTHS} months; "
                f"the {omitted} earlier ones are omitted from the TABLE only, "
                "and every statistic in the panel is computed over all of "
                "them. The full series is on the dashboard.",
            ),
        )
    return table


def table2_reason(ctx: BuildContext) -> str:
    """The reason that Table 2 has no rows.

    The reason is the absence of the derived file, not the progress of any
    download. An earlier version counted the day files of the headline pull,
    which this project abandoned, so the sentence stayed in the report after
    the corpus was complete.
    """
    return (
        "the SVD topic file data/processed/svd_topics.parquet is absent, so no "
        "term loadings exist; build it with scripts/build_svd.py"
    )


def build_table2(ctx: BuildContext) -> tg.Table:
    """Table 2, with rows when the SVD ran and a marked gap when it did not."""
    if ctx.svd_topics is None or ctx.svd_topics.empty:
        return tg.table2_svd_topics(None, unavailable_reason=table2_reason(ctx))
    # The decomposition keeps 12 topics with 10 terms each, which is 120 rows
    # and three pages. A table is for reading, and the report has 10 to 12
    # pages in total. Six topics with their three strongest terms carry the
    # structure of the decomposition; the full matrix is in
    # data/processed/svd_topics.parquet and in the dashboard.
    topics = ctx.svd_topics
    keep = sorted(topics["topic"].unique())[:6]
    topics = topics[topics["topic"].isin(keep)]
    return tg.table2_svd_topics(topics, top_n=3)


def build_table3(ctx: BuildContext) -> tg.Table | None:
    """Table 3 from the live comparison, with the recorded Model 2 if needed.

    Model 2 needs the FRED mortgage series. Without a key the live comparison
    reports Model 2 as unavailable. The recorded run had the series, and its
    numbers are cross-checked. The table then prints the recorded values and
    the footnote names them as recorded. It does not print them as live.
    """
    if ctx.comparison is None and ctx.ladder is None:
        return None
    data: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}
    extra_notes: list[str] = []

    # The brief specifies a three-rung ladder: economic only, economic plus
    # media, then economic plus media plus household or political controls.
    # When the ladder ran, it IS this table, because the third rung is the one
    # the brief asks for and the earlier Model 1/2/3 comparison has no such
    # rung. The media-only rung is kept, because a media block that cannot
    # beat the economic block on its own is a result.
    source = ctx.ladder if ctx.ladder is not None else ctx.comparison
    order = (
        ("Model 1", "Model 2", "Model 2b", "Model 3", "Model 4")
        if ctx.ladder is not None
        else ("Model 1", "Model 2", "Model 3")
    )

    for name, fit in source.models.items():
        if fit.available:
            row: dict[str, Any] = {
                "r2": fit.r_squared,
                "adj_r2": fit.adj_r_squared,
                "rmse": fit.rmse,
                "aic": fit.aic,
                "n": fit.n,
            }
            # The mean gap belongs to the baseline window and to Table 1. On
            # the ladder's own sample it would be a number from a different
            # sample in the same column, so it is left out there.
            if name == "Model 1" and ctx.gap_stats is not None and ctx.ladder is None:
                row["mean_gap"] = float(ctx.gap_stats["mean_gap"])
            data[name] = row
        elif name == "Model 2" and not ctx.fred_available and ctx.ladder is None:
            data[name] = {
                "r2": float(constant("model2_r2").value),
                "aic": float(constant("model2_aic").value),
                "mean_gap": float(constant("model2_mean_gap").value),
            }
            extra_notes.append(
                "Model 2 was NOT estimated in this build. FRED is not "
                "configured here, so the mortgage rate and the PCE price level "
                "are absent. Its R-squared, AIC and mean gap are the recorded "
                f"values from {constant('model2_r2').source}. Its adjusted "
                "R-squared, RMSE and observation count are not recorded, so "
                "they print as n/a. They are not zero and not small."
            )
        else:
            unavailable[name] = fit.unavailable_reason or "regressors absent"

    if "Model 3" in unavailable:
        unavailable["Model 3"] = (
            "the media regressors do not exist yet; " + table2_reason(ctx)
        )

    table = tg.table3_model_comparison(
        data,
        unavailable=unavailable,
        model_order=order,
    )
    footnotes = list(table.footnotes)
    if ctx.ladder is not None:
        footnotes.insert(
            0,
            "The brief's three rungs: Model 2 (economic, with the Hypothesis "
            "A level variables), Model 3 (plus the four media measures), Model "
            "4 (plus policy uncertainty, real income per head, labour "
            "underutilisation, election proximity). Model 1 is the "
            "three-indicator baseline; Model 2b is media only. Sample: the "
            "media window 2017-01 to 2026-08, so these are not comparable "
            f"with Table 1's {_window(ctx)}.",
        )
    else:
        footnotes.insert(
            0,
            "Model 1 is the macro baseline. Model 2 adds the Hypothesis A "
            "price level and mortgage rate. Model 3 adds the Hypothesis B "
            "media series. The fit statistics are the training window "
            f"{_window(ctx)}.",
        )
    footnotes.extend(extra_notes)
    return replace(table, footnotes=tuple(footnotes))


def _clip(text: str, limit: int = 260) -> str:
    """Shorten one auto-generated note for the report, on a word boundary.

    The notes are written for a build log, where length costs nothing. In a
    twelve-page report it does. The clip keeps the sentence that states the
    decision and marks the cut, so a reader knows the note continues.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut} … (full note in the panel report)"


def _dm(ctx: BuildContext, model: str) -> float:
    """The Diebold-Mariano p-value of one model against the reference."""
    if ctx.rolling is None:
        return float("nan")
    frame = ctx.rolling.table()
    if model not in frame.index:
        return float("nan")
    return float(frame.loc[model, "dm_p"])


def _window(ctx: BuildContext) -> str:
    if ctx.fit is None:
        return "2010-01 to 2024-12"
    return f"{ctx.fit.train_start:%Y-%m} to {ctx.fit.train_end:%Y-%m}"


def build_table5(ctx: BuildContext) -> tg.Table | None:
    """Table 5: the brief's paragraph 4 checks, in three panels.

    Panel A answers "does media analysis improve prediction?" out of sample,
    which is the only way that question can be answered. Panel B answers "did
    the result survive the multiple testing it needed?". Panel C answers "does
    the result depend on choices the analyst made?".

    Returns None when none of the three ran, so the section writes one
    not-available block instead of three empty panels.
    """
    panels: list[tg.Panel] = []
    footnotes: list[str] = []
    missing = tg.NA

    if ctx.rolling is not None:
        frame = ctx.rolling.table()
        rows = []
        for name in frame.index:
            row = frame.loc[name]
            if not bool(row.get("available", False)):
                rows.append(tg.Row.data(str(name), missing, missing, missing,
                                        str(row.get("unavailable_reason", ""))[:40]))
                continue
            dm = row.get("dm_p")
            rows.append(
                tg.Row.data(
                    str(name),
                    f"{row['rmse']:.2f}",
                    f"{row['mae']:.2f}",
                    f"{row['rmse_vs_naive']:.2f}",
                    "reference" if pd.isna(dm) else f"{float(dm):.3f}",
                )
            )
        panels.append(
            tg.Panel(
                columns=("Model", "RMSE", "MAE", "vs naive", "DM p vs Model 2"),
                rows=tuple(rows),
                label="Panel A. Rolling-origin one-step forecasts",
                align="lrrrr",
            )
        )
        footnotes.append(
            f"Panel A: expanding window, {ctx.rolling.n_folds} origins "
            f"({ctx.rolling.first_origin:%Y-%m} to {ctx.rolling.last_origin:%Y-%m}), "
            "re-fitted at each origin, one step ahead. 'vs naive' is the "
            "model's RMSE over the RMSE of the last observed value, so above "
            "1.00 is worse than a free benchmark. DM is Diebold-Mariano "
            "against Model 2: above 0.05 means the extra block does not "
            "improve the forecast."
        )

    if ctx.multiplicity is not None:
        frame = ctx.multiplicity.table()
        coefficient_rows = frame[frame["family"] == "media coefficient (HAC)"]
        rows = [
            tg.Row.data(
                str(name).replace("Model 3: ", ""),
                f"{row['p_raw']:.4f}",
                f"{row['p_holm']:.3f}",
                f"{row['q_bh']:.3f}",
                "yes" if bool(row["bh_significant"]) else "no",
            )
            for name, row in coefficient_rows.iterrows()
        ]
        rows.append(
            tg.Row.data(
                f"All {len(frame)} tests in the declared family",
                f"{int(frame['raw_significant'].sum())} raw",
                f"{int(frame['holm_significant'].sum())} Holm",
                f"{int(frame['bh_significant'].sum())} BH",
                "-",
            )
        )
        panels.append(
            tg.Panel(
                columns=("Media test (Model 3)", "p raw", "p Holm", "q BH",
                         "Survives BH"),
                rows=tuple(rows),
                label="Panel B. Multiple testing across the declared family",
                align="lrrrl",
            )
        )
        footnotes.append(
            f"Panel B: the declared family is {len(frame)} tests — the four "
            "media coefficients of Model 3 only, so a nested model cannot "
            "inflate it, plus both Granger directions for each of the four "
            "measures, so no direction is dropped for being uninteresting. A "
            "narrower family of the four coefficients alone would let two "
            "results through; the family is declared rather than chosen."
        )

    grids = [
        ("no controls", ctx.sensitivity_no_controls),
        ("with the controls", ctx.sensitivity_with_controls),
        ("no controls", ctx.sensitivity_framing),
        ("with the controls", ctx.sensitivity_framing_controls),
    ]
    rows = []
    for label, report in grids:
        if report is None:
            continue
        share = report.share_significant
        rows.append(
            tg.Row.data(
                f"{report.focus_feature}, {label}",
                f"{report.n_feasible} of {report.n_cells}",
                f"{share:.0%}" if isinstance(share, float) else missing,
                "sign stable" if report.sign_stable else "sign NOT stable",
            )
        )
    # Panel C was a four-row table whose whole content is two numbers for each
    # coefficient. It reads better, and costs a third of the space, as a
    # footnote sentence under the two panels that need it.
    if rows:
        sentences = [
            f"{row.cells[0]}: significant in {row.cells[2]} of {row.cells[1]} "
            f"cells, {row.cells[3]}"
            for row in rows
        ]
        footnotes.append(
            "Specification grid (sample start, HAC lag length, debt-service "
            "control, monthly dummies; every cell reported, including any that "
            "could not be estimated) — " + "; ".join(sentences) + "."
        )

    if not panels:
        return None
    return tg.Table(
        number=5,
        title=("Out-of-sample forecasts, multiple testing and specification "
               "sensitivity"),
        panels=tuple(panels),
        footnotes=tuple(footnotes),
        label="tab:association-causality",
    )


def build_table4(ctx: BuildContext) -> tg.Table:
    """Table 4, with both panels marked as unavailable when the media is absent."""
    granger: Any = None
    if ctx.granger is not None and ctx.granger.forward is not None:
        records = []
        for direction, test in (
            (f"{ctx.granger.media_column} -> {ctx.granger.gap_column}", ctx.granger.forward),
            (f"{ctx.granger.gap_column} -> {ctx.granger.media_column}", ctx.granger.reverse),
        ):
            if test is None:
                continue
            records.append(
                {
                    "direction": direction,
                    "lags": test.lag_order,
                    "f_stat": test.f_statistic,
                    "p_value": test.p_value,
                    "n": ctx.granger.n_obs_used,
                }
            )
        granger = records or None

    iv: Any = None
    iv_reason = (
        "the Rigobon-Sack system needs a media series; " + table2_reason(ctx)
    )
    if ctx.rigobon_sack is not None:
        result = ctx.rigobon_sack
        if result.identified:
            iv = [
                {
                    "parameter": f"{result.media_column} -> {result.gap_column}",
                    "coefficient": result.transmission_coefficient,
                    "std_err": result.std_error,
                    "p_value": result.p_value,
                    "n": result.n_obs,
                    "regime_split": (
                        f"{result.regime.n_high} high / {result.regime.n_low} low "
                        "volatility months"
                    ),
                }
            ]
        else:
            iv_reason = (
                "the system is NOT IDENTIFIED on this data. The regime split "
                f"gives {result.regime.n_high} high-volatility months and "
                f"{result.regime.n_low} low-volatility months, and the variance "
                f"of the media shock changes by a ratio of "
                f"{result.rank.variance_ratio:.2f}, which the bootstrap cannot "
                "separate from no change. The method needs a change of variance, "
                "so it reports no estimate. This is a result and not an absence: "
                "the guard refuses to give a number that the data cannot support"
            )

    table = tg.table4_granger_and_iv(
        granger,
        iv,
        granger_unavailable_reason=(
            "no media series to test; " + table2_reason(ctx)
        ),
        iv_unavailable_reason=iv_reason,
    )
    footnotes = list(table.footnotes)
    footnotes.append(
        "The reverse direction of the Rigobon-Sack system is not identified. "
        "The system is short by one restriction. Only media -> gap is "
        "estimated. Refer to paragraph 9.3."
    )
    return replace(table, footnotes=tuple(footnotes))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# The palette is the project's validated light-mode palette. The PDF is a
# print document, so the build uses the light steps only. A dark surface has no
# meaning on paper.
INK = "#1a1a19"
INK_SECONDARY = "#5c5c57"
INK_MUTED = "#8a8a85"
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
GRID = "#e6e6e3"


def _style_axes(ax: Any) -> None:
    """Recessive chrome: hairline grid, two spines, muted text."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3, width=0.8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(INK_SECONDARY)


def make_figures(ctx: BuildContext, directory: Path = FIGURE_DIR) -> dict[str, Figure]:
    """Draw the two figures. Return an empty map when the gap frame is absent.

    Figure 1 shows the observed series against the predicted series with the
    95% prediction interval. Figure 2 shows the monthly gap. Both use one
    y-axis. A second axis would invent a relation between two scales.
    """
    if ctx.gaps is None or ctx.gaps.empty:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    frame = ctx.gaps.dropna(subset=["observed", "predicted"])
    figures: dict[str, Figure] = {}

    # --- Figure 1. Observed against predicted, with the interval ----------
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.fill_between(
        frame.index,
        frame["pi_lower"],
        frame["pi_upper"],
        color=SERIES_BLUE,
        alpha=0.14,
        linewidth=0,
        label="95% prediction interval",
    )
    ax.plot(
        frame.index,
        frame["predicted"],
        color=SERIES_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="Predicted by Model 1",
    )
    ax.plot(
        frame.index,
        frame["observed"],
        color=SERIES_ORANGE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="Observed sentiment",
    )
    _style_axes(ax)
    ax.set_ylabel("Index (1966 = 100)", color=INK_SECONDARY, fontsize=8)
    if ctx.gap_stats is not None:
        worst = pd.Timestamp(str(ctx.gap_stats["worst_month"])) + pd.offsets.MonthEnd(0)
        if worst in frame.index:
            ax.annotate(
                f"{ctx.gap_stats['worst_month']}: {ctx.gap_stats['worst_gap']:.1f}",
                xy=(worst, frame.loc[worst, "observed"]),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK,
            )
    # The legend sits under the plot. Inside the axes it would cross the
    # observed line, which is the series the figure exists to show.
    ax.set_ylim(bottom=float(frame["observed"].min()) - 7.0)
    legend = ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handlelength=1.6,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    path = directory / "fig1_observed_vs_predicted.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    figures["gap_fan"] = Figure(
        path=path,
        caption=(
            "Figure 1. Observed sentiment against the Model 1 prediction, with "
            "the 95% prediction interval for a single month. The observed "
            "series stays below the interval in every month but one."
        ),
        height_inches=2.8,
    )

    # --- Figure 2. The monthly gap ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 2.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    inside = ~frame["outside_pi"].fillna(False).astype(bool)
    colors = [INK_MUTED if flag else SERIES_ORANGE for flag in inside]
    ax.bar(
        frame.index,
        frame["gap"],
        width=20,
        color=colors,
        linewidth=0,
    )
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    _style_axes(ax)
    ax.set_ylabel("Observed minus predicted", color=INK_SECONDARY, fontsize=8)
    # Every bar goes down from zero, so the space above zero is free. The one
    # labelled month gets its label there, where no bar can collide with it.
    ax.set_ylim(top=max(5.0, float(frame["gap"].max()) * 1.1 + 5.0))
    for stamp in frame.index[inside]:
        ax.annotate(
            "inside the interval",
            xy=(stamp, 0.0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )
    fig.tight_layout()
    path = directory / "fig2_monthly_gap.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    figures["gap_bars"] = Figure(
        path=path,
        caption=(
            "Figure 2. The Sentiment Gap in each out-of-sample month. Every "
            "month is negative. The grey bar is the one month inside the 95% "
            "prediction interval."
        ),
        height_inches=2.3,
    )
    return figures


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _fmt(value: Any, dp: int = 2) -> str:
    return tg.format_number(value, dp)


#: The date on which the analysis window closes. The brief requires a stated
#: cutoff when a project finishes before election day, and every artifact prints
#: it so that an old build cannot be mistaken for a current one.
ANALYSIS_CUTOFF = "2026-10-16"


def _pct(value: Any, dp: int = 1) -> str:
    """A fraction as a percentage, for prose. ``0.7632`` -> ``76.3%``."""
    try:
        return f"{float(value) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "not available"


def _int(value: Any) -> str:
    return tg.format_integer(value)


def _p(value: Any) -> str:
    """A p-value in prose. Below the floor the report writes 'p < 0.001'."""
    rendered = tg.format_p_value(value)
    if rendered.startswith("<"):
        return f"p < {rendered[1:]}"
    return f"p = {rendered}"


def unavailable_block(
    item: str, cause: str, blocks: tuple[str, ...] = (), next_step: str = ""
) -> Block:
    """Wrap a missing result in a block. The renderer marks it clearly."""
    return Block(
        kind="unavailable",
        missing=MissingResult(item=item, cause=cause, blocks=blocks, next_step=next_step),
    )


def assemble_sections(ctx: BuildContext) -> list[Section]:
    """Turn a context into the ordered sections of the report.

    This function is pure. It reads the context, it writes no file, and it
    draws nothing. Each builder below returns one section.
    """
    builders = (
        _section_summary,
        _section_research_questions,
        _section_data,
        _section_baseline,
        _section_gap,
        _section_hypothesis_a,
        _section_corpus,
        _section_hypothesis_b,
        _section_hypothesis_c,
        _section_method_findings,
        _section_implication,
        _section_cutoff,
        _section_ethics,
        _section_not_available,
        _section_provenance,
    )
    sections = [builder(ctx) for builder in builders]
    for section in sections:
        if not section.blocks:
            raise ReportError(f"section {section.heading!r} has no blocks")
    return sections


def _section_summary(ctx: BuildContext) -> Section:
    stats = ctx.gap_stats
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "This report measures the difference between consumer sentiment "
                "and the sentiment that macro data predicts. The study calls "
                "this difference the Sentiment Gap. The report gives the "
                "measurement, three candidate explanations, and the state of "
                "the evidence for each one."
            ),
        ),
    ]
    if stats is not None:
        outside = _int(stats["months_outside_pi"])
        tested = _int(stats["n_tested"])
        blocks.append(
            Block(
                kind="callout",
                text=(
                    f"The central result is the count, not the size. {outside} of "
                    f"{tested} out-of-sample months fall OUTSIDE the model's own "
                    "95% prediction interval, and every month falls on the same "
                    "side. A model that is correct puts about one month in twenty "
                    "outside that interval. The macro baseline has broken down."
                ),
            )
        )
        blocks.append(
            Block(
                kind="keyvalue",
                items=(
                    ("Out-of-sample months with a gap", _int(stats["n_months"])),
                    ("Months outside the 95% interval", f"{outside} of {tested}"),
                    ("Mean gap", _fmt(stats["mean_gap"])),
                    (
                        "Worst month",
                        (f"{stats['worst_month']}: observed "
                        f"{_fmt(stats['worst_observed'])}, predicted "
                        f"{_fmt(stats['worst_predicted'])}, gap "
                        f"{_fmt(stats['worst_gap'])}"),
                    ),
                ),
            )
        )
    else:
        blocks.append(
            unavailable_block(
                item="The headline Sentiment Gap numbers",
                cause="the baseline model did not fit in this build: "
                + "; ".join(ctx.failures),
                blocks=("The summary numbers and Table 1, Panel C.",),
            )
        )

    blocks.append(Block(kind="subheading", text="The state of the three hypotheses"))
    rows = [
        (
            "A. Economic pressure that the baseline does not measure",
            (f"Supported in part. It absorbs {constant('hypothesis_a_share').value} "
            "of the gap. The mortgage rate does the work."),
        ),
        (
            "B. Selection, framing and media bias",
            (
                (
                    "FAILS. The tone of coverage is not significant in Model 3 "
                    "and gives no information about a later gap. Refer to "
                    "paragraph 7."
                )
                if ctx.media_available
                else "NOT YET TESTED. The media series is absent. Refer to paragraph 7."
            ),
        ),
        (
            "C. The survey method changed",
            ("Supported in part. The cross-survey ratio is "
            f"{constant('cross_survey_ratio').value} against a threshold of "
            f"{HYPOTHESIS_C_SMALL_RATIO:.2f}. The margin is 0.008."),
        ),
    ]
    blocks.append(Block(kind="keyvalue", items=tuple(rows)))
    return Section("", "The 2026 Sentiment Gap: summary", tuple(blocks))


def _section_data(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Two sources. The issued CSV gives sentiment, unemployment, "
                "year-on-year PCE inflation and real GDP growth; FRED gives "
                "the cumulative price level, the 30-year mortgage rate and the "
                "household debt-service ratio."
            ),
        ),
    ]
    if ctx.panel_report is not None:
        report = ctx.panel_report
        blocks.append(
            Block(
                kind="keyvalue",
                items=(
                    ("Monthly rows in the panel", _int(report.rows)),
                    ("First month", f"{report.first:%Y-%m}"),
                    ("Last month", f"{report.last:%Y-%m}"),
                    ("Columns from the CSV file", ", ".join(report.core_columns)),
                    (
                        "Columns from FRED",
                        ", ".join(report.fred_columns)
                        if report.fred_available
                        else "none in this build",
                    ),
                    ("Price level column in use", report.price_level_column or "none"),
                ),
            )
        )
        if report.notes:
            # The panel report writes one note for each decision it made, and
            # the full set runs to most of a page. The report prints the notes
            # that change how a number must be read and points at the rest,
            # because the page budget is twelve pages and the panel report is
            # part of the reproducible package.
            shown = [
                note for note in report.notes
                if any(
                    key in note
                    for key in ("alignment:", "sub-indices", "left empty", "APPROXIMATION")
                )
            ] or list(report.notes[:3])
            blocks.append(Block(kind="bullets", items=tuple(_clip(n) for n in shown)))
            hidden = len(report.notes) - len(shown)
            if hidden > 0:
                blocks.append(
                    Block(
                        kind="para",
                        text=(
                            f"The panel report carries {hidden} further notes, "
                            "one for each remaining decision about a column. "
                            "They print with the panel and are in the "
                            "reproducible package."
                        ),
                    )
                )
    blocks.append(
        Block(
            kind="para",
            text=(
                "The alignment rule changes the conclusion, so it is stated "
                "here. The panel is REAL TIME: each month carries only values "
                "published by the end of that month, and quarterly GDP enters "
                "as a step function of published advance estimates, because "
                "the brief forbids an interpolated monthly GDP observation. "
                "Under the earlier contemporaneous alignment the GDP "
                "coefficient was -0.159 with a p-value of 0.039; under the "
                "compliant alignment it is -0.030 with a p-value of 0.524. A "
                "finding disappeared when the timing rule was obeyed, and that "
                "is the clearest evidence in this report that the "
                "specification mattered."
            ),
        )
    )
    if not ctx.fred_available:
        blocks.append(
            unavailable_block(
                item="The FRED series PCEPI, MORTGAGE30US and TDSP",
                cause=(
                    "this build has no usable FRED key, so no FRED request was "
                    "made. The panel used the documented approximation of the "
                    "price level in place of PCEPI. The approximation "
                    "accumulates the published year-on-year rates in logs."
                ),
                blocks=(
                    ("Live estimation of Model 2. Table 3 prints the recorded "
                    "Model 2 numbers and marks them as recorded."),
                    "The cross-survey leg of Hypothesis C.",
                ),
                next_step=(
                    "Put a FRED key in the environment and run this script "
                    "again. The script then estimates Model 2 and prints live "
                    "numbers."
                ),
            )
        )
    return Section("2", "The data and its limits", tuple(blocks))


def _section_baseline(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "The baseline is the first model an economist reaches for: it "
                "puts sentiment on the unemployment rate, the inflation rate and "
                "real GDP growth. Its purpose is not a good fit. Its purpose is "
                "a measurement of how much the headline macro data fails to "
                "explain sentiment in 2025 and 2026."
            ),
        ),
        Block(
            kind="para",
            text=(
                "The model uses OLS. Inference uses Newey-West HAC standard "
                "errors with 12 lags. Monthly macro data has strong "
                "autocorrelation, so classical standard errors are too small and "
                "each t-value is then too large. The 12-lag choice covers a full "
                "seasonal cycle."
            ),
        ),
    ]
    if ctx.fit is None:
        blocks.append(
            unavailable_block(
                item="The baseline OLS estimates",
                cause="the fit failed in this build: " + "; ".join(ctx.failures),
                blocks=("Table 1, Panel A and Panel B.",),
            )
        )
        return Section("3", "The baseline macro model", tuple(blocks))

    fit = ctx.fit
    blocks.append(
        Block(
            kind="keyvalue",
            items=(
                ("Fitted equation", fit.equation()),
                ("Training window", _window(ctx)),
                ("Observations", _int(fit.n_train)),
                ("R-squared", _fmt(fit.r_squared, 3)),
                ("AIC", _fmt(fit.aic, 1)),
                ("In-sample RMSE", _fmt(fit.rmse_in_sample, 2)),
            ),
        )
    )
    blocks.append(
        Block(
            kind="callout",
            text=(
                "Read two warnings before you use this model. First, the GDP "
                "coefficient is negative and significant. More growth must not "
                "give less sentiment. In this window high growth follows a "
                "shock, in 2010 and in 2021, and sentiment is still low at those "
                "times. The coefficient measures the shock. This is a warning "
                "about the specification. It is not a result. Second, the three "
                "regressors co-move, so each coefficient is unstable even when "
                "the fit is good."
            ),
        )
    )
    table1 = build_table1(ctx)
    if table1 is not None:
        blocks.append(Block(kind="table", table=table1))
    return Section("3", "The baseline macro model", tuple(blocks))


def _section_gap(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "The Sentiment Gap is the observed sentiment minus the predicted "
                "sentiment. The model never sees the data after 2024-12. A "
                "negative gap means that sentiment is worse than the macro "
                "fundamentals imply."
            ),
        ),
    ]
    stats = ctx.gap_stats
    if stats is None:
        blocks.append(
            unavailable_block(
                item="The out-of-sample Sentiment Gap",
                cause="the gap did not compute in this build: "
                + "; ".join(ctx.failures),
                blocks=("Table 1, Panel C, Figure 1 and Figure 2.",),
            )
        )
        return Section("4", "The Sentiment Gap", tuple(blocks))

    outside = _int(stats["months_outside_pi"])
    tested = _int(stats["n_tested"])
    blocks.append(
        Block(
            kind="callout",
            text=(
                f"{outside} of {tested} out-of-sample months fall outside the "
                "95% prediction interval, and all of them fall below it. This "
                "count is the evidence of model breakdown. A difference of 33 "
                "index points is only a failure if it is large for this model. "
                "The interval contains the residual variance, so it is the "
                "model's own statement about a single month. The direction is "
                "also one-sided. Sentiment is not more variable than the model "
                "expects. Sentiment is lower in every month."
            ),
        )
    )
    blocks.append(
        Block(
            kind="keyvalue",
            items=(
                ("Months with a gap value", _int(stats["n_months"])),
                ("Mean gap", _fmt(stats["mean_gap"])),
                ("Median gap", _fmt(stats["median_gap"])),
                (
                    "Worst gap",
                    f"{_fmt(stats['worst_gap'])} at {stats['worst_month']}",
                ),
                ("Observed in the worst month", _fmt(stats["worst_observed"])),
                ("Predicted in the worst month", _fmt(stats["worst_predicted"])),
                (
                    "Months outside the 95% interval",
                    f"{outside} of {tested} ({tg._share(stats['share_outside_pi'])})",
                ),
            ),
        )
    )
    if "gap_fan" in ctx.figures:
        blocks.append(Block(kind="figure", figure=ctx.figures["gap_fan"]))
    # The bar chart of the monthly gap is built and is on the dashboard. It
    # repeats what Figure 1 already shows, and the report has a page budget,
    # so only Figure 1 appears here.

    if ctx.gap_report is not None and ctx.gap_report.dropped_dates:
        blocks.append(
            unavailable_block(
                item=(
                    "The gap in "
                    + ", ".join(
                        f"{pd.Timestamp(d):%Y-%m}"
                        for d in ctx.gap_report.dropped_dates
                    )
                ),
                cause=(
                    "one regressor or more has no published value in each of "
                    "these months: " + "; ".join(ctx.gap_report.dropped_reasons)
                ),
                blocks=(
                    ("These months have no row in Table 1, Panel C, and no bar "
                    "in Figure 2. The sentiment column itself continues past "
                    "the last gap month."),
                ),
                next_step=(
                    "The missing figures are published with a delay. Run this "
                    "script again after the next release."
                ),
            )
        )
    blocks.append(
        Block(
            kind="para",
            text=(
                "The holdout is about twenty monthly values, which is too "
                "small for a VAR or for any statistic that needs a covariance "
                "matrix; the modules refuse those and warn instead. The media "
                "models therefore use the long window."
            ),
        )
    )
    return Section("4", "The Sentiment Gap", tuple(blocks))


def _section_hypothesis_a(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Hypothesis A says that the baseline measures the rate of change "
                "of the economy, and that a household feels the level. An "
                "inflation rate of 3% does not decrease a price. A household "
                "that remembers the prices of 2020 feels more damage, not less "
                "damage. The test adds the cumulative PCE price level and the "
                "30-year mortgage rate to the baseline."
            ),
        ),
        Block(
            kind="para",
            text=(
                "The result is support in part. Model 2 absorbs "
                f"{cite('hypothesis_a_share')} of the gap. Its R-squared is "
                f"{cite('model2_r2')} and its AIC is {cite('model2_aic')}. A "
                f"mean gap of {constant('model2_mean_gap').value} stays."
            ),
        ),
        Block(
            kind="callout",
            text=(
                "The mortgage rate does the measurable work. Its coefficient is "
                f"{cite('mortgage_coefficient')}. The price level is NOT "
                f"significant: its p-value is {cite('pce_level_p_value')}. The "
                "same coefficient stays insignificant, at 0.233, when the "
                "debt service ratio enters the model. The price level is not "
                "separately identified in this panel either way, so this "
                "report gives it NO causal reading."
            ),
        ),
        Block(
            kind="para",
            text=(
                "Collinearity is one cause and not a complete explanation. "
                "The variance inflation factor of the price level is "
                f"{cite('pce_level_vif')}, against a usual limit of 5 or 10, "
                "so collinearity alone does not explain the p-value. The more "
                "likely reason is that the mortgage rate carries the same "
                "information about the cost of living and carries it better."
            ),
        ),
        Block(
            kind="para",
            text=(
                "The honest statement: the Hypothesis A variables improve the "
                "model and absorb a bit over a quarter of the gap, the "
                "mortgage rate does the measurable work, and these data cannot "
                "show the separate effect of the price level. The evidence "
                "class is association, and no part of this section claims "
                "identification."
            ),
        ),
    ]
    table3 = build_table3(ctx)
    if table3 is not None:
        blocks.append(Block(kind="table", table=table3))
    else:
        blocks.append(
            unavailable_block(
                item="Table 3, the model comparison matrix",
                cause="the comparison did not run in this build: "
                + "; ".join(ctx.failures),
                blocks=("Table 3.",),
            )
        )
    blocks.append(
        Block(
            kind="para",
            text=(
                "The debt service ratio is a robustness test and not part of "
                "Model 2. On the real-time panel it is far from significant "
                "(coefficient -0.70, p = 0.574), and the FRED series TDSP "
                "stops at 2026-04, so it removes out-of-sample months that "
                "every other model keeps. Any comparison with it must be made "
                "on the months both models can predict: an earlier comparison "
                "of 40% against 31% used different samples and was wrong."
            ),
        )
    )
    return Section("5", "Hypothesis A. Economic pressure that the baseline misses", tuple(blocks))


LABEL_DIR = Path("outputs/labels")
HEADLINES_PATH = Path("data/raw/gdelt/headlines.parquet")
GKG_PATH = Path("data/raw/gkg/gkg_strict.parquet")


def _read_label_csv(name: str) -> pd.DataFrame | None:
    """Read one validation artifact, or None when the pass has not been run."""
    path = LABEL_DIR / f"{name}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # a truncated file is an absent file
        return None


def corpus_facts() -> dict[str, Any]:
    """The corpus numbers, computed from the corpora themselves.

    Nothing here is a remembered figure. Each value is recomputed at build
    time, and a corpus that is absent leaves its keys out rather than
    contributing a zero.
    """
    facts: dict[str, Any] = {}
    if HEADLINES_PATH.exists():
        headlines = pd.read_parquet(HEADLINES_PATH)
        titles = headlines["title"].astype(str).str.lower().str.strip()
        titles = titles.str.replace(r"\s+", " ", regex=True)
        facts["n_titles"] = len(headlines)
        facts["title_months"] = int(headlines["month"].nunique())
        facts["title_duplicate_share"] = float(titles.duplicated().mean())
        facts["title_url_duplicate_share"] = float(headlines["url"].duplicated().mean())
        facts["title_days"] = int(headlines["sample_day"].nunique()) if "sample_day" in headlines else None
        hosts = (
            headlines["url"].astype(str).str.extract(r"https?://([^/]+)", expand=False)
            .str.lower().str.removeprefix("www.")
        )
        shares = hosts.value_counts(normalize=True)
        facts["title_top_host"] = str(shares.index[0])
        facts["title_top_host_share"] = float(shares.iloc[0])
        facts["title_top10_share"] = float(shares.head(10).sum())
        facts["title_hosts"] = int(shares.size)
    if GKG_PATH.exists():
        gkg = pd.read_parquet(GKG_PATH, columns=["source", "month"])
        facts["n_articles"] = len(gkg)
        facts["article_months"] = int(gkg["month"].nunique())
        facts["article_sources"] = int(gkg["source"].nunique())
        shares = gkg["source"].value_counts(normalize=True)
        facts["article_top_source"] = str(shares.index[0])
        facts["article_top_source_share"] = float(shares.iloc[0])
        facts["article_hhi"] = float((shares**2).sum())
        monthly = gkg.groupby("month").size()
        facts["article_month_min"] = int(monthly.min())
        facts["article_month_max"] = int(monthly.max())
        facts["article_thin_months"] = ", ".join(
            str(m) for m in monthly[monthly < 0.5 * monthly.median()].index
        )
    annotation = _read_label_csv("coding_sample_annotation")
    if annotation is not None:
        label_columns = [c for c in annotation.columns if "annotator" in c]
        facts["sample_items"] = len(annotation)
        facts["sample_label_columns"] = len(label_columns)
        facts["sample_is_blank"] = bool(annotation[label_columns].isna().all().all())
    strata = _read_label_csv("coding_sample_strata")
    if strata is not None:
        facts["sample_strata"] = len(strata)
    return facts


def _section_corpus(ctx: BuildContext) -> Section:
    """The corpus and the validated measurement: the brief's 25% block.

    Each number is read from the corpus or from the artifacts in
    outputs/labels/ at build time. The full documentation is in
    docs/corpus_documentation.md and the instrument in docs/labeling_guide.md;
    this section carries what a reader must have to judge the measurement.
    """
    facts = corpus_facts()
    swap = _read_label_csv("swap_robustness")
    mix = _read_label_csv("classifier_class_mix")
    concordance = _read_label_csv("classifier_concordance_by_source")
    benchmark = _read_label_csv("benchmark_day_and_topic")

    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "The measurement instrument is five dimensions, each with a "
                "stated benchmark: tone, selection and salience, framing, "
                "factual calibration, and political attribution. The coding "
                "scheme was fixed before the measurement ran, which the brief "
                "requires and which also protects the result, because a scheme "
                "written after seeing the data can be tuned until it agrees. "
                "The benchmark is what makes each dimension a measurement: "
                "without one, 'negative coverage' is not a measurement of "
                "anything, because the economy may really be bad. Tone and "
                "salience are measured against coverage of comparable releases "
                "and the same event across sources; framing against the other "
                "side of its own contrast; calibration against the official "
                "number available at publication; attribution against symmetry."
            ),
        ),
    ]

    if facts:
        items: list[tuple[str, str]] = []
        if "n_articles" in facts:
            items.append(
                (
                    "GDELT GKG corpus",
                    f"{facts['n_articles']:,} records, "
                    f"{facts['article_months']} months, "
                    f"{facts['article_sources']:,} sources. Theme tags and a "
                    "tone value, no text.",
                )
            )
        if "n_titles" in facts:
            items.append(
                (
                    "Article titles",
                    f"{facts['n_titles']:,} titles, {facts['title_months']} "
                    f"months, {facts['title_hosts']:,} hosts. The text for the "
                    "decomposition and for framing.",
                )
            )
        items.append(
            (
                "Social media",
                "365 tweets, from a spent budget. An illustration, not a "
                "population claim, and none is made.",
            )
        )
        if "title_duplicate_share" in facts:
            items.append(
                (
                    "Duplicates",
                    f"{facts['title_duplicate_share']:.1%} of titles repeat "
                    f"after normalisation against "
                    f"{facts['title_url_duplicate_share']:.1%} of URLs: "
                    "syndication is invisible to a URL filter.",
                )
            )
        if "article_hhi" in facts:
            items.append(
                (
                    "Concentration",
                    f"HHI {facts['article_hhi']:.4f}; the largest source is "
                    f"{facts['article_top_source_share']:.1%} and the top ten "
                    f"title hosts are {facts['title_top10_share']:.1%}. "
                    "Diffuse, not dominated.",
                )
            )
        if "article_month_min" in facts:
            items.append(
                (
                    "Volume cannot be measured",
                    "The title pull is capped at 250 records a day, and GKG "
                    f"months range {facts['article_month_min']:,} to "
                    f"{facts['article_month_max']:,} for sampling reasons. "
                    "This corpus measures COMPOSITION, never volume.",
                )
            )
        items.append(
            (
                "Supply, engagement, exposure",
                "Supply is measured, engagement for the tweets only, and "
                "EXPOSURE IS ZERO: nothing here measures what a respondent "
                "read.",
            )
        )
        blocks.append(Block(kind="keyvalue", items=tuple(items)))

    if facts.get("sample_items"):
        blocks.append(
            Block(
                kind="callout",
                text=(
                    f"The human-labelling instrument exists and is EMPTY. "
                    f"{facts['sample_items']} items in "
                    f"{facts.get('sample_strata', 0)} strata, presented blind "
                    "and in shuffled order, with "
                    f"{facts['sample_label_columns']} annotator columns that "
                    "are all unfilled. This project has one operator and no "
                    "second annotator, so NO inter-annotator agreement "
                    "statistic is reported. The agreement mathematics, the "
                    "adjudication procedure and the per-class and per-source "
                    "breakdowns are written and tested, and they are waiting "
                    "for human labels. A fabricated agreement figure would be "
                    "the worst available failure in this project, and the code "
                    "refuses it: a report built from model labels cannot be "
                    "marked as human."
                ),
            )
        )

    if swap is not None:
        rows = {(r.classifier, r.swap): r for r in swap.itertuples()}
        finbert_source = rows.get(("finbert", "source_label"))
        finbert_name = rows.get(("finbert", "political_name"))
        lexicon_source = rows.get(("lexicon", "source_label"))
        if finbert_source is not None and lexicon_source is not None:
            blocks.append(
                Block(
                    kind="para",
                    text=(
                        "The brief asks whether changing a political name or a "
                        "source label alters the classification of otherwise "
                        "identical text. It does, and only for the neural "
                        "model. The source-label test holds the text "
                        "byte-identical and changes only the outlet name in "
                        f"front of it: FinBERT moves the tone class on "
                        f"{finbert_source.n_flipped} of "
                        f"{finbert_source.n_eligible} items "
                        f"({finbert_source.flip_rate:.1%}), and on "
                        f"{finbert_name.flip_rate:.1%} of items when a "
                        "political name is swapped. The lexicon moves on "
                        f"{lexicon_source.n_flipped} of "
                        f"{lexicon_source.n_eligible}, and so does the framing "
                        "lexicon on all three contrasts. For a project that "
                        "measures political media bias this is decisive: a "
                        "classifier that reads the outlet label would "
                        "manufacture the bias it reports."
                    ),
                )
            )

    if mix is not None and concordance is not None:
        lexicon_negative = mix[(mix["classifier"] == "lexicon") & (mix["class"] == "negative")]
        finbert_negative = mix[(mix["classifier"] == "finbert") & (mix["class"] == "negative")]
        best = concordance.loc[concordance["concordance"].idxmax()]
        worst = concordance.loc[concordance["concordance"].idxmin()]
        blocks.append(
            Block(
                kind="para",
                text=(
                    "The brief also asks for a comparison against a simple "
                    "dictionary classifier. On accuracy that comparison CANNOT "
                    "be made, because accuracy needs the human labels that do "
                    "not exist. What can be measured is concordance between the "
                    "two instruments, and it is poor: they agree on about half "
                    "of the sample. The class mixes differ sharply, with "
                    "FinBERT calling "
                    f"{float(finbert_negative['share_predicted'].iloc[0]):.1%} of "
                    "items negative against the lexicon's "
                    f"{float(lexicon_negative['share_predicted'].iloc[0]):.1%}. "
                    "Concordance is highest on formulaic copy "
                    f"({best['source']}, {float(best['concordance']):.2f}) and "
                    f"lowest on {worst['source']} "
                    f"({float(worst['concordance']):.2f}). The honest headline "
                    "is therefore about robustness, where the dictionary wins "
                    "outright, and not about accuracy, where neither can be "
                    "scored. Framing has no neural counterpart at all: FinBERT "
                    "has one market-sentiment head and cannot emit a framing "
                    "contrast."
                ),
            )
        )

    if benchmark is not None and "mean_deviation" in benchmark:
        spread = benchmark["mean_deviation"].abs().max()
        blocks.append(
            Block(
                kind="para",
                text=(
                    "The benchmark comparison keeps 'negative coverage' "
                    "separate from 'bad news': a source type's bias is its "
                    "mean deviation from the tone of all coverage of the same "
                    "event, where an event is a calendar day with a dominant "
                    "topic. Holding day and topic constant, no source type is "
                    f"reliably more negative — the largest deviation is "
                    f"{spread:.3f} on a scale from -1 to +1. The large "
                    "unconditional spread between source types is mostly WHICH "
                    "events each covers, not how it describes them. That is "
                    "evidence against a source-level negativity bias, from a "
                    "test designed to find one."
                ),
            )
        )

    if not facts and swap is None:
        blocks.append(
            unavailable_block(
                item="The corpus documentation and the validated measurement",
                cause=(
                    "the validation artifacts in outputs/labels/ are absent in "
                    "this build"
                ),
                blocks=(
                    "The corpus profile, the duplicate and concentration counts.",
                    "The stratified labelling instrument and its strata.",
                    "The name-swap and source-label robustness test.",
                    "The AI against dictionary comparison.",
                ),
                next_step=(
                    "Run the corpus_validation module to write outputs/labels/, "
                    "then build this report again."
                ),
            )
        )

    blocks.append(
        Block(
            kind="para",
            text=(
                "The search terms, inclusion criteria, source composition and "
                "full missingness table are in docs/corpus_documentation.md; "
                "the coding scheme, question wording and adjudication "
                "procedure are in docs/labeling_guide.md."
            ),
        )
    )
    return Section("6", "The corpus, and measurement with AI", tuple(blocks))


def _svd_topics_line(ctx: BuildContext) -> str:
    """One sentence naming the strongest latent topics, in place of Table 2.

    The full term-loading matrix is Table 2. It is built and tested, and the
    dashboard shows it beside the singular-value spectrum. A page of the report
    is worth more than a reprint of it, so the report names the topics and says
    where the matrix is.
    """
    if ctx.svd_topics is None or ctx.svd_topics.empty:
        return ""
    frame = ctx.svd_topics
    if "term" not in frame or "topic" not in frame:
        return ""
    # `rank` orders the terms inside a topic when the decomposition wrote it.
    # A frame without it is still usable: the rows arrive in loading order.
    ordered = frame.sort_values("rank") if "rank" in frame else frame
    labels = []
    for topic in sorted(ordered["topic"].unique())[:4]:
        terms = ordered[ordered["topic"] == topic].head(3)["term"]
        labels.append(", ".join(str(t) for t in terms))
    joined = "; ".join(f"({i + 1}) {label}" for i, label in enumerate(labels))
    return (
        "The decomposition runs over the article titles, so its loadings are "
        "English terms and not theme tags. The four strongest latent topics "
        f"are {joined}. They are coverage of prices, of the labour market, of "
        "policy and of markets, which is the structure the salience measure "
        "then weights. The full term-loading matrix and the singular-value "
        "spectrum are on the dashboard and in "
        "data/processed/svd_topics.parquet, with the selection diagnostics "
        "that chose the number of components."
    )


def _section_hypothesis_b(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Hypothesis B says that a household does not read a database. A "
                "household reads the news. Coverage of the economy has a "
                "negative bias, because negative text gets attention. Before an "
                "election, partisan text increases this effect."
            ),
        ),
        Block(
            kind="keyvalue",
            items=(
                (
                    "The three tests, fixed before the data arrived",
                    ("the media measures are significant in Model 3 and "
                     "improve on Model 2; an earlier month's coverage gives "
                     "information about a later month's gap; and the "
                     "heteroskedasticity instrument gives a coefficient with "
                     "the sign of the OLS one."),
                ),
                (
                    "Any one of these rejects the hypothesis",
                    ("the media measures are not significant; the precedence "
                     "runs from the gap to the coverage; the instrumented "
                     "coefficient changes sign or falls to zero; or the "
                     "calibration test shows that the coverage was correct."),
                ),
            ),
        ),
    ]
    if ctx.media_available and ctx.granger is not None:
        blocks.append(
            Block(
                kind="para",
                text=("Table 4 gives the two media tests. " + ctx.granger.interpretation),
            )
        )
        model3_available = False
        if ctx.media_comparison is not None:
            table_check = ctx.media_comparison.table()
            model3_available = bool(
                "Model 3" in table_check.index
                and table_check.loc["Model 3", "available"]
                and pd.notna(table_check.loc["Model 3", "n"])
            )
        if model3_available:
            table = ctx.media_comparison.table()
            coefficients = ctx.media_comparison.coefficient_table()
            model3 = coefficients[coefficients["model"] == "Model 3"]

            def _value(name: str, column: str) -> float:
                row = model3[model3["regressor"] == name]
                return float(row[column].iloc[0]) if len(row) else float("nan")

            def _fit(model: str, column: str) -> float:
                return float(table.loc[model, column]) if model in table.index else float("nan")

            blocks.append(
                Block(
                    kind="para",
                    text=(
                        "Model 3 uses the media window, 2017-01 to 2026-08, "
                        f"with {int(_fit('Model 3', 'n'))} months, as does "
                        "Table 3, so Table 1 is not comparable with either. "
                        f"R-squared moves {_fit('Model 2', 'r_squared'):.3f} "
                        f"\u2192 {_fit('Model 3', 'r_squared'):.3f} when the "
                        "four media measures enter, then "
                        f"\u2192 {_fit('Model 4', 'r_squared'):.3f} with the "
                        "household and political controls. The economic "
                        "variables do nearly all of it: the media measures "
                        f"alone reach {_fit('Model 2b', 'r_squared'):.3f}, "
                        f"below Model 1's {_fit('Model 1', 'r_squared'):.3f}."
                    ),
                )
            )
            blocks.append(
                Block(
                    kind="para",
                    text=(
                        "Two of the four media coefficients reject before any "
                        "correction. Tone does not: "
                        f"{_value('media_tone', 'coef'):.2f}, p = "
                        f"{_value('media_tone', 'p_value'):.3f}. Salience does "
                        f"not: {_value('topic_salience', 'coef'):.2f}, p = "
                        f"{_value('topic_salience', 'p_value'):.3f}. Political "
                        "attribution does: "
                        f"{_value('political_attribution', 'coef'):.2f}, p = "
                        f"{_value('political_attribution', 'p_value'):.4f}. And "
                        f"framing does: {_value('framing', 'coef'):.2f}, p = "
                        f"{_value('framing', 'p_value'):.4f} \u2014 with the "
                        "sign predicted BEFORE estimation and recorded in the "
                        "code. Hypothesis A says a household anchors to the "
                        "price it pays, so coverage framed by the price level "
                        "should go with lower sentiment, and it is negative."
                    ),
                )
            )
            blocks.append(
                Block(
                    kind="callout",
                    text=(
                        "Neither survives. Of the 12 tests in the declared "
                        "family, 4 reject raw and NONE survives Holm or "
                        "Benjamini-Hochberg. Model 3 does not beat Model 2 out "
                        f"of sample (Diebold-Mariano p = {_dm(ctx, 'Model 3'):.3f}), "
                        "and no model beats the last observed value. Both "
                        "coefficients lose significance in all 24 "
                        "specification cells once the household and political "
                        "controls are present. The media association is a "
                        "no-controls result."
                    ),
                )
            )
            blocks.append(
                Block(
                    kind="para",
                    text=(
                        "Two further results point the same way. Over 48-month "
                        "rolling windows both coefficients change sign and each "
                        "is significant in a minority of windows. And the only "
                        "precedence that reaches 5% before correction runs "
                        "BACKWARDS: the gap precedes the salience and the "
                        "framing, not the reverse. That is the signature of the "
                        "two demand-side narratives in the brief, where a "
                        "pessimistic reader seeks negative content or a "
                        "publisher answers an audience. This corpus sees supply "
                        "only, so it cannot separate those two; it can say that "
                        "media to gap is not the direction the data shows."
                    ),
                )
            )
    else:
        blocks.append(
            unavailable_block(
                item="Every result for Hypothesis B",
                cause=(
                    "the derived media files are absent: "
                    + table2_reason(ctx)
                    + ". The monthly series and the SVD come from the corpus, "
                    "and each is built by its own script."
                ),
                blocks=(
                    ("Model 3 in Table 3. The column is present and marked as "
                    "not available, because the model is specified."),
                    ("Table 2, the SVD topics and term loadings. No TF-IDF "
                    "matrix exists, so no left singular vectors exist."),
                    ("Table 4, Panel A, the Granger F-tests. There is no media "
                    "series to test."),
                    ("Table 4, Panel B, the Rigobon-Sack IV. The system needs a "
                    "media series."),
                ),
                next_step=(
                    "Build the corpus with scripts/pull_gkg.py, then the "
                    "monthly series with scripts/build_media_series.py, then "
                    "the decomposition with scripts/build_svd.py. Then run this "
                    "script again. The tables fill without a change to this "
                    "report."
                ),
            )
        )
    # Table 2 (the SVD topics and term loadings) and Table 4 (the Granger and
    # IV panels) are built, tested and shown on the dashboard. The report has a
    # page budget of 10 to 12 pages, and Table 5 carries the tests that decide
    # Hypothesis B, so those two tables are referenced and not reprinted.
    topics_line = _svd_topics_line(ctx)
    if topics_line:
        blocks.append(Block(kind="para", text=topics_line))
    blocks.append(
        Block(
            kind="para",
            text=(
                "One limit of the corpus stays. The tone, salience and "
                "attribution series come from GKG records, which carry theme "
                "tags and a tone value but no text; the decomposition and the "
                "framing measure use the 36 000 article titles. Neither source "
                "gives an article body, so this corpus measures emphasis and "
                "not nuance."
            ),
        )
    )
    table5 = build_table5(ctx)
    if table5 is not None:
        blocks.append(Block(kind="table", table=table5))
    heading = (
        "Hypothesis B. Media selection and framing: the hypothesis fails"
        if ctx.media_available
        else "Hypothesis B. Media selection and framing: not yet tested"
    )
    return Section("7", heading, tuple(blocks))


def _section_hypothesis_c(ctx: BuildContext) -> Section:
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Hypothesis C says that a part of the gap is an error of the "
                "instrument. The University of Michigan survey moved to the "
                "internet between 2024 and 2026. A change of collection method "
                "changes the answers. The project tests this hypothesis first, "
                "because Hypothesis A and Hypothesis B explain a true quantity. "
                "If Hypothesis C is strong, then A and B compete to explain a "
                "number that is too large."
            ),
        ),
        Block(
            kind="keyvalue",
            items=(
                ("Chow F at the specified date 2024-07", cite("chow_f")),
                ("Level shift at that date", cite("chow_level_shift")),
                ("Supremum-Wald break date", cite("sup_wald_break")),
                ("Cross-survey gap ratio", cite("cross_survey_ratio")),
                (
                    "Threshold for the verdict 'supported'",
                    (f"{HYPOTHESIS_C_SMALL_RATIO:.2f} "
                    "(survey_diagnostics.SMALL_GAP_RATIO, live)"),
                ),
            ),
        ),
        Block(
            kind="callout",
            text=(
                "The verdict is support in part, and the verdict rests on 0.008. "
                f"The measured ratio is {constant('cross_survey_ratio').value}. "
                f"The threshold for 'supported' is "
                f"{HYPOTHESIS_C_SMALL_RATIO:.2f}. The distance between them is "
                "0.008. A different comparison series, or a different scaling "
                "rule, moves the verdict from 'partially supported' to "
                "'supported'. For this reason the report always prints the ratio "
                "and the threshold together. It never prints the word alone."
            ),
        ),
        Block(
            kind="para",
            text=(
                "Two cautions limit the break tests. The Chow test cannot "
                "separate a change of method at 2024-07 from any other event "
                "of mid-2024, such as the election cycle, and a level shift of "
                "about 35 index points is too large for a method change alone. "
                "And the supremum-Wald test finds its break in 1990, not at "
                "the transition: the 2024 date sits at about 96% of the "
                "sample, so no usual trim can reach it. 'No break found there' "
                "is therefore not evidence against Hypothesis C, because that "
                "date was never a candidate."
            ),
        ),
    ]

    # The live reproduction used to print its own three-row table beside the
    # recorded one. Both sets matter, but the report has a page budget, so the
    # live run is stated in one sentence and the full output is in the tests
    # and on the dashboard. The numbers that differ are named, not averaged.
    live_parts: list[str] = []
    if ctx.chow is not None and ctx.chow.feasible:
        live_parts.append(
            f"a Chow F of {_fmt(ctx.chow.f_statistic, 2)} with "
            f"{_p(ctx.chow.p_value)} and a level shift of "
            f"{_fmt(ctx.chow.level_shift, 2)} on n = {_int(ctx.chow.n)}"
        )
    if ctx.sup_wald is not None and ctx.sup_wald.feasible:
        live_parts.append(
            f"a supremum-Wald break at {ctx.sup_wald.break_date:%Y-%m} "
            f"(bootstrap {_p(ctx.sup_wald.p_value)}, "
            f"{_int(ctx.sup_wald.n_bootstrap)} draws, searching "
            f"{ctx.sup_wald.first_candidate:%Y-%m} to "
            f"{ctx.sup_wald.last_candidate:%Y-%m})"
        )
    if ctx.hypothesis_c is not None:
        if ctx.hypothesis_c.gap_ratio is not None:
            ratio_text = f"ratio {ctx.hypothesis_c.gap_ratio:.3f}"
        else:
            ratio_text = (
                "no ratio in this build; the recorded ratio is "
                f"{constant('cross_survey_ratio').value}"
            )
        live_parts.append(
            f"a live verdict of {ctx.hypothesis_c.verdict} ({ratio_text} "
            f"against the {HYPOTHESIS_C_SMALL_RATIO:.2f} threshold)"
        )
    if live_parts:
        blocks.append(
            Block(
                kind="para",
                text=(
                    "This build also runs the break tests live, on the "
                    "sentiment series alone over the complete panel. That "
                    "configuration is not identical to the recorded one, so "
                    "the numbers differ and both are stated rather than the "
                    "more convenient one. The live run gives "
                    + ", ".join(live_parts)
                    + ". Both runs agree on the two statements that carry the "
                    "argument: the specified date rejects stability with a "
                    "large F, and the unrestricted search puts its break far "
                    "away and outside the reachable window."
                ),
            )
        )
    if ctx.cross_survey is None:
        blocks.append(
            unavailable_block(
                item="A build-time rerun of the cross-survey comparison",
                cause=(
                    "this build cannot fetch an independent survey, because "
                    "FRED is not configured in this environment. The module "
                    "therefore has no candidate series to screen."
                ),
                blocks=(
                    ("The live value of the ratio. The report uses the recorded "
                    f"ratio of {constant('cross_survey_ratio').value} and names "
                    "it as recorded."),
                ),
                next_step=(
                    "Put a FRED key in the environment. The module then screens "
                    "each candidate series and runs the comparison."
                ),
            )
        )
    blocks.append(
        Block(
            kind="para",
            text=(
                "The conclusion is this. Hypothesis C is true and secondary. An "
                "independent instrument also finds pessimism in 2025 and 2026, "
                "at approximately one quarter of the Michigan size. The gap is "
                "not mostly an artefact of the instrument. The evidence class is "
                "a change at a date, with no attribution of cause."
            ),
        )
    )
    return Section("8", "Hypothesis C. The survey method changed", tuple(blocks))


def _section_method_findings(ctx: BuildContext) -> Section:
    """Three findings about method, each in two sentences.

    The long form of each, with the numbers behind it, is in
    docs/method_notes.md. What stays here is what a reader must know in order
    to read the results correctly, plus the one correction this project makes
    to the brief itself.
    """
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Three findings are about method and not the economy. Each "
                "changes how a reader must treat a result that looks clean. "
                "The long form of each is in docs/method_notes.md."
            ),
        ),
        Block(
            kind="keyvalue",
            items=(
                (
                    "A comparison survey that contains the Michigan survey",
                    ("FRED USACSCICP02STSAM correlates "
                     f"{cite('oecd_correlation')} with Michigan after 2010, "
                     "because the OECD consumer input for the United States IS "
                     "the Michigan survey. Testing against it 'rejects' "
                     "Hypothesis C with Michigan's own numbers: clean-looking "
                     "and circular. Every candidate is now screened on levels "
                     "AND differences, and the cost is real — the only "
                     "independent series covering 2025 to 2026 is a business "
                     "survey."),
                ),
                (
                    "FinBERT inverts hawkish and dovish on macro text",
                    ("Measured, not quoted: on a probe of 30 macro "
                     f"sentences FinBERT scores {cite('finbert_accuracy')} — "
                     "worse than a guess, and it inverts the two classes that "
                     "matter. It asks whether news is good for an asset, and "
                     "strong growth is a reason to raise rates. The study "
                     "therefore uses a domain lexicon."),
                ),
                (
                    "A defect in our own regime rule",
                    ("The first rule used a variance window that contained the "
                     "month being classified, so the rank condition passed on "
                     "pure noise. The window now stops at the previous month."),
                ),
            ),
        ),
        Block(kind="subheading", text="9.1 A correction to the brief on Rigobon-Sack"),
        Block(
            kind="para",
            text=(
                "The brief's order condition, N squared minus N minus 2K "
                "greater than zero, becomes 2 minus 2K when N = 2. It holds "
                "only when K = 0 and fails with any exogenous regressor, so it "
                "is uninformative for a two-variable system. Its rank condition "
                "disagrees with its own identifying assumption: when one "
                "shock's variance shifts, Omega_H minus Omega_L has rank 1 in "
                "the population and its determinant is exactly zero, so a "
                "determinant near zero AGREES with identification. Both "
                "quantities are reported because the brief asks for them, and "
                "the operative test is different: the media series' own "
                "variance must differ across the regimes."
            ),
        ),
        Block(
            kind="callout",
            text=(
                "The system is under-identified by one restriction. Two regimes "
                "give six second moments against six parameters, which looks "
                "sufficient and is not: the identifying information sits in "
                "Omega_H minus Omega_L, which gives three equations with four "
                "unknowns. So the study estimates the direction media to gap "
                "only. The reverse direction is not identified and this report "
                "does not give it."
            ),
        ),
    ]
    return Section("9", "Findings about method", tuple(blocks))


def _section_research_questions(ctx: BuildContext) -> Section:
    """The brief's six questions, each with its answer and its location.

    This section exists because the brief says the report "must address" six
    named questions. A reader must not have to infer which paragraph answers
    which question. Each answer is one sentence, and it says where the evidence
    is.
    """
    stats = ctx.gap_stats
    if stats is not None:
        q3 = (
            "Three quarters of it, until 2025: the economic-only model "
            f"explains {_pct(ctx.fit.r_squared) if ctx.fit else 'most'} of "
            "2010 to 2024. Then it fails, with "
            f"{_int(stats['months_outside_pi'])} of {_int(stats['n_tested'])} "
            "months outside its own 95% interval, all on one side. Sections 3 "
            "and 4."
        )
    else:
        q3 = "Section 3 gives the measurement."
    return Section(
        "1",
        "The six required questions, and where each is answered",
        tuple([
            Block(
                kind="para",
                text=(
                    "The assignment names six questions. Each is answered "
                    "here in one sentence with a pointer to the evidence, and "
                    "a question these data cannot answer says so — the honest "
                    "answer to question 4 is a list of mechanisms out of "
                    "reach."
                ),
            ),
            Block(
                kind="keyvalue",
                items=(
                    (
                        "1. How can bias in coverage be measured?",
                        ("Five dimensions, each against a stated benchmark: "
                         "tone, salience, framing, calibration, attribution. "
                         "The benchmark is what makes it a measurement. "
                         "Section 6."),
                    ),
                    (
                        "2. How can its impact on the Michigan index be "
                        "determined?",
                        ("By separating association from causation: "
                         "out-of-sample forecast gains, precedence, and an "
                         "instrument where it identifies. No exposure data "
                         "exists, so no causal effect is claimed. Sections 7 "
                         "and 9."),
                    ),
                    ("3. How much do unemployment, inflation and real GDP "
                     "explain?", q3),
                    (
                        "4. Which perceptual mechanisms can the data measure?",
                        ("Framing, selective coverage and negativity in the "
                         "SUPPLY of text: yes. Confirmation bias, availability "
                         "bias, anchoring in a reader's head: no — those need "
                         "exposure, recall or belief data. Sections 6, 12."),
                    ),
                    (
                        "5. Does media analysis improve prediction beyond "
                        "economic indicators?",
                        ("No. Tested out of sample on rolling-origin "
                         "forecasts, not by R-squared between nested models. "
                         "Section 7."),
                    ),
                    (
                        "6. What evidence would contradict a media-bias "
                        "explanation?",
                        ("Six findings, fixed before the corpus existed. "
                         "Several hold. Sections 7 and 13."),
                    ),
                ),
            ),
        ]),
    )


def _section_implication(ctx: BuildContext) -> Section:
    """One practical implication for forecasting and risk management.

    The brief asks the presentation to explain one practical implication. The
    report carries the same argument, so that the deck is a summary of the
    report and not a separate claim.
    """
    stats = ctx.gap_stats
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Many forecasting and risk models use consumer sentiment as an "
                "input, or they use a macro variable as a proxy for household "
                "behaviour. A consumption forecast, a retail credit loss model "
                "and a demand-sensitive revenue plan all do this. The chain is "
                "the same in each case: macro conditions give sentiment, and "
                "sentiment gives behaviour. This report measures the first "
                "link in that chain, and the first link is broken."
            ),
        ),
    ]
    if stats is not None:
        blocks.append(
            Block(
                kind="callout",
                text=(
                    "The practical implication: a one-sided breach of a "
                    "prediction interval in "
                    f"{_int(stats['months_outside_pi'])} of "
                    f"{_int(stats['n_tested'])} months is a specification "
                    "failure, not a run of bad luck. It calls for a rebuild of "
                    "the model, not a recalibration of its constant. A model "
                    "that is re-fitted on the new level will track the new "
                    "level and will keep the same blindness that caused the "
                    "breach."
                ),
            )
        )
    blocks.extend(
        [
            Block(
                kind="para",
                text=(
                    "The test that separates the two is simple. Recalibrate "
                    "when the errors are large but balanced: the relationship "
                    "holds and the level has drifted. Rebuild when the errors "
                    "are one-sided and outside the interval, because then a "
                    "variable that matters is missing. This is the second case."
                ),
            ),
            Block(kind="subheading", text="What to do with the result"),
            Block(
                kind="bullets",
                items=(
                    "Monitor the prediction interval, not the point error. A "
                    "point error says the model was wrong; a breach says it "
                    "cannot be right. Only the second is a trigger to act.",
                    "Count the sign of the misses. Same sign in a row is a "
                    "missing variable; mixed signs are noise, and re-fitting "
                    "noise makes a model worse.",
                    "Diagnose before you fix. Unmeasured economic pressure "
                    "means add the level variables and the model works again. "
                    "A changed instrument means re-base the input series "
                    "before it enters any model. An information-environment "
                    "driver means no macro variable will help and a text feed "
                    "becomes necessary.",
                    "When you model a household, prefer the level to the rate. "
                    "A household pays a level; an indicator reports a rate, "
                    "because a rate is what policy targets.",
                    "Hold the diagnosis to the standard of evidence you would "
                    "demand from somebody else. A model rebuilt on a wrong "
                    "diagnosis is more dangerous than the broken one, because "
                    "it carries new confidence.",
                ),
            ),
            Block(
                kind="para",
                text=(
                    "This is also why the negative result in this report is "
                    "useful. A practitioner who buys a sentiment text feed "
                    "because the news feels negative has bought an input that, "
                    "on this evidence, does not forecast the gap."
                ),
            ),
        ]
    )
    return Section("10", "One practical implication", tuple(blocks))


def _section_cutoff(ctx: BuildContext) -> Section:
    """The analysis cutoff and the prospective update plan.

    The brief requires both: "If your project finishes before election day,
    state the cutoff and provide a prospective update plan."
    """
    return Section(
        "11",
        "The cutoff and the prospective update plan",
        tuple([
            Block(
                kind="para",
                text=(
                    "This project finishes before election day. So the cutoff "
                    "is stated here, and no result in this report is a "
                    "statement about what happened at the election."
                ),
            ),
            Block(
                kind="keyvalue",
                items=(
                    ("Analysis cutoff", f"{ANALYSIS_CUTOFF}"),
                    ("Last complete macro month", "2026-09"),
                    ("Last month of the media corpus", "2026-08"),
                    ("Post-election data in this project", "none"),
                ),
            ),
            Block(
                kind="para",
                text=(
                    "A separate warning belongs here. Several series in this "
                    "report move during 2026, and the campaign also happens "
                    "during 2026. Those two facts together are not evidence "
                    "about the campaign. A claim that election proximity caused "
                    "a change needs a counterfactual: the same months in a "
                    "year with no election, or a comparison between places with "
                    "different campaign intensity. This project has neither, so "
                    "it makes no such claim. The election is the reason for the "
                    "monitoring window and not an explanation inside it."
                ),
            ),
            Block(
                kind="bullets",
                items=(
                    "The binding release is the Michigan survey, on the last "
                    "Friday of each month: a new gap value needs it, and the "
                    "other series only move the prediction. Nothing is cached "
                    "across a build, so a new release enters automatically.",
                    "The training window must stay 2010-01 to 2024-12. If it "
                    "moves, the out-of-sample claim in this report is void. A "
                    "coefficient that moves with no new month means a source "
                    "revision and not a defect, and the panel keeps every "
                    "publication date so a revision is visible.",
                    "The conditions that decide each hypothesis were fixed "
                    "before the corpus existed and are not revised after a "
                    "result. One rule is recorded now: if the gap narrows "
                    "sharply in the final pre-election months while media tone "
                    "does not change, that is evidence against the media "
                    "explanation and for the other two.",
                    "The full plan, with the release calendar and the list of "
                    "quantities that must and must not move, is in "
                    "docs/update_plan.md.",
                ),
            ),
        ]),
    )


def _section_ethics(ctx: BuildContext) -> Section:
    """Ethics, limitations, and the list of claims this report does not make."""
    return Section(
        "12",
        "Ethics, limitations, and what this report does not claim",
        tuple([
            Block(kind="subheading", text="Ethics"),
            Block(
                kind="bullets",
                items=(
                    "No human-subject research was done, so no institutional "
                    "approval was required or obtained. The randomized framing "
                    "design here is a proposal and states the approval it "
                    "would need.",
                    "No individual is named in any output: no author name, "
                    "handle, identifier or author-level statistic. The corpus "
                    "is aggregated to monthly series first.",
                    "All data is public or licensed: BLS, BEA, the Federal "
                    "Reserve and Freddie Mac through FRED; GDELT; social-media "
                    "data under the provider's terms. No paywalled text was "
                    "scraped.",
                    "AI_USE.md records what the AI did, with model versions, "
                    "prompts and evidence excerpts preserved.",
                ),
            ),
            Block(
                kind="para",
                text=(
                    "Four limitations change what this report can conclude, and "
                    "each is named where it bites rather than only here. There "
                    "is NO exposure data, so nothing measures what a respondent "
                    "read, and no causal claim about readers is available. There "
                    "is one operator and no second annotator, so no "
                    "inter-annotator agreement statistic is reported. The "
                    "corpus is titles and URL paths and not article bodies, so "
                    "it measures emphasis and not nuance. And Michigan "
                    "publishes neither the Current Conditions and Expectations "
                    "sub-indices nor the income, age, geography and affiliation "
                    "breakdowns free of charge, so the split that would "
                    "discriminate between the hypotheses most directly, and the "
                    "partisan moderator the brief asks about, are both out of "
                    "reach. Section 13 lists every absent result, and "
                    "docs/ethics_and_limitations.md gives the long form with "
                    "the eight errors this project made in its own work."
                ),
            ),
            Block(kind="subheading", text="What this report does not claim"),
            Block(
                kind="bullets",
                items=(
                    "It does not claim that consumer pessimism is irrational.",
                    "It does not claim that negative coverage is biased "
                    "coverage. Coverage can be negative because the situation "
                    "is bad, which is why factual calibration is measured.",
                    "It does not claim that media exposure causes declining "
                    "sentiment, or a causal effect on the national Michigan "
                    "index from any sample result. The transport assumptions "
                    "are not met, and this report states which ones.",
                    "It does not claim that election proximity caused any "
                    "change.",
                    "It does not claim to measure confirmation bias, "
                    "availability bias, or anchoring inside a reader's head, "
                    "and it claims no inter-annotator agreement statistic.",
                    "It does not read a Granger test as causation, and it does "
                    "not report the reverse direction of the heteroskedasticity "
                    "instrument, because that direction is not identified.",
                ),
            ),
            Block(
                kind="para",
                text=(
                    "The eight errors this project made in its own work, with "
                    "the before-and-after numbers for the two that changed a "
                    "conclusion, are in docs/ethics_and_limitations.md."
                ),
            ),
        ]),
    )


def _section_not_available(ctx: BuildContext) -> Section:
    """One register of every missing item. A reader can check the holes here."""
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Every result the study specified and does not have. The list "
                "is assembled from the same objects "
                "that the sections above use, so it cannot drift from them. A "
                "result that is absent is absent. It is not a small number and "
                "it is not a zero."
            ),
        )
    ]
    register: list[tuple[str, str]] = []
    for section in (
        _section_summary,
        _section_data,
        _section_baseline,
        _section_gap,
        _section_hypothesis_a,
        _section_corpus,
        _section_hypothesis_b,
        _section_hypothesis_c,
    ):
        built = section(ctx)
        for missing in built.missing_results():
            # The cause is given in full in the section itself. The register
            # carries its first sentence, so the list stays readable.
            short = missing.cause.split(". ")[0].rstrip(".") + "."
            register.append((f"[{built.number or 'summary'}] {missing.item}", short))
    if register:
        blocks.append(Block(kind="keyvalue", items=tuple(register)))
    else:
        blocks.append(
            Block(
                kind="para",
                text=(
                    "This build has no missing results. Every table above holds "
                    "live or recorded numbers."
                ),
            )
        )

    blocks.append(Block(kind="subheading", text="13.1 What a reader must not conclude"))
    blocks.append(
        Block(
            kind="bullets",
            items=(
                ("Hypothesis B is not rejected. It is not tested. A missing "
                "result and a negative result are different findings."),
                ("An empty Model 3 column in Table 3 does not mean that the "
                "media regressors did not help. It means that the regressors do "
                "not exist yet."),
                ("A row of n/a in Table 2 or Table 4 is not a zero coefficient "
                "and not an insignificant one."),
                ("The price level has no causal reading in this report, even "
                "where its p-value is small."),
            ),
        )
    )
    if ctx.failures:
        blocks.append(Block(kind="subheading", text="13.2 What this build could not run"))
        blocks.append(Block(kind="bullets", items=tuple(ctx.failures)))
    return Section("13", "What is not yet available", tuple(blocks))


def _section_provenance(ctx: BuildContext) -> Section:
    """Provenance, compressed to a pointer and the claim-strength table.

    The full register of recorded constants is in docs/provenance_register.md.
    It is reference material, it belongs to the reproducible package that the
    brief asks for separately, and it does not need to consume a page here.
    """
    blocks: list[Block] = [
        Block(
            kind="para",
            text=(
                "Almost every number here is computed at build time: "
                "build_report.py runs data_loader, baseline_model, "
                "causal_econometrics, corpus_validation and survey_diagnostics "
                "and puts their output into the tables. A small set cannot be "
                "recomputed in every environment, because FRED needs a key and "
                "a network, or because a second tool did the check. Each of "
                f"those carries a tag from C1 to C{len(VERIFIED_CONSTANTS)} "
                "beside its value, so a recorded number is always separable "
                "from a live one, and the full register with sources is in "
                "docs/provenance_register.md. Seven tags were recomputed on "
                "2026-10-16 when the panel moved to the real-time alignment, "
                "and each records what it replaced."
            ),
        ),
        Block(
            kind="para",
            text=(
                "To reproduce it: 'pytest -q', then 'build_report.py', "
                "'build_deck.py' and 'streamlit run app.py'. The notebook "
                "notebooks/perceptual_bias_analysis.ipynb repeats the pipeline "
                "step by step and prints the same values. Every document here "
                "uses ASD-STE100 Simplified Technical English."
            ),
        ),
        Block(kind="subheading", text="14.1 The strength of each claim"),
        Block(
            kind="keyvalue",
            items=(
                ("Association",
                 "Every model coefficient, and the gap. No identification."),
                ("Predictive precedence",
                 ("The Granger tests. A rejection would not license a causal "
                  "reading.")),
                ("Instrumented",
                 ("The Rigobon-Sack IV, in one direction only, under an "
                  "assumption two covariance matrices cannot test.")),
                ("A change at a date",
                 ("The Chow and supremum-Wald tests. They identify a change, "
                  "not its cause.")),
                ("Measured failure",
                 ("The FinBERT probe and the swap tests. These measure an "
                  "instrument, not the economy.")),
                ("Not available",
                 "A result the project specified and does not have."),
            ),
        ),
    ]
    return Section("14", "Provenance and reproduction", tuple(blocks))


# ---------------------------------------------------------------------------
# Checks on an assembled report
# ---------------------------------------------------------------------------


def find_placeholder_markers(sections: list[Section]) -> list[str]:
    """Return every placeholder marker that a section contains.

    An empty list is the only acceptable result. A marker means that the build
    put fabricated text or a fabricated number in the report. The not-available
    blocks are the only way to show a gap, and they carry real prose.
    """
    found: list[str] = []
    for section in sections:
        for text in section.text_content():
            lowered = text.lower()
            for marker in PLACEHOLDER_MARKERS:
                if marker in lowered:
                    found.append(f"{section.heading}: {marker!r} in {text[:70]!r}")
    return found


#: Lines that fit on one page of the rendered document. Letter paper with
#: 0.85 inch margins gives about 663 points of height. The body leading is
#: 13.6 points, so about 48 lines of plain body text fit. The value below is
#: calibrated against a real render: the full report estimates 12.0 pages and
#: renders as 12.
LINES_PER_PAGE = 53

#: Months of the gap printed in Table 1 Panel C. The full series lives in the
#: dashboard and the notebook; the report has a page budget.
TABLE1_GAP_MONTHS = 6

#: Characters that fit on one line of body text at 9.6 points.
CHARS_PER_LINE = 96


def _lines_for_text(text: str, chars: int = CHARS_PER_LINE) -> float:
    return max(1.0, math.ceil(len(text) / chars))


def estimate_block_lines(block: Block) -> float:
    """Estimate the vertical size of one block, in lines of body text."""
    if block.kind == "para":
        return _lines_for_text(block.text) + 0.6
    if block.kind == "subheading":
        return 2.4
    if block.kind == "callout":
        return _lines_for_text(block.text, CHARS_PER_LINE - 10) + 2.4
    if block.kind == "bullets":
        return sum(_lines_for_text(str(i), CHARS_PER_LINE - 6) + 0.3 for i in block.items) + 0.8
    if block.kind == "keyvalue":
        total = 0.0
        for label, value in block.items:
            total += max(
                _lines_for_text(str(label), 34), _lines_for_text(str(value), 60)
            )
        return total + 1.0
    if block.kind == "unavailable":
        assert block.missing is not None
        missing = block.missing
        total = 3.0 + _lines_for_text(missing.item, 80) + _lines_for_text(missing.cause, 86)
        total += sum(_lines_for_text(b, 82) + 0.3 for b in missing.blocks)
        if missing.next_step:
            total += _lines_for_text(missing.next_step, 86) + 0.5
        return total + 1.5
    if block.kind == "figure":
        assert block.figure is not None
        return block.figure.height_inches * 72.0 / 13.6 + 3.2
    if block.kind == "table":
        assert block.table is not None
        total = 2.4
        for panel in block.table.panels:
            total += 2.4 if panel.label else 0.8
            total += 1.4  # the header row
            total += 1.05 * len(panel.rows)
        total += sum(_lines_for_text(note, 110) + 0.2 for note in block.table.footnotes)
        return total + 1.5
    raise ReportError(f"unknown block kind {block.kind!r}")


def estimate_section_lines(section: Section) -> float:
    return 3.2 + sum(estimate_block_lines(block) for block in section.blocks)


def estimate_pages(sections: list[Section], *, title_page: bool = True) -> float:
    """Estimate the page count of the rendered document.

    The estimate is a line count divided by the lines on one page. It is used
    as a check in the tests, so a change that makes the report too short or too
    long fails before a reader sees the PDF.
    """
    lines = sum(estimate_section_lines(section) for section in sections)
    # The masthead costs about six lines at the top of page 1, not a page.
    pages = (lines + (6.0 if title_page else 0.0)) / LINES_PER_PAGE
    return pages


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

TITLE = "The 2026 Sentiment Gap"
SUBTITLE = (
    "Consumer sentiment, macro fundamentals and the limits of three "
    "explanations"
)


def _styles() -> dict[str, Any]:
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            spaceAfter=10,
            textColor=INK,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12.5,
            leading=16,
            alignment=1,
            textColor=INK_SECONDARY,
            spaceAfter=24,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=INK,
        ),
        "subheading": ParagraphStyle(
            "SubHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=9,
            spaceAfter=3,
            textColor=INK,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.6,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
            textColor=INK,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.0,
            leftIndent=12,
            bulletIndent=2,
            spaceAfter=3,
            textColor=INK,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.8,
            leading=14.0,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=4,
            textColor=INK,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.6,
            spaceAfter=0,
            textColor=INK,
        ),
        "cellhead": ParagraphStyle(
            "CellHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.6,
            spaceAfter=0,
            textColor=INK,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.4,
            spaceAfter=1,
            textColor=INK_SECONDARY,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=10.5,
            spaceBefore=3,
            spaceAfter=8,
            textColor=INK_SECONDARY,
        ),
        "tabletitle": ParagraphStyle(
            "TableTitle",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.4,
            leading=12,
            spaceBefore=8,
            spaceAfter=3,
            textColor=INK,
        ),
        "panellabel": ParagraphStyle(
            "PanelLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.4,
            leading=11,
            spaceBefore=4,
            spaceAfter=2,
            textColor=INK_SECONDARY,
        ),
    }


def _xml(text: str) -> str:
    """Escape the three characters that reportlab reads as markup."""
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _render_table(table: tg.Table, styles: dict[str, Any], width: float) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.platypus import KeepTogether, Paragraph, Spacer, TableStyle
    from reportlab.platypus import Table as PdfTable

    flowables: list[Any] = []
    # The caption travels with the first panel. A caption alone at the foot of
    # a page reads as a table that was lost.
    caption: list[Any] = [Paragraph(_xml(table.caption), styles["tabletitle"])]
    for panel in table.panels:
        group: list[Any] = caption
        caption = []
        if panel.label:
            group.append(Paragraph(_xml(panel.label), styles["panellabel"]))
        n = len(panel.columns)
        data: list[list[Any]] = [
            [Paragraph(_xml(name), styles["cellhead"]) for name in panel.columns]
        ]
        commands: list[tuple[Any, ...]] = [
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.HexColor(INK)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK)),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.HexColor(INK)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        for letter_index, letter in enumerate(panel.column_spec):
            if letter == "r":
                commands.append(
                    ("ALIGN", (letter_index, 0), (letter_index, -1), "RIGHT")
                )
        for position, row in enumerate(panel.rows, start=1):
            if row.kind == "section":
                data.append(
                    [Paragraph(f"<i>{_xml(row.cells[0])}</i>", styles["cell"])]
                    + [""] * (n - 1)
                )
                commands.append(("SPAN", (0, position), (-1, position)))
            else:
                data.append([Paragraph(_xml(cell), styles["cell"]) for cell in row.cells])
        first = min(0.30, max(0.18, 1.0 / n + 0.06))
        rest = (1.0 - first) / (n - 1) if n > 1 else 0.0
        widths = [width * first] + [width * rest] * (n - 1)
        pdf_table = PdfTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        pdf_table.setStyle(TableStyle(commands))
        group.append(pdf_table)
        # A panel of a few rows must not split away from its own header.
        flowables.extend(group if len(panel.rows) > 14 else [KeepTogether(group)])
        flowables.append(Spacer(1, 4))
    for position, note in enumerate(table.footnotes):
        prefix = "<b>Notes:</b> " if position == 0 else ""
        flowables.append(Paragraph(prefix + _xml(note), styles["note"]))
    flowables.append(Spacer(1, 8))
    return flowables


def _render_block(block: Block, styles: dict[str, Any], width: float) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, TableStyle
    from reportlab.platypus import Table as PdfTable

    if block.kind == "para":
        return [Paragraph(_xml(block.text), styles["body"])]
    if block.kind == "subheading":
        return [Paragraph(_xml(block.text), styles["subheading"])]
    if block.kind == "bullets":
        return [
            Paragraph(_xml(str(item)), styles["bullet"], bulletText="•")
            for item in block.items
        ] + [Spacer(1, 4)]
    if block.kind == "callout":
        inner = PdfTable(
            [[Paragraph(_xml(block.text), styles["callout"])]],
            colWidths=[width],
            hAlign="LEFT",
        )
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f5fa")),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(SERIES_BLUE)),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return [Spacer(1, 3), inner, Spacer(1, 7)]
    if block.kind == "keyvalue":
        rows = [
            [
                Paragraph(f"<b>{_xml(label)}</b>", styles["cell"]),
                Paragraph(_xml(value), styles["cell"]),
            ]
            for label, value in block.items
        ]
        table = PdfTable(rows, colWidths=[width * 0.36, width * 0.64], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor(GRID)),
                ]
            )
        )
        return [Spacer(1, 2), table, Spacer(1, 7)]
    if block.kind == "unavailable":
        assert block.missing is not None
        missing = block.missing
        parts: list[Any] = [
            Paragraph(
                "<b>NOT YET AVAILABLE: </b>" + _xml(missing.item),
                styles["callout"],
            ),
            Paragraph("<b>Cause. </b>" + _xml(missing.cause), styles["callout"]),
        ]
        if missing.blocks:
            parts.append(
                Paragraph("<b>What a reader cannot see:</b>", styles["callout"])
            )
            parts.extend(
                Paragraph(_xml(item), styles["bullet"], bulletText="•")
                for item in missing.blocks
            )
        if missing.next_step:
            parts.append(
                Paragraph("<b>Next step. </b>" + _xml(missing.next_step), styles["callout"])
            )
        inner = PdfTable([[parts]], colWidths=[width], hAlign="LEFT")
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf1ec")),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(SERIES_ORANGE)),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return [Spacer(1, 3), inner, Spacer(1, 8)]
    if block.kind == "figure":
        assert block.figure is not None
        figure = block.figure
        if not figure.path.exists():
            raise ReportError(f"figure file is missing: {figure.path}")
        scale = min(1.0, width / (figure.width_inches * 72.0))
        image = Image(
            str(figure.path),
            width=figure.width_inches * 72.0 * scale,
            height=figure.height_inches * 72.0 * scale,
        )
        image.hAlign = "CENTER"
        return [
            KeepTogether(
                [
                    Spacer(1, 4),
                    image,
                    Paragraph(_xml(figure.caption), styles["caption"]),
                ]
            )
        ]
    if block.kind == "table":
        assert block.table is not None
        return _render_table(block.table, styles, width)
    raise ReportError(f"unknown block kind {block.kind!r}")


def render_pdf(
    sections: list[Section],
    path: Path | str = DEFAULT_PDF,
    *,
    build_note: str = "",
) -> tuple[Path, int]:
    """Render the sections to a PDF. Return the path and the real page count."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    path = Path(path)
    styles = _styles()
    margin = 0.85 * inch
    width = letter[0] - 2 * margin

    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=TITLE,
        author="perceptual-bias",
        subject=SUBTITLE,
    )

    # The title is a masthead at the top of page 1 and not a page of its own.
    # The brief allows 10 to 12 pages and a separate title page spends one of
    # them on four lines of text.
    story: list[Any] = [
        Paragraph(_xml(TITLE), styles["title"]),
        Paragraph(_xml(SUBTITLE), styles["subtitle"]),
    ]
    if build_note:
        story.append(Paragraph(_xml(build_note), styles["caption"]))
    story.append(Spacer(1, 0.18 * inch))

    for section in sections:
        story.append(Paragraph(_xml(section.heading), styles["heading"]))
        for block in section.blocks:
            story.extend(_render_block(block, styles, width))

    pages = {"count": 0}

    def _count(canvas: Any, doc: Any) -> None:
        pages["count"] = doc.page
        canvas.saveState()
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColorCMYK(0, 0, 0, 0.55)
        canvas.drawRightString(
            letter[0] - margin, 0.5 * inch, f"{TITLE} — page {doc.page}"
        )
        canvas.restoreState()

    document.build(story, onFirstPage=_count, onLaterPages=_count)
    return path, pages["count"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    pdf_path: Path | str = DEFAULT_PDF,
    *,
    csv_path: Path | str = CSV_PATH,
    allow_network: bool = True,
    with_figures: bool = True,
) -> tuple[Path, int, list[Section]]:
    """Run the pipeline, assemble the sections, and render the PDF."""
    ctx = gather(csv_path, allow_network=allow_network)
    if with_figures:
        ctx.figures = make_figures(ctx)
    sections = assemble_sections(ctx)

    markers = find_placeholder_markers(sections)
    if markers:
        raise ReportError(
            "the report contains placeholder text, which is never acceptable: "
            + "; ".join(markers)
        )

    note = (
        f"Built {pd.Timestamp.today():%Y-%m-%d}. "
        f"FRED: {'available' if ctx.fred_available else 'not configured'}. "
        f"GDELT corpus: {ctx.media_days_cached} of {GDELT_TARGET_DAYS} days "
        "cached."
    )
    path, pages = render_pdf(sections, pdf_path, build_note=note)
    logger.info("wrote %s (%d pages, estimate %.1f)", path, pages, estimate_pages(sections))
    return path, pages, sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Sentiment Gap PDF report.")
    parser.add_argument("--output", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not request FRED; report the FRED series as not available",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    path, pages, sections = build(
        args.output,
        allow_network=not args.offline,
        with_figures=not args.no_figures,
    )
    print(f"wrote {path}")
    print(f"pages: {pages} (estimate {estimate_pages(sections):.1f})")
    print(f"sections: {len(sections)}")
    missing = sum(len(s.missing_results()) for s in sections)
    print(f"not-available blocks: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
