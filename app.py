"""Streamlit dashboard for the 2026 Sentiment Gap study.

The dashboard shows six panels. Panels 1 to 4 have data now. Panel 4 reads the
three SVD artifacts of `scripts/build_svd.py`. Panel 5 and panel 6 are not
built in the dashboard yet; the report gives their result.

**Honest degradation is the first rule of this module.** A panel without data
does not show an empty chart. It does not show a placeholder series. It does
not show a spinner that never stops. It shows a short text block. The text
block names each input that is absent. A reader can never read an empty panel
as a result. `panel_availability` makes this decision from the columns that
exist, and `render_unavailable` prints the result.

**The module separates logic from the screen.** Each function above
`render_*` is pure. A pure function takes data and returns data or a Plotly
figure. It makes no `st.*` call and it reads no network. The tests call these
functions directly. The tests do not start a server.

**Colors come from one selected palette.** `PALETTES` holds a light palette and
a dark palette. The viewer selects the palette. The dashboard does not invert
the light colors to make the dark colors. Each figure receives the palette and
writes the colors into the figure. The dashboard also sends ``theme=None`` to
`st.plotly_chart`, because the Streamlit theme would otherwise replace these
colors.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:  # lets `streamlit run app.py` find the package
    sys.path.insert(0, str(SRC_DIR))

from perceptual_bias.baseline_model import (  # noqa: E402
    BASELINE_FEATURES,
    fit_baseline,
    gap_summary,
    sentiment_gap,
)
from perceptual_bias.causal_econometrics import (  # noqa: E402
    compare_models,
    granger_media_gap,
)
from perceptual_bias.data_loader import (  # noqa: E402
    FredSource,
    build_monthly_panel,
    training_slice,
)
from perceptual_bias.table_generator import (  # noqa: E402
    table3_model_comparison,
    table4_granger_and_iv,
    to_markdown,
)

logger = logging.getLogger(__name__)

CSV_PATH = REPO_ROOT / "data" / "raw" / "Copy_of_Perceptual_Bias_Data.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# The baseline model trains on 2010-01 to 2024-12. Each month after this date
# is out of sample. The model never saw these months.
HOLDOUT_START = "2025-01-01"
TRAIN_START = "2010-01-01"
TRAIN_END = "2024-12-31"

# The media columns that `nlp_media_svd` produces. Refer to
# `causal_econometrics.MEDIA_FEATURES`.
MEDIA_TONE_COLUMN = "media_tone"
ATTRIBUTION_COLUMN = "political_attribution"
SALIENCE_COLUMN = "topic_salience"
GAP_COLUMN = "gap"

# The brief asks for four media dimensions: tone, salience, framing and factual
# calibration. The first two are in the monthly file now. The second two come
# from `nlp_media_svd`, and that module writes them with more than one possible
# name: a single column, or one column for each contrast of the framing
# lexicon. The dashboard accepts any of these names and degrades when none of
# them is present. It never substitutes tone for framing: they measure
# different things.
FRAMING_COLUMN = "framing"
FRAMING_ALTERNATIVES: tuple[str, ...] = (
    FRAMING_COLUMN,
    "framing_level_vs_rate",
    "framing_household_vs_macro",
    "framing_current_vs_forecast",
    "level_vs_rate",
    "household_vs_macro",
    "current_vs_forecast",
)
CALIBRATION_COLUMN = "factual_calibration"
CALIBRATION_ALTERNATIVES: tuple[str, ...] = (
    CALIBRATION_COLUMN,
    "calibration_gap",
    "media_calibration",
)

# The columns that a breakdown by income, age, geography or political
# affiliation would need. No column of this group exists, because the project
# holds the published Michigan index only. Panel 13 names these columns and
# shows no chart. Refer to `DEMOGRAPHIC_NOTE`.
DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "sentiment_by_income",
    "sentiment_by_age",
    "sentiment_by_region",
    "sentiment_by_party",
)

# ---------------------------------------------------------------------------
# The derived artifacts
# ---------------------------------------------------------------------------
#
# **A panel is ready when its own derived artifact is on the disk.** An earlier
# version counted the CSV files of the abandoned GDELT headline pull and then
# called every media panel "in collection". That count measured a pull that the
# project does not use. The theme corpus comes from the GKG bulk files, and the
# panels read the small files below, not the corpus. A count of raw files can
# never again flip a panel, because no panel reads a raw file.

MEDIA_SERIES_PATH = PROCESSED_DIR / "media_monthly.parquet"
SVD_SPECTRUM_PATH = PROCESSED_DIR / "svd_spectrum.parquet"
SVD_TOPICS_PATH = PROCESSED_DIR / "svd_topics.parquet"
SVD_MONTHLY_PATH = PROCESSED_DIR / "svd_monthly.parquet"

# The headline corpus. Panel 10 reads the source of each headline from this
# file, because no smaller file holds the domain of an article.
#
# **This is not a count of files.** The panel reads one named corpus file and
# shows what is inside it. The rule above holds: readiness comes from a file
# that a panel reads, and the panel reports the size of what it read. A partial
# file gives a smaller composition, and the panel prints the count of
# headlines, so a reader can see that the corpus is partial.
HEADLINES_PATH = REPO_ROOT / "data" / "raw" / "gdelt" / "headlines.parquet"

MEDIA_COLUMNS = (MEDIA_TONE_COLUMN, SALIENCE_COLUMN, ATTRIBUTION_COLUMN)

# `nlp_media_svd` measures the framing from two text sources: the headline of
# an article and the slug of its URL. The two measures are a validity check of
# each other, and panel 7 prints their agreement. The dashboard shows the
# agreement because a low agreement is a limit on the measure.
FRAMING_SOURCE_COLUMNS: tuple[str, str] = ("framing_headline", "framing_slug")

# The optional media columns. These are not regressors of Model 3. The
# dashboard shows them, because the brief asks for four media dimensions.
MEDIA_OPTIONAL_COLUMNS: tuple[str, ...] = (
    FRAMING_ALTERNATIVES + CALIBRATION_ALTERNATIVES + FRAMING_SOURCE_COLUMNS
)

# `KSelection.spectrum_frame()` writes these columns. A file without the first
# two columns is not a spectrum, and the loader refuses it.
SPECTRUM_REQUIRED_COLUMNS = ("component", "singular_value")
SPECTRUM_OPTIONAL_COLUMNS = (
    "cumulative_energy",
    "permuted_max",
    "above_noise_floor",
    "above_permuted_component",
)
TOPICS_REQUIRED_COLUMNS = ("topic", "rank", "term")


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """One selected set of chart colors.

    The dark palette is a selected palette. It is not an inversion of the light
    palette. Each dark value was chosen for the dark surface.

    `series_3` is below 3:1 contrast on the light surface. The palette
    validator gave a warning for this color. The warning asks for relief. Each
    chart that uses `series_3` must also show direct labels or a table view.

    `sequential` is one hue in seven steps. The light ramp goes from light to
    dark, because a dark step gives more contrast on a light surface. The dark
    ramp goes from dark to light for the same cause on the dark surface. The
    dark ramp is a selected ramp. It is not the light ramp in reverse order:
    each step has its own value, and the ramp starts one step off the dark
    surface.
    """

    name: str
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series_1: str
    series_2: str
    series_3: str
    band: str
    diverging_low: str
    diverging_mid: str
    diverging_high: str
    sequential: tuple[str, ...]


LIGHT_PALETTE = Palette(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series_1="#2a78d6",
    series_2="#eb6834",
    series_3="#1baf7a",
    band="rgba(42, 120, 214, 0.10)",
    diverging_low="#d03b3b",
    diverging_mid="#f0efec",
    diverging_high="#2a78d6",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)

DARK_PALETTE = Palette(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink_primary="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series_1="#3987e5",
    series_2="#d95926",
    series_3="#199e70",
    band="rgba(57, 135, 229, 0.14)",
    diverging_low="#e66767",
    diverging_mid="#383835",
    diverging_high="#3987e5",
    sequential=("#16283c", "#1d3c5e", "#245084", "#2c67ab", "#3d82cb", "#6ba6e2", "#a3caf4"),
)

PALETTES: dict[str, Palette] = {"light": LIGHT_PALETTE, "dark": DARK_PALETTE}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def get_palette(mode: str) -> Palette:
    """Return the selected palette.

    An unknown name gives the light palette. The dashboard must still draw.
    """
    return PALETTES.get(str(mode).lower(), LIGHT_PALETTE)


def diverging_colorscale(palette: Palette) -> list[list[Any]]:
    """Blue to red with a gray midpoint.

    The midpoint must read as "nothing". A gray midpoint does this. A hue at
    the midpoint does not. The two poles are warm and cool, so they read as
    opposite.
    """
    return [
        [0.0, palette.diverging_low],
        [0.5, palette.diverging_mid],
        [1.0, palette.diverging_high],
    ]


def sequential_colorscale(palette: Palette) -> list[list[Any]]:
    """One hue in seven steps, for a magnitude scale.

    A magnitude scale uses one hue. A rainbow scale gives a hue to a value, and
    a reader then reads the hue as a class. Each palette gives its own steps,
    so the ramp gets more contrast with the value on both surfaces.
    """
    steps = list(palette.sequential)
    last = len(steps) - 1
    return [[index / last, color] for index, color in enumerate(steps)]


# ---------------------------------------------------------------------------
# Panel availability — the honest degradation rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelStatus:
    """The state of one panel, and the cause of that state.

    `missing` gives the name of each input that is absent. The screen prints
    these names. A reader must see which input is absent. A reader must not see
    an empty chart.
    """

    key: str
    number: int
    title: str
    available: bool
    missing: tuple[str, ...] = ()
    reason: str = ""
    needs_artifact: bool = False
    missing_artifacts: tuple[str, ...] = ()

    @property
    def artifact_absent(self) -> bool:
        """True when a derived file of this panel is absent.

        `needs_artifact` says that the panel reads a file. This property says
        that a file is absent now. The screen needs the second statement: a
        panel can read a file that is present and still wait for one column of
        that file, and a heading that said "the derived file is not on the
        disk" would then be false.
        """
        return bool(self.missing_artifacts)

    @property
    def label(self) -> str:
        return f"{self.number}. {self.title}"

    def summary(self) -> str:
        if self.available:
            return f"{self.label}: available"
        missing = ", ".join(self.missing) if self.missing else "unknown inputs"
        return f"{self.label}: not available ({missing})"


@dataclass(frozen=True)
class PanelSpec:
    """One panel, and the inputs that the panel needs.

    `columns` names a column of the panel frame or of the gap frame.
    `artifacts` names a derived file by its key. A panel needs both groups.

    `any_of` names a group of columns where the panel needs one member only.
    A measure can arrive with more than one name: `nlp_media_svd` can write the
    framing measure as one column or as one column for each contrast. The first
    name of the group is the name that the screen prints when no member is
    present, so a reader always sees one canonical name and also the list of
    the names that the panel accepts.
    """

    key: str
    number: int
    title: str
    columns: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()


# Each panel and the inputs that the panel needs. This table is the only place
# that states a requirement.
PANEL_REQUIREMENTS: tuple[PanelSpec, ...] = (
    PanelSpec(
        "timeline",
        1,
        "Observed and predicted sentiment",
        columns=("observed", "predicted", "pi_lower", "pi_upper"),
    ),
    PanelSpec("gap", 2, "The Sentiment Gap by month", columns=(GAP_COLUMN,)),
    PanelSpec("models", 3, "Model comparison, Models 1 to 3"),
    PanelSpec(
        "scree",
        4,
        "SVD singular value spectrum",
        artifacts=("svd_spectrum",),
    ),
    PanelSpec(
        "media_tone",
        5,
        "Media tone over time",
        columns=(MEDIA_TONE_COLUMN,),
        artifacts=("media_monthly",),
    ),
    PanelSpec(
        "granger",
        6,
        "Granger causality, media tone and the gap",
        columns=(MEDIA_TONE_COLUMN, GAP_COLUMN),
        artifacts=("media_monthly",),
    ),
    PanelSpec(
        "framing",
        7,
        "Media framing: price levels against inflation rates",
        artifacts=("media_monthly",),
        any_of=FRAMING_ALTERNATIVES,
    ),
    PanelSpec(
        "calibration",
        8,
        "Factual calibration of numerical claims",
        artifacts=("media_monthly",),
        any_of=CALIBRATION_ALTERNATIVES,
    ),
    # Panel 9 needs no column and no file. The event list is a documented
    # constant of this module, so the panel is always available.
    PanelSpec("events", 9, "Economic releases and major events"),
    PanelSpec(
        "composition",
        10,
        "Corpus composition by source",
        artifacts=("headlines",),
    ),
    PanelSpec(
        "coverage",
        11,
        "Monthly coverage of the corpus",
        artifacts=("media_monthly",),
        columns=(MEDIA_TONE_COLUMN,),
    ),
    # Panel 12 reads the panel frame, the gap frame and the fitted model. Each
    # of the three exists whenever the dashboard runs, so the panel is always
    # available.
    PanelSpec("quality", 12, "Missingness and model performance"),
    PanelSpec(
        "demographics",
        13,
        "Differences by income, age, geography and political affiliation",
        columns=DEMOGRAPHIC_COLUMNS,
    ),
)

ARTIFACT_MESSAGE = "Not yet available — the derived file is not on the disk."


@dataclass(frozen=True)
class ArtifactStatus:
    """The state of one derived file, and one fact about it.

    `fact` gives a quantity that a reader can check, for example the count of
    months. An absent file gives an empty fact. `command` gives the script that
    writes the file, so a reader can act.
    """

    key: str
    name: str
    path: Path
    present: bool
    fact: str = ""
    command: str = ""

    def summary(self) -> str:
        if self.present:
            return f"{self.name}: present. {self.fact}".strip()
        return f"{self.name}: absent. Run {self.command}." if self.command else f"{self.name}: absent."


# Each derived file, the script that writes it, and how to state its one fact.
ARTIFACT_SPECS: tuple[tuple[str, str, Path, str], ...] = (
    (
        "media_monthly",
        "The monthly media series",
        MEDIA_SERIES_PATH,
        ".venv/bin/python scripts/build_media_series.py",
    ),
    (
        "svd_spectrum",
        "The SVD spectrum",
        SVD_SPECTRUM_PATH,
        ".venv/bin/python scripts/build_svd.py",
    ),
    (
        "svd_topics",
        "The SVD topic terms",
        SVD_TOPICS_PATH,
        ".venv/bin/python scripts/build_svd.py",
    ),
    (
        "svd_monthly",
        "The monthly topic shares",
        SVD_MONTHLY_PATH,
        ".venv/bin/python scripts/build_svd.py",
    ),
    (
        "headlines",
        "The headline corpus",
        HEADLINES_PATH,
        ".venv/bin/python scripts/pull_gdelt.py",
    ),
)


def artifact_fact(key: str, frame: pd.DataFrame) -> str:
    """Give one fact about a derived file, in the unit of that file."""
    rows = int(len(frame))
    if key == "media_monthly":
        articles = ""
        if "n_articles" in frame.columns:
            articles = f"{int(frame['n_articles'].sum()):,} articles over "
        return f"{articles}{rows} months."
    if key == "svd_spectrum":
        return f"{rows} components."
    if key == "svd_topics":
        topics = int(frame["topic"].nunique()) if "topic" in frame.columns else rows
        return f"{topics} topics."
    if key == "svd_monthly":
        columns = len([name for name in frame.columns if str(name).startswith("topic_")])
        return f"{columns} topics over {rows} months."
    if key == "headlines":
        return f"{rows:,} headlines."
    return f"{rows} rows."


def artifact_status(
    frames: Mapping[str, pd.DataFrame | None],
    specs: Sequence[tuple[str, str, Path, str]] = ARTIFACT_SPECS,
) -> dict[str, ArtifactStatus]:
    """Report the state of each derived file.

    The function is pure: it reads the frames that a loader already returned.
    A `None` frame means that the file is absent or that the file is not
    usable. Both states give the same result on the screen, because both states
    mean that the panel has no input.
    """
    statuses: dict[str, ArtifactStatus] = {}
    for key, name, path, command in specs:
        frame = frames.get(key)
        present = frame is not None and not frame.empty
        statuses[key] = ArtifactStatus(
            key=key,
            name=name,
            path=Path(path),
            present=bool(present),
            fact=artifact_fact(key, frame) if present else "",
            command=command,
        )
    return statuses


def panel_availability(
    available_columns: Iterable[str],
    artifacts: Mapping[str, ArtifactStatus] | None = None,
) -> dict[str, PanelStatus]:
    """Decide which panel can draw, from the inputs that exist.

    This function is the single decision point for honest degradation. It is
    pure. The tests call it directly, and a test holds the render path against
    this result.

    A panel is available only when every column and every derived file that the
    panel needs is present. A panel with no requirement is always available:
    panel 3 renders Model 3 as "n/a" with a cause, because the project
    specified Model 3. A dropped column would instead hide the research design.

    **An absent input is a state of the build, not a result of the study.** The
    reason names the file and the script. It does not say that a hypothesis
    failed. Hypothesis B did fail, and panel 5 and panel 6 give that result
    from the data. The two statements must stay separate.
    """
    present = {str(name) for name in available_columns}
    ready = {
        key: status.present for key, status in (artifacts or {}).items()
    }
    statuses: dict[str, PanelStatus] = {}
    for spec in PANEL_REQUIREMENTS:
        missing_columns = tuple(name for name in spec.columns if name not in present)
        group_absent = bool(spec.any_of) and not any(
            name in present for name in spec.any_of
        )
        if group_absent:
            # One canonical name goes into `missing`, so the screen prints one
            # name and not seven. The reason below lists every accepted name.
            missing_columns = missing_columns + (spec.any_of[0],)
        missing_files = tuple(
            key for key in spec.artifacts if not ready.get(key, False)
        )
        missing = missing_files + missing_columns
        if not missing:
            statuses[spec.key] = PanelStatus(
                key=spec.key,
                number=spec.number,
                title=spec.title,
                available=True,
                needs_artifact=bool(spec.artifacts),
            )
            continue
        parts = []
        if missing_files:
            files = ", ".join(
                str((artifacts or {}).get(key).path.name) if (artifacts or {}).get(key) else key
                for key in missing_files
            )
            commands = sorted(
                {
                    status.command
                    for key in missing_files
                    if (status := (artifacts or {}).get(key)) is not None and status.command
                }
            )
            parts.append(f"These files are absent: {files}.")
            if commands:
                parts.append("Run " + "; ".join(commands) + ".")
        if missing_columns:
            parts.append(f"These columns are absent: {', '.join(missing_columns)}.")
        if group_absent:
            parts.append(
                "The panel accepts any one of these names: "
                + ", ".join(spec.any_of)
                + "."
            )
        statuses[spec.key] = PanelStatus(
            key=spec.key,
            number=spec.number,
            title=spec.title,
            available=False,
            missing=missing,
            reason=" ".join(parts),
            needs_artifact=bool(spec.artifacts),
            missing_artifacts=missing_files,
        )
    return statuses


@dataclass(frozen=True)
class FredStatus:
    """Whether FRED gave data, and which panels are reduced without it."""

    key_present: bool
    series: tuple[str, ...] = ()
    reduced_panels: tuple[str, ...] = ()
    message: str = ""

    @property
    def available(self) -> bool:
        return bool(self.series)


# Without FRED, two things happen to panel 3. `data_loader` substitutes an
# approximate price level, and the mortgage rate stays absent. Model 2 needs the
# mortgage rate, so Model 2 shows "n/a". Panels 1 and 2 do not change: the
# baseline uses only the columns of the supplied CSV file.
FRED_REDUCED_PANELS: tuple[str, ...] = ("3. Model comparison, Models 1 to 3",)

FRED_REDUCTION = (
    "Panel 3 is reduced. Model 2 needs the FRED mortgage rate, so Model 2 shows "
    "n/a, and the price level is an approximation from the published YoY rates. "
    "Panel 1 and panel 2 do not change."
)


def fred_status(report: Any, key_present: bool) -> FredStatus:
    """Report the FRED state in words that a reader can act on.

    A configured key does not promise data. The `PanelReport` says what FRED
    delivered. This function uses the report, not the key.
    """
    series = tuple(getattr(report, "fred_columns", ()) or ())
    if series:
        return FredStatus(
            key_present=key_present,
            series=series,
            message=f"FRED gave these series: {', '.join(series)}.",
        )
    if key_present:
        message = "A FRED key is present, but FRED gave no series. " + FRED_REDUCTION
    else:
        message = (
            "No FRED key is present. The dashboard still runs. " + FRED_REDUCTION
        )
    return FredStatus(
        key_present=key_present,
        series=(),
        reduced_panels=FRED_REDUCED_PANELS,
        message=message,
    )


def fred_key_present(environ: Mapping[str, str] | None = None) -> bool:
    """Say whether a FRED key is in the environment.

    The function returns a boolean only. It never returns the key and it never
    prints the key.
    """
    source = os.environ if environ is None else environ
    return bool(str(source.get("FRED_API_KEY", "")).strip())


# ---------------------------------------------------------------------------
# Economic releases and major events
# ---------------------------------------------------------------------------
#
# The brief asks the dashboard to show the relevant economic releases and the
# major events. This section holds that list.
#
# **Each entry gives its source.** A date without a source is not usable in a
# graded project, because a reader cannot check it. The list is short on
# purpose: it holds the dates that change the reading of the series, and it
# does not hold a date that this project did not verify.
#
# **The list holds no individual CPI or PCE release day.** BLS and BEA publish
# a release schedule for each year, and this project did not copy those
# schedules by hand. A list of approximately 70 release days from memory would
# be a list of invented dates. `RELEASE_SCHEDULE` gives what the model needs
# instead: the publisher of each series, the reference period and the
# publication lag, each with its source. `release_calendar` then gives the
# publication month of each reference month from that lag.


@dataclass(frozen=True)
class EconomicEvent:
    """One dated event, with the source of the date.

    `kind` groups the events. `mark` says whether the event gets a line on a
    time-series chart: a chart with 19 lines carries no information, so only
    the events that change the reading of the series get a line. Every event
    stays in the table of panel 9.
    """

    date: str
    label: str
    kind: str
    source: str
    note: str = ""
    mark: bool = False

    @property
    def timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.date)

    @property
    def month(self) -> str:
        return self.timestamp.strftime("%Y-%m")


# The statute that fixes a federal election day. It gives both election dates
# of this project, and a reader can check them with a calendar.
ELECTION_SOURCE = (
    "2 U.S.C. 7: the election is on the Tuesday after the first Monday of "
    "November."
)

MICHIGAN_SOURCE = (
    "University of Michigan, Surveys of Consumers, the methodology "
    "announcement on the move to web interviews (sca.isr.umich.edu). The "
    "project uses the same date in "
    "`survey_diagnostics.MICHIGAN_WEB_TRANSITION`."
)

FOMC_SOURCE = (
    "Board of Governors of the Federal Reserve System, the FOMC calendars "
    "(federalreserve.gov/monetarypolicy/fomccalendars.htm). Each date is the "
    "second day of the meeting, which is the day of the decision."
)

# The FOMC dates up to this month were checked against the published calendar
# of that year. A later date comes from the calendar that the Federal Reserve
# publishes in advance. Panel 9 prints this limit, because a reader must know
# which dates to check before a quotation.
FOMC_VERIFIED_THROUGH = "2025-12"

ECONOMIC_EVENTS: tuple[EconomicEvent, ...] = (
    EconomicEvent(
        "2024-07-31",
        "Michigan completes the move to web interviews",
        "survey method",
        MICHIGAN_SOURCE,
        "Telephone interviewing stopped from the July 2024 survey. A level "
        "shift at this date is a measurement effect. It is not a change of "
        "opinion. Hypothesis C tests this date.",
        mark=True,
    ),
    EconomicEvent(
        "2024-09-18",
        "FOMC decision: the first rate cut of the cycle",
        "policy decision",
        FOMC_SOURCE,
        "The committee reduced the target range by 50 basis points.",
        mark=True,
    ),
    EconomicEvent("2024-11-07", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent(
        "2024-11-05",
        "The 2024 general election",
        "election",
        ELECTION_SOURCE,
        "",
        mark=True,
    ),
    EconomicEvent("2024-12-18", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-01-29", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-03-19", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-05-07", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-06-18", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-07-30", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-09-17", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-10-29", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent("2025-12-10", "FOMC decision", "policy decision", FOMC_SOURCE),
    EconomicEvent(
        "2026-01-28",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-03-18",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-04-29",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-06-17",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-07-29",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-09-16",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-10-28",
        "FOMC decision",
        "policy decision",
        FOMC_SOURCE,
        "Check this date against the calendar before a quotation.",
    ),
    EconomicEvent(
        "2026-11-03",
        "The 2026 midterm election",
        "election",
        ELECTION_SOURCE,
        "The data of this project stops before this date. The report gives a "
        "plan for the months after the cutoff.",
        mark=True,
    ),
)


@dataclass(frozen=True)
class ReleaseSpec:
    """One economic release: who publishes it, when, and from which source.

    `lag_days` is the typical count of days from the end of the reference
    period to the publication. It is a typical value, not a promise: each
    publisher gives the exact day in its own schedule, and `source` names that
    schedule.
    """

    series: str
    publisher: str
    reference_period: str
    lag_days: int
    source: str


RELEASE_SCHEDULE: tuple[ReleaseSpec, ...] = (
    ReleaseSpec(
        "U-3 unemployment rate",
        "BLS",
        "the reference month",
        7,
        "BLS, the Employment Situation release schedule "
        "(bls.gov/schedule/news_release/empsit.htm).",
    ),
    ReleaseSpec(
        "CPI",
        "BLS",
        "the reference month",
        13,
        "BLS, the CPI release schedule "
        "(bls.gov/schedule/news_release/cpi.htm).",
    ),
    ReleaseSpec(
        "PCE price index",
        "BEA",
        "the reference month",
        28,
        "BEA, the Personal Income and Outlays release schedule "
        "(bea.gov/news/schedule).",
    ),
    ReleaseSpec(
        "Real GDP, the advance estimate",
        "BEA",
        "the reference quarter",
        30,
        "BEA, the Gross Domestic Product release schedule "
        "(bea.gov/news/schedule).",
    ),
    ReleaseSpec(
        "Michigan index, the final value",
        "University of Michigan",
        "the reference month",
        0,
        "University of Michigan, Surveys of Consumers, the release calendar "
        "(sca.isr.umich.edu).",
    ),
)


def events_frame(
    events: Sequence[EconomicEvent] = ECONOMIC_EVENTS,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Make the table of panel 9, in date order.

    `start` and `end` keep the events of one window. The table always holds the
    source of each date, because a date without a source is not checkable.
    """
    rows = []
    for event in events:
        stamp = event.timestamp
        if start is not None and stamp < pd.Timestamp(start):
            continue
        if end is not None and stamp > pd.Timestamp(end):
            continue
        rows.append(
            {
                "date": stamp,
                "month": event.month,
                "event": event.label,
                "kind": event.kind,
                "note": event.note,
                "source": event.source,
                "marked": bool(event.mark),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["date", "month", "event", "kind", "note", "source", "marked"],
    )
    if frame.empty:
        return frame
    return frame.sort_values("date").reset_index(drop=True)


def marked_events(
    events: Sequence[EconomicEvent] = ECONOMIC_EVENTS,
) -> tuple[EconomicEvent, ...]:
    """Give the events that get a line on a time-series chart.

    A chart with a line for each of the 21 events carries no information. The
    `mark` field selects the events that change the reading of a series: the
    two election dates, the survey transition and the first rate cut.
    """
    return tuple(event for event in events if event.mark)


def release_calendar(
    months: Sequence[str] | Sequence[pd.Timestamp],
    schedule: Sequence[ReleaseSpec] = RELEASE_SCHEDULE,
) -> pd.DataFrame:
    """Give the publication month of each series for each reference month.

    The dashboard shows this table because the brief asks for the information
    set at each prediction date. A value of a reference month is not available
    in that month: the CPI of one month publishes in the next month, and the
    advance GDP estimate of a quarter publishes after the quarter ends. The lag
    comes from `RELEASE_SCHEDULE`, which names its source.
    """
    stamps = [pd.Timestamp(month) for month in months]
    rows = []
    for stamp in stamps:
        period_end = stamp + pd.offsets.MonthEnd(0)
        for spec in schedule:
            published = period_end + pd.Timedelta(days=int(spec.lag_days))
            rows.append(
                {
                    "reference month": period_end.strftime("%Y-%m"),
                    "series": spec.series,
                    "publisher": spec.publisher,
                    "typical lag, days": int(spec.lag_days),
                    "publication month": published.strftime("%Y-%m"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "reference month",
            "series",
            "publisher",
            "typical lag, days",
            "publication month",
        ],
    )


# ---------------------------------------------------------------------------
# Data preparation — pure functions
# ---------------------------------------------------------------------------


def prepare_timeline(
    gaps: pd.DataFrame,
    holdout_start: str = HOLDOUT_START,
) -> pd.DataFrame:
    """Make the frame for panel 1.

    The function copies the input. A cached frame must not change in place.

    `out_of_sample` marks each month from `holdout_start`. The chart marks this
    period, because the model never saw it.
    """
    required = ("observed", "predicted", "pi_lower", "pi_upper")
    missing = [name for name in required if name not in gaps.columns]
    if missing:
        raise KeyError(f"timeline needs these columns: {missing}")

    frame = gaps.loc[:, list(required)].copy()
    if GAP_COLUMN in gaps.columns:
        frame[GAP_COLUMN] = gaps[GAP_COLUMN]
    if "outside_pi" in gaps.columns:
        frame["outside_pi"] = gaps["outside_pi"]
    frame = frame.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["out_of_sample"] = frame["date"] >= pd.Timestamp(holdout_start)
    frame["sample"] = frame["out_of_sample"].map(
        {True: "out of sample", False: "in sample"}
    )
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    return frame


def prepare_gap_months(gaps: pd.DataFrame, holdout_start: str = HOLDOUT_START) -> pd.DataFrame:
    """Make the frame for panel 2: one row for each out-of-sample month.

    The function keeps only the months with a gap value. A month without a gap
    value is not a zero gap. A zero bar would read as "the model was correct".
    """
    if GAP_COLUMN not in gaps.columns:
        raise KeyError(f"gap frame needs the column {GAP_COLUMN!r}")

    frame = gaps.copy()
    frame = frame.loc[frame.index >= pd.Timestamp(holdout_start)]
    frame = frame.loc[frame[GAP_COLUMN].notna()]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame.index),
            GAP_COLUMN: frame[GAP_COLUMN].astype(float).to_numpy(),
        }
    )
    out["month"] = out["date"].dt.strftime("%Y-%m")
    for name in ("observed", "predicted"):
        if name in frame.columns:
            out[name] = frame[name].astype(float).to_numpy()
    if "outside_pi" in frame.columns:
        out["outside_pi"] = frame["outside_pi"].to_numpy()
    return out.reset_index(drop=True)


def diverging_zero_domain(values: Sequence[float] | pd.Series) -> tuple[float, float]:
    """Give a color domain with zero at the middle.

    The gap is always negative in this study. A scale from the minimum to the
    maximum of the data would put the gray midpoint at a negative value, and a
    small negative gap would then read as a positive gap. A symmetric domain
    holds zero at the gray midpoint.

    An empty input gives (-1, 1), because a chart must still have a scale.
    """
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return -1.0, 1.0
    limit = float(series.abs().max())
    if limit <= 0:
        return -1.0, 1.0
    return -limit, limit


def worst_month(gap_months: pd.DataFrame) -> tuple[str, float] | None:
    """Give the month with the largest negative gap, and the gap."""
    if gap_months.empty or GAP_COLUMN not in gap_months.columns:
        return None
    row = gap_months.loc[gap_months[GAP_COLUMN].idxmin()]
    return str(row["month"]), float(row[GAP_COLUMN])


TABLE3_METRIC_MAP: dict[str, str] = {
    "r_squared": "r2",
    "adj_r_squared": "adj_r2",
    "rmse": "rmse",
    "aic": "aic",
    "n": "n",
}


def comparison_to_table3(
    comparison: Any,
    mean_gaps: Mapping[str, float] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Change a `ModelComparison` into the input of `table3_model_comparison`.

    `ModelComparison.table()` names its metrics `r_squared` and `adj_r_squared`.
    `table_generator` asks for `r2` and `adj_r2`. This function maps the names.
    It does not format a number. `table_generator` formats each number, so the
    report and the dashboard always show the same value.

    A model with `available=False` goes into the second result with its cause.
    `table3_model_comparison` then prints a column of "n/a" with a footnote.
    """
    metrics: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}
    models = getattr(comparison, "models", None)
    if not isinstance(models, Mapping):
        raise TypeError("comparison must have a 'models' mapping")

    for name, fit in models.items():
        if not getattr(fit, "available", False):
            reason = getattr(fit, "unavailable_reason", None)
            missing = tuple(getattr(fit, "missing_features", ()) or ())
            if not reason:
                reason = (
                    "the model is not estimated, because these columns are absent "
                    f"from the panel: {', '.join(missing) or 'unknown'}"
                )
            unavailable[str(name)] = str(reason)
            continue
        row: dict[str, Any] = {}
        for source, target in TABLE3_METRIC_MAP.items():
            row[target] = getattr(fit, source, None)
        row["n"] = getattr(fit, "n", None)
        if mean_gaps and name in mean_gaps:
            row["mean_gap"] = mean_gaps[name]
        metrics[str(name)] = row
    return metrics, unavailable


# The dashboard shows the fit statistics only. It does not show the mean
# out-of-sample gap of each model: it computes that gap for the baseline alone.
# An "n/a" cell in that row could read as "this model has no gap", which is not
# true. The report gives the gap of each model.
TABLE3_DASHBOARD_METRICS: tuple[str, ...] = ("r2", "adj_r2", "rmse", "aic", "n")


def model_comparison_markdown(
    comparison: Any,
    mean_gaps: Mapping[str, float] | None = None,
    metrics: Sequence[str] | None = TABLE3_DASHBOARD_METRICS,
) -> str:
    """Render table 3 with `table_generator`, as Markdown.

    The dashboard does not format these numbers. `table_generator` owns every
    rounding rule, so the dashboard and the report agree.
    """
    rows, unavailable = comparison_to_table3(comparison, mean_gaps=mean_gaps)
    order = [str(name) for name in getattr(comparison, "models", {})]
    table = table3_model_comparison(
        rows,
        unavailable=unavailable,
        model_order=order or None,
        metrics=metrics,
    )
    return to_markdown(table)


def scree_frame(basis: Any) -> pd.DataFrame:
    """Make the frame for panel 4 from an `LsaBasis`.

    The dashboard reads the spectrum from `svd_spectrum.parquet`, which
    `KSelection.spectrum_frame()` writes. This function gives the same frame
    from a basis object in memory, for a caller that holds one. Panel 4 must
    not invent a spectrum, so both paths need real values.
    """
    values = pd.Series(list(getattr(basis, "singular_values", [])), dtype="float64")
    if values.empty:
        raise ValueError("the basis has no singular values")
    energy = getattr(basis, "energy_share", None)
    cumulative = getattr(basis, "cumulative_energy", None)
    frame = pd.DataFrame(
        {
            "component": range(1, len(values) + 1),
            "singular_value": values.to_numpy(),
        }
    )
    if energy is not None and len(list(energy)) == len(values):
        frame["energy_share"] = list(energy)
    if cumulative is not None and len(list(cumulative)) == len(values):
        frame["cumulative_energy"] = list(cumulative)
    return frame


ABOVE_FLOOR_LABEL = "Above the permutation floor"
BELOW_FLOOR_LABEL = "Below the permutation floor"


def prepare_scree(spectrum: pd.DataFrame) -> pd.DataFrame:
    """Make the frame for panel 4 from the spectrum file.

    The function copies the input. A cached frame must not change in place.

    `band` gives the two classes of the chart: the components above the
    permutation floor, and the components below it. The chart colors these two
    classes, so the reader sees the decision of the floor and not a ramp.
    """
    missing = [name for name in SPECTRUM_REQUIRED_COLUMNS if name not in spectrum.columns]
    if missing:
        raise KeyError(f"the scree frame needs these columns: {missing}")
    keep = list(SPECTRUM_REQUIRED_COLUMNS) + [
        name for name in SPECTRUM_OPTIONAL_COLUMNS if name in spectrum.columns
    ]
    frame = spectrum.loc[:, keep].copy()
    frame["component"] = frame["component"].astype(int)
    if "above_noise_floor" in frame.columns:
        frame["above_noise_floor"] = frame["above_noise_floor"].astype(bool)
        frame["band"] = frame["above_noise_floor"].map(
            {True: ABOVE_FLOOR_LABEL, False: BELOW_FLOOR_LABEL}
        )
    return frame.sort_values("component").reset_index(drop=True)


@dataclass(frozen=True)
class ScreeReading:
    """The three diagnostics of the k selection, and what they say together.

    The project reports the three diagnostics together, because each one can
    fail alone. Refer to BRAINSTORMING.md paragraph 3.3.
    """

    k_max: int
    k_noise: int | None = None
    k_parallel: int | None = None
    cumulative_energy: float | None = None
    noise_floor: float | None = None

    @property
    def readings_agree(self) -> bool:
        """True when the two noise-floor methods keep the same count."""
        if self.k_noise is None or self.k_parallel is None:
            return False
        return self.k_noise == self.k_parallel

    def note(self) -> str:
        """State what the spectrum shows, in short sentences.

        The two methods do not agree on this corpus. The note says this
        plainly. A caption that gave one count would show an elbow that the
        data does not have.
        """
        lines: list[str] = []
        if self.k_noise is not None and self.k_parallel is not None:
            if self.readings_agree:
                lines.append(
                    f"The two noise-floor methods agree. Both keep {self.k_noise} "
                    "components."
                )
            else:
                lines.append(
                    "The two noise-floor methods do not agree. The permutation "
                    f"floor keeps {self.k_noise} components. The component-wise "
                    f"parallel analysis keeps {self.k_parallel} components."
                )
        if self.cumulative_energy is not None:
            lines.append(
                f"The top {self.k_max} components hold "
                f"{self.cumulative_energy:.1%} of the energy."
            )
        lines.append(
            "A corpus with few strong themes gives a high energy share and a "
            "sharp step in the bars. This corpus gives neither. The theme "
            "co-occurrence is diffuse, so the spectrum has no clear elbow. Do "
            "not read a step in these bars as the count of topics."
        )
        return " ".join(lines)


def scree_reading(spectrum: pd.DataFrame) -> ScreeReading:
    """Read the three diagnostics from the spectrum frame.

    The singular values are in descending order, so a count of the components
    above a floor is also the index of the last component above that floor.
    """
    if "singular_value" not in spectrum.columns:
        raise KeyError("the spectrum needs the column 'singular_value'")
    k_max = int(len(spectrum))
    k_noise = (
        int(spectrum["above_noise_floor"].astype(bool).sum())
        if "above_noise_floor" in spectrum.columns
        else None
    )
    k_parallel = (
        int(spectrum["above_permuted_component"].astype(bool).sum())
        if "above_permuted_component" in spectrum.columns
        else None
    )
    energy = (
        float(spectrum["cumulative_energy"].iloc[-1])
        if "cumulative_energy" in spectrum.columns
        else None
    )
    floor = (
        float(spectrum["permuted_max"].max())
        if "permuted_max" in spectrum.columns
        else None
    )
    return ScreeReading(
        k_max=k_max,
        k_noise=k_noise,
        k_parallel=k_parallel,
        cumulative_energy=energy,
        noise_floor=floor,
    )


def prepare_topic_terms(topics: pd.DataFrame, n_terms: int = 6) -> pd.DataFrame:
    """Make the topic table of panel 4: one row for each topic.

    The table holds the strongest terms of each topic in rank order. A table is
    the correct form here: the terms are text, and more than seven classes
    carry meaning.
    """
    missing = [name for name in TOPICS_REQUIRED_COLUMNS if name not in topics.columns]
    if missing:
        raise KeyError(f"the topic table needs these columns: {missing}")
    frame = topics.loc[topics["rank"] <= int(n_terms)].copy()
    frame = frame.sort_values(["topic", "rank"])
    rows: list[dict[str, Any]] = []
    for topic, part in frame.groupby("topic", sort=True):
        row: dict[str, Any] = {
            "topic": f"Topic {int(topic) + 1}",
            "terms": ", ".join(str(term) for term in part["term"]),
        }
        if "energy_share" in part.columns:
            row["energy_share"] = float(part["energy_share"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_topic_shares(monthly: pd.DataFrame) -> pd.DataFrame:
    """Make the heatmap frame of panel 4: one column for each topic.

    The index becomes a date index, because the x axis of the heatmap is time.
    The function copies the input.
    """
    columns = [name for name in monthly.columns if str(name).startswith("topic_")]
    if not columns:
        raise KeyError("the monthly file has no column with the prefix 'topic_'")
    columns.sort(key=lambda name: int(str(name).split("_")[1]))
    frame = monthly.loc[:, columns].copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame


def prepare_media_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Make the frame for panel 5: the media series with a date column.

    The function keeps only the months with a media value. The corpus starts in
    2017-01, and the panel starts in 2010-01. A month before the corpus is not
    a month with a tone of zero.
    """
    if MEDIA_TONE_COLUMN not in panel.columns:
        raise KeyError(f"panel 5 needs the column {MEDIA_TONE_COLUMN!r}")
    keep = [name for name in MEDIA_COLUMNS if name in panel.columns]
    frame = panel.loc[:, keep].copy()
    frame = frame.loc[frame[MEDIA_TONE_COLUMN].notna()]
    frame = frame.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    return frame


def granger_frame(
    panel: pd.DataFrame,
    gaps: pd.DataFrame,
    media_column: str = MEDIA_TONE_COLUMN,
    gap_column: str = GAP_COLUMN,
) -> pd.DataFrame:
    """Join the media series and the gap on the months that hold both.

    **The test uses the long window.** The out-of-sample gap has about twenty
    months, and `granger_media_gap` refuses inference at that size. The gap of
    an in-sample month is a residual of the fitted model, and the long window
    gives about 115 months. The function states this choice in one place, so panel 6 and the
    report use the same rows.
    """
    for name, frame in ((media_column, panel), (gap_column, gaps)):
        if name not in frame.columns:
            raise KeyError(f"panel 6 needs the column {name!r}")
    joined = pd.DataFrame(
        {
            media_column: panel[media_column],
            gap_column: gaps[gap_column],
        }
    )
    return joined.dropna()


def granger_records(report: Any) -> list[dict[str, Any]]:
    """Change a `GrangerReport` into the records of `table4_granger_and_iv`.

    Both directions always go into the result. Hypothesis B fails if coverage
    follows sentiment, so a one-direction table would hide the test.
    """
    records: list[dict[str, Any]] = []
    for attribute in ("forward", "reverse"):
        test = getattr(report, attribute, None)
        if test is None:
            continue
        records.append(
            {
                "direction": test.direction,
                "lags": test.lag_order,
                "f_stat": test.f_statistic,
                "p_value": test.p_value,
                "n": getattr(report, "n_obs_used", None),
            }
        )
    return records


# The dashboard does not run the Rigobon-Sack estimator. The reason states
# this. A reader must know where a number came from, and the report holds that
# result.
IV_NOT_RUN = (
    "the dashboard does not run the Rigobon-Sack estimator; the report gives "
    "its result"
)


def granger_markdown(report: Any, iv_reason: str = IV_NOT_RUN) -> str:
    """Render table 4 with `table_generator`, as Markdown.

    `table_generator` owns every rounding rule, so the dashboard and the report
    give the same numbers. Both directions always stay in the table.
    """
    table = table4_granger_and_iv(
        granger_records(report) or None,
        iv=None,
        iv_unavailable_reason=iv_reason,
        granger_unavailable_reason="the test did not run",
    )
    return to_markdown(table)


# ---------------------------------------------------------------------------
# The framing measure and the factual-calibration measure
# ---------------------------------------------------------------------------


def present_columns(
    columns: Iterable[str], accepted: Sequence[str] = FRAMING_ALTERNATIVES
) -> tuple[str, ...]:
    """Give the accepted names that are present, in the order of `accepted`.

    The function holds the name rule in one place. `panel_availability` uses
    the same group of names, so the panel and the chart can never disagree
    about which measure exists.
    """
    present = {str(name) for name in columns}
    return tuple(name for name in accepted if name in present)


# The three framing contrasts of the brief, with the name that each contrast
# can carry. The order is the order of the palette, so a contrast always keeps
# its color.
FRAMING_CONTRASTS: tuple[tuple[str, ...], ...] = (
    ("framing_level_vs_rate", "level_vs_rate"),
    ("framing_household_vs_macro", "household_vs_macro"),
    ("framing_current_vs_forecast", "current_vs_forecast"),
)


@dataclass(frozen=True)
class FramingAgreement:
    """How well the two text sources of the framing measure agree.

    The two sources are the headline of an article and the slug of its URL.
    Each source holds different words, so a high correlation is evidence that
    the measure holds a signal of the coverage and not a property of one text
    field. `n_months` gives the count of months that hold both values, because
    a correlation of few months is not usable.
    """

    columns: tuple[str, ...]
    n_months: int
    correlation: float

    def note(self) -> str:
        return (
            "The framing measure comes from two text sources: "
            f"{self.columns[0]} and {self.columns[1]}. The two agree at "
            f"r = {self.correlation:.3f} over {self.n_months} months that hold "
            "both values. The two sources hold different words, so the "
            "agreement is a validity check of the measure. It is not a "
            "measure of accuracy: both sources can emphasise the same frame "
            "and both can be wrong about an article."
        )


def framing_agreement(
    frame: pd.DataFrame, columns: Sequence[str] = FRAMING_SOURCE_COLUMNS
) -> FramingAgreement | None:
    """Give the agreement of the two text sources, or None when it is absent.

    The function returns None when the frame holds one source only, or when
    fewer than three months hold both values. None means "the dashboard says
    nothing", which is the honest answer. It never returns a correlation of
    zero in that state: a zero would read as "the two sources disagree".
    """
    present = tuple(name for name in columns if name in frame.columns)
    if len(present) < 2:
        return None
    both = frame.loc[:, list(present)].dropna()
    if len(both) < 3:
        return None
    value = float(both[present[0]].corr(both[present[1]]))
    if pd.isna(value):
        return None
    return FramingAgreement(
        columns=present, n_months=int(len(both)), correlation=value
    )


def framing_series(columns: Iterable[str]) -> tuple[str, ...]:
    """Select the framing columns that the chart draws. At most three.

    **The chart draws the three contrasts when the file holds them.** The
    column `framing` can be a composite of the contrasts or an alias of one of
    them, so a chart with `framing` beside the contrasts can draw the same
    numbers two times. The three contrasts answer the question of the brief
    directly, so they come first.

    The result never holds more than three names. The palette gives three
    series colors in a fixed order, and a fourth series would repeat a color.
    A repeated color makes two series look like one series.
    """
    present = {str(name) for name in columns}
    contrasts = tuple(
        next(name for name in group if name in present)
        for group in FRAMING_CONTRASTS
        if any(name in present for name in group)
    )
    if contrasts:
        return contrasts
    return present_columns(columns, (FRAMING_COLUMN,))


def prepare_measure_series(
    panel: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Make a date frame for one group of media measures.

    The function copies the input, and it keeps only the months that hold a
    value in one column or more. A month before the corpus starts is not a
    month with a measure of zero.
    """
    keep = [name for name in columns if name in panel.columns]
    if not keep:
        raise KeyError(f"the panel holds none of these columns: {list(columns)}")
    frame = panel.loc[:, keep].copy()
    frame = frame.loc[frame.notna().any(axis=1)]
    frame = frame.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    return frame


# ---------------------------------------------------------------------------
# Corpus composition, coverage, missingness and model performance
# ---------------------------------------------------------------------------


OTHER_DOMAIN_LABEL = "Other domains"


def url_domain(url: str) -> str:
    """Give the domain of one URL, in lower case and without `www.`.

    The function uses the text only. It makes no request. A URL without a
    domain gives an empty string, and `domain_composition` counts it as
    unknown. A damaged URL is a fact about the corpus.
    """
    text = str(url).strip().lower()
    if "//" in text:
        text = text.split("//", 1)[1]
    text = text.split("/", 1)[0]
    text = text.split("?", 1)[0].split("@")[-1].split(":", 1)[0]
    if text.startswith("www."):
        text = text[4:]
    return text


# A host of three labels or more usually belongs to a larger publisher: a
# local radio station of one network, or one section of one site. These two-part
# endings keep a country domain together, for example `bbc.co.uk`.
COMPOUND_SUFFIXES: tuple[str, ...] = ("co", "com", "org", "net", "gov", "ac", "edu")


def registrable_domain(host: str) -> str:
    """Fold a host to the domain that a publisher registers.

    **A network of local sites would otherwise hide its own size.** One
    publisher in this corpus has 267 station hosts, and each host holds few
    headlines. A count by host puts that publisher at the bottom of the table.
    A count by registrable domain puts it in the first rows, which is the fact
    that a reader needs.

    The rule uses the text only. It is not a public-suffix list: a host with a
    two-part country ending keeps three labels, and every other host keeps two.
    A longer suffix list would give a different answer for a small group of
    hosts, and panel 10 prints the rule for that reason.
    """
    parts = [part for part in str(host).strip().lower().split(".") if part]
    if len(parts) <= 2:
        return ".".join(parts)
    if len(parts[-1]) == 2 and parts[-2] in COMPOUND_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def domain_composition(
    headlines: pd.DataFrame,
    url_column: str = "url",
    top_n: int = 12,
    fold_hosts: bool = True,
) -> pd.DataFrame:
    """Count the headlines of each domain, with the small domains in one row.

    **The composition is a fact about the corpus, not a result of the study.**
    One domain can hold a large share of a GDELT query, and a reader must see
    that before a reader reads a tone series. The rows after `top_n` go into
    one row, because 400 bars carry no information. The row keeps the count of
    the domains that it holds.
    """
    hosts, domains = _corpus_domains(headlines, url_column, fold_hosts)
    counts = domains.value_counts()
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame(
            columns=["domain", "n_headlines", "share", "n_domains", "n_hosts"]
        )
    hosts_by_domain = pd.DataFrame({"host": hosts, "domain": domains})
    host_counts = hosts_by_domain.groupby("domain")["host"].nunique()
    head = counts.iloc[: int(top_n)]
    rows = [
        {
            "domain": str(name),
            "n_headlines": int(value),
            "share": float(value) / total,
            "n_domains": 1,
            "n_hosts": int(host_counts.get(name, 1)),
        }
        for name, value in head.items()
    ]
    tail = counts.iloc[int(top_n) :]
    if not tail.empty:
        rows.append(
            {
                "domain": OTHER_DOMAIN_LABEL,
                "n_headlines": int(tail.sum()),
                "share": float(tail.sum()) / total,
                "n_domains": int(len(tail)),
                "n_hosts": int(host_counts.loc[tail.index].sum()),
            }
        )
    return pd.DataFrame(rows)


def _corpus_domains(
    headlines: pd.DataFrame, url_column: str, fold_hosts: bool
) -> tuple[pd.Series, pd.Series]:
    """Give the host and the domain of each row. One rule, one place."""
    if url_column not in headlines.columns:
        raise KeyError(f"the headline frame needs the column {url_column!r}")
    hosts = headlines[url_column].map(url_domain).replace("", "unknown")
    domains = hosts.map(registrable_domain) if fold_hosts else hosts
    return hosts, domains.replace("", "unknown")


# A domain with this share or more of the corpus is a concentration. The value
# is a selection, and panel 10 prints it with the measured share, so a reader
# can apply a different limit. The limit exists because the word "dominates"
# needs a number behind it.
CONCENTRATION_LIMIT = 0.10


@dataclass(frozen=True)
class CompositionReport:
    """What the composition of the corpus says about the corpus.

    `top_share` is the share of the largest domain. The note states the
    measured share and compares it with `CONCENTRATION_LIMIT`. **The note has
    no fixed verdict**: an earlier text said that one domain gives the corpus
    its tone, and that sentence would stay on the screen after a new pull made
    it false.
    """

    n_headlines: int
    n_domains: int
    n_hosts: int
    top_domain: str
    top_share: float
    top_hosts: int
    top5_share: float

    @property
    def concentrated(self) -> bool:
        return self.top_share >= CONCENTRATION_LIMIT

    def note(self) -> str:
        lines = [
            f"The corpus holds {self.n_headlines:,} headlines from "
            f"{self.n_domains:,} domains and {self.n_hosts:,} hosts. "
            f"{self.top_domain} is the largest domain with "
            f"{self.top_share:.1%} of the headlines, and the five largest "
            f"domains give {self.top5_share:.1%} together."
        ]
        if self.concentrated:
            lines.append(
                f"The share of {self.top_domain} is at or above the limit of "
                f"{CONCENTRATION_LIMIT:.0%}, so this corpus measures that "
                "publisher more than it measures the national press. Each "
                "claim about coverage has this limit."
            )
        else:
            lines.append(
                f"No domain reaches the limit of {CONCENTRATION_LIMIT:.0%}, so "
                "no single publisher gives this corpus its tone. The corpus is "
                "thin in a different way: it holds few headlines from each of "
                "many domains, so a claim about one publisher is not possible."
            )
        if self.top_hosts > 1:
            lines.append(
                f"{self.top_domain} uses {self.top_hosts} hosts. The table "
                "counts the hosts of one publisher together. A count by host "
                "would put a network of local sites at the end of the table."
            )
        return " ".join(lines)


def composition_report(
    headlines: pd.DataFrame, url_column: str = "url", fold_hosts: bool = True
) -> CompositionReport:
    """Read the composition facts from the headline frame."""
    hosts, domains = _corpus_domains(headlines, url_column, fold_hosts)
    counts = domains.value_counts()
    if counts.empty:
        raise ValueError("the headline frame holds no URL")
    total = int(counts.sum())
    top = str(counts.index[0])
    return CompositionReport(
        n_headlines=total,
        n_domains=int(len(counts)),
        n_hosts=int(hosts.nunique()),
        top_domain=top,
        top_share=float(counts.iloc[0]) / total,
        top_hosts=int(hosts.loc[domains == top].nunique()),
        top5_share=float(counts.iloc[:5].sum()) / total,
    )


# A month with fewer articles than this share of the median month is a thin
# month. The rule is a share of the median and not a fixed count, because the
# count of articles of the corpus depends on the query and on the year. The
# value of one half is a selection. Panel 11 prints the rule and the limit, so
# a reader can apply a different rule to the table.
THIN_COVERAGE_SHARE = 0.5


@dataclass(frozen=True)
class CoverageReport:
    """The thin months of the corpus, and the rule that selected them."""

    n_months: int
    median_articles: float
    threshold: float
    thin_months: tuple[str, ...]
    share_thin: float
    min_articles: int
    min_month: str

    def note(self) -> str:
        return (
            f"The rule is {THIN_COVERAGE_SHARE:.0%} of the median month. The "
            f"median month holds {self.median_articles:,.0f} articles, so the "
            f"limit is {self.threshold:,.0f} articles. "
            f"{len(self.thin_months)} of {self.n_months} months "
            f"({self.share_thin:.1%}) are below the limit. The thinnest month "
            f"is {self.min_month} with {self.min_articles:,} articles. A tone "
            "value of a thin month holds more noise than a tone value of a "
            "full month."
        )


def coverage_report(
    media: pd.DataFrame,
    count_column: str = "n_articles",
    share: float = THIN_COVERAGE_SHARE,
) -> CoverageReport:
    """Report the months with thin coverage, from the count of articles."""
    if count_column not in media.columns:
        raise KeyError(f"the media frame needs the column {count_column!r}")
    counts = media[count_column].dropna().astype(float)
    if counts.empty:
        raise ValueError("the media frame holds no count of articles")
    median = float(counts.median())
    threshold = median * float(share)
    thin = counts.loc[counts < threshold]
    index = pd.to_datetime(pd.Series(counts.index))
    labels = {
        position: stamp.strftime("%Y-%m")
        for position, stamp in zip(counts.index, index, strict=False)
    }
    smallest = counts.idxmin()
    return CoverageReport(
        n_months=int(len(counts)),
        median_articles=median,
        threshold=threshold,
        thin_months=tuple(labels[position] for position in thin.index),
        share_thin=float(len(thin)) / float(len(counts)),
        min_articles=int(counts.min()),
        min_month=labels[smallest],
    )


def prepare_coverage(
    media: pd.DataFrame, count_column: str = "n_articles"
) -> pd.DataFrame:
    """Make the frame of panel 11: the count of articles of each month."""
    if count_column not in media.columns:
        raise KeyError(f"the media frame needs the column {count_column!r}")
    frame = media.loc[:, [count_column]].dropna().copy()
    frame = frame.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    frame[count_column] = frame[count_column].astype(float)
    return frame


def missing_month_table(
    panel: pd.DataFrame,
    gaps: pd.DataFrame,
    features: Sequence[str] = BASELINE_FEATURES,
    holdout_start: str = HOLDOUT_START,
) -> pd.DataFrame:
    """Name each out-of-sample month without a gap value, and the cause.

    **A month without a gap value is not a month with a gap of zero.** The
    table gives the name of the input that is absent in that month, so a reader
    sees the cause. The rule comes from the model: the model cannot predict a
    month without every regressor of the model.
    """
    start = pd.Timestamp(holdout_start)
    months = panel.loc[panel.index >= start]
    rows = []
    for stamp in months.index:
        absent = [
            str(name)
            for name in features
            if name not in panel.columns or pd.isna(panel.loc[stamp, name])
        ]
        has_gap = (
            GAP_COLUMN in gaps.columns
            and stamp in gaps.index
            and bool(pd.notna(gaps.loc[stamp, GAP_COLUMN]))
        )
        if has_gap and not absent:
            continue
        rows.append(
            {
                "month": pd.Timestamp(stamp).strftime("%Y-%m"),
                "gap value": "present" if has_gap else "absent",
                "absent inputs": ", ".join(absent) if absent else "none",
            }
        )
    return pd.DataFrame(rows, columns=["month", "gap value", "absent inputs"])


@dataclass(frozen=True)
class ModelPerformance:
    """The performance indicators that the brief asks the dashboard to show.

    The in-sample RMSE and the out-of-sample mean absolute gap are two
    different quantities. The first says how well the model fitted the months
    that it saw. The second says how far the model was in the months that it
    did not see. The dashboard gives both, because the difference between them
    is the result of this project.
    """

    n_train: int
    rmse_in_sample: float
    n_tested: int
    mean_abs_gap: float
    months_inside_pi: int
    share_inside_pi: float

    def note(self) -> str:
        return (
            f"The model fitted {self.n_train} months with an RMSE of "
            f"{self.rmse_in_sample:.2f} index points. In the "
            f"{self.n_tested} out-of-sample months the mean absolute gap is "
            f"{self.mean_abs_gap:.2f} points, and "
            f"{self.months_inside_pi} of {self.n_tested} months "
            f"({self.share_inside_pi:.1%}) are inside the 95% prediction "
            "interval. A correct model holds approximately 95% of the months "
            "inside that interval."
        )


def model_performance(
    fit: Any,
    gaps: pd.DataFrame,
    holdout_start: str = HOLDOUT_START,
) -> ModelPerformance:
    """Read the performance indicators from the fit and the gap frame.

    `outside_pi` is a nullable boolean. A month without a test is not a month
    inside the interval, so the function counts only the months that hold a
    value.
    """
    for name in (GAP_COLUMN, "outside_pi"):
        if name not in gaps.columns:
            raise KeyError(f"the performance report needs the column {name!r}")
    holdout = gaps.loc[gaps.index >= pd.Timestamp(holdout_start)]
    tested = holdout.loc[holdout["outside_pi"].notna()]
    outside = tested["outside_pi"].astype(bool)
    n_tested = int(len(tested))
    inside = int((~outside).sum())
    gap_values = holdout[GAP_COLUMN].dropna().astype(float)
    return ModelPerformance(
        n_train=int(getattr(fit, "n_train", 0)),
        rmse_in_sample=float(getattr(fit, "rmse_in_sample", float("nan"))),
        n_tested=n_tested,
        mean_abs_gap=float(gap_values.abs().mean()) if not gap_values.empty else float("nan"),
        months_inside_pi=inside,
        share_inside_pi=float(inside) / n_tested if n_tested else float("nan"),
    )


def performance_tiles(performance: ModelPerformance) -> list[StatTile]:
    """Make the performance numbers of panel 12 as tiles.

    The out-of-sample mean absolute gap is the lead figure, because it is the
    number that the study is about. The in-sample RMSE stands beside it: the
    difference between the two numbers is the result.
    """
    return [
        StatTile(
            "Mean absolute gap, out of sample",
            f"{performance.mean_abs_gap:.2f}",
            f"Index points over {performance.n_tested} months",
            lead=True,
        ),
        StatTile(
            "RMSE, in sample",
            f"{performance.rmse_in_sample:.2f}",
            f"Index points over {performance.n_train} training months",
        ),
        StatTile(
            "Inside the 95% interval",
            f"{performance.months_inside_pi} of {performance.n_tested}",
            f"{performance.share_inside_pi:.1%} of the months with a test",
        ),
    ]


def panel_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    """Count the months with a value and without a value, for each column.

    The table gives the first month and the last month of each column, because
    a column that starts late is a different state from a column with holes. A
    media column starts in 2017-01, and that is the start of the corpus. It is
    not a gap in the data.
    """
    rows = []
    index = pd.to_datetime(pd.Series(panel.index))
    for name in panel.columns:
        values = panel[name]
        present = values.notna().to_numpy()
        months = index.loc[present]
        rows.append(
            {
                "column": str(name),
                "months with a value": int(present.sum()),
                "months without a value": int((~present).sum()),
                "first month": months.min().strftime("%Y-%m") if len(months) else "none",
                "last month": months.max().strftime("%Y-%m") if len(months) else "none",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "months with a value",
            "months without a value",
            "first month",
            "last month",
        ],
    )


# The one honest statement of panel 13. The project holds the published
# Michigan index only. A split by income, age, geography or affiliation needs
# the microdata of the survey, and the project does not hold it.
DEMOGRAPHIC_NOTE = (
    "The brief asks for differences by income, age, geography and political "
    "affiliation where the data permit. **The data of this project do not "
    "permit it.** The project holds the published Michigan index, which is "
    "one number for each month for the whole country. A split needs the "
    "Michigan microdata: the response of each person, with the weight of that "
    "person and the demographic fields of that person. The Surveys of "
    "Consumers publish those files, and a user must obtain them from the "
    "University of Michigan or from ICPSR. The dashboard shows no split, "
    "because a split that comes from a model and not from the microdata is "
    "an invention."
)

DEMOGRAPHIC_PLAN = (
    "What a split would need: the Michigan microdata for 2010 to 2026; the "
    "survey weights of each year; the documented income, age, region and "
    "party fields; and the same gap method applied inside each group, with "
    "the smaller sample of each group in the prediction interval. Michigan "
    "documents a difference of economic perception by party, so affiliation "
    "is a moderator and also a confounder. The report gives this limit."
)


# ---------------------------------------------------------------------------
# Insight titles — a title states the finding, not the axes
# ---------------------------------------------------------------------------


def timeline_title(summary: pd.Series) -> str:
    """State the panel 1 finding."""
    outside = int(summary.get("months_outside_pi", 0))
    tested = int(summary.get("n_tested", 0))
    return (
        f"Observed sentiment fell below the model's 95% prediction interval in "
        f"{outside} of {tested} out-of-sample months"
    )


def gap_title(summary: pd.Series) -> str:
    """State the panel 2 finding."""
    mean_gap = float(summary.get("mean_gap", float("nan")))
    worst = summary.get("worst_month", "n/a")
    worst_gap = float(summary.get("worst_gap", float("nan")))
    return (
        f"Every out-of-sample month has a negative gap; the mean is {mean_gap:.2f} "
        f"and {worst} is the worst at {worst_gap:.2f}"
    )


def scree_title(reading: ScreeReading) -> str:
    """State the panel 4 finding.

    The finding is the absence of an elbow. A title that said "the spectrum of
    the theme matrix" would hide it.
    """
    if reading.cumulative_energy is None:
        return f"The spectrum of the theme matrix, {reading.k_max} components"
    return (
        f"The spectrum has no clear elbow: the top {reading.k_max} components "
        f"hold only {reading.cumulative_energy:.1%} of the energy"
    )


def energy_title(reading: ScreeReading) -> str:
    """State the panel 4 energy finding."""
    if reading.cumulative_energy is None:
        return "The cumulative energy of the spectrum"
    return (
        f"The energy climbs in small steps and reaches "
        f"{reading.cumulative_energy:.1%} at component {reading.k_max}"
    )


def media_tone_title(frame: pd.DataFrame, column: str = MEDIA_TONE_COLUMN) -> str:
    """State the panel 5 finding: the series moves, and the gap does not follow."""
    if column not in frame.columns or frame[column].dropna().empty:
        raise KeyError(f"the media frame needs values in {column!r}")
    values = frame[column].astype(float)
    return (
        f"{column} moves from {values.min():.3f} to {values.max():.3f} over "
        f"{int(values.notna().sum())} months"
    )


def granger_title(report: Any) -> str:
    """State the panel 6 finding from the report, not from a fixed text."""
    forward = getattr(report, "forward", None)
    reverse = getattr(report, "reverse", None)
    rejects = [
        test for test in (forward, reverse) if test is not None and test.rejects()
    ]
    if not rejects:
        return (
            "Granger finds no predictive precedence in either direction, so "
            "Hypothesis B fails its own falsification condition"
        )
    directions = "; ".join(test.direction for test in rejects)
    return f"Granger rejects the null in this direction: {directions}"


def granger_verdict(report: Any) -> str:
    """State what the Granger result does to Hypothesis B, with its numbers.

    **The numbers come from the report.** Paragraph 2.2 of BRAINSTORMING.md
    fixed the falsification condition before the corpus existed: if the tone
    does not precede the gap, Hypothesis B fails. This function states the
    result of that condition, and it prints the p-value that the test gave. A
    fixed number in a sentence cannot follow a new estimation window.
    """
    tests = [
        test
        for test in (getattr(report, "forward", None), getattr(report, "reverse", None))
        if test is not None
    ]
    if not tests:
        return (
            "The Granger test gave no result, so this panel states nothing "
            "about Hypothesis B."
        )
    values = "; ".join(
        f"{test.direction}: F = {test.f_statistic:.3f}, p = {test.p_value:.3f}"
        for test in tests
    )
    rejects = [test for test in tests if test.rejects()]
    if rejects:
        return (
            "**One direction rejects its null.** " + values + ". A rejection is "
            "predictive precedence. It is not a cause: a common cause can move "
            "both series together. Paragraph 2.2 of BRAINSTORMING.md fixed "
            "this condition before the corpus existed."
        )
    return (
        "**Hypothesis B fails its own falsification condition, and that is a "
        "result.** " + values + ". The tone of coverage gives no information "
        "about the gap of a later month, and the gap gives no information "
        "about the tone. Paragraph 2.2 of BRAINSTORMING.md fixed this "
        "condition before the corpus existed. The model comparison of panel 3 "
        "gives the p-value of each media regressor, and the report gives the "
        "Rigobon-Sack result."
    )


def framing_title(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """State the panel 7 finding from the data.

    The title gives the mean of each contrast, because the sign of the mean is
    the finding: a positive mean is the price-level frame, and a negative mean
    is the inflation-rate frame.
    """
    present = [name for name in columns if name in frame.columns]
    if not present:
        raise KeyError(f"the framing frame holds none of these columns: {list(columns)}")
    parts = []
    for name in present:
        values = frame[name].dropna().astype(float)
        if values.empty:
            continue
        parts.append(f"{name} has a mean of {values.mean():.3f}")
    if not parts:
        raise KeyError("the framing frame holds no value")
    months = int(frame[present].notna().any(axis=1).sum())
    return f"Over {months} months, " + "; ".join(parts)


def calibration_title(frame: pd.DataFrame, column: str) -> str:
    """State the panel 8 finding from the data."""
    if column not in frame.columns or frame[column].dropna().empty:
        raise KeyError(f"the calibration frame needs values in {column!r}")
    values = frame[column].dropna().astype(float)
    return (
        f"{column} moves from {values.min():.3f} to {values.max():.3f} over "
        f"{int(values.notna().sum())} months, with a mean of {values.mean():.3f}"
    )


def composition_title(report: CompositionReport) -> str:
    """State the panel 10 finding: one domain holds a large share."""
    return (
        f"{report.top_domain} gives {report.top_share:.1%} of the "
        f"{report.n_headlines:,} headlines, from {report.n_domains:,} domains"
    )


def coverage_title(report: CoverageReport) -> str:
    """State the panel 11 finding: how many months are thin."""
    return (
        f"{len(report.thin_months)} of {report.n_months} months hold fewer "
        f"than {report.threshold:,.0f} articles, which is "
        f"{THIN_COVERAGE_SHARE:.0%} of the median month"
    )


def quality_title(performance: ModelPerformance, missing: pd.DataFrame) -> str:
    """State the panel 12 finding: the model misses out of sample."""
    return (
        f"The in-sample RMSE is {performance.rmse_in_sample:.2f} points, the "
        f"out-of-sample mean absolute gap is {performance.mean_abs_gap:.2f} "
        f"points, and {performance.months_inside_pi} of "
        f"{performance.n_tested} months are inside the 95% interval; "
        f"{len(missing)} months hold no gap value"
    )


# ---------------------------------------------------------------------------
# Figures — pure Plotly builders
# ---------------------------------------------------------------------------


def _style_axes(figure: go.Figure, palette: Palette, x_title: str, y_title: str) -> go.Figure:
    """Apply the recessive chrome that every figure shares.

    The gridlines are solid hairlines one step off the surface. A dashed
    gridline reads as a threshold, and these lines carry no data.
    """
    figure.update_layout(
        paper_bgcolor=palette.surface,
        plot_bgcolor=palette.surface,
        font=dict(family=FONT_FAMILY, color=palette.ink_secondary, size=13),
        margin=dict(l=72, r=28, t=86, b=64),
        hoverlabel=dict(
            bgcolor=palette.surface,
            bordercolor=palette.axis,
            font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=13),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette.ink_secondary),
        ),
    )
    axis_common = dict(
        showgrid=True,
        gridcolor=palette.grid,
        gridwidth=1,
        griddash="solid",
        zeroline=False,
        linecolor=palette.axis,
        linewidth=1,
        tickfont=dict(color=palette.ink_muted, size=12),
        title=dict(font=dict(color=palette.ink_secondary, size=13)),
    )
    figure.update_xaxes(title_text=x_title, **axis_common)
    figure.update_yaxes(title_text=y_title, **axis_common)
    return figure


def _title_layout(figure: go.Figure, palette: Palette, title: str, subtitle: str = "") -> go.Figure:
    text = title if not subtitle else f"{title}<br><sub>{subtitle}</sub>"
    figure.update_layout(
        title=dict(
            text=text,
            font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=17),
            x=0,
            xanchor="left",
            y=0.97,
            yanchor="top",
        )
    )
    return figure


def add_event_marks(
    figure: go.Figure,
    palette: Palette,
    events: Sequence[EconomicEvent] = (),
    x_min: Any = None,
    x_max: Any = None,
    labels: bool = True,
) -> go.Figure:
    """Put a hairline on a time-series chart for each event.

    **The line is chrome, not a series.** It uses the muted ink and a dotted
    line, so it never reads as data. A solid line is the out-of-sample edge and
    a dashed line is a threshold, so the dot keeps the three meanings separate.

    An event outside the window of the chart gets no line: a line at the edge
    of a chart points at a month that the chart does not show. The label stands
    along the line, because a horizontal label over a narrow month collides
    with the next label.
    """
    if not events:
        return figure
    low = pd.Timestamp(x_min) if x_min is not None else None
    high = pd.Timestamp(x_max) if x_max is not None else None
    for event in events:
        stamp = event.timestamp
        if low is not None and stamp < low:
            continue
        if high is not None and stamp > high:
            continue
        annotation: dict[str, Any] = {}
        if labels:
            annotation = dict(
                annotation_text=event.label,
                annotation_position="top left",
                annotation_textangle=-90,
                annotation_font=dict(
                    family=FONT_FAMILY, color=palette.ink_muted, size=11
                ),
            )
        figure.add_vline(
            x=stamp,
            line=dict(color=palette.ink_muted, width=1, dash="dot"),
            layer="below",
            **annotation,
        )
    return figure


def figure_observed_vs_predicted(
    timeline: pd.DataFrame,
    palette: Palette,
    title: str,
    holdout_start: str = HOLDOUT_START,
    events: Sequence[EconomicEvent] = (),
) -> go.Figure:
    """Panel 1. One y axis. Two series. A 95% prediction interval band.

    The band is a prediction interval, not a confidence interval on the mean.
    The question is whether one month agrees with the model, so the band holds
    the residual variance.

    The figure marks the out-of-sample period with a gray wash and a hairline.
    The wash uses a chrome color, not a series color, because the period is not
    a series.
    """
    figure = go.Figure()
    if timeline.empty:
        raise ValueError("the timeline frame is empty")

    dates = timeline["date"]
    figure.add_trace(
        go.Scatter(
            x=pd.concat([dates, dates[::-1]]),
            y=pd.concat([timeline["pi_upper"], timeline["pi_lower"][::-1]]),
            fill="toself",
            fillcolor=palette.band,
            line=dict(width=0),
            name="95% prediction interval",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=timeline["predicted"],
            mode="lines",
            name="Predicted by the macro model",
            line=dict(color=palette.series_1, width=2, shape="linear"),
            customdata=timeline[["pi_lower", "pi_upper", "sample"]],
            hovertemplate=(
                "<b>%{y:.2f}</b> predicted<br>%{x|%Y-%m}<br>"
                "95%% interval %{customdata[0]:.2f} to %{customdata[1]:.2f}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=timeline["observed"],
            mode="lines",
            name="Observed sentiment",
            line=dict(color=palette.series_2, width=2),
            hovertemplate="<b>%{y:.2f}</b> observed<br>%{x|%Y-%m}<extra></extra>",
        )
    )

    start = pd.Timestamp(holdout_start)
    if (dates >= start).any():
        figure.add_vrect(
            x0=start,
            x1=dates.max(),
            fillcolor=palette.ink_muted,
            opacity=0.10,
            line_width=0,
            layer="below",
        )
        figure.add_vline(
            x=start,
            line=dict(color=palette.axis, width=1),
        )
        # The label sits to the left of the hairline. The out-of-sample period
        # is a narrow slice at the right edge, so a label to the right of the
        # line would go outside the plot.
        figure.add_annotation(
            x=start,
            y=1.0,
            yref="paper",
            xanchor="right",
            yanchor="top",
            xshift=-6,
            text="Out of sample →<br>the model never saw 2025 and later",
            showarrow=False,
            font=dict(family=FONT_FAMILY, color=palette.ink_secondary, size=12),
            bgcolor=palette.surface,
        )

    add_event_marks(figure, palette, events, x_min=dates.min(), x_max=dates.max())

    _style_axes(figure, palette, "Month", "Consumer sentiment index")
    _title_layout(
        figure,
        palette,
        title,
        "The band is a 95% prediction interval. It holds the residual variance.",
    )
    figure.update_layout(
        hovermode="x unified",
        showlegend=True,
    )
    figure.update_xaxes(showspikes=True, spikecolor=palette.axis, spikethickness=1, spikemode="across")
    return figure


def figure_gap_by_month(
    gap_months: pd.DataFrame,
    palette: Palette,
    title: str,
) -> go.Figure:
    """Panel 2. A diverging bar chart that is anchored at zero.

    The color scale runs blue to red through a gray midpoint. The domain is
    symmetric, so zero sits at the gray midpoint. A rainbow scale would give a
    hue to the midpoint, and the midpoint must read as "nothing".

    The chart labels the worst month directly. The table view holds every
    value, so no value is only in a tooltip.
    """
    if gap_months.empty:
        raise ValueError("the gap frame is empty")

    low, high = diverging_zero_domain(gap_months[GAP_COLUMN])
    figure = go.Figure(
        go.Bar(
            x=gap_months["month"],
            y=gap_months[GAP_COLUMN],
            marker=dict(
                color=gap_months[GAP_COLUMN],
                colorscale=diverging_colorscale(palette),
                cmin=low,
                cmid=0.0,
                cmax=high,
                cornerradius=4,
                colorbar=dict(
                    title=dict(
                        text="Gap",
                        font=dict(color=palette.ink_secondary, size=12),
                    ),
                    tickfont=dict(color=palette.ink_muted, size=11),
                    outlinewidth=0,
                    thickness=10,
                ),
            ),
            hovertemplate="<b>%{y:.2f}</b> gap<br>%{x}<extra></extra>",
            name="Sentiment Gap",
            showlegend=False,
        )
    )

    worst = worst_month(gap_months)
    if worst is not None:
        month, value = worst
        figure.add_annotation(
            x=month,
            y=value,
            text=f"{month}: {value:.2f}",
            showarrow=False,
            yshift=-18,
            font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=12),
        )

    _style_axes(figure, palette, "Month", "Observed minus predicted sentiment")
    _title_layout(
        figure,
        palette,
        title,
        "The scale is diverging and anchored at zero. Gray is zero. Red is a negative gap.",
    )
    figure.update_layout(bargap=0.35)
    figure.add_hline(y=0, line=dict(color=palette.axis, width=1))
    return figure


SCREE_SUBTITLE = (
    "The bars are the singular values, in descending order. The step line is "
    "the permuted maximum of each component. The dashed line is the permutation "
    "floor."
)


def figure_scree(scree: pd.DataFrame, palette: Palette, title: str) -> go.Figure:
    """Panel 4. The singular value spectrum against the noise floor.

    One y axis: the permuted values have the unit of a singular value, so the
    floor belongs on the same scale as the bars.

    **The chart colors a decision, not a magnitude.** The bars above the
    permutation floor get the accent color. The other bars get the muted ink.
    A ramp over 40 bars would color the bar length a second time, and it would
    also suggest an order that the floor does not make.

    The function builds one series when the frame holds no floor column. It
    never invents a floor.
    """
    if scree.empty:
        raise ValueError("the scree frame is empty")

    figure = go.Figure()
    has_floor = "above_noise_floor" in scree.columns
    if has_floor:
        groups = (
            (scree.loc[scree["above_noise_floor"]], ABOVE_FLOOR_LABEL, palette.series_1),
            (scree.loc[~scree["above_noise_floor"]], BELOW_FLOOR_LABEL, palette.ink_muted),
        )
        for part, label, color in groups:
            if part.empty:
                continue
            figure.add_trace(
                go.Bar(
                    x=part["component"],
                    y=part["singular_value"],
                    name=label,
                    marker=dict(color=color, cornerradius=4),
                    hovertemplate="<b>%{y:.2f}</b><br>component %{x}<extra></extra>",
                    showlegend=True,
                )
            )
    else:
        figure.add_trace(
            go.Bar(
                x=scree["component"],
                y=scree["singular_value"],
                marker=dict(color=palette.series_1, cornerradius=4),
                hovertemplate="<b>%{y:.4f}</b><br>component %{x}<extra></extra>",
                showlegend=False,
            )
        )

    if "permuted_max" in scree.columns:
        figure.add_trace(
            go.Scatter(
                x=scree["component"],
                y=scree["permuted_max"],
                mode="lines",
                line=dict(color=palette.series_2, width=2, shape="hv"),
                name="Permuted maximum of each component",
                hovertemplate="<b>%{y:.2f}</b> permuted<br>component %{x}<extra></extra>",
                showlegend=True,
            )
        )
        floor = float(scree["permuted_max"].max())
        # The dashed line is a threshold, not a gridline. The gridlines stay
        # solid, so the dash reads as "a limit" and not as "a grid".
        figure.add_hline(
            y=floor,
            line=dict(color=palette.series_2, width=1, dash="dash"),
            annotation_text=f"Permutation floor {floor:.1f}",
            annotation_position="top right",
            annotation_font=dict(family=FONT_FAMILY, color=palette.ink_secondary, size=12),
        )

    # One direct label: the first component. A label on each of 40 bars is
    # unreadable, and the table view holds every value.
    first = scree.iloc[0]
    figure.add_annotation(
        x=int(first["component"]),
        y=float(first["singular_value"]),
        text=f"{float(first['singular_value']):.1f}",
        showarrow=False,
        yshift=14,
        font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=12),
    )

    _style_axes(figure, palette, "Component", "Singular value")
    _title_layout(figure, palette, title, SCREE_SUBTITLE if has_floor else "The values are in descending order.")
    figure.update_layout(bargap=0.35, barmode="overlay")
    figure.update_xaxes(dtick=1 if len(scree) <= 12 else 5)
    return figure


def figure_cumulative_energy(
    scree: pd.DataFrame,
    palette: Palette,
    title: str,
    marks: Sequence[tuple[int, str]] = (),
) -> go.Figure:
    """Panel 4. The cumulative energy, on its own axis and its own chart.

    The energy is a share, and the singular value is not. Two units never share
    one axis, so this is a second chart and not a second axis on the scree.
    """
    if "cumulative_energy" not in scree.columns:
        raise KeyError("the energy chart needs the column 'cumulative_energy'")
    figure = go.Figure(
        go.Scatter(
            x=scree["component"],
            y=scree["cumulative_energy"],
            mode="lines",
            line=dict(color=palette.series_1, width=2),
            fill="tozeroy",
            fillcolor=palette.band,
            name="Cumulative energy",
            hovertemplate="<b>%{y:.1%}</b> of the energy<br>top %{x} components<extra></extra>",
            showlegend=False,
        )
    )
    last = scree.iloc[-1]
    figure.add_annotation(
        x=int(last["component"]),
        y=float(last["cumulative_energy"]),
        text=f"{float(last['cumulative_energy']):.1%} at k={int(last['component'])}",
        showarrow=False,
        xanchor="right",
        yshift=16,
        font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=12),
    )
    for component, label in marks:
        row = scree.loc[scree["component"] == int(component)]
        if row.empty:
            continue
        figure.add_vline(
            x=int(component),
            line=dict(color=palette.axis, width=1),
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(family=FONT_FAMILY, color=palette.ink_secondary, size=12),
        )

    _style_axes(figure, palette, "Count of components kept", "Share of the energy")
    _title_layout(
        figure,
        palette,
        title,
        "A sharp corpus reaches a high share in a few components.",
    )
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    figure.update_xaxes(dtick=1 if len(scree) <= 12 else 5)
    return figure


def gap_heatmap_frame(gaps: pd.DataFrame) -> pd.DataFrame:
    """Lay the gap out as years by months, for a heatmap.

    The brief asks for a heatmap of the gap. A heatmap answers a question that
    a bar chart cannot: does the gap cluster in a part of the year, or does it
    hold across every month? A month with no gap value stays empty. It does
    not become a zero, because a zero would read as "the model was correct".
    """
    if gaps.empty or GAP_COLUMN not in gaps.columns:
        return pd.DataFrame()
    # Only the holdout. Before 2025 the model fitted the data, so a difference
    # there is a residual and not a gap. A heatmap that mixed the two would
    # invite a reader to compare a fit with a forecast.
    frame = gaps.loc[HOLDOUT_START:, [GAP_COLUMN]].dropna().copy()
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    wide = frame.pivot_table(index="year", columns="month", values=GAP_COLUMN, aggfunc="mean")
    return wide.reindex(columns=range(1, 13))


def figure_gap_heatmap(wide: pd.DataFrame, palette: Palette, title: str) -> go.Figure:
    """One cell for each month. The scale is diverging and anchored at zero."""
    limit = float(max(abs(wide.min().min()), abs(wide.max().max()))) if not wide.empty else 1.0
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    figure = go.Figure(
        go.Heatmap(
            z=wide.to_numpy(),
            x=months,
            y=[str(year) for year in wide.index],
            colorscale=diverging_colorscale(palette),
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            hovertemplate="%{y} %{x}<br>gap %{z:.2f}<extra></extra>",
            colorbar=dict(title="Gap", thickness=12),
        )
    )
    _title_layout(figure, palette, title, "Gray is zero. A darker cell is a larger miss.")
    _style_axes(figure, palette, "", "")
    figure.update_yaxes(autorange="reversed")
    return figure


def figure_topic_heatmap(shares: pd.DataFrame, palette: Palette, title: str) -> go.Figure:
    """Panel 4. The share of each topic in each month, as a heatmap.

    A heatmap is the form for magnitude in a grid. The scale is sequential: one
    hue, light to dark on the light surface and dark to light on the dark
    surface. The color bar gives the scale, and the table view holds the
    values.
    """
    if shares.empty:
        raise ValueError("the topic share frame is empty")
    labels = [f"Topic {int(str(name).split('_')[1])}" for name in shares.columns]
    figure = go.Figure(
        go.Heatmap(
            x=shares.index,
            y=labels,
            z=shares.to_numpy().T,
            colorscale=sequential_colorscale(palette),
            hovertemplate="<b>%{z:.1%}</b> of the month<br>%{y}<br>%{x|%Y-%m}<extra></extra>",
            colorbar=dict(
                title=dict(
                    text="Share",
                    font=dict(color=palette.ink_secondary, size=12),
                ),
                tickformat=".0%",
                tickfont=dict(color=palette.ink_muted, size=11),
                outlinewidth=0,
                thickness=10,
            ),
        )
    )
    _style_axes(figure, palette, "Month", "Topic")
    _title_layout(
        figure,
        palette,
        title,
        "A flat row is a topic with a stable share. The shares hold no large step.",
    )
    figure.update_yaxes(showgrid=False, autorange="reversed")
    figure.update_xaxes(showgrid=False)
    return figure


def figure_media_tone(
    frame: pd.DataFrame,
    palette: Palette,
    title: str,
    column: str = MEDIA_TONE_COLUMN,
    y_title: str = "Media tone, from -1 to +1",
    subtitle: str = "",
    zero_line: bool = False,
    events: Sequence[EconomicEvent] = (),
) -> go.Figure:
    """Panel 5. One media series over time, one series and one axis.

    Panel 5 calls this function one time for each series that it shows. Two
    series with different units never share one axis, so each series gets its
    own chart. A single series needs no legend box: the title names it.
    """
    if column not in frame.columns:
        raise KeyError(f"the media frame needs the column {column!r}")
    figure = go.Figure(
        go.Scatter(
            x=frame["date"],
            y=frame[column],
            mode="lines",
            line=dict(color=palette.series_1, width=2),
            name=column,
            hovertemplate="<b>%{y:.3f}</b><br>%{x|%Y-%m}<extra></extra>",
            showlegend=False,
        )
    )
    values = frame[column].astype(float)
    if not values.dropna().empty:
        # Two direct labels: the first month and the last month. The extreme
        # months stay in the table view.
        for position, anchor in ((0, "left"), (len(frame) - 1, "right")):
            row = frame.iloc[position]
            figure.add_annotation(
                x=row["date"],
                y=float(row[column]),
                text=f"{float(row[column]):.3f}",
                showarrow=False,
                xanchor=anchor,
                yshift=14,
                font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=12),
            )
    if zero_line:
        figure.add_hline(y=0, line=dict(color=palette.axis, width=1))
    if len(frame):
        add_event_marks(
            figure,
            palette,
            events,
            x_min=frame["date"].min(),
            x_max=frame["date"].max(),
        )
    _style_axes(figure, palette, "Month", y_title)
    _title_layout(figure, palette, title, subtitle)
    figure.update_layout(hovermode="x unified")
    return figure


FRAMING_SUBTITLE = (
    "Each contrast is a ratio in the range -1 to +1. A positive value is the "
    "price-level frame, the household frame or the current-conditions frame. "
    "A negative value is the opposite side of the same contrast."
)


def figure_framing(
    frame: pd.DataFrame,
    palette: Palette,
    title: str,
    columns: Sequence[str],
    subtitle: str = FRAMING_SUBTITLE,
    events: Sequence[EconomicEvent] = (),
) -> go.Figure:
    """Panel 7. The framing contrasts over time, on one axis.

    Each contrast is a ratio in the same range, so the contrasts share one
    axis. This is not a dual axis: there is one scale, and each series uses it.

    The series colors come from the palette in a fixed order. `series_3` is
    below 3:1 contrast on the light surface, so each series also gets a direct
    label at its last month, and the table view holds every value.
    """
    present = [name for name in columns if name in frame.columns]
    if not present:
        raise KeyError(f"the framing frame holds none of these columns: {list(columns)}")
    if frame.empty:
        raise ValueError("the framing frame is empty")

    colors = (palette.series_1, palette.series_2, palette.series_3)
    if len(present) > len(colors):
        # A fourth series would repeat the first color, and two series of the
        # same color read as one series. `framing_series` keeps the count at
        # three or below.
        raise ValueError(
            f"the chart draws {len(colors)} series at most; it received "
            f"{len(present)}: {present}"
        )
    figure = go.Figure()
    for position, name in enumerate(present):
        color = colors[position % len(colors)]
        values = frame[name].astype(float)
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>%{{y:.3f}}</b> {name}<br>%{{x|%Y-%m}}<extra></extra>",
                showlegend=len(present) > 1,
            )
        )
        tail = frame.loc[values.notna()]
        if tail.empty:
            continue
        last = tail.iloc[-1]
        figure.add_annotation(
            x=last["date"],
            y=float(last[name]),
            text=name,
            showarrow=False,
            xanchor="right",
            yshift=12,
            font=dict(family=FONT_FAMILY, color=palette.ink_secondary, size=11),
        )

    figure.add_hline(y=0, line=dict(color=palette.axis, width=1))
    add_event_marks(
        figure,
        palette,
        events,
        x_min=frame["date"].min(),
        x_max=frame["date"].max(),
        labels=False,
    )
    _style_axes(figure, palette, "Month", "Framing ratio, from -1 to +1")
    _title_layout(figure, palette, title, subtitle)
    figure.update_layout(hovermode="x unified")
    return figure


def figure_domain_composition(
    composition: pd.DataFrame,
    palette: Palette,
    title: str,
) -> go.Figure:
    """Panel 10. The share of each domain, as horizontal bars.

    A horizontal bar is the form for a ranked magnitude with a text label: the
    domain names are long, and a horizontal bar gives each name a full line.

    The bars use one color. The length carries the magnitude, so a color ramp
    over the bars would encode the same number two times.
    """
    if composition.empty:
        raise ValueError("the composition frame is empty")
    for name in ("domain", "share"):
        if name not in composition.columns:
            raise KeyError(f"the composition frame needs the column {name!r}")
    frame = composition.sort_values("share")
    figure = go.Figure(
        go.Bar(
            x=frame["share"],
            y=frame["domain"],
            orientation="h",
            marker=dict(color=palette.series_1, cornerradius=4),
            customdata=frame[["n_headlines"]].to_numpy(),
            hovertemplate=(
                "<b>%{x:.1%}</b> of the corpus<br>%{y}<br>"
                "%{customdata[0]:,} headlines<extra></extra>"
            ),
            showlegend=False,
        )
    )
    top = frame.iloc[-1]
    figure.add_annotation(
        x=float(top["share"]),
        y=str(top["domain"]),
        text=f"{float(top['share']):.1%}",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font=dict(family=FONT_FAMILY, color=palette.ink_primary, size=12),
    )
    _style_axes(figure, palette, "Share of the headlines", "")
    _title_layout(
        figure,
        palette,
        title,
        "The small domains are in one row. The row gives the count of domains "
        "that it holds.",
    )
    figure.update_xaxes(tickformat=".0%", rangemode="tozero")
    figure.update_yaxes(showgrid=False)
    figure.update_layout(bargap=0.3)
    return figure


def figure_monthly_coverage(
    coverage: pd.DataFrame,
    palette: Palette,
    title: str,
    threshold: float | None = None,
    count_column: str = "n_articles",
) -> go.Figure:
    """Panel 11. The count of articles of each month, against the thin limit.

    One axis: the bars and the limit have the same unit. The limit is a dashed
    line, because a dashed line is a threshold in this dashboard. The bars
    below the limit get the muted ink, so the chart colors a decision and not
    a magnitude.
    """
    if coverage.empty:
        raise ValueError("the coverage frame is empty")
    if count_column not in coverage.columns:
        raise KeyError(f"the coverage frame needs the column {count_column!r}")

    figure = go.Figure()
    if threshold is None:
        figure.add_trace(
            go.Bar(
                x=coverage["date"],
                y=coverage[count_column],
                marker=dict(color=palette.series_1, cornerradius=4),
                hovertemplate="<b>%{y:,.0f}</b> articles<br>%{x|%Y-%m}<extra></extra>",
                showlegend=False,
            )
        )
    else:
        thin = coverage[count_column] < float(threshold)
        groups = (
            (coverage.loc[~thin], "At or above the limit", palette.series_1),
            (coverage.loc[thin], "Below the limit", palette.ink_muted),
        )
        for part, label, color in groups:
            if part.empty:
                continue
            figure.add_trace(
                go.Bar(
                    x=part["date"],
                    y=part[count_column],
                    name=label,
                    marker=dict(color=color, cornerradius=4),
                    hovertemplate=(
                        "<b>%{y:,.0f}</b> articles<br>%{x|%Y-%m}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )
        figure.add_hline(
            y=float(threshold),
            line=dict(color=palette.series_2, width=1, dash="dash"),
            annotation_text=f"Thin limit {float(threshold):,.0f} articles",
            annotation_position="top right",
            annotation_font=dict(
                family=FONT_FAMILY, color=palette.ink_secondary, size=12
            ),
        )

    _style_axes(figure, palette, "Month", "Articles in the corpus")
    _title_layout(
        figure,
        palette,
        title,
        "A thin month gives a noisy tone value. The limit is a share of the "
        "median month.",
    )
    figure.update_layout(bargap=0.2, barmode="overlay")
    figure.update_yaxes(rangemode="tozero")
    return figure


# ---------------------------------------------------------------------------
# The page chrome — the header, the status strip, the stat tiles and the CSS
# ---------------------------------------------------------------------------
#
# Each function here returns a string. The tests call these functions and read
# the string. No function makes an `st.*` call.


@dataclass(frozen=True)
class StatusChip:
    """One reading of the status strip.

    `warn` gives the chip a red edge. The value also gives the state in words,
    so the state never depends on the color alone.
    """

    label: str
    value: str
    warn: bool = False


@dataclass(frozen=True)
class StatTile:
    """One headline number.

    `lead` makes the tile the one large figure of the page. The page has one
    lead tile, because a page with four large figures has no hierarchy.
    """

    label: str
    value: str
    detail: str = ""
    lead: bool = False


def status_strip_html(chips: Sequence[StatusChip]) -> str:
    """Render the status strip. The caller gives the readings."""
    cells = []
    for chip in chips:
        classes = "pb-chip pb-chip--warn" if chip.warn else "pb-chip"
        cells.append(
            f'<span class="{classes}"><span class="pb-chip-label">'
            f"{escape(chip.label)}</span>{escape(chip.value)}</span>"
        )
    return '<div class="pb-strip">' + "".join(cells) + "</div>"


def header_html(title: str, subtitle: str, chips: Sequence[StatusChip] = ()) -> str:
    """Render the page header: the title, one line of subtitle, the strip."""
    return (
        '<div class="pb-header">'
        f'<h1 class="pb-title">{escape(title)}</h1>'
        f'<p class="pb-subtitle">{escape(subtitle)}</p>'
        f"{status_strip_html(chips)}"
        "</div>"
    )


def stat_tiles_html(tiles: Sequence[StatTile]) -> str:
    """Render the headline numbers as a grid of tiles."""
    cells = []
    for tile in tiles:
        classes = "pb-tile pb-tile--lead" if tile.lead else "pb-tile"
        detail = (
            f'<div class="pb-tile-detail">{escape(tile.detail)}</div>'
            if tile.detail
            else ""
        )
        cells.append(
            f'<div class="{classes}">'
            f'<div class="pb-tile-label">{escape(tile.label)}</div>'
            f'<div class="pb-tile-value">{escape(tile.value)}</div>'
            f"{detail}</div>"
        )
    return '<div class="pb-tiles">' + "".join(cells) + "</div>"


def section_html(heading: str, caption: str = "") -> str:
    """Render one section heading, with an optional line below it."""
    body = f'<p class="pb-caption">{escape(caption)}</p>' if caption else ""
    return f'<div class="pb-section-block"><h2 class="pb-section">{escape(heading)}</h2>{body}</div>'


RULE_HTML = '<hr class="pb-rule">'

# One CSS block for the page.
#
# **Each rule uses a class of this file, with three exceptions.** The three
# rules in group 1 use a Streamlit class name. Streamlit can rename an internal
# class at any upgrade. If that happens, the page keeps the Streamlit default
# background, text color and width, and every other rule in this block still
# works, because each other rule uses a `pb-` class that this file writes. The
# block holds no layout rule for a chart: a figure carries its own colors.
#
# The template uses `string.Template`, because CSS uses the brace character and
# an f-string would need each brace two times.
CSS_TEMPLATE = Template(
    """
<style>
/* 1. The three Streamlit selectors. They are cosmetic. */
.stApp { background-color: $page; }
.stApp, .stApp p, .stApp li, .stApp label { color: $ink; }
.stApp .block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1280px; }

/* 2. The type scale. One step between each level. */
.pb-header { margin: 0 0 0.25rem; }
.pb-title { font-family: $font; font-size: 2.05rem; font-weight: 650;
            line-height: 1.15; letter-spacing: -0.01em; color: $ink; margin: 0; }
.pb-subtitle { font-size: 1.02rem; line-height: 1.5; color: $ink_secondary;
               margin: 0.5rem 0 0; max-width: 68ch; }
.pb-section-block { margin: 0 0 0.9rem; }
.pb-section { font-family: $font; font-size: 1.2rem; font-weight: 620;
              color: $ink; margin: 0; letter-spacing: -0.005em; }
.pb-caption { font-size: 0.88rem; line-height: 1.5; color: $ink_muted;
              margin: 0.3rem 0 0; max-width: 84ch; }
.pb-rule { height: 1px; background: $grid; border: 0; margin: 2rem 0 1.5rem; }

/* 3. The status strip. The words give the state; the color repeats it. */
.pb-strip { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.1rem 0 0; }
.pb-chip { display: inline-flex; gap: 0.45rem; align-items: baseline;
           padding: 0.3rem 0.65rem; border: 1px solid $axis; border-radius: 999px;
           background: $surface; font-size: 0.84rem; color: $ink_secondary; }
.pb-chip-label { font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase;
                 color: $ink_muted; }
.pb-chip--warn { border-color: $warn; color: $ink; }

/* 4. The stat tiles. The lead tile is the one large figure of the page. */
.pb-tiles { display: grid; gap: 0.75rem; margin: 1.4rem 0 0.25rem;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.pb-tile { background: $surface; border: 1px solid $grid; border-radius: 10px;
           padding: 0.95rem 1.05rem; }
.pb-tile-label { font-size: 0.74rem; letter-spacing: 0.07em; text-transform: uppercase;
                 color: $ink_muted; }
.pb-tile-value { font-size: 1.8rem; font-weight: 620; line-height: 1.1; color: $ink;
                 margin-top: 0.35rem; font-variant-numeric: proportional-nums; }
.pb-tile--lead .pb-tile-value { font-size: 2.5rem; }
.pb-tile-detail { font-size: 0.84rem; line-height: 1.45; color: $ink_secondary;
                  margin-top: 0.35rem; }
</style>
"""
)

# The count of rules that use a Streamlit class name. A test holds this number,
# because each such rule is a risk at an upgrade.
STREAMLIT_CSS_RULES = 3


def custom_css(palette: Palette) -> str:
    """Return the one CSS block of the page, in the colors of the palette."""
    return CSS_TEMPLATE.substitute(
        page=palette.page,
        surface=palette.surface,
        ink=palette.ink_primary,
        ink_secondary=palette.ink_secondary,
        ink_muted=palette.ink_muted,
        grid=palette.grid,
        axis=palette.axis,
        warn=palette.diverging_low,
        font=FONT_FAMILY,
    )


def status_chips(
    fred: FredStatus,
    artifacts: Mapping[str, ArtifactStatus],
    built_at: str | None = None,
) -> list[StatusChip]:
    """Make the readings of the status strip.

    The strip gives three facts: the state of FRED, the size of the corpus and
    the time of the last build. Each fact is a state of the build. None of them
    is a result of the study.
    """
    chips = [
        StatusChip(
            "FRED",
            f"{len(fred.series)} series" if fred.available else "not available",
            warn=not fred.available,
        )
    ]
    corpus = artifacts.get("media_monthly")
    if corpus is not None:
        chips.append(
            StatusChip("Corpus", corpus.fact if corpus.present else "absent", warn=not corpus.present)
        )
    spectrum = artifacts.get("svd_spectrum")
    if spectrum is not None:
        chips.append(
            StatusChip("SVD", spectrum.fact if spectrum.present else "absent", warn=not spectrum.present)
        )
    chips.append(StatusChip("Last build", built_at or "no derived file"))
    return chips


def headline_tiles(summary: pd.Series, corpus: ArtifactStatus | None = None) -> list[StatTile]:
    """Make the headline numbers of the page.

    The mean gap is the lead figure, because the study is about that one
    number. The corpus tile is a size, not a result.
    """
    tiles = [
        StatTile(
            "Mean gap, out of sample",
            f"{float(summary['mean_gap']):.2f}",
            "Observed minus predicted sentiment, 2025-01 and later",
            lead=True,
        ),
        StatTile(
            "Worst month",
            str(summary["worst_month"]),
            f"The gap is {float(summary['worst_gap']):.2f}",
        ),
        StatTile(
            "Outside the 95% interval",
            f"{int(summary['months_outside_pi'])} of {int(summary['n_tested'])}",
            "Months with an observed value below the prediction interval",
        ),
    ]
    if corpus is not None and corpus.present:
        tiles.append(StatTile("Media corpus", corpus.fact, "GDELT GKG economic themes"))
    return tiles


# ---------------------------------------------------------------------------
# Cached work — the dashboard must not refit on each interaction
# ---------------------------------------------------------------------------


def _streamlit():
    """Import Streamlit inside a function.

    The tests import this module. The import must not start Streamlit work.
    """
    import streamlit as st

    return st


def load_environment() -> bool:
    """Load `.env` and say whether a FRED key is present.

    The function returns a boolean. It never returns the key.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning("could not load .env (%s); continuing", exc)
    return fred_key_present()


def load_processed_frame(
    path: str | Path,
    required: Sequence[str] = (),
    *,
    label: str = "the file",
) -> pd.DataFrame | None:
    """Read one derived file, or return None when the file is not usable.

    **An absent file is a normal state.** A build script writes each file, and
    the dashboard must run before that script runs. A damaged file and a file
    without the columns that a panel needs give the same answer, because a
    panel cannot draw from either one. The function writes a log line and
    returns None. It never raises, because one bad file must not stop the app.
    """
    file = Path(path)
    if not file.exists():
        return None
    try:
        frame = pd.read_parquet(file)
    except Exception as exc:  # noqa: BLE001 - a damaged file must not stop the app
        logger.warning("%s did not load (%s)", label, exc)
        return None
    if frame.empty:
        logger.warning("%s is empty", label)
        return None
    missing = [name for name in required if name not in frame.columns]
    if missing:
        logger.warning("%s has no column %s", label, ", ".join(missing))
        return None
    return frame


def load_media_series(path: str | Path = MEDIA_SERIES_PATH) -> pd.DataFrame | None:
    """Read the monthly media series, or return None when it is absent.

    The series comes from `scripts/build_media_series.py`, which reads the
    GKG theme corpus. The function keeps the three regressor columns and the
    count of articles. The count is a fact about the corpus, and the header
    shows it.
    """
    series = load_processed_frame(path, label="the media series")
    if series is None:
        return None
    keep = [
        column
        for column in (*MEDIA_COLUMNS, *MEDIA_OPTIONAL_COLUMNS, "n_articles")
        if column in series.columns
    ]
    return series.loc[:, keep] if keep else None


def load_headlines(path: str | Path = HEADLINES_PATH) -> pd.DataFrame | None:
    """Read the URL and the month of each headline, or None when absent.

    The function keeps two columns only. The corpus holds 36 000 rows, and
    panel 10 needs the domain of each row and nothing else. The function never
    returns the title, because no panel shows a title.
    """
    frame = load_processed_frame(path, ("url",), label="the headline corpus")
    if frame is None:
        return None
    keep = [name for name in ("url", "month") if name in frame.columns]
    return frame.loc[:, keep]


def load_svd_spectrum(path: str | Path = SVD_SPECTRUM_PATH) -> pd.DataFrame | None:
    """Read the singular value spectrum, or return None when it is absent."""
    return load_processed_frame(
        path, SPECTRUM_REQUIRED_COLUMNS, label="the SVD spectrum"
    )


def load_svd_topics(path: str | Path = SVD_TOPICS_PATH) -> pd.DataFrame | None:
    """Read the strongest terms of each topic, or None when they are absent."""
    return load_processed_frame(path, TOPICS_REQUIRED_COLUMNS, label="the SVD topics")


def load_svd_monthly(path: str | Path = SVD_MONTHLY_PATH) -> pd.DataFrame | None:
    """Read the topic share of each month, or None when it is absent."""
    return load_processed_frame(path, label="the monthly topic shares")


def load_artifacts() -> dict[str, pd.DataFrame | None]:
    """Read every derived file that a panel needs.

    The function returns one mapping. `artifact_status` then makes the state of
    each file, and `panel_availability` makes the state of each panel. One read
    serves both, so the header and the panels can never disagree.
    """
    return {
        "media_monthly": load_media_series(),
        "svd_spectrum": load_svd_spectrum(),
        "svd_topics": load_svd_topics(),
        "svd_monthly": load_svd_monthly(),
        "headlines": load_headlines(),
    }


def last_build_time(paths: Sequence[str | Path] = ()) -> str | None:
    """Give the time of the newest derived file, or None when none exists."""
    stamps = [Path(path).stat().st_mtime for path in paths if Path(path).exists()]
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps)).strftime("%Y-%m-%d %H:%M")


ARTIFACT_PATHS: tuple[Path, ...] = tuple(spec[2] for spec in ARTIFACT_SPECS)


def build_panel(csv_path: str, use_fred: bool) -> tuple[pd.DataFrame, Any]:
    """Build the monthly panel. FRED gets one call for the whole dashboard.

    Each later step receives this panel as an argument. An earlier version
    rebuilt the panel in each step, and FRED then answered three times. One
    answer could fail, and two steps could then disagree about the data.

    The media series is joined when it exists. The join is a left join, so the
    months before the corpus starts keep an empty value. The panel then tells
    `panel_availability` which media panels can show a result.
    """
    fred = FredSource() if use_fred else FredSource(api_key="")
    panel, report = build_monthly_panel(csv_path, fred=fred)
    media = load_media_series()
    if media is not None:
        # The three regressor columns, and the framing and calibration columns
        # when they exist. `n_articles` stays out: it is a fact about the
        # corpus, not a regressor, and panel 11 reads it from the media frame.
        keep = [
            name
            for name in (*MEDIA_COLUMNS, *MEDIA_OPTIONAL_COLUMNS)
            if name in media.columns
        ]
        if keep:
            panel = panel.join(media.loc[:, keep], how="left")
    return panel, report


def build_fit(_panel: pd.DataFrame, csv_path: str, use_fred: bool) -> Any:
    """Fit the baseline on 2010-01 to 2024-12.

    The leading underscore of `_panel` keeps the frame out of the cache key.
    `csv_path` and `use_fred` identify the panel, so the key stays correct and
    small. This is the documented Streamlit pattern for an unhashable argument.
    """
    train = training_slice(_panel, start=TRAIN_START, end=TRAIN_END)
    return fit_baseline(train, features=BASELINE_FEATURES)


def build_gaps(
    _fit: Any, _panel: pd.DataFrame, csv_path: str, use_fred: bool
) -> tuple[pd.DataFrame, Any]:
    """Give the gap frame for every month that the model can predict."""
    return sentiment_gap(_fit, _panel)


def build_comparison(
    _panel: pd.DataFrame,
    csv_path: str,
    use_fred: bool,
    start: str = TRAIN_START,
    end: str = TRAIN_END,
) -> Any:
    """Fit Models 1 to 3 on the estimation window.

    The window is the training window. R-squared and AIC are only comparable
    with the baseline headline when the rows are the same rows. A longer window
    would mix the out-of-sample months into the fit statistics.
    """
    return compare_models(_panel.loc[start:end])


def build_composition(
    _headlines: pd.DataFrame, rows: int = 0
) -> tuple[CompositionReport, pd.DataFrame]:
    """Count the domains of the headline corpus one time for each build.

    The corpus holds 36 000 rows, and the count takes approximately 90
    milliseconds. The palette control reruns the page, so the count would run
    again at each change of the palette. The leading underscore keeps the frame
    out of the cache key, and `rows` says when the corpus changed. This is the
    pattern of `build_granger`.
    """
    return composition_report(_headlines), domain_composition(_headlines)


def build_granger(
    _frame: pd.DataFrame,
    csv_path: str,
    use_fred: bool,
    rows: int = 0,
) -> Any:
    """Run the Granger test in both directions on the joined frame.

    `rows` goes into the cache key. The frame has a leading underscore, so
    Streamlit does not hash it, and the count of rows then says when the frame
    changed.
    """
    return granger_media_gap(_frame)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_unavailable(status: PanelStatus, extra: str = "") -> None:
    """Print the "not yet available" state of one panel.

    The block has no chart, no series and no spinner. It names each input that
    is absent. A reader cannot read this block as a result.

    A caller must send only a panel with `available=False`. A test holds this
    rule: the render path and `panel_availability` must never disagree.
    """
    st = _streamlit()
    with st.container(border=True):
        st.markdown(f'<h2 class="pb-section">{escape(status.label)}</h2>', unsafe_allow_html=True)
        # The heading states what is absent now, and not what the panel reads.
        # A panel can read a file that is present and still wait for one column
        # of that file.
        st.markdown(
            f"**{ARTIFACT_MESSAGE if status.artifact_absent else 'Not yet available.'}**"
        )
        if status.missing:
            st.markdown(
                "This panel needs these inputs. They are absent:\n"
                + "\n".join(f"- `{name}`" for name in status.missing)
            )
        if status.reason:
            st.caption(status.reason)
        if extra:
            st.caption(extra)
        st.caption(
            "An absent input is not a negative result. This panel shows no chart, "
            "because an empty chart would read as a result."
        )


def render_section(heading: str, caption: str = "") -> None:
    """Print one section heading and its line of text."""
    st = _streamlit()
    st.markdown(section_html(heading, caption), unsafe_allow_html=True)


PAGE_TITLE = "The 2026 Sentiment Gap"
PAGE_SUBTITLE = (
    "A macro model trains on 2010-01 to 2024-12, then predicts 2025 and 2026. "
    "The gap is the observed sentiment minus the predicted sentiment. The six "
    "panels test three causes of that gap."
)


def main() -> None:  # pragma: no cover - the screen layer
    """Draw the dashboard."""
    st = _streamlit()
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":material/insights:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Cache the heavy work. `cache_data` returns a copy of a frame.
    # `cache_resource` keeps a fitted model, which must not be copied.
    # The FRED calls get a TTL, so a bad key or an outage does not hold a bad
    # result for the life of the process.
    cached_panel = st.cache_data(ttl="6h", show_spinner=False)(build_panel)
    cached_fit = st.cache_resource(show_spinner=False)(build_fit)
    cached_gaps = st.cache_data(ttl="6h", show_spinner=False)(build_gaps)
    cached_comparison = st.cache_resource(show_spinner=False)(build_comparison)
    cached_granger = st.cache_resource(show_spinner=False)(build_granger)
    cached_artifacts = st.cache_data(ttl="6h", show_spinner=False)(load_artifacts)
    cached_composition = st.cache_data(ttl="6h", show_spinner=False)(build_composition)

    key_present = load_environment()

    st.session_state.setdefault("palette_mode", "light")
    st.session_state.setdefault("show_tables", True)
    palette = get_palette(st.session_state["palette_mode"])

    # The selected palette paints the page, so the page and the charts agree.
    st.markdown(custom_css(palette), unsafe_allow_html=True)

    # The header holds the numbers of the data. The data loads below, so the
    # header gets a slot here and its content later.
    header_slot = st.container()

    with st.status("Loading the panel and fitting the baseline", expanded=False) as status_box:
        try:
            panel, report = cached_panel(str(CSV_PATH), key_present)
            fit = cached_fit(panel, str(CSV_PATH), key_present)
            gaps, prediction_report = cached_gaps(
                fit, panel, str(CSV_PATH), key_present
            )
            frames = cached_artifacts()
            status_box.update(label="The panel and the baseline are ready", state="complete")
        except Exception as exc:
            status_box.update(label="The panel did not load", state="error")
            st.error(f"The dashboard cannot load the data: {exc}")
            return

    fred = fred_status(report, key_present)
    artifacts = artifact_status(frames)

    # The columns of the panel and the gap frame, with the derived files, make
    # the state of each panel. This is the only decision point.
    statuses = panel_availability(set(panel.columns) | set(gaps.columns), artifacts)
    summary = gap_summary(gaps.loc[gaps.index >= pd.Timestamp(HOLDOUT_START)])
    built_at = last_build_time(ARTIFACT_PATHS)

    with header_slot:
        st.markdown(
            header_html(
                PAGE_TITLE, PAGE_SUBTITLE, status_chips(fred, artifacts, built_at)
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            stat_tiles_html(headline_tiles(summary, artifacts.get("media_monthly"))),
            unsafe_allow_html=True,
        )
        if not fred.available:
            st.warning(fred.message)
        st.markdown(RULE_HTML, unsafe_allow_html=True)

    # One control row above every panel. A panel holds no control of the page.
    with st.container(border=True):
        render_section("View", "These two controls change every panel below.")
        controls = st.columns([1.3, 1.2, 2.5])
        with controls[0]:
            st.radio(
                "Palette",
                options=("light", "dark"),
                key="palette_mode",
                horizontal=True,
                help="The dashboard selects a palette. It does not invert the colors.",
            )
        with controls[1]:
            st.toggle("Show the table views", key="show_tables")
        with controls[2]:
            if fred.available:
                st.caption(fred.message)
            st.caption(
                "Each chart has a table view. The table holds every value of "
                "the chart."
            )

    st.markdown(RULE_HTML, unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "1. Sentiment over time",
            "2. The gap by month",
            "3. Model comparison",
            "4. SVD spectrum",
            "5. Media series",
            "6. Granger causality",
            "7-8. Framing and calibration",
            "9. Events and releases",
            "10-12. Data quality",
            "13. Demographic splits",
        ]
    )

    with tabs[0]:
        _render_timeline(palette, gaps, summary, statuses["timeline"], prediction_report)
    with tabs[1]:
        _render_gap(palette, gaps, summary, statuses["gap"])
    with tabs[2]:
        _render_models(cached_comparison, panel, key_present, statuses["models"], fred)
    with tabs[3]:
        _render_scree(
            palette,
            statuses["scree"],
            frames.get("svd_spectrum"),
            frames.get("svd_topics"),
            frames.get("svd_monthly"),
        )
    with tabs[4]:
        _render_media(palette, panel, statuses["media_tone"])
    with tabs[5]:
        _render_granger(cached_granger, panel, gaps, key_present, statuses["granger"])
    with tabs[6]:
        _render_framing(palette, panel, statuses["framing"])
        st.markdown(RULE_HTML, unsafe_allow_html=True)
        _render_calibration(palette, panel, statuses["calibration"])
    with tabs[7]:
        _render_events(palette, gaps, statuses["events"])
    with tabs[8]:
        _render_composition(
            palette,
            frames.get("headlines"),
            statuses["composition"],
            cached_composition,
        )
        st.markdown(RULE_HTML, unsafe_allow_html=True)
        _render_coverage(palette, frames.get("media_monthly"), statuses["coverage"])
        st.markdown(RULE_HTML, unsafe_allow_html=True)
        _render_quality(palette, panel, gaps, fit, prediction_report, statuses["quality"])
    with tabs[9]:
        _render_demographics(statuses["demographics"])

    st.markdown(RULE_HTML, unsafe_allow_html=True)

    # The footer states the state of the build. It reads the statuses, so it
    # can never disagree with a panel.
    pending = [status for status in statuses.values() if not status.available]
    if pending:
        st.caption(
            "These panels have no input yet: "
            + "; ".join(status.summary() for status in pending)
        )
    else:
        st.caption("Every panel has its input.")
    st.caption(
        "The derived files: "
        + " ".join(status.summary() for status in artifacts.values())
    )
    for note in getattr(report, "notes", ())[:6]:
        st.caption(note)
    for note in getattr(fit, "notes", ())[:4]:
        st.caption(note)


def _render_timeline(palette, gaps, summary, status: PanelStatus, prediction_report) -> None:  # pragma: no cover
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    @st.fragment
    def body() -> None:
        timeline = prepare_timeline(gaps)
        earliest = int(timeline["date"].dt.year.min())
        latest = int(timeline["date"].dt.year.max())
        st.session_state.setdefault("timeline_from", max(earliest, 2010))
        start_year = st.slider(
            "First year on the chart",
            min_value=earliest,
            max_value=max(earliest, latest - 1),
            key="timeline_from",
            help="The chart always keeps the out-of-sample months.",
        )
        shown = timeline.loc[timeline["date"].dt.year >= start_year]
        figure = figure_observed_vs_predicted(
            shown, palette, timeline_title(summary), events=marked_events()
        )
        st.plotly_chart(figure, width="stretch", theme=None, key="timeline_chart")
        if st.session_state.get("show_tables", True):
            st.dataframe(
                shown.loc[:, ["month", "observed", "predicted", "pi_lower", "pi_upper", "sample"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "month": st.column_config.TextColumn("Month"),
                    "observed": st.column_config.NumberColumn("Observed", format="%.2f"),
                    "predicted": st.column_config.NumberColumn("Predicted", format="%.2f"),
                    "pi_lower": st.column_config.NumberColumn("95% lower", format="%.2f"),
                    "pi_upper": st.column_config.NumberColumn("95% upper", format="%.2f"),
                    "sample": st.column_config.TextColumn("Sample"),
                },
            )
        if getattr(prediction_report, "dropped_reasons", ()):
            st.caption("The model removed these months: " + "; ".join(prediction_report.dropped_reasons))

    body()


def _render_gap(palette, gaps, summary, status: PanelStatus) -> None:  # pragma: no cover
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    @st.fragment
    def body() -> None:
        gap_months = prepare_gap_months(gaps)
        figure = figure_gap_by_month(gap_months, palette, gap_title(summary))
        st.plotly_chart(figure, width="stretch", theme=None, key="gap_chart")
        wide = gap_heatmap_frame(gaps)
        if not wide.empty:
            st.plotly_chart(
                figure_gap_heatmap(wide, palette, "The gap by year and month"),
                width="stretch",
                theme=None,
                key="gap_heatmap",
            )
        if st.session_state.get("show_tables", True):
            columns = ["month", GAP_COLUMN]
            for name in ("observed", "predicted"):
                if name in gap_months.columns:
                    columns.append(name)
            st.dataframe(
                gap_months.loc[:, columns],
                width="stretch",
                hide_index=True,
                column_config={
                    "month": st.column_config.TextColumn("Month"),
                    GAP_COLUMN: st.column_config.NumberColumn("Gap", format="%.2f"),
                    "observed": st.column_config.NumberColumn("Observed", format="%.2f"),
                    "predicted": st.column_config.NumberColumn("Predicted", format="%.2f"),
                },
            )
        st.caption(
            "A month without a gap value is not in the chart. The loader names "
            "each month that it removes."
        )

    body()


def _render_models(cached_comparison, panel, key_present: bool, status: PanelStatus, fred: FredStatus) -> None:  # pragma: no cover
    st = _streamlit()
    with st.spinner("Fitting Models 1 to 3"):
        try:
            comparison = cached_comparison(panel, str(CSV_PATH), key_present)
        except Exception as exc:
            st.error(f"The model comparison did not run: {exc}")
            return

    st.markdown("#### 3. Model comparison, Models 1 to 3")
    st.markdown(
        "`table_generator` renders this table. The dashboard does not format "
        "these numbers, so the dashboard and the report agree. The window is "
        f"{TRAIN_START[:7]} to {TRAIN_END[:7]}, the estimation window of the baseline."
    )
    st.markdown(model_comparison_markdown(comparison))
    st.caption(comparison.summary())
    st.caption(
        "The table shows the fit statistics only. The mean out-of-sample gap of "
        "each model is in the report. An 'n/a' in that row could read as 'this "
        "model has no gap'."
    )
    # The cause comes from the model object. An earlier version printed a fixed
    # sentence about the corpus here, and that sentence named a module constant
    # that the file did not define, so the branch raised a NameError on the one
    # screen that needed it. The reason of the fit cannot drift from the fit.
    unavailable = {
        str(name): str(
            getattr(fit, "unavailable_reason", "")
            or "these columns are absent from the panel: "
            + ", ".join(getattr(fit, "missing_features", ()) or ("unknown",))
        )
        for name, fit in comparison.models.items()
        if not fit.available
    }
    if unavailable:
        st.info(
            "These models show n/a, with the cause of each one: "
            + " ".join(f"{name}: {reason}." for name, reason in unavailable.items())
            + " An absent column is a state of the build. It is not the "
            "finding that the media regressors did not help."
        )
    if not fred.available:
        st.caption(fred.message)



def read_scree(spectrum: pd.DataFrame) -> ScreeReading:
    """Read the k-selection diagnostics out of the saved spectrum.

    The two permutation readings disagree on this corpus. The noise floor keeps
    few components. The parallel analysis keeps all of them. Paragraph 3.3 of
    BRAINSTORMING.md predicted this: the matrix is not centred and it is not
    negative, so a permutation inside a column keeps the mean direction. The
    panel gives both readings and does not choose one.
    """
    k_noise = int(spectrum["above_noise_floor"].sum()) if "above_noise_floor" in spectrum else 0
    k_parallel = (
        int(spectrum["above_permuted_component"].sum())
        if "above_permuted_component" in spectrum
        else 0
    )
    cumulative = float(spectrum["cumulative_energy"].iloc[-1]) if len(spectrum) else 0.0
    floor = float(spectrum["permuted_max"].max()) if "permuted_max" in spectrum else 0.0
    return ScreeReading(
        k_max=int(len(spectrum)),
        k_noise=k_noise,
        k_parallel=k_parallel,
        cumulative_energy=cumulative,
        noise_floor=floor,
    )


def _render_scree(palette, status: PanelStatus, spectrum, topics, monthly) -> None:  # pragma: no cover
    st = _streamlit()
    if not status.available or spectrum is None or spectrum.empty:
        render_unavailable(status)
        return

    reading = read_scree(spectrum)
    st.plotly_chart(
        figure_scree(spectrum, palette, scree_title(reading)),
        width="stretch",
        theme=None,
        key="scree_chart",
    )
    st.plotly_chart(
        figure_cumulative_energy(
            spectrum,
            palette,
            energy_title(reading),
            marks=((reading.k_noise, "noise floor"),),
        ),
        width="stretch",
        theme=None,
        key="energy_chart",
    )
    st.caption(
        "This corpus has no clear elbow. The two readings of the noise floor "
        f"disagree: the floor keeps {reading.k_noise} components and the parallel "
        f"analysis keeps {reading.k_parallel}. The first {reading.k_max} components "
        f"hold {reading.cumulative_energy:.1%} of the energy. Theme co-occurrence "
        "is diffuse, so a chart with a sharp elbow would not be true."
    )

    if topics is not None and not topics.empty and st.session_state.get("show_tables", True):
        st.markdown("**The strongest theme tags of each topic**")
        wide = (
            topics.sort_values(["topic", "rank"])
            .groupby("topic")["term"]
            .apply(lambda terms: ", ".join(terms.head(6)))
            .reset_index()
        )
        wide.columns = ["Topic", "Theme tags, strongest first"]
        st.dataframe(wide, width="stretch", hide_index=True)

    if monthly is not None and not monthly.empty:
        st.plotly_chart(
            figure_topic_heatmap(monthly, palette, "The share of each topic in each month"),
            width="stretch",
            theme=None,
            key="topic_heatmap",
        )


def _render_media(palette, panel, status: PanelStatus) -> None:  # pragma: no cover
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    frame = prepare_media_series(panel)
    st.plotly_chart(
        figure_media_tone(
            frame,
            palette,
            media_tone_title(frame),
            zero_line=True,
            events=marked_events(),
        ),
        width="stretch",
        theme=None,
        key="media_tone_chart",
    )
    values = frame[MEDIA_TONE_COLUMN].astype(float)
    st.caption(
        "The index is (positive - negative) / (positive + negative). It uses "
        "the tone that GDELT calculates for each article. A value below zero "
        f"is negative coverage. The mean of the series is {values.mean():.3f} "
        f"over {int(values.notna().sum())} months, and the range is "
        f"{values.min():.3f} to {values.max():.3f}."
    )
    if st.session_state.get("show_tables", True):
        extra = [c for c in (SALIENCE_COLUMN, ATTRIBUTION_COLUMN) if c in frame.columns]
        table = frame.loc[:, ["month", MEDIA_TONE_COLUMN] + extra].copy()
        table = table.rename(columns={"month": "Month"})
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                MEDIA_TONE_COLUMN: st.column_config.NumberColumn("Media tone", format="%.4f"),
                "topic_salience": st.column_config.NumberColumn("Distress share", format="%.4f"),
                "political_attribution": st.column_config.NumberColumn(
                    "Political frame share", format="%.4f"
                ),
            },
        )


def _render_granger(cached_granger, panel, gaps, key_present: bool, status: PanelStatus) -> None:  # pragma: no cover
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    frame = gaps.loc[:, [GAP_COLUMN]].join(
        panel.loc[:, [MEDIA_TONE_COLUMN]], how="inner"
    ).dropna()
    report = cached_granger(frame, str(CSV_PATH), key_present, len(frame))
    if report is None:
        render_unavailable(status, "The Granger test did not return a result.")
        return

    st.markdown(f"**{granger_title(report)}**")
    records = granger_records(report)
    if records:
        st.dataframe(
            pd.DataFrame(records),
            width="stretch",
            hide_index=True,
        )
    # The text comes from the test, not from a fixed sentence. An earlier
    # version printed one p-value of Model 3 in words. That number moved when
    # the panel changed its alignment, and the sentence could not move with it.
    st.info(granger_verdict(report))
    st.caption(
        "A Granger test measures predictive precedence. It does not measure a "
        "cause. A common cause can move both series together."
    )


def _render_framing(palette, panel, status: PanelStatus) -> None:  # pragma: no cover
    """Panel 7. The framing measure of the brief."""
    st = _streamlit()
    if not status.available:
        render_unavailable(
            status,
            "Framing is the emphasis on price levels against inflation rates, "
            "on household experience against national aggregates, and on "
            "current conditions against forecasts. `nlp_media_svd` holds the "
            "lexicon of these three contrasts. The monthly file must carry the "
            "measure before this panel can show it. The panel shows no chart "
            "now, and it does not use the tone series in place of framing: "
            "tone and framing measure different things.",
        )
        return

    columns = framing_series(panel.columns)
    frame = prepare_measure_series(panel, columns)
    st.plotly_chart(
        figure_framing(
            frame,
            palette,
            framing_title(frame, columns),
            columns,
            events=marked_events(),
        ),
        width="stretch",
        theme=None,
        key="framing_chart",
    )
    st.caption(
        "A positive value is the price-level frame, the household frame or the "
        "current-conditions frame. Hypothesis A gives the reason to measure "
        "this: a household anchors to the price that it pays, and not to the "
        "rate of change of that price. A framing measure shows which side the "
        "coverage emphasises. It does not show what a reader believes."
    )
    st.caption(
        "The chart draws these columns: "
        + ", ".join(f"`{name}`" for name in columns)
        + ". The panel draws the three contrasts of the brief when the file "
        "holds them, because a composite column can repeat a contrast."
    )
    agreement = framing_agreement(panel)
    if agreement is None:
        st.caption(
            "The panel holds one text source of the framing measure only, so "
            "it gives no agreement between the sources."
        )
    else:
        st.caption(agreement.note())
    months = int(frame["date"].notna().sum())
    st.caption(
        f"The measure covers {months} months, from {frame['month'].iloc[0]} to "
        f"{frame['month'].iloc[-1]}. A month outside this window holds no "
        "framing value. It does not hold a framing value of zero."
    )
    if st.session_state.get("show_tables", True):
        table = frame.loc[:, ["month", *columns]].rename(columns={"month": "Month"})
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(name, format="%.4f")
                for name in columns
            },
        )


def _render_calibration(palette, panel, status: PanelStatus) -> None:  # pragma: no cover
    """Panel 8. The factual-calibration measure of the brief."""
    st = _streamlit()
    if not status.available:
        render_unavailable(
            status,
            "Factual calibration asks whether a numerical claim agrees with "
            "the official value that was available at the date of publication. "
            "`nlp_media_svd` holds the claim lexicon and the `calibration_gap` "
            "function for this measure, and the monthly file must carry the "
            "result before this panel can show it. The panel shows no chart "
            "now. A count of numerical claims is not a calibration measure: a "
            "calibration measure needs the official value of each claim.",
        )
        return

    columns = present_columns(panel.columns, CALIBRATION_ALTERNATIVES)
    frame = prepare_measure_series(panel, columns)
    column = columns[0]
    st.plotly_chart(
        figure_media_tone(
            frame,
            palette,
            calibration_title(frame, column),
            column=column,
            y_title="Factual calibration",
            subtitle=(
                "The measure compares a numerical claim with the official "
                "value of the same month."
            ),
            zero_line=True,
            events=marked_events(),
        ),
        width="stretch",
        theme=None,
        key="calibration_chart",
    )
    st.caption(
        "A value near to zero is coverage that agrees with the official value. "
        "A value away from zero is coverage that states a different magnitude. "
        "A difference is not proof of an intention: a claim can come from an "
        "older release, from a revision, or from a different definition of the "
        "same measure."
    )
    if st.session_state.get("show_tables", True):
        table = frame.loc[:, ["month", *columns]].rename(columns={"month": "Month"})
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(name, format="%.4f")
                for name in columns
            },
        )


def _render_events(palette, gaps, status: PanelStatus) -> None:  # pragma: no cover
    """Panel 9. The economic releases and the major events."""
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    events = events_frame()
    marked = events_frame(marked_events())
    st.markdown(
        f"**{len(events)} dated events, of which {len(marked)} get a line on "
        "the time-series charts**"
    )
    st.caption(
        "Each row gives the source of its date. A chart shows a line for the "
        "two election dates, for the survey transition and for the first rate "
        "cut of the cycle. A chart with a line for each event would carry no "
        "information, and every event stays in this table."
    )
    st.dataframe(
        events.loc[:, ["month", "event", "kind", "marked", "note", "source"]],
        width="stretch",
        hide_index=True,
        column_config={
            "month": st.column_config.TextColumn("Month"),
            "event": st.column_config.TextColumn("Event", width="medium"),
            "kind": st.column_config.TextColumn("Kind"),
            "marked": st.column_config.CheckboxColumn("On the charts"),
            "note": st.column_config.TextColumn("Remark", width="medium"),
            "source": st.column_config.TextColumn("Source", width="large"),
        },
    )
    st.caption(
        f"The FOMC dates up to {FOMC_VERIFIED_THROUGH} were checked against "
        "the published calendar of that year. A later date comes from the "
        "calendar that the Federal Reserve publishes in advance. Check a 2026 "
        "date against that calendar before you quote it."
    )
    st.warning(
        "This table holds no individual CPI release day and no individual PCE "
        "release day. BLS and BEA publish a schedule for each year, and this "
        "project did not copy those schedules. The table below gives what the "
        "model needs instead: the publication lag of each series, with its "
        "source."
    )

    st.markdown("**The release schedule of each series**")
    schedule = pd.DataFrame(
        [
            {
                "series": spec.series,
                "publisher": spec.publisher,
                "reference period": spec.reference_period,
                "typical lag, days": spec.lag_days,
                "source": spec.source,
            }
            for spec in RELEASE_SCHEDULE
        ]
    )
    st.dataframe(
        schedule,
        width="stretch",
        hide_index=True,
        column_config={"source": st.column_config.TextColumn("Source", width="large")},
    )

    recent = [stamp for stamp in gaps.index[-6:]]
    if recent:
        st.markdown("**The information set of the last six months**")
        st.caption(
            "A value of a reference month is not available in that month. The "
            "table gives the publication month of each series, from the lag "
            "above. The model must use only the values that a publisher had "
            "released by the prediction date."
        )
        st.dataframe(
            release_calendar(recent),
            width="stretch",
            hide_index=True,
        )


def _render_composition(
    palette, headlines, status: PanelStatus, cached_composition=build_composition
) -> None:  # pragma: no cover
    """Panel 10. The composition of the corpus by source."""
    st = _streamlit()
    if not status.available or headlines is None or headlines.empty:
        render_unavailable(
            status,
            "The composition comes from the domain of each headline. No "
            "smaller file holds the domain of an article, so this panel needs "
            "the headline corpus itself.",
        )
        return

    report, composition = cached_composition(headlines, len(headlines))
    st.plotly_chart(
        figure_domain_composition(composition, palette, composition_title(report)),
        width="stretch",
        theme=None,
        key="composition_chart",
    )
    if report.concentrated:
        st.warning(report.note())
    else:
        st.info(report.note())
    st.caption(
        "The corpus of the tone series is a different corpus: the tone, the "
        "salience and the attribution come from the GDELT GKG theme files, "
        "which hold no domain name of this form. The composition here is the "
        "composition of the headline set, which gives the SVD terms. A reader "
        "must not read one composition as the composition of the other corpus."
    )
    if st.session_state.get("show_tables", True):
        table = composition.rename(
            columns={
                "domain": "Domain",
                "n_headlines": "Headlines",
                "share": "Share",
                "n_domains": "Domains in the row",
                "n_hosts": "Hosts",
            }
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Headlines": st.column_config.NumberColumn("Headlines", format="%d"),
                "Share": st.column_config.NumberColumn("Share", format="%.3f"),
                "Hosts": st.column_config.NumberColumn("Hosts", format="%d"),
            },
        )


def _render_coverage(palette, media, status: PanelStatus) -> None:  # pragma: no cover
    """Panel 11. The months with thin coverage."""
    st = _streamlit()
    if not status.available or media is None or "n_articles" not in getattr(media, "columns", ()):
        render_unavailable(
            status,
            "The coverage of a month is the count of articles of that month. "
            "The monthly media file carries that count in the column "
            "`n_articles`.",
        )
        return

    report = coverage_report(media)
    coverage = prepare_coverage(media)
    st.plotly_chart(
        figure_monthly_coverage(
            coverage, palette, coverage_title(report), threshold=report.threshold
        ),
        width="stretch",
        theme=None,
        key="coverage_chart",
    )
    st.caption(report.note())
    if report.thin_months:
        st.caption("The thin months: " + ", ".join(report.thin_months) + ".")
    if st.session_state.get("show_tables", True):
        table = coverage.loc[:, ["month", "n_articles"]].rename(
            columns={"month": "Month", "n_articles": "Articles"}
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Articles": st.column_config.NumberColumn("Articles", format="%d")
            },
        )


def _render_quality(
    palette, panel, gaps, fit, prediction_report, status: PanelStatus
) -> None:  # pragma: no cover
    """Panel 12. The missing months and the performance of the model."""
    st = _streamlit()
    if not status.available:
        render_unavailable(status)
        return

    try:
        performance = model_performance(fit, gaps)
    except KeyError as exc:
        st.error(f"The performance report did not run: {exc}")
        return
    missing = missing_month_table(panel, gaps)

    st.markdown(f"**{quality_title(performance, missing)}**")
    st.markdown(
        stat_tiles_html(performance_tiles(performance)), unsafe_allow_html=True
    )
    st.caption(performance.note())

    st.markdown("**The months without a gap value, and the cause of each one**")
    if missing.empty:
        st.caption("Each out-of-sample month holds a gap value.")
    else:
        st.dataframe(
            missing,
            width="stretch",
            hide_index=True,
            column_config={
                "month": st.column_config.TextColumn("Month"),
                "gap value": st.column_config.TextColumn("Gap value"),
                "absent inputs": st.column_config.TextColumn(
                    "Absent inputs", width="medium"
                ),
            },
        )
    st.caption(
        "A month without a gap value is not a month with a gap of zero. The "
        "model cannot predict a month without every regressor of the model, "
        "and the table names the regressor that is absent."
    )
    for note in getattr(prediction_report, "dropped_reasons", ()):
        st.caption(note)

    st.markdown("**The missing values of each column of the panel**")
    st.dataframe(
        panel_missingness(panel),
        width="stretch",
        hide_index=True,
        column_config={
            "column": st.column_config.TextColumn("Column"),
            "months with a value": st.column_config.NumberColumn(
                "Months with a value", format="%d"
            ),
            "months without a value": st.column_config.NumberColumn(
                "Months without a value", format="%d"
            ),
            "first month": st.column_config.TextColumn("First month"),
            "last month": st.column_config.TextColumn("Last month"),
        },
    )
    st.caption(
        "A column that starts late is not a column with a value of zero before "
        "that month. The media columns start in 2017-01, because the corpus "
        "starts there."
    )


def _render_demographics(status: PanelStatus) -> None:  # pragma: no cover
    """Panel 13. The split that the data do not permit."""
    st = _streamlit()
    if not status.available:
        render_unavailable(status, DEMOGRAPHIC_NOTE + " " + DEMOGRAPHIC_PLAN)
        return

    # This branch runs only if a future build joins the microdata columns. The
    # panel then still shows no split: a split needs the survey weights and the
    # sample size of each group, and a reviewer must check both first.
    st = _streamlit()
    with st.container(border=True):
        st.markdown(
            f'<h2 class="pb-section">{escape(status.label)}</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "The demographic columns are present now. This dashboard shows no "
            "split yet. A split needs the survey weight of each group and the "
            "sample size of each group, because a small group gives a wide "
            "prediction interval."
        )
        st.caption(DEMOGRAPHIC_PLAN)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------
#
# **This block is the last block of the file.** Streamlit runs this file as the
# main module, so the call below runs at the position of the call. An earlier
# version put this block in the middle of the file, and `main` then called
# `_render_scree` before Python defined it. The import of the module and the
# tests gave no error, because Python resolves a name at call time. Only a
# person who opened tab 4 saw the failure. A test holds the position of this
# block for that reason.
if __name__ == "__main__":  # pragma: no cover
    main()
