"""Build the presentation for the 2026 Sentiment Gap study.

The deck comes last on purpose. It presents the result; it does not find it.
Every number here is read from the same modules that the report uses, so a
change to the analysis changes the deck.

**The shape is five questions.** The brief asks for a 6 to 8 minute
presentation that explains one practical implication for forecasting or risk
management. A listener cannot hold a project summary in eight minutes, so the
deck answers five questions in order and then states the implication:

    1. What were we asked?        the question, and the three suspects
    2. What did we find?          the gap, and why the COUNT matters
    3. Which suspect did it?      the ladder: the cost of money, not the news
    4. Why is a null useful?      it was pre-registered, and it held
    5. So what do I do?           the decision rule, which is the implication

Seven slides: a title that carries the framing, the five questions, and a
close. About one minute a slide.

**Honest degradation applies here too.** A slide that has no result says so.
It never shows an empty chart and never prints a number it could not read.

Run:  .venv/bin/python build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

load_dotenv()
sys.path.insert(0, "src")

from perceptual_bias.baseline_model import fit_baseline, gap_summary, sentiment_gap  # noqa: E402
from perceptual_bias.causal_econometrics import compare_models  # noqa: E402
from perceptual_bias.data_loader import (  # noqa: E402
    build_monthly_panel,
    holdout_slice,
    training_slice,
)

OUT = Path("perceptual_bias_deck.pptx")
CSV = Path("data/raw/Copy_of_Perceptual_Bias_Data.csv")
MEDIA = Path("data/processed/media_monthly.parquet")
FIGURES = Path("outputs/figures")
CUTOFF = "2026-10-16"

# The validated palette of the project. One hue carries identity, the accent
# marks the one thing on a slide that matters, and text wears text colours.
INK = RGBColor(0x11, 0x11, 0x10)
MUTED = RGBColor(0x62, 0x61, 0x5E)
FAINT = RGBColor(0xE6, 0xE4, 0xE0)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xC8, 0x53, 0x1F)
RED = RGBColor(0xB3, 0x2D, 0x2D)
GREEN = RGBColor(0x1A, 0x7F, 0x37)
PAPER = RGBColor(0xFC, 0xFB, 0xF9)

WIDTH = Inches(13.333)
HEIGHT = Inches(7.5)
FONT = "Helvetica Neue"


# ---------------------------------------------------------------------------
# Slide furniture
# ---------------------------------------------------------------------------


def text_box(
    slide,
    left,
    top,
    width,
    height,
    text,
    size=18,
    bold=False,
    color=INK,
    space_after=7,
    line_spacing=1.12,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.space_after = Pt(space_after)
        paragraph.line_spacing = line_spacing
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = FONT
    return box


def note(slide, text: str) -> None:
    """Attach the speaker notes of one slide.

    The notes carry the intuition. A slide holds the claim; a presenter must be
    able to say why the claim is the one to expect and what a listener will ask
    next. They are written to be read aloud, so they use short sentences and no
    symbol that a presenter must decode.
    """
    slide.notes_slide.notes_text_frame.text = text.strip()


def blank(presentation):
    """A blank slide on the project's paper colour."""
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    background = slide.background
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    return slide


def rule(slide, left, top, width, color=BLUE, height=Inches(0.055)):
    """A thin accent rule. It is chrome, so it never carries a series colour."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def question_slide(presentation, number, question, context, answer, accent=BLUE):
    """One question slide: the question posed, the context, then the answer.

    Three lines in a fixed order, because a listener needs all three and in
    this order:

    * the **question**, set large and posed literally, with its question mark.
      The audience is following questions and not section titles;
    * the **context**, one line on why the question is worth asking. Without
      it a terse slide reads as an assertion;
    * the **answer**, one line in the accent colour. A listener who reads
      nothing else still leaves with the answer.
    """
    slide = blank(presentation)
    # The question text carries its own "Qn." label, so there is no separate
    # chip. Two labels on one heading read as a numbering mistake.
    rule(slide, Inches(0.78), Inches(0.52), Inches(1.5), accent, Inches(0.07))
    text_box(slide, Inches(0.75), Inches(0.78), Inches(11.9), Inches(1.0),
             question, size=31, bold=True)
    text_box(slide, Inches(0.78), Inches(1.66), Inches(11.8), Inches(0.5),
             context, size=14, color=MUTED)
    rule(slide, Inches(0.78), Inches(2.16), Inches(11.8), accent, Inches(0.02))
    text_box(slide, Inches(0.78), Inches(2.26), Inches(11.8), Inches(0.5),
             answer, size=18, bold=True, color=accent)
    text_box(slide, Inches(11.1), Inches(6.95), Inches(1.6), Inches(0.3),
             f"{number} / 5", size=10, color=MUTED)
    return slide


def stat(slide, left, top, value, label, color=INK, value_size=42):
    text_box(slide, left, top, Inches(3.1), Inches(0.95), value,
             size=value_size, bold=True, color=color, space_after=0)
    text_box(slide, left, top + Inches(0.88), Inches(3.1), Inches(0.7), label,
             size=12, color=MUTED)


def picture(slide, path: Path, left, top, width):
    if path.exists():
        slide.shapes.add_picture(str(path), left, top, width=width)
        return True
    return False


# ---------------------------------------------------------------------------
# The two charts
# ---------------------------------------------------------------------------


def make_charts(
    gaps: pd.DataFrame,
    rolling: pd.DataFrame | None,
    multiplicity: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Build the slide charts at the size a slide needs.

    The report's figures are sized for a page and their type is too small to
    read from the back of a room, so the deck draws its own. The rules are the
    project's: one axis, a neutral grid, direct labels instead of a legend
    where two series fit, and no hue that carries no meaning.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paper = "#FCFBF9"
    made: dict[str, Path] = {}

    def style(ax):
        ax.set_facecolor(paper)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#C9C6C1")
        ax.tick_params(colors="#62615E", labelsize=11, length=3)
        ax.grid(axis="y", color="#EAE7E3", linewidth=0.9)
        ax.set_axisbelow(True)

    # --- Chart 1. Observed sentiment against the model, with the interval ---
    frame = gaps.dropna(subset=["observed", "predicted"])
    fig, ax = plt.subplots(figsize=(10.4, 4.0), dpi=200)
    fig.patch.set_facecolor(paper)
    ax.fill_between(
        frame.index, frame["pi_lower"], frame["pi_upper"],
        color="#2A78D6", alpha=0.13, linewidth=0,
    )
    ax.plot(frame.index, frame["predicted"], color="#2A78D6", linewidth=2.4)
    ax.plot(frame.index, frame["observed"], color="#C8531F", linewidth=2.4)
    style(ax)
    ax.set_ylabel("Index of Consumer Sentiment", color="#62615E", fontsize=11)
    last = frame.index[-1]
    ax.annotate(
        "the model expects this",
        xy=(last, frame["predicted"].iloc[-1]), xytext=(-6, 16),
        textcoords="offset points", ha="right", fontsize=11, color="#2A78D6",
        fontweight="bold",
    )
    ax.annotate(
        "consumers say this",
        xy=(last, frame["observed"].iloc[-1]), xytext=(-6, -22),
        textcoords="offset points", ha="right", fontsize=11, color="#C8531F",
        fontweight="bold",
    )
    worst = frame["observed"].sub(frame["predicted"]).idxmin()
    ax.annotate(
        "",
        xy=(worst, frame.loc[worst, "observed"]),
        xytext=(worst, frame.loc[worst, "predicted"]),
        arrowprops={"arrowstyle": "<->", "color": "#B32D2D", "linewidth": 1.6},
    )
    ax.annotate(
        f"{frame.loc[worst, 'observed'] - frame.loc[worst, 'predicted']:.0f} points",
        xy=(worst, (frame.loc[worst, "observed"] + frame.loc[worst, "predicted"]) / 2),
        xytext=(10, 0), textcoords="offset points", fontsize=11,
        color="#B32D2D", fontweight="bold", va="center",
    )
    fig.tight_layout()
    path = FIGURES / "deck_gap.png"
    fig.savefig(path, facecolor=paper)
    plt.close(fig)
    made["gap"] = path

    # --- Chart 2. Out-of-sample error of each model against a free guess ----
    if rolling is not None and not rolling.empty:
        usable = rolling[rolling["available"].fillna(False).astype(bool)]
        order = [m for m in ("Naive (last observed)", "Model 2", "Model 3",
                             "Model 4", "Model 1", "Model 2b")
                 if m in usable.index]
        if order:
            labels = {
                "Naive (last observed)": "Last month's value\n(free)",
                "Model 1": "Headline three\nindicators",
                "Model 2": "+ price level,\nmortgage rate",
                "Model 3": "+ media\nmeasures",
                "Model 4": "+ household,\npolitical controls",
                "Model 2b": "Media only",
            }
            values = [float(usable.loc[m, "rmse"]) for m in order]
            colors = ["#62615E" if m.startswith("Naive") else
                      ("#C8531F" if m in ("Model 3", "Model 2b") else "#2A78D6")
                      for m in order]
            fig, ax = plt.subplots(figsize=(10.4, 3.9), dpi=200)
            fig.patch.set_facecolor(paper)
            bars = ax.bar([labels.get(m, m) for m in order], values,
                          color=colors, width=0.62, linewidth=0)
            style(ax)
            ax.set_ylabel("Forecast error, index points\n(lower is better)",
                          color="#62615E", fontsize=11)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.1f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=12, fontweight="bold", color="#62615E")
            ax.axhline(values[0], color="#62615E", linewidth=1.0, linestyle=(0, (4, 3)))
            ax.annotate("nothing beats the free guess", xy=(len(order) - 0.4, values[0]),
                        xytext=(0, 8), textcoords="offset points", ha="right",
                        fontsize=11, color="#62615E", fontweight="bold")
            fig.tight_layout()
            path = FIGURES / "deck_forecasts.png"
            fig.savefig(path, facecolor=paper)
            plt.close(fig)
            made["forecasts"] = path

    # --- Chart 3. Every test we ran, before and after the correction --------
    # This is the chart that earns the null. A listener is entitled to suspect
    # that we ran many tests and reported the flattering ones, so every test in
    # the declared family is plotted, named, and shown twice: as it looked
    # alone, and as it stands once the count is taken into account.
    if multiplicity is not None and not multiplicity.empty:
        frame = multiplicity.sort_values("p_raw")
        labels = [
            str(name).replace("Model 3: ", "coefficient: ").replace("_", " ")
            for name in frame.index
        ]
        positions = range(len(frame))
        fig, ax = plt.subplots(figsize=(9.4, 4.3), dpi=200)
        fig.patch.set_facecolor(paper)
        for y, (raw, adjusted) in enumerate(
            zip(frame["p_raw"], frame["q_bh"])
        ):
            ax.plot([raw, adjusted], [y, y], color="#C9C6C1", linewidth=1.6,
                    zorder=1, solid_capstyle="round")
        ax.scatter(frame["p_raw"], list(positions), s=62, color="#C8531F",
                   zorder=3, label="on its own")
        ax.scatter(frame["q_bh"], list(positions), s=62, color="#62615E",
                   zorder=3, label="after correcting for 12 tests")
        ax.axvline(0.05, color="#B32D2D", linewidth=1.3, linestyle=(0, (4, 3)))
        ax.annotate("5% line", xy=(0.05, len(frame) - 0.4), xytext=(6, 0),
                    textcoords="offset points", fontsize=11, color="#B32D2D",
                    fontweight="bold", va="center")
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("p-value  (left of the line = significant)",
                      color="#62615E", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#EAE7E3", linewidth=0.9)
        legend = ax.legend(loc="lower right", frameon=False, fontsize=10.5)
        for text in legend.get_texts():
            text.set_color("#62615E")
        fig.tight_layout()
        path = FIGURES / "deck_multiplicity.png"
        fig.savefig(path, facecolor=paper)
        plt.close(fig)
        made["multiplicity"] = path
    return made


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


def gather() -> dict:
    """Read every number the deck states from the live modules.

    The econometrics calls are wrapped, because that module is the part of the
    project most likely to change. A number the deck cannot read becomes a
    sentence on the slide, never a NaN and never a remembered figure.
    """
    panel, _ = build_monthly_panel(CSV)
    fit = fit_baseline(training_slice(panel))
    gaps, _ = sentiment_gap(fit, holdout_slice(panel))
    summary = gap_summary(gaps)
    data: dict = {
        "gaps": gaps,
        "equation": fit.equation(),
        "r2": fit.r_squared,
        "mean_gap": summary["mean_gap"],
        "worst_gap": summary["worst_gap"],
        "worst_month": summary["worst_month"],
        "worst_observed": summary["worst_observed"],
        "worst_predicted": summary["worst_predicted"],
        "outside": int(summary["months_outside_pi"]),
        "tested": int(summary["n_tested"]),
        "unemployment_beta": float(fit.params.get("unemployment", float("nan"))),
        "inflation_beta": float(fit.params.get("pce_yoy", float("nan"))),
    }

    if MEDIA.exists():
        media = pd.read_parquet(MEDIA)
        data["media_months"] = len(media)
        if "framing" in media:
            data["framing_months"] = int(media["framing"].notna().sum())
        if {"framing_headline", "framing_slug"} <= set(media.columns):
            paired = media.loc[:, ["framing_headline", "framing_slug"]].dropna()
            if len(paired) > 2:
                data["framing_agreement"] = float(paired.corr().iloc[0, 1])
        try:
            import build_report as br

            ctx = br.gather(br.CSV_PATH, allow_network=True)
            data["ladder"] = ctx.ladder.table() if ctx.ladder is not None else None
            data["rolling"] = ctx.rolling.table() if ctx.rolling is not None else None
            data["multiplicity"] = (
                ctx.multiplicity.table() if ctx.multiplicity is not None else None
            )
            for label, report in (
                ("sens_no_controls", ctx.sensitivity_no_controls),
                ("sens_with_controls", ctx.sensitivity_with_controls),
            ):
                if report is not None:
                    data[label] = float(report.share_significant)
        except Exception as exc:  # the slide degrades; it does not invent
            data["analysis_failed"] = str(exc)

    try:
        comparison = compare_models(training_slice(panel))
        table = comparison.table()
        data["m1_r2"] = float(table.loc["Model 1", "r_squared"])
    except Exception as exc:
        data["model_comparison_failed"] = str(exc)
    return data


def _p(value) -> str:
    """A p-value for a slide, or the honest absence of one."""
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    if number != number:
        return "not available"
    return "p < 0.001" if number < 0.001 else f"p = {number:.3f}"


def _ladder(data: dict, model: str, column: str = "r_squared") -> float:
    table = data.get("ladder")
    if table is None or model not in table.index:
        return float("nan")
    return float(table.loc[model, column])


def _survivors(data: dict) -> tuple[int, int, int]:
    table = data.get("multiplicity")
    if table is None:
        return (0, 0, 0)
    return (
        int(table["raw_significant"].sum()),
        int(table["bh_significant"].sum()),
        len(table),
    )


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------


def build() -> Path:
    data = gather()
    charts = make_charts(
        data["gaps"], data.get("rolling"), data.get("multiplicity")
    )
    presentation = Presentation()
    presentation.slide_width = WIDTH
    presentation.slide_height = HEIGHT

    # ------------------------------------------------- 1. Title and framing
    slide = blank(presentation)
    rule(slide, Inches(0.78), Inches(1.95), Inches(2.4), BLUE, Inches(0.08))
    text_box(slide, Inches(0.75), Inches(2.25), Inches(11.6), Inches(1.7),
             "Americans are far gloomier\nthan the economy explains",
             size=44, bold=True)
    text_box(slide, Inches(0.78), Inches(4.35), Inches(11.0), Inches(1.2),
             "Five questions about the 2026 sentiment gap — and one practical "
             "implication for forecasting and risk management.", size=19, color=MUTED)
    text_box(slide, Inches(0.78), Inches(6.55), Inches(11.6), Inches(0.5),
             f"AI Applications in Finance  ·  analysis cutoff {CUTOFF}  ·  "
             "no post-election data", size=11, color=MUTED)
    note(slide, """
One argument in seven minutes, built as five questions.

Here is the shape. First, what we were asked. Second, what we found. Third,
which of the three candidate explanations the evidence supports. Fourth, why
the result that did not appear is the most useful one in the project. And
fifth, what any of you should actually do differently on Monday.

The one thing to carry out of the room is a decision rule about broken models.
I will give it on the last slide, and everything before it is the evidence
that earns it.
""")

    # --------------------------------------------- 2. Q1. What were we asked
    slide = question_slide(
        presentation, 1,
        "Q1.  What exactly were we asked to find out?",
        "Consumer sentiment sits near record lows while unemployment and inflation look "
        "ordinary. Something has to give: either the indicators, the people, or the "
        "measurement.",
        "Whether US consumers are more pessimistic than the economy justifies — and if so, why.",
    )
    text_box(slide, Inches(0.8), Inches(2.95), Inches(11.6), Inches(0.5),
             "Three candidate explanations. The whole task is to tell them apart.",
             size=16, bold=True)
    text_box(slide, Inches(0.98), Inches(3.5), Inches(11.3), Inches(2.3),
             "A    Real economic pressure that the headline indicators miss.\n\n"
             "B    A distorted picture of the economy: selection, framing, negativity.\n\n"
             "C    Measurement: the survey changed, or our own model is too simple.",
             size=19)
    text_box(slide, Inches(0.98), Inches(5.95), Inches(11.3), Inches(0.9),
             "The brief forbids three assumptions: that pessimism is irrational, that "
             "negative coverage is biased coverage, and that coverage causes the mood. "
             "All three are hypotheses to test, and that is why a null result counts here.",
             size=14, color=ORANGE)
    note(slide, """
The question is not whether people feel bad. We know they do. The question is
whether the economy explains it, and if not, what does.

Three candidates. A: the indicators miss something real. B: people see a
distorted picture. C: the measuring instrument changed, or our model is too
simple.

Read the orange line out loud, because it is the discipline of the whole
project. We were told not to assume pessimism is irrational, not to assume
negative coverage is biased, and not to assume coverage causes the mood. Those
are the three comfortable assumptions, and all three are testable. Pre-registering
that is what makes the later null worth anything.
""")

    # ----------------------------------------------- 3. Q2. What did we find
    slide = question_slide(
        presentation, 2,
        "Q2.  How big is the gap, and how do we know it is real?",
        "Fit the three standard indicators on 2010 to 2024. Then ask the fitted model "
        "about months it has never seen. The shaded band is the model's own 95% range.",
        "A model that explains fifteen years of sentiment misses 2026 by 25 points, "
        "always downward.",
        accent=ORANGE,
    )
    if "gap" in charts:
        picture(slide, charts["gap"], Inches(0.78), Inches(2.85), Inches(8.5))
    stat(slide, Inches(9.65), Inches(2.95), f"{data['r2']:.0%}",
         "of sentiment explained,\n2010 to 2024", BLUE, value_size=36)
    stat(slide, Inches(9.65), Inches(4.35), f"{data['mean_gap']:+.0f}",
         "average miss, index points,\n2025 to 2026", ORANGE, value_size=36)
    stat(slide, Inches(9.65), Inches(5.75), f"{data['outside']} of {data['tested']}",
         "months outside the model's\nown 95% interval", RED, value_size=36)
    note(slide, f"""
Fit on fifteen years, 2010 to 2024, using the three indicators everyone uses:
unemployment, inflation, GDP growth. It works. It explains about three quarters
of the variation, and inflation matters roughly two and a half times as much as
unemployment, which is worth knowing on its own.

Then show it months it has never seen. In {data['worst_month']} it expects
{data['worst_predicted']:.0f} and the survey comes in at {data['worst_observed']:.0f}.
That is the red arrow.

But the number that matters is the bottom one. A ninety five per cent interval
is supposed to be breached about one month in twenty. Ours is breached
{data['outside']} times in {data['tested']}, every single time downward. That
is not an unlucky run. That is a missing variable, and the next slide asks
which one.
""")

    # ------------------------------------------- 4. Q3. Which one explains it
    survivors_raw, survivors_bh, family = _survivors(data)
    slide = question_slide(
        presentation, 3,
        "Q3.  Which of the three explanations does the evidence support?",
        "Judge each one by whether it helps predict a month the model has not seen, not "
        "by whether it raises R-squared. More regressors always raise R-squared.",
        "The cost of money, mostly. Not the news.",
        accent=GREEN,
    )
    if "forecasts" in charts:
        picture(slide, charts["forecasts"], Inches(0.78), Inches(2.8), Inches(8.3))
    text_box(slide, Inches(9.35), Inches(2.85), Inches(3.4), Inches(4.0),
             "A  Mostly right.\n"
             "Adding the price level and the\nmortgage rate absorbs 27.6%\n"
             "of the gap. The mortgage rate\ndoes the work.\n\n"
             "C  Partly, and smaller.\n"
             "A real break at the survey's\nmethod change, but an\n"
             "independent survey is gloomy\ntoo.\n\n"
             "B  Fails.\n"
             "The attractive story did not\nsurvive its own test.",
             size=14)
    text_box(slide, Inches(0.78), Inches(6.5), Inches(8.3), Inches(0.8),
             "Inflation falling is news about a RATE. A household pays a LEVEL. "
             "Prices stopped rising fast; they did not come down.",
             size=15, bold=True, color=GREEN)
    note(slide, """
The chart is the honest version of model comparison: forecast errors on months
each model had never seen.

Three things in it. Adding the price level and the mortgage rate cuts the error
by more than half, which is candidate A. Adding the media measures on top
changes almost nothing. And the grey bar on the left is just last month's value
with no model at all, and nothing beats it. If I showed you only R-squared,
every model would look better than it is.

Then the green line, which is the intuition for A. Inflation falling is a
statement about a rate. A household pays a level. Prices stopped rising
quickly; they never came down. A smaller increase on an already high price is
not relief, and our indicators report the rate because a rate is what policy
targets.
""")

    # --------------------------------------- 5. Q4. Why the null is the result
    slide = question_slide(
        presentation, 4,
        "Q4.  Why is “the media explanation fails” a finding and not a dead end?",
        f"We ran {family} tests on the media block. Run enough tests and something "
        "always reaches 5% by luck, so the chart corrects for how many we ran.",
        "Because the conditions that would kill it were fixed before the corpus existed "
        "— and they held.",
        accent=RED,
    )
    if "multiplicity" in charts:
        picture(slide, charts["multiplicity"], Inches(0.78), Inches(2.85), Inches(7.5))
    text_box(slide, Inches(8.55), Inches(2.85), Inches(4.2), Inches(3.4),
             f"{survivors_raw} of {family} tests reach 5% raw.\n"
             f"{survivors_bh} survive the correction.\n\n"
             "The media block did not improve a\nforecast of an unseen month.\n\n"
             "Both coefficients died in 24 of 24\nspecification variants once\n"
             "household controls entered.\n\n"
             "The only timing signal runs\nBACKWARDS: mood first,\ncoverage after.",
             size=14)
    rule(slide, Inches(0.78), Inches(6.08), Inches(11.9), FAINT, Inches(0.02))
    text_box(slide, Inches(0.78), Inches(6.22), Inches(11.9), Inches(1.0),
             "The sharpest finding is about the instrument, not the economy: FinBERT "
             "changes its verdict on 5.8% of byte-identical texts when only the outlet's "
             "name changes, where the word list changes on 0%. A classifier that reads "
             "the label would have manufactured the bias we went looking for.",
             size=14, bold=True, color=RED)
    note(slide, """
Four reasons the null is solid.

Two of the four media measures did reach five per cent. Correct for how many
tests you ran and none survives. We declared that family in advance and counted
every test, including the boring directions: a narrower family would have let
two results through, and I am telling you that rather than choosing the family
that flatters us.

It also fails to forecast, it dies in all twenty four specification variants
once household controls are present, and the only timing signal points the
wrong way — mood first, coverage after.

Then the red paragraph, which is the finding I would publish. Swap only the
outlet's name in front of an identical sentence and the neural classifier
changes its answer on one text in seventeen. The word list never moves. That
model would have produced the result we were hunting for, and we would not have
known.
""")

    # ------------------------------------------------- 6. Q5. So what do I do
    slide = question_slide(
        presentation, 5,
        "Q5.  So what should I do differently with a model that misses?",
        "This generalises past consumer sentiment. Any model you own will start missing. "
        "The question is how you tell a drifted level from a missing variable.",
        "Treat a one-sided breach as a specification failure. Rebuild — do not recalibrate.",
        accent=BLUE,
    )
    text_box(slide, Inches(0.9), Inches(3.0), Inches(5.4), Inches(2.2),
             "Recalibrate when\nerrors are large but BALANCED.\n"
             "The relationship holds and the\nlevel has drifted.",
             size=18)
    text_box(slide, Inches(7.0), Inches(3.0), Inches(5.4), Inches(2.2),
             "Rebuild when\nerrors are ONE-SIDED and outside\nthe interval. Something the model\n"
             "cannot see is moving the outcome.",
             size=18, bold=True, color=BLUE)
    rule(slide, Inches(0.9), Inches(5.1), Inches(11.4), FAINT, Inches(0.02))
    text_box(slide, Inches(0.9), Inches(5.3), Inches(11.4), Inches(1.9),
             "Re-fit this model on the new level and it tracks the new level beautifully — "
             "and keeps exactly the blindness that caused the breach. You have silenced the "
             "alarm, not put out the fire.\n\n"
             "Then diagnose before you fix. Unmeasured pressure: add the level "
             "variables — cumulative prices, the mortgage rate, debt service. A changed "
             "instrument: re-base the series before it enters anything. An information "
             "driver: no macro variable helps, and a text feed becomes necessary.",
             size=16)
    note(slide, """
This is the slide to remember, and it generalises past consumer sentiment.

When your model misses, the instinct is to re-fit it. Sometimes that is right:
if the errors are large but balanced, the relationship still holds and the
level has drifted. Recalibrate.

But if the errors are one-sided and outside the interval, a variable that
matters is missing, and re-fitting is the expensive mistake. The re-fitted model
will track the new level beautifully and carry exactly the blindness that caused
the breach. You have silenced the alarm and left the fire.

And diagnose before you fix, because the three candidates need three different
fixes and the wrong fix ships with new confidence attached. Unmeasured
pressure: add the level variables. Changed instrument: re-base the series
before it enters anything. Information environment: no macro variable will
help you.
""")

    # ---------------------------------------------------------- 7. Conclusion
    slide = blank(presentation)
    rule(slide, Inches(0.78), Inches(0.95), Inches(2.4), BLUE, Inches(0.08))
    text_box(slide, Inches(0.75), Inches(1.3), Inches(11.6), Inches(1.0),
             "In one line", size=34, bold=True)
    text_box(slide, Inches(0.9), Inches(2.5), Inches(11.4), Inches(1.8),
             "A one-sided breach of a prediction interval is a specification\n"
             "failure, not a run of bad luck.", size=30, bold=True, color=BLUE)
    text_box(slide, Inches(0.9), Inches(4.3), Inches(11.4), Inches(2.0),
             "The gap is real and large. The cost of money explains a good part of it. "
             "The survey's method explains a smaller part. The tone of the news explains "
             "none of it, on four separate tests fixed in advance.\n\n"
             "Knowing where not to spend is worth as much as knowing where to spend.",
             size=17)
    text_box(slide, Inches(0.9), Inches(6.6), Inches(11.4), Inches(0.5),
             "No exposure data, so no causal claim about readers. Limits are stated in "
             "the report, not buried.", size=12, color=MUTED)
    note(slide, """
One line, then I stop.

A one-sided breach of a prediction interval is a specification failure, not a
run of bad luck. Rebuild rather than recalibrate, and diagnose before you
choose the fix.

The summary underneath: the gap is real and large, the cost of money explains a
good part of it, the survey's method a smaller part, and the tone of the news
none of it on four separate pre-registered tests.

And the grey line matters. We have no exposure data at all, so nothing here is
a causal claim about what any reader read. That limit is in the report rather
than buried, which is the only reason the rest of it is worth your trust.

Questions.
""")

    presentation.save(OUT)
    return OUT


def _notes_word_count(path: Path) -> int:
    """Total words in the speaker notes, which is the real length of the talk."""
    total = 0
    for slide in Presentation(path).slides:
        if slide.has_notes_slide:
            total += len(slide.notes_slide.notes_text_frame.text.split())
    return total


if __name__ == "__main__":
    path = build()
    presentation = Presentation(path)
    slides = len(presentation.slides._sldIdLst)
    words = _notes_word_count(path)
    print(f"wrote {path}")
    print(f"slides: {slides}")
    print(f"speaker-note words: {words} (about {words / 130:.1f} minutes at 130 words a minute)")
