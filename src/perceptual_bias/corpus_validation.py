"""Corpus documentation, stratified sampling and AI-measurement validation.

This module is the code behind section 2 ("Build a balanced media corpus") and
the validation part of section 3 ("Measure coverage and candidate biases with
AI") of the assignment brief. It does five jobs:

1. `profile_corpus` measures a corpus and names every problem in it:
   duplicates, source concentration, wire and press-release content, foreign
   content, thin months and junk text. The result feeds
   `docs/corpus_documentation.md`.
2. `draw_stratified_sample` draws a reproducible stratified sample of at least
   300 items. The strata are year, source type and predicted tone class.
3. `write_annotation_files` writes the labeling instrument: a blind annotation
   CSV with empty label columns for two annotators, and a separate key file.
4. `compute_agreement` and `adjudicate` give the agreement mathematics
   (Cohen's kappa, percent agreement, agreement by class and by source) and the
   adjudication rule.
5. `compare_classifiers`, `swap_robustness_test` and `same_event_cross_source`
   give the AI-against-dictionary comparison, the name and source swap test,
   and the same-event cross-source benchmark.

**Honesty rule, and it is enforced in code.** This module cannot make human
labels. No function in this module writes a label into an annotator column. A
function that produces labels from a model must set `provenance="automated"`,
and `AgreementReport.labels_are_human` is then False and
`AgreementReport.outstanding` says that the human labels of the brief are not
done. `require_human_labels` raises `ValidationError` if a caller tries to
present automated labels as human labels. The two automated labelers (the
project lexicon and FinBERT) are a test of the agreement code. They are not a
substitute for the two human annotators.

**No look-ahead.** The sampler and the tone-class thresholds use fixed
constants. Nothing in this module fits a transform on the full sample and then
applies it to a part of the sample. `draw_stratified_sample` takes a `max_date`
argument, so a sample for a given prediction date can exclude later items.

**Nothing is dropped in silence.** Each public function returns a dataclass
that gives the count and the cause of each removed row.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from perceptual_bias.nlp_media_svd import (
    FRAMING_LEXICON,
    HEADLINE_FRAMING_CONTRAST,
    PROBE_LABELS,
    finbert_probe,
    score_framing,
    score_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ADJUDICATION_PROCEDURE",
    "ANNOTATOR_COLUMNS",
    "ATTRIBUTION_CLASSES",
    "BENCHMARK_TOPICS",
    "CALIBRATION_CLASSES",
    "DIMENSION_QUESTIONS",
    "EVENT_KEYS",
    "FRAMING_AI_LIMITATION",
    "FRAMING_BALANCED",
    "FRAMING_BALANCED_MAX",
    "FRAMING_CLASSES",
    "FRAMING_CONTRASTS",
    "FRAMING_SIDE_NAMES",
    "HEADLINE_FRAMING_CONTRAST",
    "LABEL_QUESTIONS",
    "NAME_SWAPS",
    "NOT_FRAMED",
    "OUTSTANDING_HUMAN_WORK",
    "PROVENANCE_VALUES",
    "SALIENCE_CLASSES",
    "SOURCE_LABEL_SWAPS",
    "SOURCE_TYPES",
    "TONE_CLASSES",
    "TONE_LABEL_A",
    "TONE_LABEL_B",
    "TONE_NEGATIVE_MAX",
    "TONE_POSITIVE_MIN",
    "AgreementReport",
    "BenchmarkReport",
    "ClassifierComparison",
    "CorpusProfile",
    "SampleReport",
    "StratifiedSample",
    "SwapRobustnessReport",
    "ValidationError",
    "adjudicate",
    "agreement_by_dimension",
    "agreement_for_question",
    "annotator_pair",
    "classify_source",
    "cohen_kappa",
    "compare_classifiers",
    "compute_agreement",
    "domain_of",
    "dominant_topic",
    "draw_stratified_sample",
    "finbert_tone_labels",
    "framing_class_from_ratio",
    "framing_labeler",
    "join_labels",
    "lexicon_framing_labels",
    "lexicon_tone_labels",
    "normalise_text",
    "percent_agreement",
    "predict_framing_class",
    "predict_framing_classes",
    "predict_tone_class",
    "profile_corpus",
    "require_human_labels",
    "same_event_cross_source",
    "swap_robustness_test",
    "swap_text",
    "write_annotation_files",
]


# --------------------------------------------------------------------------
# 0. Errors and configuration
# --------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a corpus audit or a label set is not as specified.

    The module raises this class before pandas raises a `KeyError`, and it
    raises this class when a caller tries to call automated labels human
    labels. Refer to `require_human_labels`.
    """


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValidationError(f"frame is missing required columns: {missing}")


#: The three tone classes of the coding scheme. Refer to
#: `docs/labeling_guide.md`. The order is fixed, because it is the row and
#: column order of each confusion matrix.
TONE_CLASSES: tuple[str, str, str] = ("negative", "neutral", "positive")

#: Thresholds that convert a continuous tone score in [-1, +1] into a class.
#: These are fixed constants and not sample quantiles. A quantile threshold
#: would use the full sample to make the class of each item, which is
#: look-ahead and which also makes the class prevalence an artefact of the
#: sample.
TONE_POSITIVE_MIN = 0.20
TONE_NEGATIVE_MAX = -0.20

#: Source types. "other" holds local outlets and small outlets. The headline
#: corpus has 5132 distinct domains, and a hand-made list cannot name the long
#: tail. The documentation gives this limitation.
SOURCE_TYPES: tuple[str, ...] = (
    "national_news",
    "aggregator",
    "wire_or_pr",
    "algo_finance",
    "foreign",
    "social",
    "other",
)

#: Generic top-level domains. A domain with a different top-level domain counts
#: as foreign, because GDELT collects worldwide coverage and the project needs
#: US coverage.
_GENERIC_TLDS: frozenset[str] = frozenset(
    {"com", "org", "net", "gov", "edu", "us", "io", "co", "info", "biz", "tv", "fm", "news"}
)

_WIRE_OR_PR: tuple[str, ...] = (
    "prnewswire",
    "globenewswire",
    "businesswire",
    "business-wire",
    "newswire",
    "einpresswire",
    "accesswire",
    "prweb",
    "presswire",
    "newsfilecorp",
)

_AGGREGATOR: tuple[str, ...] = (
    "msn.com",
    "yahoo.com",
    "news.google",
    "google.com",
    "flipboard",
    "bignewsnetwork",
    "miragenews",
    "headtopics",
    "newsbreak",
    "apple.news",
    "smartnews",
    "onenewspage",
    "newsnow",
)

#: Domains whose articles are generated from market data by a template. These
#: are the "suspected automated activity" of the brief in the news corpus.
_ALGO_FINANCE: tuple[str, ...] = (
    "marketscreener",
    "4-traders",
    "wkrb13",
    "dailypolitical",
    "americanbankingnews",
    "tickerreport",
    "stocknewsapi",
    "zacks",
    "benzinga",
    "tipranks",
    "simplywall",
    "joker.com",
    "etfdailynews",
    "modernreaders",
    "themarketsdaily",
    "transcriptdaily",
)

_NATIONAL_NEWS: tuple[str, ...] = (
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "cnn.com",
    "foxnews.com",
    "foxbusiness.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    "usatoday.com",
    "npr.org",
    "bloomberg.com",
    "cnbc.com",
    "marketwatch.com",
    "forbes.com",
    "businessinsider.com",
    "politico.com",
    "thehill.com",
    "axios.com",
    "newsweek.com",
    "time.com",
    "latimes.com",
    "barrons.com",
    "fortune.com",
    "theatlantic.com",
    "vox.com",
    "zerohedge.com",
    "fool.com",
    "theepochtimes.com",
    "breitbart.com",
    "huffpost.com",
    "salon.com",
    "slate.com",
)

#: Outlets that are not US outlets, and whose top-level domain is generic.
_FOREIGN_NAMES: tuple[str, ...] = (
    "indiatimes",
    "business-standard",
    "menafn",
    "finanznachrichten",
    "scmp.com",
    "aljazeera",
    "theguardian",
    "bbc.co",
    "ft.com",
    "economist.com",
    "dailymail",
    "telegraph.co",
    "straitstimes",
    "japantimes",
    "koreaherald",
    "arabnews",
    "thehindu",
    "livemint",
    "moneycontrol",
    "zawya",
    "sakshieducation",
    "miragenews",
)

#: Names that the swap test exchanges. Each pair is exchanged in both
#: directions in one pass, so the text cannot collapse to one name.
NAME_SWAPS: tuple[tuple[str, str], ...] = (
    ("trump", "biden"),
    ("republican", "democrat"),
    ("republicans", "democrats"),
    ("gop", "democratic party"),
    ("powell", "yellen"),
    ("maga", "progressive"),
    ("conservative", "liberal"),
)

#: Source labels that the swap test exchanges. The label is put in front of the
#: text, which is how a label reaches a model that reads only text.
SOURCE_LABEL_SWAPS: tuple[tuple[str, str], ...] = (
    ("Fox News", "MSNBC"),
    ("Breitbart", "The Guardian"),
    ("Wall Street Journal", "New York Times"),
)

# --------------------------------------------------------------------------
# 0b. The five dimensions of the coding scheme
# --------------------------------------------------------------------------

#: The three framing contrasts, in the order of `FRAMING_LEXICON`.
FRAMING_CONTRASTS: tuple[str, ...] = tuple(FRAMING_LEXICON)

#: The human name of each side of each contrast. The left side is the positive
#: end of the ratio in `nlp_media_svd.FramingScore.ratio`, and the right side is
#: the negative end. An annotator writes one of these words, so the human label
#: and the lexicon ratio sit on the same scale and can be compared directly.
FRAMING_SIDE_NAMES: dict[str, tuple[str, str]] = {
    "level_vs_rate": ("level", "rate"),
    "household_vs_macro": ("household", "macro"),
    "current_vs_forecast": ("current", "forecast"),
}

#: The label for a document that frames neither side of a contrast.
#:
#: **This class is not "balanced".** A document that names no price and no
#: inflation rate is silent about that contrast. A document that names both
#: equally is balanced. The lexicon gives NaN for the first and 0.0 for the
#: second, and the labeling guide gives the annotator the same two answers. A
#: scheme that merges them would turn silence into balance and would make the
#: measure agree with itself by construction.
NOT_FRAMED = "not_framed"

#: The label for a document that frames both sides of a contrast about equally.
FRAMING_BALANCED = "balanced"

#: Width of the balanced band. A ratio inside +-0.20 reads as balanced. The
#: value is the same distance from zero as `TONE_POSITIVE_MIN`, so one
#: threshold argument does not have to be remembered for each dimension.
FRAMING_BALANCED_MAX = 0.20

#: The permitted labels of each framing contrast.
FRAMING_CLASSES: dict[str, tuple[str, ...]] = {
    contrast: (left, right, FRAMING_BALANCED, NOT_FRAMED)
    for contrast, (left, right) in FRAMING_SIDE_NAMES.items()
}

#: Selection and salience: which economic fact the item puts first. The list
#: is the brief's own list ("layoffs, hiring, rising prices, falling inflation,
#: growth, and recession risk") with two answers added that an annotator needs:
#: an economic item about something else, and an item that is not economic.
SALIENCE_CLASSES: tuple[str, ...] = (
    "layoffs",
    "hiring",
    "rising_prices",
    "falling_inflation",
    "growth",
    "recession_risk",
    "other_economic",
    "not_economic",
)

#: Factual calibration: whether a numerical claim matches the official data
#: that was available at publication. "no_numeric_claim" is a frequent and
#: correct answer for a headline, and it must not be read as "correct".
CALIBRATION_CLASSES: tuple[str, ...] = (
    "matches_official_data",
    "overstates",
    "understates",
    "no_numeric_claim",
    "cannot_check",
)

#: Political attribution: which actor gets the credit or the blame. The brief
#: warns that criticism is not the same thing as bias, so the scheme records
#: the direction and the actor, and it makes no judgement about fairness.
ATTRIBUTION_CLASSES: tuple[str, ...] = (
    "credit_administration",
    "blame_administration",
    "credit_federal_reserve",
    "blame_federal_reserve",
    "credit_other_actor",
    "blame_other_actor",
    "no_attribution",
)

#: The label questions of the instrument, and the permitted answers of each
#: one. The five dimensions of the brief are: tone, selection and salience,
#: framing (three contrasts, so three questions), factual calibration, and
#: political attribution. One dictionary holds all of them, so a new dimension
#: reaches the annotation file, the agreement code and the documentation
#: together and cannot reach one of the three alone.
LABEL_QUESTIONS: dict[str, tuple[str, ...]] = {
    "tone": TONE_CLASSES,
    "salience": SALIENCE_CLASSES,
    **{f"framing_{contrast}": classes for contrast, classes in FRAMING_CLASSES.items()},
    "calibration": CALIBRATION_CLASSES,
    "attribution": ATTRIBUTION_CLASSES,
}

#: The five named dimensions, and the questions that measure each one.
DIMENSION_QUESTIONS: dict[str, tuple[str, ...]] = {
    "tone": ("tone",),
    "selection_and_salience": ("salience",),
    "framing": tuple(f"framing_{contrast}" for contrast in FRAMING_CONTRASTS),
    "factual_calibration": ("calibration",),
    "political_attribution": ("attribution",),
}


def annotator_pair(question: str) -> tuple[str, str]:
    """Give the two column names of one label question.

    The names are `<question>_annotator_a` and `<question>_annotator_b`. The
    function raises for an unknown question, so a typed column name cannot
    create a silent empty column in the instrument.
    """
    if question not in LABEL_QUESTIONS:
        raise ValidationError(
            f"unknown label question {question!r}; the questions are "
            f"{tuple(LABEL_QUESTIONS)}"
        )
    return f"{question}_annotator_a", f"{question}_annotator_b"


def _all_annotator_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for question in LABEL_QUESTIONS:
        columns.extend(annotator_pair(question))
    columns.extend(("notes_annotator_a", "notes_annotator_b"))
    return tuple(columns)


#: Every annotator column of the instrument: two per label question, and one
#: note column for each annotator. A function in this module never writes a
#: value into any of them.
ANNOTATOR_COLUMNS: tuple[str, ...] = _all_annotator_columns()

#: The default tone columns. Tone is the dimension that the lexicon and FinBERT
#: can both produce, so it is the dimension of the agreement defaults.
TONE_LABEL_A, TONE_LABEL_B = annotator_pair("tone")

#: Why the AI-against-dictionary comparison cannot cover framing.
FRAMING_AI_LIMITATION = (
    "FinBERT has one output head and it gives market sentiment. It cannot give "
    "a framing contrast, and no off-the-shelf model in this project gives one. "
    "The framing dimension therefore has no neural counterpart to compare "
    "against: its validation is the human labels against the framing lexicon, "
    "and that comparison needs the human annotators."
)

#: Permitted values of the `provenance` argument.
PROVENANCE_VALUES: tuple[str, str, str] = ("human", "automated", "mixed")

#: The adjudication procedure. Two annotators label each item. This list gives
#: what to do when they do not agree.
ADJUDICATION_PROCEDURE: tuple[str, ...] = (
    ("1. Each annotator labels all items alone. No annotator sees the labels of "
    "the other annotator, and no annotator sees the model labels."),
    ("2. Calculate percent agreement and Cohen's kappa on the full set, by "
    "class and by source type, before adjudication."),
    ("3. Keep each item where the two labels agree. The agreed label becomes "
    "the gold label."),
    ("4. For each item where the two labels do not agree, a third person who "
    "did not label the item reads the item and the two notes. This person "
    "selects one of the two labels or the label 'unusable'."),
    ("5. If the third person selects 'unusable', remove the item from the gold "
    "set and record the cause. Do not substitute a model label."),
    ("6. Record each adjudication decision with the item id, the two labels, "
    "the gold label and the cause. The record is the file "
    "'outputs/labels/adjudication_log.csv'."),
    ("7. If the two annotators disagree on more than 20 percent of the items, "
    "stop. Make the decision rules in docs/labeling_guide.md more exact, then "
    "label a new sample. Do not adjudicate a bad instrument."),
    ("8. Report the agreement before adjudication. The agreement after "
    "adjudication is 100 percent by construction, and it is not a measurement."),
    ("9. Adjudicate each question of the instrument on its own. An annotator "
    "can agree about tone and disagree about framing, and one gold label for "
    "the item would hide that. Refer to LABEL_QUESTIONS."),
    ("10. For a framing question, keep 'not_framed' and 'balanced' apart in the "
    "adjudication. A third person who makes them one class destroys the "
    "measure that the framing lexicon reports as NaN against 0.0."),
)

#: What the brief asks for that this module cannot produce. Each report that
#: has no human labels carries this text, so it cannot be lost.
OUTSTANDING_HUMAN_WORK: tuple[str, ...] = (
    ("Two independent human annotators must label the 300-item sample. The "
    "columns 'label_annotator_a' and 'label_annotator_b' are empty."),
    ("Agreement before adjudication (Cohen's kappa and percent agreement) "
    "cannot be reported until the human labels exist."),
    ("AI performance by class and by source needs a human gold standard. Until "
    "then, the lexicon-against-FinBERT table is concordance between two "
    "models, and it is not accuracy."),
    ("The adjudication log needs a third person. Refer to "
    "ADJUDICATION_PROCEDURE."),
    ("The five dimensions of the coding scheme (tone, selection and salience, "
    "framing in three contrasts, factual calibration, political attribution) "
    "each need a human label. Refer to LABEL_QUESTIONS."),
    ("The framing dimension has no neural counterpart, so its only validation "
    "is human against lexicon. Refer to FRAMING_AI_LIMITATION."),
)


# --------------------------------------------------------------------------
# 1. Small text and source helpers
# --------------------------------------------------------------------------

_DOMAIN_PATTERN = re.compile(r"^(?:https?://)?([^/?#]+)", re.IGNORECASE)
_PORT_PATTERN = re.compile(r":\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def domain_of(url: str | float | None) -> str:
    """Get the host name from a URL, in lower case and without "www." or a port.

    The headline corpus holds hosts such as "example.com:443". A port makes a
    second apparent outlet from one outlet, so the port is removed. This
    function gives an empty string for a value that is not a URL. It does not
    raise, because the caller counts the empty values and gives them in a
    report.
    """
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return ""
    match = _DOMAIN_PATTERN.match(str(url).strip())
    if match is None:
        return ""
    host = match.group(1).lower()
    host = _PORT_PATTERN.sub("", host)
    return host.removeprefix("www.")


def normalise_text(text: str | float | None) -> str:
    """Make a comparison key for a text: lower case, letters and digits only.

    Syndication changes the punctuation and the case of a headline, but not the
    words. An exact match on the raw string finds 31 percent of the duplicates
    in the headline corpus. This key finds 32 percent.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    # The apostrophe is removed and not replaced by a space, which is what
    # `nlp_media_svd.tokenize` does: "what's" gives "whats" and not "what s".
    lowered = str(text).lower().replace("’", "'").replace("'", "")
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return _SPACES.sub(" ", cleaned).strip()


def classify_source(source: str | float | None, *, corpus: str = "news") -> str:
    """Put one source into one of `SOURCE_TYPES`.

    The order of the tests is important. A press-release host is tested first,
    because a press release that an aggregator carries is still a press
    release. A foreign top-level domain is tested after the US national list,
    because a US outlet can use a ".co" domain.

    `corpus="social"` gives "social" for each row. A platform account is not a
    news outlet, and the two must not share a stratum.
    """
    if corpus == "social":
        return "social"
    host = domain_of(source) or str(source or "").strip().lower()
    if not host:
        return "other"
    if any(name in host for name in _WIRE_OR_PR):
        return "wire_or_pr"
    if any(name in host for name in _ALGO_FINANCE):
        return "algo_finance"
    if any(name in host for name in _AGGREGATOR):
        return "aggregator"
    if any(host == name or host.endswith("." + name) or name in host for name in _NATIONAL_NEWS):
        return "national_news"
    if any(name in host for name in _FOREIGN_NAMES):
        return "foreign"
    tld = host.rsplit(".", 1)[-1]
    if tld and tld not in _GENERIC_TLDS:
        return "foreign"
    return "other"


def predict_tone_class(text: str | float | None) -> str:
    """Give the tone class of one text from the project lexicon.

    This is the dictionary classifier of the brief, and it is also the
    stratification variable. The thresholds are `TONE_POSITIVE_MIN` and
    `TONE_NEGATIVE_MAX`.

    A text with no polar term gets "neutral". The lexicon tone of such a text
    is 0.0, which is the same number as balanced coverage. The two conditions
    are different, and `LexiconScore.polar_total` separates them. For a class
    label the two conditions give the same class, because an annotator who
    reads a headline with no economic polarity also selects "neutral".
    """
    score = score_text("" if text is None else str(text))
    if score.tone >= TONE_POSITIVE_MIN:
        return "positive"
    if score.tone <= TONE_NEGATIVE_MAX:
        return "negative"
    return "neutral"


def framing_class_from_ratio(
    value: float, contrast: str, *, balanced_max: float = FRAMING_BALANCED_MAX
) -> str:
    """Convert one framing ratio into one of `FRAMING_CLASSES[contrast]`.

    NaN gives `NOT_FRAMED` and not `FRAMING_BALANCED`. The rule comes from
    `nlp_media_svd.FramingScore.ratio`: a document that frames nothing is not a
    document that frames both sides equally. The two answers must stay apart on
    the human scale as well, or the comparison between the human label and the
    lexicon ratio would count silence as agreement with balance.
    """
    if contrast not in FRAMING_SIDE_NAMES:
        raise ValidationError(
            f"unknown framing contrast {contrast!r}; the contrasts are {FRAMING_CONTRASTS}"
        )
    left, right = FRAMING_SIDE_NAMES[contrast]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NOT_FRAMED
    numeric = float(value)
    if numeric >= balanced_max:
        return left
    if numeric <= -balanced_max:
        return right
    return FRAMING_BALANCED


def predict_framing_class(text: str | float | None, contrast: str) -> str:
    """Framing class of one text for one contrast, from the framing lexicon.

    Negation is not applied, because `score_framing` does not apply it. The
    docstring of that function gives the cause: a framing measure has no
    valence to flip, and "prices are not higher than in 2020" still frames the
    question as a question about the price level.
    """
    score = score_framing("" if text is None else str(text))
    return framing_class_from_ratio(score.ratio(contrast), contrast)


def predict_framing_classes(text: str | float | None) -> dict[str, str]:
    """Framing class of one text for every contrast, in one pass of the text."""
    score = score_framing("" if text is None else str(text))
    return {
        contrast: framing_class_from_ratio(score.ratio(contrast), contrast)
        for contrast in FRAMING_CONTRASTS
    }


def lexicon_framing_labels(texts: Sequence[str], contrast: str) -> list[str]:
    """Framing class from the lexicon for each text. Always available."""
    return [predict_framing_class(text, contrast) for text in texts]


def framing_labeler(contrast: str) -> Callable[[Sequence[str]], list[str]]:
    """Make a labeler for one framing contrast, for the swap robustness test.

    The swap test takes a function from texts to labels. This factory gives
    that function for a framing contrast, so the name-swap test and the
    source-label test cover framing and not only tone. The brief asks whether a
    change of political name alters the classification, and framing is a
    classification.
    """
    if contrast not in FRAMING_SIDE_NAMES:
        raise ValidationError(
            f"unknown framing contrast {contrast!r}; the contrasts are {FRAMING_CONTRASTS}"
        )

    def labeler(texts: Sequence[str]) -> list[str]:
        return lexicon_framing_labels(texts, contrast)

    return labeler


def _item_id(parts: Sequence[object]) -> str:
    """Stable 12-character id from the item fields.

    The id comes from the content and not from the row number, so a new pull of
    the corpus or a different sample size gives the same id to the same item.
    An annotation file and a label file can then be joined after a re-run.
    """
    joined = "\u001f".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"itm-{digest}"


# --------------------------------------------------------------------------
# 2. Corpus profile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusProfile:
    """What one corpus holds, and what is wrong with it.

    Each share is a fraction of `rows`. `dropped_reasons` is empty here,
    because a profile removes no row: the profile counts the problems and
    leaves the rows in place. The sampler removes rows, and `SampleReport`
    names them.
    """

    name: str
    unit: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    months_present: int
    months_in_span: int
    missing_months: tuple[str, ...]
    thin_months: tuple[tuple[str, int], ...]
    median_rows_per_month: float
    n_sources: int
    top_sources: tuple[tuple[str, int], ...]
    top1_share: float
    top10_share: float
    source_hhi: float
    source_type_shares: Mapping[str, float]
    duplicate_exact_share: float
    duplicate_normalised_share: float
    duplicate_url_share: float
    wire_or_pr_share: float
    algo_finance_share: float
    foreign_share: float
    short_text_share: float
    empty_text_share: float
    engagement_columns: tuple[str, ...]
    has_exposure_data: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def thin_month_count(self) -> int:
        return len(self.thin_months)

    def summary(self) -> str:
        span = "no rows"
        if self.start is not None and self.end is not None:
            span = f"{self.start:%Y-%m} to {self.end:%Y-%m}"
        return (
            f"{self.name}: {self.rows} {self.unit} ({span}), "
            f"{self.months_present}/{self.months_in_span} months present, "
            f"{self.n_sources} sources, top-1 {self.top1_share:.1%}, "
            f"top-10 {self.top10_share:.1%}, "
            f"duplicate text {self.duplicate_normalised_share:.1%}, "
            f"wire/PR {self.wire_or_pr_share:.1%}, "
            f"template finance {self.algo_finance_share:.1%}, "
            f"foreign {self.foreign_share:.1%}, "
            f"exposure data {'yes' if self.has_exposure_data else 'no'}"
        )

    def to_frame(self) -> pd.DataFrame:
        """One-row frame, for a report table."""
        return pd.DataFrame(
            [
                {
                    "corpus": self.name,
                    "unit": self.unit,
                    "rows": self.rows,
                    "start": self.start,
                    "end": self.end,
                    "months_present": self.months_present,
                    "months_in_span": self.months_in_span,
                    "missing_months": len(self.missing_months),
                    "thin_months": self.thin_month_count,
                    "median_rows_per_month": self.median_rows_per_month,
                    "n_sources": self.n_sources,
                    "top1_share": self.top1_share,
                    "top10_share": self.top10_share,
                    "source_hhi": self.source_hhi,
                    "duplicate_exact_share": self.duplicate_exact_share,
                    "duplicate_normalised_share": self.duplicate_normalised_share,
                    "duplicate_url_share": self.duplicate_url_share,
                    "wire_or_pr_share": self.wire_or_pr_share,
                    "algo_finance_share": self.algo_finance_share,
                    "foreign_share": self.foreign_share,
                    "short_text_share": self.short_text_share,
                    "empty_text_share": self.empty_text_share,
                    "has_exposure_data": self.has_exposure_data,
                }
            ]
        )


def _month_strings(dates: pd.Series) -> pd.Series:
    return pd.Index(pd.to_datetime(dates)).to_period("M").astype(str).to_series(index=dates.index)


def profile_corpus(
    frame: pd.DataFrame,
    *,
    name: str,
    unit: str,
    date_column: str = "date",
    source_column: str = "source",
    text_column: str | None = "text",
    url_column: str | None = "url",
    engagement_columns: Sequence[str] = (),
    corpus_kind: str = "news",
    thin_month_quantile: float = 0.25,
    short_text_tokens: int = 3,
) -> CorpusProfile:
    """Measure one corpus and give the real numbers for the documentation.

    `unit` says what one row is: "headlines", "article records" or "posts". The
    brief asks for this, because a headline corpus and a full-article corpus
    answer different questions.

    A month is thin when its row count is below `thin_month_quantile` of the
    monthly counts and also below half of the median month. Both conditions are
    needed: a quantile alone calls the lowest quarter of any corpus thin, even
    a corpus with equal months.

    `has_exposure_data` is always False. The project has no data on what a
    survey respondent read. `engagement_columns` holds the columns that measure
    attention, such as tweet views and likes. Attention is not exposure, and
    the two names are kept apart for that reason.
    """
    _require_columns(frame, (date_column, source_column))
    if text_column is not None:
        _require_columns(frame, (text_column,))
    rows = len(frame)
    if rows == 0:
        raise ValidationError(f"corpus {name!r} is empty; a profile needs at least one row")

    dates = pd.to_datetime(frame[date_column])
    if getattr(dates.dtype, "tz", None) is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    months = pd.Index(dates).to_period("M")
    month_counts = pd.Series(months.astype(str)).value_counts().sort_index()
    span = pd.period_range(months.min(), months.max(), freq="M").astype(str)
    missing = tuple(month for month in span if month not in set(month_counts.index))
    median_month = float(month_counts.median())
    cut = float(month_counts.quantile(thin_month_quantile))
    thin = tuple(
        (month, int(count))
        for month, count in month_counts.items()
        if count <= cut and count < 0.5 * median_month
    )

    sources = frame[source_column].astype("string").fillna("")
    source_counts = sources.value_counts()
    shares = source_counts / rows
    hhi = float((shares**2).sum())
    top1 = float(shares.iloc[0]) if len(shares) else 0.0
    top10 = float(shares.head(10).sum())

    types = sources.map(lambda value: classify_source(value, corpus=corpus_kind))
    type_shares = {
        source_type: float((types == source_type).mean()) for source_type in SOURCE_TYPES
    }

    if text_column is None:
        duplicate_exact = float("nan")
        duplicate_norm = float("nan")
        short_share = float("nan")
        empty_share = float("nan")
    else:
        texts = frame[text_column]
        duplicate_exact = float(texts.duplicated().mean())
        keys = texts.map(normalise_text)
        duplicate_norm = float(keys.duplicated().mean())
        token_counts = keys.str.split().str.len().fillna(0)
        short_share = float((token_counts <= short_text_tokens).mean())
        empty_share = float((keys == "").mean())

    if url_column is not None and url_column in frame.columns:
        duplicate_url = float(frame[url_column].duplicated().mean())
    else:
        duplicate_url = float("nan")

    present = tuple(column for column in engagement_columns if column in frame.columns)
    missing_engagement = tuple(
        column for column in engagement_columns if column not in frame.columns
    )
    notes: list[str] = []
    if missing_engagement:
        notes.append(
            "engagement columns requested but not present: " + ", ".join(missing_engagement)
        )
    if text_column is None:
        notes.append(
            "no text column: this corpus gives metadata and a tone vector, not words, "
            "so it cannot enter the human-label sample"
        )
    notes.append("no exposure data: the project has no record of what a respondent read")

    return CorpusProfile(
        name=name,
        unit=unit,
        rows=rows,
        start=start,
        end=end,
        months_present=len(month_counts),
        months_in_span=len(span),
        missing_months=missing,
        thin_months=thin,
        median_rows_per_month=median_month,
        n_sources=len(source_counts),
        top_sources=tuple((str(key), int(value)) for key, value in source_counts.head(10).items()),
        top1_share=top1,
        top10_share=top10,
        source_hhi=hhi,
        source_type_shares=type_shares,
        duplicate_exact_share=duplicate_exact,
        duplicate_normalised_share=duplicate_norm,
        duplicate_url_share=duplicate_url,
        wire_or_pr_share=type_shares["wire_or_pr"],
        algo_finance_share=type_shares["algo_finance"],
        foreign_share=type_shares["foreign"],
        short_text_share=short_share,
        empty_text_share=empty_share,
        engagement_columns=present,
        has_exposure_data=False,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 3. Stratified sample
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleReport:
    """What the sampler removed before it drew the sample, and the cause.

    `dropped_examples` keeps up to five normalised keys for each cause, so a
    reviewer can see what a cause removed and does not need to trust the count.
    """

    requested: int
    population_rows: int
    eligible_rows: int
    drawn_rows: int
    dropped_rows: int
    dropped_reasons: Mapping[str, int]
    dropped_examples: Mapping[str, tuple[str, ...]]
    empty_strata: int
    capped_strata: tuple[str, ...]
    framing_quota: int
    framing_oversample: Mapping[str, int]
    max_date: pd.Timestamp | None
    seed: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        detail = ", ".join(f"{cause} x{count}" for cause, count in sorted(self.dropped_reasons.items()))
        return (
            f"{self.drawn_rows} items drawn of {self.requested} requested; "
            f"{self.eligible_rows} eligible of {self.population_rows} rows; "
            f"dropped: {detail or 'none'}"
        )


@dataclass(frozen=True)
class StratifiedSample:
    """A reproducible stratified sample and its strata table.

    `frame` has one row per item with the columns `item_id`, `corpus`, `date`,
    `source`, `source_type`, `year`, `predicted_tone_class`, `stratum`,
    `sampling_layer`, one `predicted_framing_<contrast>` class and one
    `<contrast>_ratio` value for each framing contrast, `frames_any_contrast`
    and `text`. `strata` gives, for each stratum, the population count, the
    target count and the drawn count.

    The sample has no label column. A label column is a human product, and this
    class does not make one.
    """

    frame: pd.DataFrame
    strata: pd.DataFrame
    report: SampleReport
    seed: int

    @property
    def n_items(self) -> int:
        return len(self.frame)

    @property
    def meets_brief_minimum(self) -> bool:
        """True when the sample has the 300 items that the brief asks for."""
        return self.n_items >= 300

    def summary(self) -> str:
        return (
            f"stratified sample: {self.n_items} items in {len(self.strata)} strata "
            f"(seed {self.seed}); brief minimum of 300 "
            f"{'met' if self.meets_brief_minimum else 'NOT met'}"
        )


def _allocate(population: pd.Series, total: int) -> pd.Series:
    """Give each stratum a target count, proportional and with a floor of one.

    The method is the largest-remainder method. Each non-empty stratum gets at
    least one item, because a stratum with no item cannot show a problem in
    that stratum. A stratum cannot get more items than it holds. If the caps
    make the total too small, the remaining items go to the largest strata that
    are not full.
    """
    population = population[population > 0]
    if population.empty:
        return population.astype(int)
    if total >= int(population.sum()):
        return population.astype(int)

    share = population / float(population.sum())
    exact = share * total
    target = np.minimum(np.floor(exact).astype(int), population.to_numpy())
    target = np.maximum(target, 1)
    target = np.minimum(target, population.to_numpy())
    allocated = pd.Series(target, index=population.index, dtype=int)

    remainder = exact - np.floor(exact)
    order = pd.Series(remainder, index=population.index).sort_values(ascending=False).index
    while int(allocated.sum()) < total:
        progressed = False
        for key in order:
            if int(allocated.sum()) >= total:
                break
            if allocated[key] < population[key]:
                allocated[key] += 1
                progressed = True
        if not progressed:
            break
    # The floor of one can overshoot the target when there are many strata.
    order_desc = allocated.sort_values(ascending=False).index
    while int(allocated.sum()) > total:
        progressed = False
        for key in order_desc:
            if int(allocated.sum()) <= total:
                break
            if allocated[key] > 1:
                allocated[key] -= 1
                progressed = True
        if not progressed:
            break
    return allocated


def draw_stratified_sample(
    sources: Mapping[str, pd.DataFrame],
    *,
    n: int = 300,
    seed: int = 20261016,
    date_column: str = "date",
    source_column: str = "source",
    text_column: str = "text",
    max_date: pd.Timestamp | str | None = None,
    min_tokens: int = 4,
    framing_quota: int = 0,
) -> StratifiedSample:
    """Draw a stratified sample of at least `n` items from one or more corpora.

    `sources` maps a corpus name to a frame. A corpus with the name "tweets",
    or a name that holds "tweet" or "social", counts as a social corpus, and
    each of its items gets the source type "social".

    The strata are the product of the calendar year, the source type and the
    predicted tone class from the lexicon. Tone class is a stratum variable
    because the class prevalence in the corpus is far from equal: a sample that
    is drawn at random gives few positive items, and a per-class performance
    table then rests on a handful of items.

    The sampler removes a row for one of four causes only, and `SampleReport`
    gives the count and examples of each cause:

    - `empty_text`: the text is absent or has no letters.
    - `short_text`: fewer than `min_tokens` tokens. These are navigational
      strings such as "Business" or "Latest Articles". 2.2 percent of the
      headline corpus is such a string. A human cannot label them, so they must
      not enter the instrument.
    - `duplicate_normalised_text`: the same normalised text as an earlier row.
      The duplicate rate of the headline corpus is 32 percent, so a sample that
      keeps duplicates spends a third of the annotator budget on repeats.
    - `after_max_date`: the date is after `max_date`.

    `max_date` is the no-look-ahead control. A validation sample for a
    prediction date must not hold an item from after that date.

    `framing_quota` asks for at least that many items that frame each framing
    contrast. The framing lexicon is silent on most headlines, so a
    proportional sample cannot support a framing agreement number. The added
    items carry `sampling_layer = "framing_oversample[<contrast>]"`, and a
    prevalence estimate must use the stratified layer only. A quota of 0 turns
    the layer off and gives a purely proportional sample.
    """
    if n <= 0:
        raise ValidationError(f"n must be positive, got {n}")
    if not sources:
        raise ValidationError("no corpora given to sample from")

    cutoff = pd.Timestamp(max_date) if max_date is not None else None
    dropped: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    def record(cause: str, keys: Sequence[str]) -> None:
        if not len(keys):
            return
        dropped[cause] = dropped.get(cause, 0) + len(keys)
        bucket = examples.setdefault(cause, [])
        for key in keys:
            if len(bucket) >= 5:
                break
            bucket.append(str(key)[:120])

    pooled: list[pd.DataFrame] = []
    population_rows = 0
    for corpus_name, raw in sources.items():
        _require_columns(raw, (date_column, source_column, text_column))
        population_rows += len(raw)
        corpus_kind = (
            "social"
            if ("tweet" in corpus_name.lower() or "social" in corpus_name.lower())
            else "news"
        )
        work = pd.DataFrame(
            {
                "corpus": corpus_name,
                "date": pd.to_datetime(raw[date_column]),
                "source": raw[source_column].astype("string").fillna(""),
                "text": raw[text_column].astype("string").fillna(""),
            }
        )
        if getattr(work["date"].dtype, "tz", None) is not None:
            work["date"] = work["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        work["normalised"] = work["text"].map(normalise_text)

        if cutoff is not None:
            late = work["date"] > cutoff
            record(f"after_max_date[{corpus_name}]", work.loc[late, "normalised"].tolist())
            work = work.loc[~late]

        empty = work["normalised"] == ""
        record(f"empty_text[{corpus_name}]", work.loc[empty, "text"].astype(str).tolist())
        work = work.loc[~empty]

        short = work["normalised"].str.split().str.len() < min_tokens
        record(f"short_text[{corpus_name}]", work.loc[short, "normalised"].tolist())
        work = work.loc[~short]

        work = work.sort_values(["date", "source", "normalised"], kind="stable")
        duplicated = work["normalised"].duplicated()
        record(
            f"duplicate_normalised_text[{corpus_name}]",
            work.loc[duplicated, "normalised"].tolist(),
        )
        work = work.loc[~duplicated]

        work["source_type"] = work["source"].map(
            lambda value, kind=corpus_kind: classify_source(value, corpus=kind)
        )
        pooled.append(work)

    eligible = pd.concat(pooled, ignore_index=True)
    if eligible.empty:
        raise ValidationError(
            "no eligible items after the text filters; "
            f"dropped: {dropped}"
        )

    eligible["year"] = eligible["date"].dt.year.astype(int)
    eligible["predicted_tone_class"] = eligible["text"].map(predict_tone_class)
    framing_scores = [score_framing(text) for text in eligible["text"].astype(str)]
    for contrast in FRAMING_CONTRASTS:
        ratios = [score.ratio(contrast) for score in framing_scores]
        eligible[f"{contrast}_ratio"] = ratios
        eligible[f"predicted_framing_{contrast}"] = [
            framing_class_from_ratio(value, contrast) for value in ratios
        ]
    eligible["frames_any_contrast"] = [score.framing_total > 0 for score in framing_scores]
    eligible["stratum"] = (
        eligible["year"].astype(str)
        + "|"
        + eligible["source_type"]
        + "|"
        + eligible["predicted_tone_class"]
    )
    eligible["item_id"] = [
        _item_id((row.corpus, row.date, row.source, row.normalised))
        for row in eligible.itertuples()
    ]
    identical = eligible["item_id"].duplicated()
    record("duplicate_item_id", eligible.loc[identical, "normalised"].tolist())
    eligible = eligible.loc[~identical]

    population = eligible["stratum"].value_counts().sort_index()
    allocated = _allocate(population, n)

    rng = np.random.default_rng(seed)
    chosen: list[pd.DataFrame] = []
    capped: list[str] = []
    for stratum, target in allocated.items():
        block = eligible.loc[eligible["stratum"] == stratum]
        take = int(min(target, len(block)))
        if take < target:
            capped.append(str(stratum))
        if take <= 0:
            continue
        positions = rng.choice(len(block), size=take, replace=False)
        chosen.append(block.iloc[np.sort(positions)])

    sample = pd.concat(chosen, ignore_index=True) if chosen else eligible.head(0).copy()
    sample["sampling_layer"] = "stratified"

    # The framing oversample. The framing lexicon is silent on most headlines:
    # 9.2 percent of the headline corpus frames level against rate, and only
    # 2.2 to 2.5 percent frames the other two contrasts. A proportional sample
    # of 300 items therefore gives about ten framed items for two of the three
    # contrasts, which cannot support an agreement number. This layer adds
    # items that frame a contrast, and it balances the two sides so that the
    # added items cannot make the framing of the corpus look one-sided.
    #
    # The cost is recorded and it is real: the sample is no longer a
    # proportional sample of the corpus. A prevalence estimate must use the
    # rows where `sampling_layer == "stratified"` only. The agreement and the
    # performance numbers may use every row, because they condition on the item.
    oversample_counts: dict[str, int] = {}
    if framing_quota > 0:
        taken_ids = set(sample["item_id"])
        extra: list[pd.DataFrame] = []
        for contrast in FRAMING_CONTRASTS:
            column = f"predicted_framing_{contrast}"
            have = int((sample[column] != NOT_FRAMED).sum())
            need = int(framing_quota - have)
            if need <= 0:
                oversample_counts[contrast] = 0
                continue
            pool = eligible.loc[
                (eligible[column] != NOT_FRAMED) & (~eligible["item_id"].isin(taken_ids))
            ]
            sides = [side for side in FRAMING_CLASSES[contrast] if side != NOT_FRAMED]
            added = 0
            per_side = {side: 0 for side in sides}
            for position in range(need):
                side = sides[position % len(sides)]
                per_side[side] += 1
            for side in sides:
                block = pool.loc[(pool[column] == side) & (~pool["item_id"].isin(taken_ids))]
                take = int(min(per_side[side], len(block)))
                if take <= 0:
                    continue
                positions = rng.choice(len(block), size=take, replace=False)
                picked = block.iloc[np.sort(positions)].copy()
                picked["sampling_layer"] = f"framing_oversample[{contrast}]"
                taken_ids.update(picked["item_id"])
                extra.append(picked)
                added += take
            oversample_counts[contrast] = added
            if added < need:
                notes_oversample = (
                    f"the framing quota for {contrast} asked for {framing_quota} framed "
                    f"items and the corpus could give {have + added}"
                )
                logger.info(notes_oversample)
        if extra:
            sample = pd.concat([sample, *extra], ignore_index=True)

    # Present the items in a shuffled order. A file that is ordered by stratum
    # tells the annotator the predicted class of each run of items.
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "presentation_order", np.arange(1, len(sample) + 1))

    drawn = sample["stratum"].value_counts()
    strata = pd.DataFrame(
        {
            "population": population,
            "target": allocated.reindex(population.index).fillna(0).astype(int),
            "drawn": drawn.reindex(population.index).fillna(0).astype(int),
        }
    )
    strata.index.name = "stratum"
    strata[["year", "source_type", "predicted_tone_class"]] = (
        strata.index.to_series().str.split("|", expand=True).to_numpy()
    )

    notes: list[str] = []
    if len(sample) < n:
        notes.append(
            f"only {len(sample)} items available after filtering; the request was {n}"
        )
    if capped:
        notes.append(
            f"{len(capped)} strata are smaller than their target and gave all their items"
        )
    notes.append(
        "the sample has no label column; the annotator columns are made by "
        "write_annotation_files and they stay empty"
    )
    for contrast, added in oversample_counts.items():
        framed = int((sample[f"predicted_framing_{contrast}"] != NOT_FRAMED).sum())
        notes.append(
            f"framing {contrast}: {framed} items frame this contrast "
            f"({added} added by the oversample layer)"
        )
    if oversample_counts:
        notes.append(
            "a prevalence estimate must use the rows where sampling_layer is "
            "'stratified'; the framing oversample is not proportional"
        )

    report = SampleReport(
        requested=n,
        population_rows=population_rows,
        eligible_rows=len(eligible),
        drawn_rows=len(sample),
        dropped_rows=int(sum(dropped.values())),
        dropped_reasons=dict(sorted(dropped.items())),
        dropped_examples={cause: tuple(values) for cause, values in sorted(examples.items())},
        empty_strata=int((strata["drawn"] == 0).sum()),
        capped_strata=tuple(sorted(capped)),
        framing_quota=int(framing_quota),
        framing_oversample=dict(sorted(oversample_counts.items())),
        max_date=cutoff,
        seed=seed,
        notes=tuple(notes),
    )
    columns = [
        "item_id",
        "presentation_order",
        "corpus",
        "date",
        "source",
        "source_type",
        "year",
        "predicted_tone_class",
        "stratum",
        "sampling_layer",
        *[f"predicted_framing_{contrast}" for contrast in FRAMING_CONTRASTS],
        *[f"{contrast}_ratio" for contrast in FRAMING_CONTRASTS],
        "frames_any_contrast",
        "text",
    ]
    return StratifiedSample(
        frame=sample[columns].copy(), strata=strata.reset_index(), report=report, seed=seed
    )


def write_annotation_files(
    sample: StratifiedSample,
    directory: str | Path,
    *,
    prefix: str = "coding_sample",
) -> dict[str, Path]:
    """Write the blind annotation file, the key file and the strata file.

    Three files, and the split is deliberate:

    The default prefix is "coding_sample" and not "tone_sample": the
    instrument covers the five dimensions of the brief, and tone is one of
    them.

    - `<prefix>_annotation.csv`: what an annotator receives. It holds the item
      id, the presentation order, the text and the four empty columns of
      `ANNOTATOR_COLUMNS`: two label columns for each of the seven questions of
      the five dimensions, and one note column for each annotator. It does
      **not** hold the source, the date or the
      predicted class. A visible source name or model class would anchor the
      annotator, and the swap test in this module exists because that effect is
      real.
    - `<prefix>_key.csv`: the item id with the source, the date, the stratum
      and the predicted class. The analyst holds this file and joins it after
      the labels arrive. Refer to `join_labels`.
    - `<prefix>_strata.csv`: the population, the target and the drawn count of
      each stratum.

    The function returns the three paths. The label columns are empty strings.
    This function never writes a label.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    annotation = sample.frame[["item_id", "presentation_order", "text"]].copy()
    for column in ANNOTATOR_COLUMNS:
        annotation[column] = ""
    annotation = annotation.sort_values("presentation_order", kind="stable")

    key_columns = [
        "item_id",
        "presentation_order",
        "corpus",
        "date",
        "source",
        "source_type",
        "year",
        "predicted_tone_class",
        "stratum",
        "sampling_layer",
        *[f"predicted_framing_{contrast}" for contrast in FRAMING_CONTRASTS],
        *[f"{contrast}_ratio" for contrast in FRAMING_CONTRASTS],
    ]
    key = sample.frame[[name for name in key_columns if name in sample.frame.columns]].copy()

    paths = {
        "annotation": folder / f"{prefix}_annotation.csv",
        "key": folder / f"{prefix}_key.csv",
        "strata": folder / f"{prefix}_strata.csv",
    }
    annotation.to_csv(paths["annotation"], index=False)
    key.to_csv(paths["key"], index=False)
    sample.strata.to_csv(paths["strata"], index=False)
    logger.info("annotation instrument written to %s", folder)
    return paths


def join_labels(sample: StratifiedSample, labels: pd.DataFrame) -> pd.DataFrame:
    """Join a filled annotation file back onto the sample key.

    The join is on `item_id` and it is an inner join on the sample. An id in
    the label file that the sample does not hold raises `ValidationError`,
    because it means the annotator worked from a different sample and the
    agreement numbers would then cover two instruments.
    """
    _require_columns(labels, ("item_id",))
    known = set(sample.frame["item_id"])
    unknown = sorted(set(labels["item_id"].astype(str)) - known)
    if unknown:
        raise ValidationError(
            f"{len(unknown)} item ids in the label file are not in the sample, "
            f"first few: {unknown[:5]}"
        )
    merged = sample.frame.merge(
        labels.drop(columns=[c for c in ("text", "presentation_order") if c in labels.columns]),
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    return merged


# --------------------------------------------------------------------------
# 4. Agreement and adjudication
# --------------------------------------------------------------------------


def _clean_labels(values: pd.Series, labels: Sequence[str]) -> pd.Series:
    """Lower-case the labels and make an unknown or empty value missing."""
    cleaned = values.astype("string").str.strip().str.lower()
    allowed = {label.lower() for label in labels}
    return cleaned.where(cleaned.isin(allowed))


def percent_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    """Share of items where the two label sequences are the same."""
    if len(first) != len(second):
        raise ValidationError(f"{len(first)} labels against {len(second)} labels")
    if len(first) == 0:
        return float("nan")
    matches = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    return float(matches / len(first))


def cohen_kappa(
    first: Sequence[str], second: Sequence[str], labels: Sequence[str] = TONE_CLASSES
) -> float:
    """Cohen's kappa for two annotators on a nominal scale.

    The formula is ``(po - pe) / (1 - pe)``. `pe` uses the product of the two
    marginal distributions, which is Cohen's assumption: each annotator keeps
    their own class prevalence.

    The function gives NaN, and not 1.0, when `pe` is 1.0. That happens when
    both annotators give one class to every item. Agreement is then complete
    and also without information, and 1.0 would hide that.

    The function is written out and not taken from a library, so the marginals
    and the expected table can be inspected against the report.
    """
    if len(first) != len(second):
        raise ValidationError(f"{len(first)} labels against {len(second)} labels")
    total = len(first)
    if total == 0:
        return float("nan")
    index = {label: position for position, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(first, second, strict=True):
        if a not in index:
            raise ValidationError(f"unknown label {a!r} from annotator A")
        if b not in index:
            raise ValidationError(f"unknown label {b!r} from annotator B")
        table[index[a], index[b]] += 1.0
    observed = float(np.trace(table) / total)
    row_marginal = table.sum(axis=1) / total
    column_marginal = table.sum(axis=0) / total
    expected = float((row_marginal * column_marginal).sum())
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


@dataclass(frozen=True)
class AgreementReport:
    """Agreement between two label sets, with the provenance of the labels.

    `labels_are_human` is True only when `provenance == "human"`. The field
    exists so that a report, a table or a dashboard cannot show an agreement
    number without the question "whose labels?" being answered in the same
    object.

    `n_missing_a` and `n_missing_b` count the items that the annotator left
    empty. Those items are not in the kappa calculation, and they are not
    counted as agreement.
    """

    provenance: str
    label_a_name: str
    label_b_name: str
    n_items: int
    n_both_labelled: int
    n_missing_a: int
    n_missing_b: int
    n_invalid: int
    percent_agreement: float
    cohen_kappa: float
    confusion: pd.DataFrame
    by_class: pd.DataFrame
    by_source: pd.DataFrame
    disagreements: pd.DataFrame
    outstanding: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def labels_are_human(self) -> bool:
        return self.provenance == "human"

    @property
    def n_disagreements(self) -> int:
        return len(self.disagreements)

    def summary(self) -> str:
        head = (
            f"[{self.provenance.upper()} LABELS] {self.label_a_name} against "
            f"{self.label_b_name}: "
        )
        if self.n_both_labelled == 0:
            return head + (
                f"0 of {self.n_items} items have both labels; agreement cannot be "
                "calculated. Human labels are OUTSTANDING."
            )

        body = (
            f"{self.n_both_labelled} of {self.n_items} items labelled, "
            f"percent agreement {self.percent_agreement:.3f}, "
            f"kappa {self.cohen_kappa:.3f}"
        )
        if not self.labels_are_human:
            body += " -- these are MODEL labels, not human labels"
        return head + body


def require_human_labels(report: AgreementReport) -> AgreementReport:
    """Pass a human-label report through, and raise for any other kind.

    A caller that needs the human agreement of the brief calls this function.
    A model-label report then stops the caller instead of entering a table that
    says "inter-annotator agreement".
    """
    if not report.labels_are_human:
        raise ValidationError(
            f"this report uses {report.provenance} labels "
            f"({report.label_a_name} against {report.label_b_name}); the brief asks for "
            "two independent human annotators. Refer to OUTSTANDING_HUMAN_WORK."
        )
    if report.n_both_labelled == 0:
        raise ValidationError("the human label columns are empty; no agreement to report")
    return report


def _confusion(
    first: Sequence[str], second: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    table = pd.DataFrame(
        0,
        index=pd.Index(labels, name="annotator_a"),
        columns=pd.Index(labels, name="annotator_b"),
        dtype=int,
    )
    for a, b in zip(first, second, strict=True):
        table.loc[a, b] += 1
    return table


def compute_agreement(
    frame: pd.DataFrame,
    *,
    label_a: str = TONE_LABEL_A,
    label_b: str = TONE_LABEL_B,
    provenance: str,
    source_column: str | None = "source_type",
    labels: Sequence[str] = TONE_CLASSES,
) -> AgreementReport:
    """Agreement between two label columns, by class and by source.

    `provenance` is a required keyword argument, and it must be one of
    `PROVENANCE_VALUES`. There is no default. A default of "human" would let a
    model comparison become an inter-annotator table by accident, and a default
    of "automated" would be wrong for the real instrument. The caller must say.

    The by-class table gives, for each class, the count that each annotator
    used, the count where both used it, and the specific agreement
    ``2*both/(a+b)``. Specific agreement is reported because kappa is one
    number for the whole instrument and it cannot say which class the two
    annotators read differently.

    An item where either label is empty or is not in `labels` is not in the
    agreement calculation. The counts `n_missing_a`, `n_missing_b` and
    `n_invalid` give those items, so the denominator can be checked.
    """
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError(
            f"provenance must be one of {PROVENANCE_VALUES}, got {provenance!r}"
        )
    _require_columns(frame, (label_a, label_b))
    if source_column is not None and source_column not in frame.columns:
        raise ValidationError(f"source column {source_column!r} is not in the frame")

    raw_a = frame[label_a].astype("string").str.strip()
    raw_b = frame[label_b].astype("string").str.strip()
    clean_a = _clean_labels(raw_a, labels)
    clean_b = _clean_labels(raw_b, labels)

    missing_a = int(((raw_a.isna()) | (raw_a == "")).sum())
    missing_b = int(((raw_b.isna()) | (raw_b == "")).sum())
    given_a = ~((raw_a.isna()) | (raw_a == ""))
    given_b = ~((raw_b.isna()) | (raw_b == ""))
    invalid = int(((given_a & clean_a.isna()) | (given_b & clean_b.isna())).sum())

    usable = clean_a.notna() & clean_b.notna()
    values_a = clean_a[usable].tolist()
    values_b = clean_b[usable].tolist()

    if values_a:
        agreement = percent_agreement(values_a, values_b)
        kappa = cohen_kappa(values_a, values_b, labels)
        confusion = _confusion(values_a, values_b, labels)
    else:
        agreement = float("nan")
        kappa = float("nan")
        confusion = _confusion([], [], labels)

    class_rows = []
    for label in labels:
        count_a = int(sum(1 for value in values_a if value == label))
        count_b = int(sum(1 for value in values_b if value == label))
        both = int(
            sum(1 for a, b in zip(values_a, values_b, strict=True) if a == label and b == label)
        )
        denominator = count_a + count_b
        class_rows.append(
            {
                "class": label,
                "n_annotator_a": count_a,
                "n_annotator_b": count_b,
                "n_both": both,
                "specific_agreement": float(2 * both / denominator)
                if denominator
                else float("nan"),
            }
        )
    by_class = pd.DataFrame(class_rows)

    source_rows = []
    if source_column is not None and usable.any():
        sub = frame.loc[usable, [source_column]].copy()
        sub["a"] = values_a
        sub["b"] = values_b
        for source_value, block in sub.groupby(source_column, dropna=False):
            pair_a = block["a"].tolist()
            pair_b = block["b"].tolist()
            source_rows.append(
                {
                    "source": str(source_value),
                    "n_items": len(block),
                    "percent_agreement": percent_agreement(pair_a, pair_b),
                    "cohen_kappa": cohen_kappa(pair_a, pair_b, labels)
                    if len(block) > 1
                    else float("nan"),
                }
            )
    by_source = pd.DataFrame(
        source_rows, columns=["source", "n_items", "percent_agreement", "cohen_kappa"]
    )

    disagree_mask = usable & (clean_a != clean_b)
    keep = [
        column
        for column in ("item_id", "source", "source_type", "date", "text", label_a, label_b)
        if column in frame.columns
    ]
    disagreements = frame.loc[disagree_mask, keep].copy()

    outstanding: list[str] = []
    notes: list[str] = []
    if provenance != "human":
        outstanding.extend(OUTSTANDING_HUMAN_WORK)
        notes.append(
            f"these labels are {provenance}: {label_a} and {label_b} come from models "
            "and they exercise the agreement code. They are not the two independent "
            "human annotators that the brief asks for."
        )
    if not values_a:
        outstanding.append(
            "both label columns are empty for every item, so no agreement number exists"
        )
    if missing_a or missing_b:
        notes.append(
            f"items without a label: {missing_a} for {label_a}, {missing_b} for {label_b}"
        )
    if invalid:
        notes.append(f"{invalid} items hold a label outside {tuple(labels)} and are excluded")

    return AgreementReport(
        provenance=provenance,
        label_a_name=label_a,
        label_b_name=label_b,
        n_items=len(frame),
        n_both_labelled=int(usable.sum()),
        n_missing_a=missing_a,
        n_missing_b=missing_b,
        n_invalid=invalid,
        percent_agreement=agreement,
        cohen_kappa=kappa,
        confusion=confusion,
        by_class=by_class,
        by_source=by_source,
        disagreements=disagreements,
        outstanding=tuple(outstanding),
        notes=tuple(notes),
    )


def agreement_for_question(
    frame: pd.DataFrame,
    question: str,
    *,
    provenance: str,
    source_column: str | None = "source_type",
) -> AgreementReport:
    """Agreement for one of the `LABEL_QUESTIONS` of the instrument.

    The function finds the two column names and the permitted labels of the
    question, so a caller cannot pair the tone columns with the framing label
    set. For a framing question the label set holds `NOT_FRAMED` as its own
    class: an item that frames nothing and an item that frames both sides
    equally are two different answers, and they must not agree with each other.
    """
    label_a, label_b = annotator_pair(question)
    return compute_agreement(
        frame,
        label_a=label_a,
        label_b=label_b,
        provenance=provenance,
        source_column=source_column,
        labels=LABEL_QUESTIONS[question],
    )


def agreement_by_dimension(
    frame: pd.DataFrame,
    *,
    provenance: str,
    source_column: str | None = "source_type",
) -> pd.DataFrame:
    """One agreement row for each question of each of the five dimensions.

    A question whose two columns are not in the frame gives a row with the note
    "column absent" and not an exception, because the instrument may be filled
    one dimension at a time. A question whose columns are present but empty
    gives a row with NaN agreement, which is the honest value before the human
    labels arrive.
    """
    rows = []
    for dimension, questions in DIMENSION_QUESTIONS.items():
        for question in questions:
            label_a, label_b = annotator_pair(question)
            if label_a not in frame.columns or label_b not in frame.columns:
                rows.append(
                    {
                        "dimension": dimension,
                        "question": question,
                        "n_items": len(frame),
                        "n_both_labelled": 0,
                        "percent_agreement": float("nan"),
                        "cohen_kappa": float("nan"),
                        "labels_are_human": False,
                        "note": "column absent",
                    }
                )
                continue
            report = agreement_for_question(
                frame, question, provenance=provenance, source_column=source_column
            )
            rows.append(
                {
                    "dimension": dimension,
                    "question": question,
                    "n_items": report.n_items,
                    "n_both_labelled": report.n_both_labelled,
                    "percent_agreement": report.percent_agreement,
                    "cohen_kappa": report.cohen_kappa,
                    "labels_are_human": report.labels_are_human,
                    "note": "human labels outstanding"
                    if report.n_both_labelled == 0
                    else "",
                }
            )
    return pd.DataFrame(rows)


def adjudicate(
    frame: pd.DataFrame,
    *,
    label_a: str = TONE_LABEL_A,
    label_b: str = TONE_LABEL_B,
    adjudicated_column: str = "tone_adjudicated",
    labels: Sequence[str] = TONE_CLASSES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply step 3 to step 6 of `ADJUDICATION_PROCEDURE`.

    The function gives ``(gold, log)``. `gold` holds the items with a gold
    label. `log` holds one row for each item that needs a human decision or
    that has no gold label, with the cause.

    The function **does not** select a label for a disagreement. A rule that
    selects one of two human labels by machine makes the third person in the
    procedure unnecessary and it also makes the gold set a model product. If
    the frame already holds `adjudicated_column`, the function uses the value
    that the third person entered there.
    """
    _require_columns(frame, (label_a, label_b))
    clean_a = _clean_labels(frame[label_a], labels)
    clean_b = _clean_labels(frame[label_b], labels)
    manual = (
        _clean_labels(frame[adjudicated_column], labels)
        if adjudicated_column in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )

    gold = pd.Series(pd.NA, index=frame.index, dtype="string")
    agreed = clean_a.notna() & clean_b.notna() & (clean_a == clean_b)
    gold[agreed] = clean_a[agreed]
    resolved = manual.notna()
    gold[resolved] = manual[resolved]

    cause = pd.Series(pd.NA, index=frame.index, dtype="string")
    cause[agreed] = "both annotators agree"
    cause[resolved] = "third person decided"
    cause[clean_a.notna() & clean_b.notna() & (clean_a != clean_b) & ~resolved] = (
        "disagreement: a third person must decide"
    )
    cause[(clean_a.isna() | clean_b.isna()) & ~resolved] = (
        "one or both labels are absent: the item is not labelled yet"
    )

    keep = [
        column
        for column in ("item_id", "source", "source_type", "date", "text")
        if column in frame.columns
    ]
    out = frame[keep].copy()
    out[label_a] = clean_a
    out[label_b] = clean_b
    out[adjudicated_column] = gold
    out["adjudication_cause"] = cause

    log = out.loc[gold.isna() | resolved].copy()
    gold_frame = out.loc[gold.notna()].copy()
    return gold_frame, log


# --------------------------------------------------------------------------
# 5. Two automated labelers, and the AI-against-dictionary comparison
# --------------------------------------------------------------------------


def lexicon_tone_labels(texts: Sequence[str]) -> list[str]:
    """Tone class from the project lexicon for each text. Always available."""
    return [predict_tone_class(text) for text in texts]


def finbert_tone_labels(
    texts: Sequence[str],
    *,
    allow_download: bool = False,
    batch_size: int = 16,
    model: str = "ProsusAI/finbert",
) -> tuple[list[str | None], bool, str | None]:
    """Tone class from FinBERT for each text, or a reason why it cannot run.

    The function gives ``(labels, available, reason)``. When the weights are not
    in the local HuggingFace cache, `labels` holds None for every text,
    `available` is False and `reason` says so. Nothing is downloaded unless
    `allow_download=True`. A test suite that pulls 440 MB of weights is not a
    test suite.

    The function calls `nlp_media_svd.finbert_probe` with the texts and a
    constant intended label, and it reads the raw FinBERT label from the
    result. The model is therefore loaded by one piece of code in the project
    and not two. FinBERT's native labels are positive, negative and neutral,
    which are the tone classes of `TONE_CLASSES`. The policy-stance map of
    `nlp_media_svd` is **not** used here: that map answers the hawkish and
    dovish question, and this comparison is about tone.
    """
    if not texts:
        return [], False, "no texts given"
    probe = [(str(text), PROBE_LABELS[2]) for text in texts]
    result = finbert_probe(
        probe=probe, model=model, allow_download=allow_download, batch_size=batch_size
    )
    if not result.available:
        return [None] * len(texts), False, result.reason
    raw = result.predictions["finbert_label"].astype(str).str.lower().tolist()
    labels: list[str | None] = [
        value if value in TONE_CLASSES else None for value in raw
    ]
    unknown = sorted({value for value in raw if value not in TONE_CLASSES})
    reason = f"labels outside {TONE_CLASSES}: {unknown}" if unknown else None
    return labels, True, reason


@dataclass(frozen=True)
class ClassifierComparison:
    """The lexicon against FinBERT, with or without a human reference.

    `reference_is_human` says which kind of table this is:

    - True: `by_class` holds precision, recall and F1 for each classifier
      against the human gold labels, and `by_source` holds accuracy by source.
      This is the "AI performance by class and source" of the brief.
    - False: there is no gold standard. `confusion` is then the cross-table of
      the two models, `by_class` holds the class counts of each model, and the
      only honest headline number is concordance. `outstanding` says so.
    """

    finbert_available: bool
    finbert_reason: str | None
    model: str
    n_items: int
    n_scored: int
    reference_column: str | None
    reference_is_human: bool
    concordance: float
    concordance_kappa: float
    confusion: pd.DataFrame
    by_class: pd.DataFrame
    by_source: pd.DataFrame
    labels: pd.DataFrame
    outstanding: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if not self.finbert_available:
            return f"lexicon against FinBERT not run: {self.finbert_reason}"
        if self.reference_is_human:
            return (
                f"lexicon against FinBERT on {self.n_scored} items, scored against "
                f"human labels in {self.reference_column!r}"
            )
        return (
            f"lexicon against FinBERT on {self.n_scored} items: concordance "
            f"{self.concordance:.3f}, kappa {self.concordance_kappa:.3f}. "
            "NO human gold standard, so this is agreement and not accuracy."
        )


def _class_metrics(truth: Sequence[str], predicted: Sequence[str], label: str) -> dict[str, float]:
    true_positive = sum(1 for t, p in zip(truth, predicted, strict=True) if t == label and p == label)
    false_positive = sum(1 for t, p in zip(truth, predicted, strict=True) if t != label and p == label)
    false_negative = sum(1 for t, p in zip(truth, predicted, strict=True) if t == label and p != label)
    support = true_positive + false_negative
    precision = (
        float(true_positive / (true_positive + false_positive))
        if (true_positive + false_positive)
        else float("nan")
    )
    recall = float(true_positive / support) if support else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    return {"support": float(support), "precision": precision, "recall": recall, "f1": f1}


def compare_classifiers(
    frame: pd.DataFrame,
    *,
    text_column: str = "text",
    source_column: str = "source_type",
    reference_column: str | None = None,
    reference_is_human: bool = False,
    allow_download: bool = False,
    finbert_label_fn: Callable[[Sequence[str]], tuple[list[str | None], bool, str | None]]
    | None = None,
    model: str = "ProsusAI/finbert",
) -> ClassifierComparison:
    """Compare the dictionary classifier with FinBERT on the sample.

    The brief asks for the AI approach against "a simple dictionary or
    conventional classifier". The project lexicon is the dictionary, and
    FinBERT is the AI classifier.

    `reference_column` names a human gold-label column. When it is None, the
    function reports concordance between the two models and marks the accuracy
    table as outstanding. When it is given and `reference_is_human` is False,
    the function still computes the metrics but it keeps the warning in
    `outstanding`: a model reference gives a performance table against a model,
    and that is not the brief's measurement.

    `finbert_label_fn` exists for the tests. A test gives a small deterministic
    function instead of the model, so the test needs no weights and no network.
    """
    _require_columns(frame, (text_column,))
    if source_column not in frame.columns:
        raise ValidationError(f"source column {source_column!r} is not in the frame")
    if reference_column is not None:
        _require_columns(frame, (reference_column,))
    if reference_is_human and reference_column is None:
        raise ValidationError("reference_is_human is True but no reference column was given")

    texts = frame[text_column].astype(str).tolist()
    lexicon = lexicon_tone_labels(texts)
    label_fn = finbert_label_fn or (
        lambda values: finbert_tone_labels(values, allow_download=allow_download)
    )
    finbert, available, reason = label_fn(texts)
    if len(finbert) != len(texts):
        raise ValidationError(
            f"the FinBERT labeler gave {len(finbert)} labels for {len(texts)} texts"
        )

    labels = pd.DataFrame(
        {
            "item_id": frame["item_id"].to_numpy()
            if "item_id" in frame.columns
            else np.arange(len(frame)),
            "source": frame[source_column].astype(str).to_numpy(),
            "label_lexicon": lexicon,
            "label_finbert": pd.Series(finbert, dtype="string"),
            "labeler_kind": "automated",
        }
    )
    if reference_column is not None:
        labels["label_reference"] = _clean_labels(
            frame[reference_column].reset_index(drop=True), TONE_CLASSES
        )

    scored = labels["label_finbert"].notna()
    outstanding: list[str] = []
    notes: list[str] = []

    if available and scored.any():
        pair_lexicon = labels.loc[scored, "label_lexicon"].tolist()
        pair_finbert = labels.loc[scored, "label_finbert"].tolist()
        concordance = percent_agreement(pair_lexicon, pair_finbert)
        concordance_kappa = cohen_kappa(pair_lexicon, pair_finbert, TONE_CLASSES)
        confusion = pd.DataFrame(
            0,
            index=pd.Index(TONE_CLASSES, name="lexicon"),
            columns=pd.Index(TONE_CLASSES, name="finbert"),
            dtype=int,
        )
        for a, b in zip(pair_lexicon, pair_finbert, strict=True):
            confusion.loc[a, b] += 1
    else:
        concordance = float("nan")
        concordance_kappa = float("nan")
        confusion = pd.DataFrame(
            0,
            index=pd.Index(TONE_CLASSES, name="lexicon"),
            columns=pd.Index(TONE_CLASSES, name="finbert"),
            dtype=int,
        )

    class_rows = []
    source_rows = []
    if reference_column is not None:
        usable = labels["label_reference"].notna() & scored
        if usable.any():
            truth = labels.loc[usable, "label_reference"].tolist()
            for name, column in (("lexicon", "label_lexicon"), ("finbert", "label_finbert")):
                predicted = labels.loc[usable, column].tolist()
                for label in TONE_CLASSES:
                    row = {"classifier": name, "class": label}
                    row.update(_class_metrics(truth, predicted, label))
                    class_rows.append(row)
            block = labels.loc[usable].copy()
            for source_value, part in block.groupby("source", dropna=False):
                row = {"source": str(source_value), "n_items": len(part)}
                for name, column in (("lexicon", "label_lexicon"), ("finbert", "label_finbert")):
                    row[f"accuracy_{name}"] = percent_agreement(
                        part["label_reference"].tolist(), part[column].tolist()
                    )
                source_rows.append(row)
        else:
            notes.append("the reference column has no usable label, so no accuracy was computed")
        if not reference_is_human:
            outstanding.append(
                f"the reference column {reference_column!r} is not human. The precision, "
                "recall and F1 numbers measure agreement with a model."
            )
    else:
        for name, column in (("lexicon", "label_lexicon"), ("finbert", "label_finbert")):
            for label in TONE_CLASSES:
                class_rows.append(
                    {
                        "classifier": name,
                        "class": label,
                        "n_predicted": int((labels[column].fillna("") == label).sum()),
                        "share_predicted": float(
                            (labels[column].fillna("") == label).mean()
                        ),
                    }
                )
        if scored.any():
            for source_value, part in labels.loc[scored].groupby("source", dropna=False):
                source_rows.append(
                    {
                        "source": str(source_value),
                        "n_items": len(part),
                        "concordance": percent_agreement(
                            part["label_lexicon"].tolist(), part["label_finbert"].tolist()
                        ),
                    }
                )
        outstanding.extend(OUTSTANDING_HUMAN_WORK)
        notes.append(
            "no reference labels: the table gives the class mix of each model and their "
            "concordance. Neither model is correct by definition."
        )

    notes.append(FRAMING_AI_LIMITATION)
    if not available:
        outstanding.append(
            "FinBERT did not run, so the AI-against-dictionary comparison has only the "
            "dictionary side"
        )
        notes.append(str(reason))
    elif reason:
        notes.append(str(reason))

    return ClassifierComparison(
        finbert_available=bool(available),
        finbert_reason=reason,
        model=model,
        n_items=len(frame),
        n_scored=int(scored.sum()),
        reference_column=reference_column,
        reference_is_human=bool(reference_is_human),
        concordance=concordance,
        concordance_kappa=concordance_kappa,
        confusion=confusion,
        by_class=pd.DataFrame(class_rows),
        by_source=pd.DataFrame(source_rows),
        labels=labels,
        outstanding=tuple(outstanding),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 6. Name and source swap robustness
# --------------------------------------------------------------------------


def _swap_pattern(pairs: Sequence[tuple[str, str]]) -> re.Pattern[str]:
    words = sorted({word for pair in pairs for word in pair}, key=len, reverse=True)
    escaped = [re.escape(word) for word in words]
    return re.compile(r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])", re.IGNORECASE)


def _match_case(original: str, replacement: str) -> str:
    """Give the replacement the case pattern of the original word."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def swap_text(text: str, pairs: Sequence[tuple[str, str]] = NAME_SWAPS) -> tuple[str, int]:
    """Exchange each word of each pair with its partner, in one pass.

    The function gives ``(new_text, n_swaps)``. One pass is important. Two
    passes would turn "Trump" into "Biden" and then "Biden" into "Trump", and
    the text would not change. The pattern uses word limits, so "Trumpet" does
    not become "Bidenet".
    """
    mapping: dict[str, str] = {}
    for first, second in pairs:
        mapping[first.lower()] = second
        mapping[second.lower()] = first
    pattern = _swap_pattern(pairs)
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        found = match.group(1)
        target = mapping.get(found.lower())
        if target is None:
            return found
        count += 1
        return _match_case(found, target)

    return pattern.sub(replace, str(text)), count


@dataclass(frozen=True)
class SwapRobustnessReport:
    """How often a swap of political names or source labels changes a class.

    A classifier that measures economic tone must give the same class to
    "Trump says prices are falling" and "Biden says prices are falling". The
    flip rate is the measurement. `n_eligible` is the count of items where the
    swap changed the text at all: an item with no political name cannot test
    anything, and keeping it in the denominator would make any classifier look
    robust.
    """

    classifier: str
    swap_kind: str
    dimension: str
    n_items: int
    n_eligible: int
    n_scored: int
    n_flipped: int
    flip_rate: float
    flips_by_class: pd.DataFrame
    flips_by_source: pd.DataFrame
    transitions: pd.DataFrame
    examples: pd.DataFrame
    available: bool = True
    reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        head = f"{self.swap_kind} swap test, {self.dimension}, {self.classifier}"
        if not self.available:
            return f"{head}: not run: {self.reason}"
        if self.n_eligible == 0:
            return (
                f"{head}: no item of {self.n_items} holds a swappable token, so the "
                "test is empty"
            )
        return (
            f"{head}: {self.n_flipped} of {self.n_scored} eligible items change class "
            f"({self.flip_rate:.1%})"
        )


def swap_robustness_test(
    frame: pd.DataFrame,
    *,
    labeler: Callable[[Sequence[str]], Sequence[str | None]],
    classifier: str,
    text_column: str = "text",
    source_column: str = "source_type",
    swap_kind: str = "political_name",
    pairs: Sequence[tuple[str, str]] = NAME_SWAPS,
    source_labels: Sequence[tuple[str, str]] = SOURCE_LABEL_SWAPS,
    labels: Sequence[str] = TONE_CLASSES,
    dimension: str = "tone",
    max_examples: int = 20,
) -> SwapRobustnessReport:
    """Test whether a name swap or a source-label swap changes the class.

    `swap_kind` selects the manipulation:

    - "political_name": exchange the words of `pairs` inside the text.
    - "source_label": put a source label in front of the text, then exchange
      the label for its partner. The text itself does not change. A classifier
      that reads only the words must then give the same class, and a different
      class is proof that the label and not the content drives the measurement.
      The source label reaches a real model through the prompt or through a
      prefix, so a prefix is the honest test for a text classifier.

    `labeler` takes a list of texts and gives a list of class labels, or None
    for a text that it could not score. The argument is a function so that the
    same test runs for the lexicon and for FinBERT, and so that a test can give
    a small deterministic function.

    `labels` and `dimension` make the test cover any dimension of the coding
    scheme and not only tone. A framing test passes
    ``labeler=framing_labeler(contrast)``,
    ``labels=FRAMING_CLASSES[contrast]`` and ``dimension=f"framing_{contrast}"``.
    The brief asks whether a change of political name alters the
    classification, and a framing contrast is a classification.
    """
    _require_columns(frame, (text_column,))
    if source_column not in frame.columns:
        raise ValidationError(f"source column {source_column!r} is not in the frame")
    if swap_kind not in ("political_name", "source_label"):
        raise ValidationError(f"unknown swap_kind {swap_kind!r}")

    original_texts = frame[text_column].astype(str).tolist()
    sources = frame[source_column].astype(str).tolist()

    if swap_kind == "political_name":
        base = original_texts
        swapped: list[str] = []
        eligible: list[bool] = []
        for text in original_texts:
            new_text, count = swap_text(text, pairs)
            swapped.append(new_text)
            eligible.append(count > 0)
    else:
        if not source_labels:
            raise ValidationError("the source-label swap needs at least one label pair")
        base = []
        swapped = []
        eligible = []
        n_pairs = len(source_labels)
        for position, text in enumerate(original_texts):
            first, second = source_labels[position % n_pairs]
            base.append(f"{first}: {text}")
            swapped.append(f"{second}: {text}")
            eligible.append(True)

    labels_base = list(labeler(base))
    labels_swapped = list(labeler(swapped))
    if len(labels_base) != len(base) or len(labels_swapped) != len(swapped):
        raise ValidationError(
            f"the labeler gave {len(labels_base)}/{len(labels_swapped)} labels "
            f"for {len(base)} texts"
        )

    work = pd.DataFrame(
        {
            "item_id": frame["item_id"].to_numpy()
            if "item_id" in frame.columns
            else np.arange(len(frame)),
            "source": sources,
            "text_original": base,
            "text_swapped": swapped,
            "label_original": pd.Series(labels_base, dtype="string"),
            "label_swapped": pd.Series(labels_swapped, dtype="string"),
            "eligible": eligible,
        }
    )
    scored = work["eligible"] & work["label_original"].notna() & work["label_swapped"].notna()
    available = bool(scored.any()) or not work["eligible"].any()
    reason = None
    if work["eligible"].any() and not scored.any():
        available = False
        reason = "the labeler gave no label for any eligible item"

    work["flipped"] = scored & (work["label_original"] != work["label_swapped"])
    n_flipped = int(work["flipped"].sum())
    n_scored = int(scored.sum())
    flip_rate = float(n_flipped / n_scored) if n_scored else float("nan")

    by_class_rows = []
    for label in labels:
        block = work.loc[scored & (work["label_original"] == label)]
        by_class_rows.append(
            {
                "class_original": label,
                "n_items": len(block),
                "n_flipped": int(block["flipped"].sum()),
                "flip_rate": float(block["flipped"].mean()) if len(block) else float("nan"),
            }
        )
    by_source_rows = []
    for source_value, block in work.loc[scored].groupby("source", dropna=False):
        by_source_rows.append(
            {
                "source": str(source_value),
                "n_items": len(block),
                "n_flipped": int(block["flipped"].sum()),
                "flip_rate": float(block["flipped"].mean()),
            }
        )

    transitions = pd.DataFrame(
        0,
        index=pd.Index(labels, name="original"),
        columns=pd.Index(labels, name="swapped"),
        dtype=int,
    )
    for _, row in work.loc[scored].iterrows():
        first = str(row["label_original"])
        second = str(row["label_swapped"])
        if first in labels and second in labels:
            transitions.loc[first, second] += 1

    examples = work.loc[work["flipped"]].head(max_examples)[
        ["item_id", "source", "text_original", "text_swapped", "label_original", "label_swapped"]
    ].copy()

    notes = [
        ("the flip rate uses the eligible items as the denominator: an item without a "
        "swappable token cannot change and it is not evidence of robustness"),
    ]
    if swap_kind == "source_label":
        notes.append(
            "the text is identical in the two conditions; only the source label in front "
            "of the text is different"
        )
        notes.append(
            "the label pair of an item comes from its position in the frame, so the "
            "flip rate by source type also reflects which pair that source received"
        )
    ineligible = int((~work["eligible"]).sum())
    if ineligible:
        notes.append(f"{ineligible} items hold no swappable token and are excluded")

    return SwapRobustnessReport(
        classifier=classifier,
        swap_kind=swap_kind,
        dimension=dimension,
        n_items=len(frame),
        n_eligible=int(work["eligible"].sum()),
        n_scored=n_scored,
        n_flipped=n_flipped,
        flip_rate=flip_rate,
        flips_by_class=pd.DataFrame(by_class_rows),
        flips_by_source=pd.DataFrame(by_source_rows),
        transitions=transitions,
        examples=examples,
        available=available,
        reason=reason,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 7. The same-event cross-source benchmark
# --------------------------------------------------------------------------


#: Topic terms for the "same day and same topic" event key. The six topics are
#: the brief's own salience list, and they are the same six answers that
#: `SALIENCE_CLASSES` gives a human annotator. One list serves the benchmark
#: and the labeling guide, so a human label and a machine event key cannot
#: define the topic differently.
BENCHMARK_TOPICS: dict[str, tuple[str, ...]] = {
    "layoffs": ("layoff", "layoffs", "job cuts", "jobless", "fired", "redundancies"),
    "hiring": ("hiring", "hires", "jobs added", "payrolls", "job gains", "employment rose"),
    "rising_prices": ("prices", "price", "cost", "costs", "cost of living", "expensive"),
    "falling_inflation": ("inflation", "cpi", "disinflation", "deflation", "price growth"),
    "growth": ("gdp", "growth", "grew", "expanded", "output", "economy grew"),
    "recession_risk": ("recession", "downturn", "contraction", "slump", "slowdown"),
}

_COMPILED_TOPICS: dict[str, tuple[tuple[str, ...], ...]] = {
    topic: tuple(tuple(term.split()) for term in terms)
    for topic, terms in BENCHMARK_TOPICS.items()
}

#: The permitted event keys of `same_event_cross_source`.
EVENT_KEYS: tuple[str, str] = ("normalised_text", "day_and_topic")


def dominant_topic(text: str | float | None) -> str:
    """Give the topic of one text from `BENCHMARK_TOPICS`, or "none".

    The topic with the most hits wins. A tie goes to the topic that appears
    first in `BENCHMARK_TOPICS`, which is a fixed order and not a data-derived
    order, so the function cannot change its answer when the corpus changes.
    """
    tokens = normalise_text(text).split()
    if not tokens:
        return "none"
    best = "none"
    best_hits = 0
    for topic, terms in _COMPILED_TOPICS.items():
        hits = 0
        for term in terms:
            width = len(term)
            for start in range(len(tokens) - width + 1):
                if tuple(tokens[start : start + width]) == term:
                    hits += 1
        if hits > best_hits:
            best = topic
            best_hits = hits
    return best


@dataclass(frozen=True)
class BenchmarkReport:
    """The same-event cross-source benchmark.

    **The benchmark.** For each event, the benchmark value is the mean tone of
    all items about that event. The bias of a source is then the mean of its
    deviation from that benchmark, over the events that the source covered.
    The construction holds the event constant, so a source that covers more bad
    news cannot look negative for that reason alone. This answers the brief's
    instruction to define the benchmark against which each proposed bias is
    measured.

    **What it cannot do.** An event key that comes from a shared normalised
    headline finds syndication and wire copy. It does not find two independent
    articles about the same release that use different words. The measured
    deviation is therefore a floor on cross-source variation and not an
    estimate of it. `n_events` and `median_sources_per_event` give the size of
    the comparison.
    """

    event_key: str
    n_items: int
    n_events: int
    n_items_in_events: int
    median_sources_per_event: float
    mean_within_event_tone_range: float
    class_disagreement_rate: float
    by_source: pd.DataFrame
    events: pd.DataFrame
    dropped_reasons: Mapping[str, int]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if self.n_events == 0:
            return (
                f"same-event benchmark on {self.n_items} items: no event appears in two "
                "or more source types, so no comparison is possible"
            )
        return (
            f"same-event benchmark: {self.n_events} events across source types, "
            f"{self.n_items_in_events} items, mean within-event tone range "
            f"{self.mean_within_event_tone_range:.3f}, class disagreement "
            f"{self.class_disagreement_rate:.1%}"
        )


def same_event_cross_source(
    frame: pd.DataFrame,
    *,
    text_column: str = "text",
    source_column: str = "source",
    date_column: str = "date",
    corpus_kind: str = "news",
    event_key: str = "normalised_text",
    min_source_types: int = 2,
    min_tokens: int = 4,
) -> BenchmarkReport:
    """Compare the tone of the same event across source types.

    `event_key` selects what "the same event" means, and the choice changes
    what the comparison can find:

    - `"normalised_text"`: two items are the same event when their normalised
      text is the same. This finds syndicated copy and wire copy. It gives a
      clean comparison and a **degenerate** one: the text is identical, so the
      lexicon must give the same tone, and the measured deviation is zero by
      construction. The result is a control. A number other than zero would
      show a defect in the scorer.
    - `"day_and_topic"`: two items are the same event when they appeared on the
      same calendar day and have the same dominant topic from
      `BENCHMARK_TOPICS`. The text is then different, so the comparison can
      find a real difference between source types. This is the benchmark that
      the brief describes as "coverage of similar economic releases". It is
      weaker than a release-level key, because two items on one day about
      prices need not be about one release.

    An event enters the comparison when at least `min_source_types` different
    source types carry it. A single outlet that repeats its own headline is not
    a cross-source comparison, and such an event is removed with a record.
    """
    if event_key not in EVENT_KEYS:
        raise ValidationError(f"event_key must be one of {EVENT_KEYS}, got {event_key!r}")
    _require_columns(frame, (text_column, source_column, date_column))
    dropped: dict[str, int] = {}

    work = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column]),
            "source": frame[source_column].astype("string").fillna(""),
            "text": frame[text_column].astype("string").fillna(""),
        }
    )
    if getattr(work["date"].dtype, "tz", None) is not None:
        work["date"] = work["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    work["normalised"] = work["text"].map(normalise_text)

    too_short = work["normalised"].str.split().str.len().fillna(0) < min_tokens
    dropped["short_or_empty_text"] = int(too_short.sum())
    work = work.loc[~too_short]

    if event_key == "normalised_text":
        work["event_key"] = work["normalised"]
    else:
        work["topic"] = work["text"].map(dominant_topic)
        off_topic = work["topic"] == "none"
        dropped["no_benchmark_topic"] = int(off_topic.sum())
        work = work.loc[~off_topic]
        work["event_key"] = work["date"].dt.strftime("%Y-%m-%d") + "|" + work["topic"]

    work["source_type"] = work["source"].map(
        lambda value: classify_source(value, corpus=corpus_kind)
    )
    scores = [score_text(text) for text in work["text"].astype(str)]
    work["tone"] = [score.tone for score in scores]
    work["tone_class"] = [
        "positive"
        if score.tone >= TONE_POSITIVE_MIN
        else "negative"
        if score.tone <= TONE_NEGATIVE_MAX
        else "neutral"
        for score in scores
    ]

    grouped = work.groupby("event_key")
    type_counts = grouped["source_type"].nunique()
    keep_keys = type_counts[type_counts >= min_source_types].index
    dropped["event_in_one_source_type_only"] = int((type_counts < min_source_types).sum())
    events_frame = work.loc[work["event_key"].isin(keep_keys)].copy()

    if events_frame.empty:
        return BenchmarkReport(
            event_key=event_key,
            n_items=len(frame),
            n_events=0,
            n_items_in_events=0,
            median_sources_per_event=float("nan"),
            mean_within_event_tone_range=float("nan"),
            class_disagreement_rate=float("nan"),
            by_source=pd.DataFrame(
                columns=["source_type", "n_events", "n_items", "mean_deviation", "mean_tone"]
            ),
            events=pd.DataFrame(
                columns=[
                    "event_key",
                    "n_items",
                    "n_source_types",
                    "benchmark_tone",
                    "tone_range",
                    "n_classes",
                    "first_date",
                ]
            ),
            dropped_reasons=dict(sorted(dropped.items())),
            notes=(
                ("no event is carried by two or more source types; the corpus is too "
                "sparse for this benchmark at this grouping"),
            ),
        )

    benchmark = events_frame.groupby("event_key")["tone"].transform("mean")
    events_frame["benchmark_tone"] = benchmark
    events_frame["deviation"] = events_frame["tone"] - benchmark

    events = events_frame.groupby("event_key").agg(
        n_items=("tone", "size"),
        n_source_types=("source_type", "nunique"),
        benchmark_tone=("benchmark_tone", "first"),
        tone_range=("tone", lambda values: float(values.max() - values.min())),
        n_classes=("tone_class", "nunique"),
        first_date=("date", "min"),
    )

    by_source = (
        events_frame.groupby("source_type")
        .agg(
            n_events=("event_key", "nunique"),
            n_items=("tone", "size"),
            mean_deviation=("deviation", "mean"),
            mean_tone=("tone", "mean"),
        )
        .reset_index()
        .sort_values("mean_deviation")
    )

    notes = [
        ("the benchmark for an event is the mean tone of every item about that event, "
        "so a source cannot look negative because it covers worse events"),
    ]
    if event_key == "normalised_text":
        notes.append(
            "the event key is the normalised text, so the comparison finds syndicated "
            "and wire copy. The text is identical, so a deviation of zero is the "
            "correct result and this run is a control and not a measurement"
        )
    else:
        notes.append(
            "the event key is the calendar day and the dominant topic, so the texts "
            "are different and a deviation can be real. Two items on one day about "
            "one topic need not be about one release, so the key is weaker than a "
            "release-level key"
        )
    return BenchmarkReport(
        event_key=event_key,
        n_items=len(frame),
        n_events=len(events),
        n_items_in_events=len(events_frame),
        median_sources_per_event=float(events["n_source_types"].median()),
        mean_within_event_tone_range=float(events["tone_range"].mean()),
        class_disagreement_rate=float((events["n_classes"] > 1).mean()),
        by_source=by_source,
        events=events.reset_index().sort_values("n_items", ascending=False),
        dropped_reasons=dict(sorted(dropped.items())),
        notes=tuple(notes),
    )
