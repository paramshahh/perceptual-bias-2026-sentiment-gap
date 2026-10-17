"""Media-text pipeline for Hypothesis B: corpus, LSA, lexicons, FinBERT probe.

Hypothesis B claims the Sentiment Gap is partly a measurement of the
*information environment* rather than of the economy. That requires four things,
and this module supplies them in order:

1. **A corpus.** GDELT's Doc 2.0 API is free, needs no key, and indexes
   worldwide news metadata. It is pulled day by day and cached to
   ``data/raw/gdelt/`` as parquet, so a re-run, a report rebuild or a dashboard
   reload costs no network. Everything downstream of the cache is pure
   computation and runs offline, which is also how the tests exercise it.
2. **A term-document matrix** (TF-IDF, sparse, English stopwords).
3. **Latent semantic structure** via truncated SVD, *without demeaning* — see
   `fit_lsa` and BRAINSTORMING.md §3.2. The non-demeaning choice is load-bearing
   here, not an oversight: it is what makes "this document contains none of the
   dictionary terms" map to the origin instead of to minus-the-mean.
4. **Scores**: a domain phrase lexicon over the four dimensions of
   BRAINSTORMING.md §4.3, and an empirical FinBERT probe rather than a citation.

Two conventions inherited from `data_loader` and `baseline_model` and kept
deliberately:

* **Nothing is dropped silently.** Corpus ingestion returns a `CorpusReport`
  naming every discarded document and why; the term-document matrix names the
  documents that survived vectorisation with no vocabulary terms at all (they
  are *kept*, as zero columns, because they are legitimate off-topic documents
  and discarding them would bias daily denominators).
* **No look-ahead.** `fit_term_document` / `apply_term_document` and `fit_lsa` /
  `project_documents` are separate calls. The vocabulary, the IDF weights and
  the SVD basis are estimated on the training window and then applied unchanged
  to the holdout. Fitting on the pooled sample and reporting holdout numbers
  would embed the answer in the transform.

What this module deliberately does **not** do: it never calls twitterapi.io
(`twitter_source` owns that paid budget), and it never downloads model weights
implicitly — the FinBERT probe is an explicit function call that reports itself
unavailable rather than pulling 440 MB behind the caller's back.
"""

from __future__ import annotations

import hashlib
import logging
import time
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Default cache directory for raw GDELT pulls, relative to the repo root.
GDELT_CACHE_DIR = Path("data/raw/gdelt")

#: Economic query terms. GDELT's Doc API ANDs multiple keywords and has no
#: boolean OR for a plain keyword list, so each term is queried separately and
#: the results unioned. Kept short: every extra term is another HTTP request.
ECONOMIC_TERMS: tuple[str, ...] = (
    "inflation",
    "cost of living",
    "grocery prices",
    "gas prices",
    "recession",
    "layoffs",
    "consumer sentiment",
    "economy",
)

#: GDELT Doc API hard-caps an article list at 250 records per query.
GDELT_MAX_RECORDS = 250

# TF-IDF defaults. ``min_df=2`` drops hapax legomena (overwhelmingly typos,
# proper nouns and tokenisation debris in news headlines); ``max_df=0.8`` drops
# terms present in most documents, which carry no discriminating information but
# would dominate the leading singular triplet.
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.8
DEFAULT_NGRAM_RANGE = (1, 2)

#: How many tokens before a lexicon hit are searched for a negator.
NEGATION_WINDOW = 3

#: Tokens that invert the polarity of a following polar phrase.
NEGATORS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "none",
    "cannot",
    "cant",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "hasnt",
    "havent",
    "hardly",
    "barely",
    "scarcely",
    "without",
    "fails",
    "failed",
    "failing",
    "far from",
    "nothing",
    "nowhere",
)

#: The four scoring dimensions of BRAINSTORMING.md §4.3, as an auditable
#: module-level dictionary. Phrases are lowercase and whitespace-tokenised, so
#: multi-word entries match as contiguous token n-grams.
#:
#: Two dimensions are *polar* (`negative`, `positive`) and feed the tone index;
#: the other three are *descriptive counters* and are never polarity-flipped by
#: negation, because "not a crisis" still makes crisis the frame of the
#: sentence — it is salient either way. That asymmetry is intentional.
LEXICON: dict[str, tuple[str, ...]] = {
    "negative": (
        "recession",
        "downturn",
        "slump",
        "slowdown",
        "contraction",
        "crisis",
        "collapse",
        "crash",
        "layoffs",
        "job cuts",
        "unemployment rising",
        "squeeze",
        "squeezed",
        "struggling",
        "struggle",
        "hardship",
        "pain",
        "painful",
        "worse",
        "worsening",
        "deteriorating",
        "deterioration",
        "weak",
        "weakening",
        "weakness",
        "falling behind",
        "cant afford",
        "unaffordable",
        "soaring prices",
        "surging prices",
        "high prices",
        "price shock",
        "sticker shock",
        "stagnant",
        "stagnation",
        "gloom",
        "gloomy",
        "grim",
        "bleak",
        "fears",
        "worries",
        "anxiety",
        "pessimism",
        "pessimistic",
        "cost of living crisis",
        "belt tightening",
        "debt burden",
        "foreclosure",
        "bankruptcy",
        "distress",
    ),
    "positive": (
        "recovery",
        "recovering",
        "rebound",
        "rebounding",
        "expansion",
        "growth",
        "growing",
        "boom",
        "booming",
        "strong",
        "strength",
        "strengthening",
        "robust",
        "resilient",
        "resilience",
        "improving",
        "improvement",
        "improved",
        "gains",
        "upbeat",
        "optimism",
        "optimistic",
        "confidence",
        "cooling inflation",
        "easing inflation",
        "prices falling",
        "wage gains",
        "hiring",
        "job gains",
        "record low unemployment",
        "affordable",
        "relief",
        "solid",
        "healthy",
        "stabilising",
        "stabilizing",
        "momentum",
    ),
    # Selection / salience: words that make a story *prominent* regardless of
    # its sign. High salience density is how a 3% economy gets covered like a
    # catastrophe without a single false statement.
    "selection_salience": (
        "crisis",
        "unprecedented",
        "record",
        "record high",
        "historic",
        "alarming",
        "shocking",
        "warning",
        "warns",
        "emergency",
        "spiral",
        "spiralling",
        "spiraling",
        "breaking point",
        "worst",
        "highest since",
        "lowest since",
        "decades",
        "crushing",
        "devastating",
        "dire",
        "exclusive",
        "revealed",
    ),
    # Factual calibration: phrases making a *checkable magnitude claim*. Each
    # also appears in CALIBRATION_CLAIMS with the inflation rate it implies, so
    # the claim can be scored against the macro panel rather than taken on tone.
    "factual_calibration": (
        "soaring",
        "skyrocketing",
        "spiking",
        "surging",
        "exploding",
        "runaway inflation",
        "hyperinflation",
        "out of control",
        "rampant",
        "climbing",
        "rising",
        "elevated",
        "easing",
        "cooling",
        "moderating",
        "plummeting",
        "collapsing",
        "slowing",
        "flat",
        "steady",
    ),
    # Political attribution: who is blamed or credited, and whether the article
    # asserts or merely reports an assertion. The attribution verbs matter as
    # much as the partisan markers — "critics warn X" is reporting, not claiming.
    "political_attribution": (
        "biden",
        "trump",
        "white house",
        "administration",
        "congress",
        "democrats",
        "democrat",
        "republicans",
        "republican",
        "gop",
        "the fed",
        "federal reserve",
        "midterms",
        "midterm",
        "election",
        "campaign",
        "blame",
        "blames",
        "blamed",
        "credit",
        "credits",
        "critics say",
        "critics warn",
        "opponents say",
        "accused",
        "accuses",
        "slammed",
        "defended",
        "according to",
        "claims",
        "alleges",
    ),
}

#: Polarity dimensions: the only ones that enter the tone index and the only
#: ones negation flips.
POLAR_DIMENSIONS: tuple[str, str] = ("negative", "positive")

#: The framing dimension, which the assignment requires as its own measure.
#:
#: The brief defines framing as the "emphasis on price levels versus inflation
#: rates, national aggregates versus household experiences, and current
#: conditions versus forecasts". Each contrast is a PAIR, and the measure is a
#: ratio between the two sides, not a count of one side. A count would only
#: measure how much an outlet writes about the economy.
#:
#: Each contrast returns a value in [-1, +1]:
#:   level_vs_rate      +1 = all price-LEVEL language, -1 = all RATE language
#:   household_vs_macro +1 = all HOUSEHOLD language,   -1 = all AGGREGATE language
#:   current_vs_forecast +1 = all CURRENT language,    -1 = all FORECAST language
#:
#: A positive `level_vs_rate` is the framing that Hypothesis A describes: a
#: household anchors to the price it pays, not to the rate of change. So this
#: measure connects the media block to the economic block, instead of being a
#: fifth number with no argument behind it.
FRAMING_LEXICON: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "level_vs_rate": (
        (
            # Single-word anchors first. A headline is about nine words long, so
            # a lexicon of only long phrases produces a month with two hits and
            # a ratio of +-1, which is noise wearing the costume of a measure.
            # "price" names the amount a household pays; "inflation" names its
            # rate of change. That is the contrast the brief asks for, and it is
            # the standard word-frequency frame indicator.
            "price", "prices", "pricing", "cost", "costs", "costly",
            "expensive", "affordability", "affordable", "unaffordable",
            "bill", "bills", "fare", "fares", "rent", "rents",
            "price level", "price levels", "cost of living", "grocery bill",
            "still cost", "more expensive than", "compared with 2020",
            "since the pandemic", "cumulative", "sticker shock", "price tag",
            "costs more", "higher prices", "elevated prices", "grocery prices",
            "gas prices", "food prices", "housing costs", "out of reach",
        ),
        (
            "inflation", "inflationary", "cpi", "deflation", "disinflation",
            "inflation rate", "rate of inflation", "year over year",
            "annual rate", "monthly rate", "cooling", "cooled",
            "inflation fell", "inflation eased", "inflation slowed",
            "percent annual", "price growth", "rate of increase",
            "basis points", "annualized", "pace of price",
        ),
    ),
    "household_vs_macro": (
        (
            "household", "families", "shoppers", "consumers say", "at the pump",
            "at the checkout", "paycheck", "rent", "grocery", "kitchen table",
            "working families", "budget",
        ),
        (
            "gross domestic product", "gdp", "aggregate", "national economy",
            "economy grew", "economy expanded", "headline figure",
            "macroeconomic", "nationwide", "overall economy",
        ),
    ),
    "current_vs_forecast": (
        (
            "currently", "right now", "this month", "today", "latest reading",
            "reported", "data showed", "came in at",
        ),
        (
            "forecast", "expected to", "projected", "outlook", "will likely",
            "could reach", "analysts expect", "next year", "ahead of",
            "braces for", "warns of",
        ),
    ),
}


#: Implied YoY inflation (percent) behind each magnitude claim, used by
#: `calibration_gap`. These are judgements, not measurements, which is exactly
#: why they live in a module-level dict where a reviewer can argue with them
#: instead of being buried in a function body. They are deliberately generous to
#: the press: "soaring" is read as 8%, not 20%.
CALIBRATION_CLAIMS: dict[str, float] = {
    "hyperinflation": 25.0,
    "runaway inflation": 15.0,
    "out of control": 12.0,
    "skyrocketing": 10.0,
    "exploding": 10.0,
    "rampant": 9.0,
    "soaring": 8.0,
    "spiking": 8.0,
    "surging": 7.0,
    "climbing": 5.0,
    "rising": 4.0,
    "elevated": 4.0,
    "moderating": 2.5,
    "slowing": 2.5,
    "cooling": 2.0,
    "easing": 2.0,
    "steady": 2.0,
    "flat": 1.0,
    "plummeting": 0.0,
    "collapsing": 0.0,
}

#: FinBERT emits market-sentiment labels; the probe needs policy stance. This is
#: the *charitable* mapping: a sentence that is good news for risk assets is read
#: as dovish, bad news as hawkish. BRAINSTORMING.md §4.1's canonical failure
#: ("strong growth and a tight labour market") is precisely a sentence where
#: this charitable mapping still gets the policy stance backwards.
FINBERT_LABEL_MAP: dict[str, str] = {
    "positive": "dovish",
    "negative": "hawkish",
    "neutral": "neutral",
}

FINBERT_MODEL = "ProsusAI/finbert"

PROBE_LABELS: tuple[str, str, str] = ("hawkish", "dovish", "neutral")

#: Hand-built macro probe set: 30 sentences with intended policy-stance labels.
#: Written to span the failure modes named in BRAINSTORMING.md §4.1 — plain
#: statements, conditionals, hedged persistence and attributed assertions — not
#: to be easy.
FINBERT_PROBE: tuple[tuple[str, str], ...] = (
    # --- hawkish (10) ---
    (
        (
            "Strong productivity growth and robust labour market conditions "
            "continued through the quarter."
        ),
        "hawkish",
    ),
    ("If inflation fails to moderate, further tightening may be warranted.", "hawkish"),
    ("Wage growth remains well above levels consistent with the target.", "hawkish"),
    ("The labour market is exceptionally tight and job openings keep climbing.", "hawkish"),
    ("Core services inflation has proven stickier than previously anticipated.", "hawkish"),
    ("Demand continues to outpace supply across much of the economy.", "hawkish"),
    ("The committee judges that additional policy firming will be appropriate.", "hawkish"),
    ("Consumer spending accelerated again despite higher borrowing costs.", "hawkish"),
    ("Inflation expectations have begun to drift above the longer-run anchor.", "hawkish"),
    ("Economic activity expanded at a solid pace and risks are to the upside.", "hawkish"),
    # --- dovish (10) ---
    ("Payroll growth has slowed markedly and the unemployment rate has risen.", "dovish"),
    ("Disinflation is now broad-based across goods and services categories.", "dovish"),
    ("Downside risks to employment have become more prominent.", "dovish"),
    ("The committee sees scope to begin reducing the policy rate.", "dovish"),
    ("Credit conditions have tightened sharply and lending has contracted.", "dovish"),
    ("Manufacturing output declined for a fourth consecutive month.", "dovish"),
    ("Inflation is expected to prove transitory.", "dovish"),
    ("Consumer demand has weakened and firms report falling order books.", "dovish"),
    ("Policy is now judged to be meaningfully restrictive.", "dovish"),
    ("Housing activity has cooled considerably under higher mortgage rates.", "dovish"),
    # --- neutral (10) ---
    ("The committee will continue to assess incoming data.", "neutral"),
    ("Personal consumption expenditures rose 0.3 percent in the month.", "neutral"),
    ("The next meeting is scheduled for the middle of next month.", "neutral"),
    ("Critics warn the economy is deteriorating ahead of the vote.", "neutral"),
    ("Supporters say the administration has delivered a strong recovery.", "neutral"),
    ("Analysts are divided on the outlook for the second half.", "neutral"),
    ("The report revised the prior month's estimate modestly higher.", "neutral"),
    ("Survey respondents cited both prices and employment as concerns.", "neutral"),
    ("The statement left the target range unchanged.", "neutral"),
    ("Data releases this week include retail sales and industrial production.", "neutral"),
)


class MediaError(Exception):
    """Raised when a corpus or text transform cannot be built as specified.

    Mirrors `DataError` / `ModelError`: callers should never have to catch
    pandas', sklearn's or scipy's internal exception types to discover that a
    column was missing or a vocabulary came out empty.
    """


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise MediaError(f"frame is missing required columns: {missing}")


# --------------------------------------------------------------------------
# 1. GDELT ingestion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusReport:
    """What a corpus pull or tidy actually produced, and what it discarded.

    `dropped_reasons` is a reason -> count mapping rather than a bare total:
    "400 documents dropped" is not auditable, "400 non-English, 12 duplicate
    URLs" is.
    """

    requested_terms: tuple[str, ...]
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    raw_rows: int
    kept_rows: int
    dropped_rows: int
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    cached_chunks: int = 0
    fetched_chunks: int = 0
    failed_chunks: tuple[str, ...] = field(default_factory=tuple)
    network_available: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def offline(self) -> bool:
        """True when nothing was fetched over the network this run.

        Note this is *not* a failure: a fully cached run is also offline, which
        is the intended steady state.
        """
        return self.fetched_chunks == 0

    def summary(self) -> str:
        span = "no rows"
        if self.start is not None and self.end is not None:
            span = f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
        parts = [
            f"{self.kept_rows} documents kept of {self.raw_rows} raw ({span})",
            f"cache: {self.cached_chunks} chunks, network: {self.fetched_chunks} chunks",
        ]
        if self.dropped_reasons:
            detail = ", ".join(
                f"{reason} x{count}" for reason, count in sorted(self.dropped_reasons.items())
            )
            parts.append(f"dropped: {detail}")
        if self.failed_chunks:
            parts.append(f"failed chunks: {len(self.failed_chunks)}")
        return " | ".join(parts)


def _term_key(terms: Sequence[str]) -> str:
    """Stable short hash of the query terms, for cache file names."""
    joined = "|".join(sorted(terms))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def tidy_articles(raw: pd.DataFrame, terms: Sequence[str] = ()) -> tuple[pd.DataFrame, CorpusReport]:
    """Normalise a raw GDELT article list into the project's tidy schema.

    Returns ``(frame, report)`` where the frame has columns
    ``date, title, url, domain, text, language, sourcecountry`` indexed 0..n-1,
    and the report names every discarded row.

    **On the ``text`` column.** GDELT's Doc 2.0 API serves *metadata*, not
    article bodies — there is no full text to be had without scraping each URL,
    which the project does not do. ``text`` is therefore the headline. That is a
    real limitation and it is stated here rather than in a footnote: headlines
    are the part of coverage most readers actually consume and the part where
    framing is most concentrated, but a headline-only corpus cannot measure
    body-text hedging. Downstream code reads ``text`` and does not care where it
    came from, so swapping in scraped bodies later changes nothing else.
    """
    expected = ("date", "title", "url", "domain", "text", "language", "sourcecountry")
    dropped: dict[str, int] = {}

    if raw is None or raw.empty:
        empty = pd.DataFrame({name: pd.Series(dtype="object") for name in expected})
        empty["date"] = pd.Series(dtype="datetime64[ns]")
        return empty[list(expected)], CorpusReport(
            requested_terms=tuple(terms),
            start=None,
            end=None,
            raw_rows=0,
            kept_rows=0,
            dropped_rows=0,
            notes=("no raw rows to tidy",),
        )

    frame = raw.copy()
    raw_rows = len(frame)

    # GDELT names the timestamp `seendate` (e.g. 20260512T120000Z).
    if "seendate" in frame.columns:
        stamps = pd.to_datetime(frame["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
        fallback = pd.to_datetime(frame["seendate"], errors="coerce", utc=True, format="mixed")
        stamps = stamps.fillna(fallback)
    elif "date" in frame.columns:
        stamps = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
    else:
        raise MediaError("raw articles have neither a `seendate` nor a `date` column")

    frame["date"] = stamps.dt.tz_convert(None).dt.normalize()

    for column in ("title", "url", "domain", "language", "sourcecountry"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = frame[column].astype("object")

    def _drop(mask: pd.Series, reason: str) -> None:
        nonlocal frame
        count = int(mask.sum())
        if count:
            dropped[reason] = dropped.get(reason, 0) + count
            frame = frame.loc[~mask]

    _drop(frame["date"].isna(), "unparseable date")
    _drop(frame["url"].isna() | (frame["url"].astype(str).str.strip() == ""), "missing url")
    _drop(
        frame["title"].isna() | (frame["title"].astype(str).str.strip() == ""),
        "missing or empty title",
    )

    language = frame["language"].astype(str).str.strip().str.lower()
    non_english = ~language.isin({"english", "en", "eng", "", "nan", "<na>", "none"})
    _drop(non_english, "non-English")

    duplicates = frame.duplicated(subset="url", keep="first")
    _drop(duplicates, "duplicate url")

    frame["title"] = frame["title"].astype(str).str.strip()
    frame["text"] = frame["title"]
    frame["domain"] = frame["domain"].astype(str).str.strip().str.lower()

    tidy = frame.loc[:, list(expected)].sort_values(["date", "url"]).reset_index(drop=True)

    report = CorpusReport(
        requested_terms=tuple(terms),
        start=tidy["date"].min() if not tidy.empty else None,
        end=tidy["date"].max() if not tidy.empty else None,
        raw_rows=raw_rows,
        kept_rows=len(tidy),
        dropped_rows=raw_rows - len(tidy),
        dropped_reasons=dropped,
        notes=("`text` is the headline: GDELT Doc 2.0 serves metadata, not bodies",),
    )
    return tidy, report


class GdeltSource:
    """Cached GDELT Doc 2.0 client that degrades instead of exploding.

    `available` says the `gdeltdoc` package imported and a client was built; it
    does **not** promise the network works. Whether GDELT really delivered is
    decided by what `fetch_range` manages to return, exactly as `FredSource`
    separates `configured` from the panel it actually produced.
    """

    def __init__(
        self,
        cache_dir: str | Path = GDELT_CACHE_DIR,
        client: object | None = None,
        max_records: int = GDELT_MAX_RECORDS,
        allow_network: bool = True,
        min_interval_seconds: float = 6.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self.allow_network = bool(allow_network)
        self._client = client
        self._filters = None
        self._unavailable_reason: str | None = None
        # GDELT throttles bursts. When it does, `gdeltdoc` raises with an EMPTY
        # message, because the throttle response is not the JSON it expects —
        # so a rate limit looks exactly like a malformed query. Measured: ten
        # rapid probes were enough to start failing every call, including one
        # that had succeeded moments earlier. Space the requests out.
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_request_at: float = 0.0

        if client is not None:
            # Injected client (used by tests and by any offline replay).
            return
        if not allow_network:
            self._unavailable_reason = "network disabled by caller (allow_network=False)"
            return
        try:
            from gdeltdoc import Filters, GdeltDoc

            self._client = GdeltDoc()
            self._filters = Filters
        except Exception as exc:  # pragma: no cover - import environment dependent
            self._unavailable_reason = f"gdeltdoc unavailable ({exc})"
            logger.warning("GDELT client unavailable (%s); cache-only mode", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _cache_path(self, start: pd.Timestamp, end: pd.Timestamp, terms: Sequence[str]) -> Path:
        return self.cache_dir / f"{start:%Y%m%d}_{end:%Y%m%d}_{_term_key(terms)}.parquet"

    def _throttle(self) -> None:
        """Wait so that consecutive GDELT requests are spaced out.

        Does nothing when `min_interval_seconds` is 0, which is what tests and
        injected stub clients use.
        """
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if 0 < elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def _fetch_chunk(
        self, start: pd.Timestamp, end: pd.Timestamp, terms: Sequence[str]
    ) -> pd.DataFrame:
        """One network pull: every term queried, results concatenated.

        GDELT ANDs a keyword list, so a single multi-keyword query would return
        only documents mentioning *all* the terms — a handful of articles, not a
        corpus. Each term is therefore its own query and the union is
        deduplicated downstream in `tidy_articles`.
        """
        if self._client is None:
            raise MediaError(f"no GDELT client available: {self._unavailable_reason}")

        frames: list[pd.DataFrame] = []
        for term in terms:
            if self._filters is not None:
                query = self._filters(
                    keyword=term,
                    start_date=f"{start:%Y-%m-%d}",
                    end_date=f"{end:%Y-%m-%d}",
                    language="english",
                    num_records=self.max_records,
                )
            else:
                # Injected stub clients take the plain arguments.
                query = {
                    "keyword": term,
                    "start_date": f"{start:%Y-%m-%d}",
                    "end_date": f"{end:%Y-%m-%d}",
                }
            self._throttle()
            got = self._client.article_search(query)
            if got is not None and len(got):
                frames.append(pd.DataFrame(got))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        terms: Sequence[str] = ECONOMIC_TERMS,
        chunk_days: int = 1,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, CorpusReport]:
        """Fetch English economic news for ``[start, end]``, cache-first.

        The range is split into ``chunk_days``-wide chunks because the Doc API
        caps an article list at 250 records: one query over a month returns the
        250 articles GDELT happens to rank first, which is a sample of unknown
        shape. Daily chunks give a roughly even daily draw, which is what a
        daily tone index needs.

        Each chunk is cached as parquet under ``data/raw/gdelt/`` keyed by
        ``(chunk start, chunk end, term hash)``, so re-runs cost no network and
        a partially failed run resumes where it stopped. Chunk failures are
        recorded in the report rather than raised: a corpus with three missing
        days is still usable and the report says which days are missing.
        """
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise MediaError(f"end {end_ts:%Y-%m-%d} precedes start {start_ts:%Y-%m-%d}")
        if chunk_days < 1:
            raise MediaError(f"chunk_days must be >= 1, got {chunk_days}")
        terms = tuple(terms)
        if not terms:
            raise MediaError("no query terms supplied")

        chunks: list[pd.DataFrame] = []
        cached = fetched = 0
        failed: list[str] = []
        notes: list[str] = []

        cursor = start_ts
        while cursor <= end_ts:
            chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
            path = self._cache_path(cursor, chunk_end, terms)
            if path.exists() and not refresh:
                try:
                    chunks.append(pd.read_parquet(path))
                    cached += 1
                except Exception as exc:  # pragma: no cover - corrupt cache
                    failed.append(f"{cursor:%Y-%m-%d}: unreadable cache ({exc})")
            elif not self.available:
                failed.append(f"{cursor:%Y-%m-%d}: {self._unavailable_reason or 'no client'}")
            else:
                try:
                    got = self._fetch_chunk(cursor, chunk_end, terms)
                except Exception as exc:
                    # Network, rate-limit and parse failures are all the same
                    # thing from here: this chunk has no data and the report
                    # must say so.
                    failed.append(f"{cursor:%Y-%m-%d}: fetch failed ({exc})")
                    logger.warning("GDELT fetch failed for %s: %s", f"{cursor:%Y-%m-%d}", exc)
                else:
                    fetched += 1
                    if len(got):
                        got.to_parquet(path, index=False)
                        chunks.append(got)
                    else:
                        notes.append(f"{cursor:%Y-%m-%d}: GDELT returned no articles")
            cursor = chunk_end + pd.Timedelta(days=1)

        raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        tidy, tidy_report = tidy_articles(raw, terms=terms)

        if failed:
            notes.append(
                f"{len(failed)} of chunks unavailable; corpus has gaps (see failed_chunks)"
            )
        if not self.available and self._unavailable_reason:
            notes.append(f"network unavailable: {self._unavailable_reason}")

        report = CorpusReport(
            requested_terms=terms,
            start=tidy_report.start,
            end=tidy_report.end,
            raw_rows=tidy_report.raw_rows,
            kept_rows=tidy_report.kept_rows,
            dropped_rows=tidy_report.dropped_rows,
            dropped_reasons=tidy_report.dropped_reasons,
            cached_chunks=cached,
            fetched_chunks=fetched,
            failed_chunks=tuple(failed),
            network_available=self.available,
            notes=tuple(list(tidy_report.notes) + notes),
        )
        return tidy, report


# --------------------------------------------------------------------------
# 2. TF-IDF term-document matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TermDocumentMatrix:
    """A sparse TF-IDF term-document matrix, ``m`` terms by ``n`` documents.

    Orientation is terms-by-documents, matching BRAINSTORMING.md §3.1's
    ``X = U_k S_k V_k^T``, not sklearn's documents-by-terms. Getting this
    backwards silently swaps term loadings for document coordinates, so the
    shape is asserted on construction.

    `empty_doc_ids` names documents that vectorised to an all-zero column — no
    vocabulary term present. They are **kept**, not dropped: under the
    non-demeaned decomposition such a document maps to the origin and
    contributes nothing, which is the correct semantics (see `fit_lsa`), and
    dropping them would shrink the denominator of every daily index.
    """

    matrix: sp.csr_matrix
    terms: tuple[str, ...]
    doc_ids: tuple[object, ...]
    vectorizer: TfidfVectorizer
    fitted: bool
    empty_doc_ids: tuple[object, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        rows, cols = self.matrix.shape
        if rows != len(self.terms) or cols != len(self.doc_ids):
            raise MediaError(
                f"matrix shape {self.matrix.shape} does not match "
                f"{len(self.terms)} terms x {len(self.doc_ids)} documents"
            )

    @property
    def n_terms(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[1]

    @property
    def density(self) -> float:
        total = self.n_terms * self.n_docs
        return float(self.matrix.nnz / total) if total else 0.0

    @property
    def frobenius_sq(self) -> float:
        """``||X||_F^2``, the total energy the SVD's singular values share out."""
        return float(np.square(self.matrix.data).sum())

    def summary(self) -> str:
        return (
            f"{self.n_terms} terms x {self.n_docs} docs, density {self.density:.4f}, "
            f"{len(self.empty_doc_ids)} documents with no vocabulary terms"
        )


def _prepare_documents(
    documents: Sequence[str] | pd.Series, doc_ids: Sequence[object] | None
) -> tuple[list[str], tuple[object, ...]]:
    if isinstance(documents, pd.Series):
        ids = tuple(documents.index) if doc_ids is None else tuple(doc_ids)
        texts = ["" if pd.isna(value) else str(value) for value in documents.to_numpy()]
    else:
        texts = ["" if value is None else str(value) for value in documents]
        ids = tuple(range(len(texts))) if doc_ids is None else tuple(doc_ids)
    if len(ids) != len(texts):
        raise MediaError(f"{len(texts)} documents but {len(ids)} document ids")
    if not texts:
        raise MediaError("no documents supplied")
    return texts, ids


def fit_term_document(
    documents: Sequence[str] | pd.Series,
    doc_ids: Sequence[object] | None = None,
    min_df: float = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int | None = None,
    sublinear_tf: bool = True,
) -> TermDocumentMatrix:
    """Fit TF-IDF on the **training** documents and return ``X`` (terms x docs).

    Sparse throughout: the matrix is never densified, here or anywhere
    downstream, because the whole case for `svds` over a dense eigendecomposition
    rests on it (BRAINSTORMING.md §3.2).

    ``sublinear_tf=True`` uses ``1 + log(tf)``. News headlines repeat a term two
    or three times at most, and where they do the second mention carries far
    less information than the first; the log damping is the standard LSA choice
    for that reason.

    The fitted vocabulary and IDF vector are the look-ahead surface: they must be
    estimated here, on the training window only, and applied to the holdout by
    `apply_term_document`.
    """
    texts, ids = _prepare_documents(documents, doc_ids)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        dtype=np.float64,
    )
    try:
        doc_term = vectorizer.fit_transform(texts)
    except ValueError as exc:
        # The usual cause is min_df/max_df pruning the vocabulary to nothing on
        # a small corpus. Say that, rather than letting sklearn's message out.
        raise MediaError(
            f"TF-IDF vocabulary came out empty (min_df={min_df}, max_df={max_df}, "
            f"{len(texts)} documents): {exc}"
        ) from exc

    matrix = sp.csr_matrix(doc_term.T)
    terms = tuple(vectorizer.get_feature_names_out())
    empty = _empty_columns(matrix, ids)
    notes = [
        (
            f"TF-IDF fitted on {len(texts)} documents (min_df={min_df}, "
            f"max_df={max_df}, ngram_range={ngram_range}, sublinear_tf={sublinear_tf})"
        )
    ]
    if empty:
        notes.append(
            f"{len(empty)} documents contain no vocabulary term and map to the "
            "origin; kept, not dropped"
        )
    return TermDocumentMatrix(
        matrix=matrix,
        terms=terms,
        doc_ids=ids,
        vectorizer=vectorizer,
        fitted=True,
        empty_doc_ids=empty,
        notes=tuple(notes),
    )


def apply_term_document(
    fitted: TermDocumentMatrix,
    documents: Sequence[str] | pd.Series,
    doc_ids: Sequence[object] | None = None,
) -> TermDocumentMatrix:
    """Apply an already-fitted TF-IDF to holdout documents. No refit.

    Terms absent from the training vocabulary are discarded — which is the
    point. A holdout document written entirely in new vocabulary maps to the
    origin, and that is an honest statement about a basis estimated before it
    existed, not a bug.
    """
    if not fitted.fitted:
        raise MediaError("cannot apply a term-document matrix that was never fitted")
    texts, ids = _prepare_documents(documents, doc_ids)
    doc_term = fitted.vectorizer.transform(texts)
    matrix = sp.csr_matrix(doc_term.T)
    empty = _empty_columns(matrix, ids)
    notes = [f"TF-IDF applied (not refitted) to {len(texts)} holdout documents"]
    if empty:
        notes.append(f"{len(empty)} holdout documents contain no training-vocabulary term")
    return TermDocumentMatrix(
        matrix=matrix,
        terms=fitted.terms,
        doc_ids=ids,
        vectorizer=fitted.vectorizer,
        fitted=False,
        empty_doc_ids=empty,
        notes=tuple(notes),
    )


def _empty_columns(matrix: sp.csr_matrix, ids: Sequence[object]) -> tuple[object, ...]:
    column_nnz = np.asarray(sp.csc_matrix(matrix).getnnz(axis=0)).ravel()
    return tuple(ids[i] for i in np.flatnonzero(column_nnz == 0))


# --------------------------------------------------------------------------
# 3. Truncated SVD / LSA, without demeaning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LsaBasis:
    """A truncated SVD of the term-document matrix: ``X ~ U_k S_k V_k^T``.

    `term_loadings` is ``U_k`` (m x k), latent topics in term space;
    `doc_coordinates` is ``V_k`` (n x k), one row per training document;
    `singular_values` is the diagonal of ``S_k``, descending.

    **Not demeaned, on purpose.** Two reasons, the second of which is the real
    one (BRAINSTORMING.md §3.2):

    *Sparsity.* Subtracting any mean makes every zero entry non-zero, taking
    storage from O(nnz) to O(mn) and destroying the property `svds` exploits.

    *Zero-prior drift.* A document containing none of the dictionary terms maps
    to the zero vector and contributes nothing, which is the correct semantics —
    *absence of signal*, not *average signal*. After demeaning it would acquire
    minus-the-mean and be assigned a position in topic space it never earned, so
    daily salience aggregates would drift with however many off-topic documents
    happened to be collected that day. The origin must mean "no terms present".

    So this is not PCA on the covariance matrix; it diagonalises the second
    moment ``X^T X`` and answers "which directions of co-occurrence carry most
    energy", which is the question latent *semantic* analysis asks.
    """

    singular_values: np.ndarray
    term_loadings: np.ndarray
    doc_coordinates: np.ndarray
    terms: tuple[str, ...]
    doc_ids: tuple[object, ...]
    frobenius_sq: float
    demeaned: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def k(self) -> int:
        return len(self.singular_values)

    @property
    def energy_share(self) -> np.ndarray:
        """Per-component share of total ``||X||_F^2``."""
        if self.frobenius_sq <= 0:
            return np.zeros_like(self.singular_values)
        return np.square(self.singular_values) / self.frobenius_sq

    @property
    def cumulative_energy(self) -> np.ndarray:
        return np.cumsum(self.energy_share)

    def summary(self) -> str:
        return (
            f"k={self.k}, sigma_1={self.singular_values[0]:.4f}, "
            f"cumulative energy={self.cumulative_energy[-1]:.3f}, demeaned={self.demeaned}"
        )


def _orient(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix each component's sign so its largest-magnitude term loading is positive.

    The SVD determines ``(u_i, v_i)`` only up to a joint sign flip. Without a
    convention the top-terms table for a topic can come back as the *negative*
    of the same topic between runs or platforms, which would make Table 2
    irreproducible. Flipping both vectors together leaves ``U S V^T``
    unchanged.
    """
    for column in range(u.shape[1]):
        pivot = int(np.argmax(np.abs(u[:, column])))
        if u[pivot, column] < 0:
            u[:, column] *= -1.0
            v[:, column] *= -1.0
    return u, v


def fit_lsa(tdm: TermDocumentMatrix, k: int, random_state: int | None = 0) -> LsaBasis:
    """Truncated SVD of ``X`` with `scipy.sparse.linalg.svds`, no demeaning.

    ``svds`` returns triplets in ascending order of singular value; they are
    reversed here so component 0 is always the leading one.

    `k` must satisfy ``1 <= k < min(m, n)``: `svds` is an iterative Lanczos
    method on the sparse operator and cannot return the full spectrum. Asking
    for too many is a caller error with a clear message rather than a scipy
    traceback.
    """
    if k < 1:
        raise MediaError(f"k must be at least 1, got {k}")
    limit = min(tdm.n_terms, tdm.n_docs)
    if k >= limit:
        raise MediaError(
            f"k={k} must be strictly less than min(n_terms, n_docs)={limit}; "
            "svds cannot return the full spectrum"
        )
    if tdm.matrix.nnz == 0:
        raise MediaError("term-document matrix is entirely zero; nothing to decompose")

    rng = np.random.default_rng(random_state)
    v0 = rng.normal(size=limit)
    u, s, vt = svds(tdm.matrix.asfptype(), k=k, v0=v0)

    order = np.argsort(s)[::-1]
    s = np.asarray(s, dtype=float)[order]
    u = np.asarray(u, dtype=float)[:, order]
    v = np.asarray(vt, dtype=float)[order, :].T
    u, v = _orient(u, v)

    return LsaBasis(
        singular_values=s,
        term_loadings=u,
        doc_coordinates=v,
        terms=tdm.terms,
        doc_ids=tdm.doc_ids,
        frobenius_sq=tdm.frobenius_sq,
        demeaned=False,
        notes=(
            (
                "non-demeaned truncated SVD (second-moment, not covariance); "
                "a document with no vocabulary terms maps to the origin"
            ),
        ),
    )


def project_documents(basis: LsaBasis, tdm: TermDocumentMatrix) -> pd.DataFrame:
    """Place holdout documents in the training topic space. No refit.

    ``V = X^T U_k S_k^{-1}`` — the same algebra that produced ``V_k`` on the
    training matrix, applied to new columns with the basis held fixed. This is
    where the no-look-ahead rule is actually enforced for the SVD: nothing about
    the holdout's own co-occurrence structure enters ``U_k`` or ``S_k``.

    Because there is no demeaning, an all-zero holdout column projects to an
    all-zero row. That property is tested, not assumed.
    """
    if tuple(tdm.terms) != tuple(basis.terms):
        raise MediaError(
            "term-document matrix vocabulary does not match the basis; "
            "use apply_term_document with the fitted matrix"
        )
    inverse = np.zeros_like(basis.singular_values)
    nonzero = basis.singular_values > 0
    inverse[nonzero] = 1.0 / basis.singular_values[nonzero]
    coordinates = (tdm.matrix.T @ basis.term_loadings) * inverse
    return pd.DataFrame(
        np.asarray(coordinates),
        index=pd.Index(tdm.doc_ids, name="doc_id"),
        columns=[f"topic_{i}" for i in range(basis.k)],
    )


def top_terms_per_topic(basis: LsaBasis, n_terms: int = 10) -> pd.DataFrame:
    """Top-|loading| terms per latent topic — the report's Table 2.

    Ranked by absolute loading, with the signed loading reported alongside.
    Beyond the first component, loadings are genuinely signed: a topic's
    negative tail is the vocabulary that *contrasts* with its positive tail, and
    hiding the sign would misrepresent the component as a word list.
    """
    if n_terms < 1:
        raise MediaError(f"n_terms must be at least 1, got {n_terms}")
    rows: list[dict[str, object]] = []
    for topic in range(basis.k):
        loadings = basis.term_loadings[:, topic]
        order = np.argsort(np.abs(loadings))[::-1][:n_terms]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "topic": topic,
                    "rank": rank,
                    "term": basis.terms[index],
                    "loading": float(loadings[index]),
                    "singular_value": float(basis.singular_values[topic]),
                    "energy_share": float(basis.energy_share[topic]),
                }
            )
    return pd.DataFrame(rows)


def reconstruction_error(tdm: TermDocumentMatrix, basis: LsaBasis) -> float:
    """Relative Frobenius error ``||X - U_k S_k V_k^T||_F / ||X||_F``.

    Computed from the spectrum rather than by forming the dense residual:
    Eckart-Young gives ``||X - X_k||_F^2 = ||X||_F^2 - sum(sigma_i^2)``, so the
    error is available in closed form and densifying an m x n matrix to measure
    it would defeat the point of the sparse decomposition.
    """
    total = tdm.frobenius_sq
    if total <= 0:
        raise MediaError("term-document matrix has zero energy; error is undefined")
    captured = float(np.square(basis.singular_values).sum())
    residual = max(total - captured, 0.0)  # clamp float noise at k -> rank
    return float(np.sqrt(residual / total))


# --------------------------------------------------------------------------
# 4. k-selection diagnostics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KSelection:
    """k-selection evidence, returned as data. Nothing here prints or plots.

    All three diagnostics of BRAINSTORMING.md §3.3 are returned together, so the
    report quotes the disagreement rather than whichever one flatters the
    chosen k.

    Two readings of the permutation null are returned because they disagree in a
    way that matters:

    * `k_noise` — components whose singular value exceeds the **largest**
      singular value any permutation produced. This is the form §3.3 specifies
      and it is deliberately conservative.
    * `k_parallel` — the leading run of components beating the permuted
      distribution **at their own position** (Horn's parallel analysis). Less
      conservative, and the one to prefer when the two disagree and the
      corpus is dense.

    **Why they can disagree sharply.** ``X`` is non-negative and is not demeaned,
    so a large part of ``sigma_1`` is the overall "mean" direction — and a
    within-column permutation preserves exactly that. On a dense matrix the
    permuted ``sigma_1`` can therefore match or exceed the real one and
    `k_noise` collapses to zero even when real topics exist. At realistic TF-IDF
    sparsity (a few percent) the effect is small and the two readings agree.
    This is a property of permuting a non-demeaned non-negative matrix, not a
    defect in the corpus, and the report should say so rather than quietly
    switching diagnostics.
    """

    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    noise_floor: float
    permuted_spectrum: np.ndarray
    permuted_component_max: np.ndarray
    k_energy: int
    k_noise: int
    k_parallel: int
    energy_target: float
    n_permutations: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def k_max(self) -> int:
        return len(self.singular_values)

    def spectrum_frame(self) -> pd.DataFrame:
        """Tidy scree table: one row per component, for plotting elsewhere."""
        return pd.DataFrame(
            {
                "component": np.arange(1, self.k_max + 1),
                "singular_value": self.singular_values,
                "cumulative_energy": self.cumulative_energy,
                "permuted_max": self.permuted_component_max,
                "above_noise_floor": self.singular_values > self.noise_floor,
                "above_permuted_component": self.singular_values
                > self.permuted_component_max,
            }
        )

    def summary(self) -> str:
        return (
            f"k_energy={self.k_energy} (target {self.energy_target:.0%}), "
            f"k_noise={self.k_noise} (floor {self.noise_floor:.4f} from "
            f"{self.n_permutations} permutations), k_parallel={self.k_parallel}"
        )


def _permute_within_columns(matrix: sp.csc_matrix, rng: np.random.Generator) -> sp.csc_matrix:
    """Shuffle each column's non-zero values across row positions.

    Within-column shuffling preserves every column's sparsity pattern size and
    its norm — so total ``||X||_F^2`` is unchanged — while destroying which
    *terms* co-occur in which document. The permuted spectrum is therefore the
    spectrum of a matrix with identical document-level marginals and no semantic
    structure, which is exactly the null the noise floor needs.
    """
    out = matrix.copy().tocsc()
    n_rows = out.shape[0]
    for column in range(out.shape[1]):
        start, stop = out.indptr[column], out.indptr[column + 1]
        count = stop - start
        if count == 0:
            continue
        out.indices[start:stop] = np.sort(
            rng.choice(n_rows, size=count, replace=False).astype(out.indices.dtype)
        )
    out.has_sorted_indices = True
    return out


def select_k(
    tdm: TermDocumentMatrix,
    k_max: int | None = None,
    energy_target: float = 0.9,
    n_permutations: int = 10,
    random_state: int | None = 0,
) -> KSelection:
    """Compute the three k diagnostics: spectrum, cumulative energy, noise floor.

    The noise floor is the **maximum** singular value observed across
    `n_permutations` within-column shuffles, which is the practical, assumption
    free stand-in for the Marchenko-Pastur bulk edge: components whose singular
    value never exceeds anything a structureless matrix of the same shape and
    sparsity produced are not distinguishable from text noise.

    Taking the max (rather than a quantile) is the conservative choice — it
    under-counts components — and is stated as such because a generous k is how
    an LSA pipeline ends up fitting headline noise and calling it a topic. The
    full permuted spectrum is returned too, with `k_parallel` as the
    component-wise (parallel-analysis) reading; `KSelection` documents when and
    why the two disagree.

    Returns data only. `k_energy`, `k_noise` and `k_parallel` are candidates;
    the caller picks and the report shows all three.
    """
    if not 0 < energy_target <= 1:
        raise MediaError(f"energy_target must be in (0, 1], got {energy_target}")
    if n_permutations < 1:
        raise MediaError(f"n_permutations must be at least 1, got {n_permutations}")

    limit = min(tdm.n_terms, tdm.n_docs) - 1
    if limit < 1:
        raise MediaError(
            f"matrix is too small to decompose: {tdm.n_terms} terms x {tdm.n_docs} docs"
        )
    k_max = limit if k_max is None else min(int(k_max), limit)
    if k_max < 1:
        raise MediaError(f"k_max must be at least 1, got {k_max}")

    basis = fit_lsa(tdm, k=k_max, random_state=random_state)
    singular_values = basis.singular_values
    cumulative = basis.cumulative_energy

    rng = np.random.default_rng(random_state)
    source = sp.csc_matrix(tdm.matrix)
    spectra: list[np.ndarray] = []
    for _ in range(n_permutations):
        shuffled = _permute_within_columns(source, rng)
        if shuffled.nnz == 0:
            spectra.append(np.zeros(k_max))
            continue
        values = svds(shuffled.asfptype(), k=k_max, return_singular_vectors=False)
        spectra.append(np.sort(np.asarray(values, dtype=float))[::-1])
    permuted_spectrum = np.vstack(spectra)
    permuted_component_max = permuted_spectrum.max(axis=0)
    noise_floor = float(permuted_component_max.max())

    above_target = np.flatnonzero(cumulative >= energy_target)
    k_energy = int(above_target[0] + 1) if above_target.size else k_max
    k_noise = int(np.sum(singular_values > noise_floor))
    beats_own_position = singular_values > permuted_component_max
    failures = np.flatnonzero(~beats_own_position)
    k_parallel = int(failures[0]) if failures.size else k_max

    notes = [
        f"k_max={k_max} singular triplets computed (limit min(m,n)-1={limit})",
        (
            "noise floor = max singular value over within-column permutations "
            "(conservative stand-in for the Marchenko-Pastur bulk edge)"
        ),
    ]
    if k_energy == k_max and cumulative[-1] < energy_target:
        notes.append(
            f"cumulative energy only reaches {cumulative[-1]:.3f} at k_max; "
            f"the {energy_target:.0%} target is not attained within k_max"
        )
    if k_noise == 0:
        notes.append(
            "no component exceeds the permutation noise floor: this corpus shows "
            "no latent structure distinguishable from noise"
        )
    if k_noise == k_max:
        notes.append(
            "every computed component exceeds the noise floor; k_max may be binding"
        )
    if k_parallel != k_noise:
        notes.append(
            f"the two permutation readings disagree (k_noise={k_noise}, "
            f"k_parallel={k_parallel}); at density {tdm.density:.4f} the "
            "non-demeaned mean direction is partly preserved by the permutation"
        )

    return KSelection(
        singular_values=singular_values,
        cumulative_energy=cumulative,
        noise_floor=noise_floor,
        permuted_spectrum=permuted_spectrum,
        permuted_component_max=permuted_component_max,
        k_energy=k_energy,
        k_noise=k_noise,
        k_parallel=k_parallel,
        energy_target=float(energy_target),
        n_permutations=int(n_permutations),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 5. Domain phrase lexicon, negation-aware
# --------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Phrases built once at import: {dimension: [(token tuple, phrase string)]},
# longest first so "cost of living crisis" wins over "crisis".
_COMPILED_LEXICON: dict[str, list[tuple[tuple[str, ...], str]]] = {
    dimension: sorted(
        ((tuple(_TOKEN_PATTERN.findall(phrase.lower())), phrase) for phrase in phrases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for dimension, phrases in LEXICON.items()
}

_NEGATOR_TOKENS: set[str] = set()
_NEGATOR_BIGRAMS: set[tuple[str, str]] = set()
for _negator in NEGATORS:
    _parts = tuple(_TOKEN_PATTERN.findall(_negator.lower()))
    if len(_parts) == 1:
        _NEGATOR_TOKENS.add(_parts[0])
    elif len(_parts) == 2:
        _NEGATOR_BIGRAMS.add(_parts)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with contraction apostrophes stripped.

    ``don't`` -> ``dont`` so that the negator list can be written without
    apostrophe variants, and ``inflation's`` -> ``inflations`` consistently
    rather than depending on the source's choice of apostrophe character.
    """
    if text is None:
        return []
    normalised = str(text).lower().replace("’", "'")
    return [token.replace("'", "") for token in _TOKEN_PATTERN.findall(normalised)]


def _is_negated(tokens: Sequence[str], position: int, window: int = NEGATION_WINDOW) -> bool:
    """Is there a negator within `window` tokens before `position`?

    A fixed backward window is crude — it will catch "no one doubts the recovery
    is strong" as a negated positive — but it handles the dominant news-headline
    construction ("not improving", "no recovery in sight", "fails to improve")
    and it is inspectable. The alternative, dependency parsing, is a much larger
    dependency for a gain this corpus cannot demonstrate.
    """
    start = max(0, position - window)
    for index in range(start, position):
        if tokens[index] in _NEGATOR_TOKENS:
            return True
        if index + 1 < position and (tokens[index], tokens[index + 1]) in _NEGATOR_BIGRAMS:
            return True
    return False


@dataclass(frozen=True)
class LexiconScore:
    """Lexicon counts for one document across the four §4.3 dimensions.

    `positive` / `negative` are *post-negation* counts: "not improving" counts
    as negative, not positive. `negated_hits` records how many flips happened,
    so the negation handling can be audited on real text instead of trusted.
    """

    n_tokens: int
    positive: int
    negative: int
    selection_salience: int
    factual_calibration: int
    political_attribution: int
    negated_hits: int
    matched_phrases: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def polar_total(self) -> int:
        """Denominator of the tone index: polar hits only, not all tokens."""
        return self.positive + self.negative

    @property
    def tone(self) -> float:
        """``(Pos - Neg) / Total`` in [-1, +1]; 0.0 when no polar term appears.

        A document with no polar language is tonally neutral, which is 0.0 — the
        same number the index gives for balanced coverage. The two are
        distinguishable via `polar_total`, and daily aggregation uses summed
        counts rather than averaged per-document tones precisely so that a day
        of silent documents cannot pull the index around.
        """
        total = self.polar_total
        if total == 0:
            return 0.0
        return float((self.positive - self.negative) / total)

    @property
    def salience_density(self) -> float:
        """Salience hits per token: how loudly a story is told, length-adjusted."""
        if self.n_tokens == 0:
            return 0.0
        return float(self.selection_salience / self.n_tokens)


def score_text(text: str) -> LexiconScore:
    """Score one document against `LEXICON`, negation-aware.

    Matching is longest-phrase-first with non-overlapping consumption **within a
    dimension**: "cost of living crisis" consumes its tokens so "crisis" cannot
    double-count inside it. Dimensions are scored independently of one another,
    because they answer different questions and a phrase may legitimately be
    both salient and a magnitude claim.

    Negation flips polarity for `positive`/`negative` only; see `LEXICON` for
    why the descriptive dimensions are not flipped.
    """
    tokens = tokenize(text)
    counts = {dimension: 0 for dimension in LEXICON}
    negated_hits = 0
    matched: list[tuple[str, str]] = []

    for dimension, phrases in _COMPILED_LEXICON.items():
        consumed = np.zeros(len(tokens), dtype=bool)
        for phrase_tokens, phrase in phrases:
            width = len(phrase_tokens)
            if width == 0 or width > len(tokens):
                continue
            for start in range(len(tokens) - width + 1):
                if consumed[start : start + width].any():
                    continue
                if tuple(tokens[start : start + width]) != phrase_tokens:
                    continue
                consumed[start : start + width] = True
                target = dimension
                if dimension in POLAR_DIMENSIONS and _is_negated(tokens, start):
                    target = "positive" if dimension == "negative" else "negative"
                    negated_hits += 1
                counts[target] += 1
                matched.append((target, phrase))

    return LexiconScore(
        n_tokens=len(tokens),
        positive=counts["positive"],
        negative=counts["negative"],
        selection_salience=counts["selection_salience"],
        factual_calibration=counts["factual_calibration"],
        political_attribution=counts["political_attribution"],
        negated_hits=negated_hits,
        matched_phrases=tuple(matched),
    )


def score_documents(
    frame: pd.DataFrame, text_column: str = "text", date_column: str = "date"
) -> pd.DataFrame:
    """Score every document in a tidy corpus frame; one row in, one row out.

    No document is dropped: a document with no lexicon hits scores zeros, which
    is information (it is off-topic or neutrally written), not an absence.
    """
    _require_columns(frame, (text_column, date_column))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                date_column,
                "n_tokens",
                "positive",
                "negative",
                "polar_total",
                "tone",
                "selection_salience",
                "factual_calibration",
                "political_attribution",
                "negated_hits",
                "salience_density",
            ]
        )
    rows = []
    for value in frame[text_column].to_numpy():
        score = score_text("" if pd.isna(value) else str(value))
        rows.append(
            {
                "n_tokens": score.n_tokens,
                "positive": score.positive,
                "negative": score.negative,
                "polar_total": score.polar_total,
                "tone": score.tone,
                "selection_salience": score.selection_salience,
                "factual_calibration": score.factual_calibration,
                "political_attribution": score.political_attribution,
                "negated_hits": score.negated_hits,
                "salience_density": score.salience_density,
            }
        )
    scored = pd.DataFrame(rows, index=frame.index)
    scored.insert(0, date_column, pd.to_datetime(frame[date_column]).to_numpy())
    return scored


def daily_tone_index(
    frame: pd.DataFrame, text_column: str = "text", date_column: str = "date"
) -> pd.DataFrame:
    """Daily media tone index ``S_media = (Pos - Neg) / Total`` in [-1, +1].

    Aggregation is over **summed counts**, not averaged per-document tones. The
    difference matters: averaging tones gives a two-word headline the same weight
    as a densely polar one, and gives every polar-free document a vote of 0.0,
    so a day with many off-topic documents would be pulled towards neutral purely
    by corpus composition. Summing counts makes the index a statement about the
    polar language published that day.

    Days where no polar term appears at all get ``S_media = NaN``, not 0.0 — the
    `baseline_model` convention that a missing observation must not be scored as
    a measured one. `n_documents` and `polar_total` are returned alongside so a
    thinly covered day can be weighted or excluded downstream rather than
    quietly treated as equal to a well covered one.
    """
    scored = score_documents(frame, text_column=text_column, date_column=date_column)
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "n_documents",
                "n_tokens",
                "positive",
                "negative",
                "polar_total",
                "S_media",
                "selection_salience",
                "factual_calibration",
                "political_attribution",
                "negated_hits",
                "salience_density",
            ]
        )

    scored = scored.copy()
    scored["day"] = pd.to_datetime(scored[date_column]).dt.normalize()
    grouped = scored.groupby("day", sort=True)
    daily = grouped.agg(
        n_documents=("tone", "size"),
        n_tokens=("n_tokens", "sum"),
        positive=("positive", "sum"),
        negative=("negative", "sum"),
        polar_total=("polar_total", "sum"),
        selection_salience=("selection_salience", "sum"),
        factual_calibration=("factual_calibration", "sum"),
        political_attribution=("political_attribution", "sum"),
        negated_hits=("negated_hits", "sum"),
    )
    total = daily["polar_total"].astype(float)
    daily["S_media"] = ((daily["positive"] - daily["negative"]) / total).where(total > 0)
    daily["salience_density"] = (
        daily["selection_salience"] / daily["n_tokens"].astype(float)
    ).where(daily["n_tokens"] > 0)
    daily.index.name = "date"
    ordered = [
        "n_documents",
        "n_tokens",
        "positive",
        "negative",
        "polar_total",
        "S_media",
        "selection_salience",
        "factual_calibration",
        "political_attribution",
        "negated_hits",
        "salience_density",
    ]
    return daily.loc[:, ordered]


# --------------------------------------------------------------------------
# Framing: the fifth required dimension.
# --------------------------------------------------------------------------

# Compiled once at import, in the same longest-phrase-first form the polarity
# lexicon uses, so "cost of living" wins over nothing and "year over year"
# cannot be split into two unrelated tokens.
_COMPILED_FRAMING: dict[str, tuple[list[tuple[tuple[str, ...], str]], list[tuple[tuple[str, ...], str]]]] = {
    contrast: (
        sorted(
            ((tuple(_TOKEN_PATTERN.findall(phrase.lower())), phrase) for phrase in left),
            key=lambda item: len(item[0]),
            reverse=True,
        ),
        sorted(
            ((tuple(_TOKEN_PATTERN.findall(phrase.lower())), phrase) for phrase in right),
            key=lambda item: len(item[0]),
            reverse=True,
        ),
    )
    for contrast, (left, right) in FRAMING_LEXICON.items()
}

#: The contrast that carries a pre-registered directional prediction, and so the
#: one that the single `framing` column reports. Refer to `framing_index`.
HEADLINE_FRAMING_CONTRAST = "level_vs_rate"


def _count_side(tokens: Sequence[str], phrases: Sequence[tuple[tuple[str, ...], str]]) -> tuple[int, list[str]]:
    """Non-overlapping phrase count for one side of one contrast."""
    if not tokens:
        return 0, []
    consumed = np.zeros(len(tokens), dtype=bool)
    hits = 0
    matched: list[str] = []
    for phrase_tokens, phrase in phrases:
        width = len(phrase_tokens)
        if width == 0 or width > len(tokens):
            continue
        for start in range(len(tokens) - width + 1):
            if consumed[start : start + width].any():
                continue
            if tuple(tokens[start : start + width]) != phrase_tokens:
                continue
            consumed[start : start + width] = True
            hits += 1
            matched.append(phrase)
    return hits, matched


@dataclass(frozen=True)
class FramingScore:
    """Framing counts for one document, two sides per contrast.

    Both sides are kept, never only the ratio. A ratio of +1.0 from one hit and
    a ratio of +1.0 from forty hits are the same number and very different
    evidence, and the daily index needs the raw counts to aggregate correctly.
    """

    n_tokens: int
    counts: dict[str, tuple[int, int]]
    matched_phrases: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    def ratio(self, contrast: str) -> float:
        """``(left - right) / (left + right)`` in [-1, +1]; NaN when silent.

        NaN, not 0.0. A document that frames nothing is not a document that
        frames both sides equally, and the model must not be allowed to read the
        first as the second. This is the same rule the tone index applies to a
        day with no polar language.
        """
        left, right = self.counts.get(contrast, (0, 0))
        total = left + right
        if total == 0:
            return float("nan")
        return float((left - right) / total)

    @property
    def framing_total(self) -> int:
        """Every framing hit in the document, across all three contrasts."""
        return sum(left + right for left, right in self.counts.values())


def score_framing(text: str) -> FramingScore:
    """Score one document against `FRAMING_LEXICON`.

    Negation is deliberately **not** applied. A negator flips a valence, and
    framing has no valence to flip: "prices are not higher than in 2020" still
    frames the question as one about the price level rather than the inflation
    rate, which is precisely what the measure claims to detect. Applying the
    polarity machinery here would corrupt the measure to look consistent.

    Each contrast is counted over its own token field, so a sentence may be both
    household-framed and current-framed. Within one side of one contrast the
    matching is non-overlapping, so "cost of living" is one hit and not two.
    """
    tokens = tokenize(text)
    counts: dict[str, tuple[int, int]] = {}
    matched: list[tuple[str, str, str]] = []
    for contrast, (left_phrases, right_phrases) in _COMPILED_FRAMING.items():
        left, left_hits = _count_side(tokens, left_phrases)
        right, right_hits = _count_side(tokens, right_phrases)
        counts[contrast] = (left, right)
        matched.extend((contrast, "left", phrase) for phrase in left_hits)
        matched.extend((contrast, "right", phrase) for phrase in right_hits)
    return FramingScore(n_tokens=len(tokens), counts=counts, matched_phrases=tuple(matched))


def _framing_columns() -> list[str]:
    columns = ["n_tokens", "framing_total"]
    for contrast in FRAMING_LEXICON:
        columns.extend([f"{contrast}_left", f"{contrast}_right", contrast])
    columns.append("framing")
    return columns


def score_framing_documents(
    frame: pd.DataFrame, text_column: str = "text", date_column: str = "date"
) -> pd.DataFrame:
    """Score framing for every document in a tidy corpus frame.

    One row in, one row out, and no document dropped. The `framing` column
    repeats the headline contrast so a downstream consumer that wants one
    framing number does not have to choose one itself.
    """
    _require_columns(frame, (text_column, date_column))
    if frame.empty:
        return pd.DataFrame(columns=[date_column, *_framing_columns()])

    rows = []
    for value in frame[text_column].to_numpy():
        score = score_framing("" if pd.isna(value) else str(value))
        row: dict[str, object] = {
            "n_tokens": score.n_tokens,
            "framing_total": score.framing_total,
        }
        for contrast in FRAMING_LEXICON:
            left, right = score.counts[contrast]
            row[f"{contrast}_left"] = left
            row[f"{contrast}_right"] = right
            row[contrast] = score.ratio(contrast)
        row["framing"] = row[HEADLINE_FRAMING_CONTRAST]
        rows.append(row)

    scored = pd.DataFrame(rows, index=frame.index).loc[:, _framing_columns()]
    scored.insert(0, date_column, pd.to_datetime(frame[date_column]).to_numpy())
    return scored


def framing_index(
    frame: pd.DataFrame, text_column: str = "text", date_column: str = "date"
) -> pd.DataFrame:
    """Daily framing index, one ratio per contrast, in [-1, +1].

    Aggregation sums the two sides across the day and then takes the ratio; it
    does not average the per-document ratios. The reason is the one
    `daily_tone_index` documents: a per-document average gives a headline with a
    single framing word the same vote as a full article that frames the question
    twelve times, and it has no way to express "nothing framed today" other than
    by inventing a number.

    The `framing` column repeats ``level_vs_rate``, which is the contrast with a
    direction predicted in advance: Hypothesis A says a household anchors to the
    price it pays rather than to the rate of change, so coverage that frames by
    price LEVEL (positive) is coverage aligned with the anchoring account, while
    coverage that frames by the falling inflation RATE (negative) is coverage
    that contradicts it. The other two contrasts are reported because the brief
    requires them, but no sign was predicted for them beforehand and none is
    claimed now.

    A day with no framing language in any document gets NaN for that contrast,
    never 0.0.
    """
    scored = score_framing_documents(frame, text_column=text_column, date_column=date_column)
    columns = ["n_documents", "framing_total"]
    for contrast in FRAMING_LEXICON:
        columns.extend([f"{contrast}_left", f"{contrast}_right", contrast])
    columns.append("framing")
    if scored.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "date"
        return empty

    scored = scored.copy()
    scored["day"] = pd.to_datetime(scored[date_column]).dt.normalize()
    aggregation: dict[str, tuple[str, str]] = {
        "n_documents": ("framing_total", "size"),
        "framing_total": ("framing_total", "sum"),
    }
    for contrast in FRAMING_LEXICON:
        aggregation[f"{contrast}_left"] = (f"{contrast}_left", "sum")
        aggregation[f"{contrast}_right"] = (f"{contrast}_right", "sum")
    daily = scored.groupby("day", sort=True).agg(**aggregation)

    for contrast in FRAMING_LEXICON:
        left = daily[f"{contrast}_left"].astype(float)
        right = daily[f"{contrast}_right"].astype(float)
        total = left + right
        daily[contrast] = ((left - right) / total).where(total > 0)
    daily["framing"] = daily[HEADLINE_FRAMING_CONTRAST]
    daily.index.name = "date"
    return daily.loc[:, columns]


def daily_topic_salience(
    basis: LsaBasis,
    tdm: TermDocumentMatrix,
    dates: Sequence[object] | pd.Series,
    normalise: bool = True,
) -> pd.DataFrame:
    """Daily salience weight per latent topic, from the SVD document coordinates.

    Topic ``j``'s weight on day ``t`` is ``sum_{d in t} |V_dj| * sigma_j`` —
    absolute coordinate because a topic's negative tail is still that topic
    being discussed, scaled by the singular value because a component carrying
    little energy should not count the same as the leading one. With
    ``normalise=True`` the weights are shares summing to 1 within each day,
    which is the "coverage share of negative-valence topics" quantity
    BRAINSTORMING.md §4.3 asks for.

    Days whose documents all map to the origin — no vocabulary term anywhere —
    get NaN rather than a uniform or zero share. The zero-prior property of the
    non-demeaned decomposition is what makes that distinction available at all.
    """
    coordinates = project_documents(basis, tdm)
    dates = pd.Series(list(dates) if not isinstance(dates, pd.Series) else dates.to_numpy())
    if len(dates) != len(coordinates):
        raise MediaError(f"{len(coordinates)} documents but {len(dates)} dates")

    weighted = np.abs(coordinates.to_numpy()) * basis.singular_values
    frame = pd.DataFrame(weighted, columns=list(coordinates.columns))
    frame["day"] = pd.to_datetime(dates.to_numpy()).normalize()
    daily = frame.groupby("day", sort=True).sum()

    if normalise:
        totals = daily.sum(axis=1)
        daily = daily.div(totals.where(totals > 0), axis=0)
    daily.index.name = "date"
    return daily


@dataclass(frozen=True)
class CalibrationReport:
    """Factual calibration: do the magnitude claims match the macro panel?

    This is the dimension that makes Hypothesis B falsifiable rather than merely
    illustrable (BRAINSTORMING.md §4.3). Tone alone cannot separate "coverage is
    negatively biased" from "coverage accurately describes a bad situation";
    comparing a claimed magnitude against the realised series can.

    `overstatement` is ``implied - actual`` in percentage points: positive means
    coverage described inflation as hotter than it was. `days_scored` and
    `days_unscored` are reported because a calibration statistic computed on the
    handful of days that happened to contain a magnitude word would otherwise
    look like a corpus-wide finding.
    """

    frame: pd.DataFrame
    days_scored: int
    days_unscored: int
    mean_implied: float
    mean_actual: float
    mean_overstatement: float
    claims_used: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"{self.days_scored} days with a magnitude claim ({self.days_unscored} without): "
            f"implied {self.mean_implied:.2f}% vs actual {self.mean_actual:.2f}%, "
            f"overstatement {self.mean_overstatement:+.2f}pp"
        )


def calibration_gap(
    frame: pd.DataFrame,
    actual_yoy: pd.Series,
    text_column: str = "text",
    date_column: str = "date",
    claims: dict[str, float] | None = None,
) -> CalibrationReport:
    """Compare the inflation rate coverage *implies* against the published rate.

    Each document's implied rate is the mean of `CALIBRATION_CLAIMS` values for
    the magnitude phrases it contains; a document with no magnitude phrase is
    unscored (NaN), never imputed. Daily implied rates are then compared to
    `actual_yoy` — a monthly series, reindexed onto the day by month, since the
    published YoY figure is what was knowable that month.

    The mapping from phrase to implied rate is a judgement and lives in
    `CALIBRATION_CLAIMS` where it can be argued with. A reviewer who thinks
    "soaring" implies 6% rather than 8% can pass their own `claims` dict and see
    whether the conclusion moves.
    """
    _require_columns(frame, (text_column, date_column))
    claims = dict(CALIBRATION_CLAIMS if claims is None else claims)
    compiled = sorted(
        ((tuple(tokenize(phrase)), phrase) for phrase in claims),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    implied: list[float] = []
    for value in frame[text_column].to_numpy():
        tokens = tokenize("" if pd.isna(value) else str(value))
        consumed = np.zeros(len(tokens), dtype=bool)
        hits: list[float] = []
        for phrase_tokens, phrase in compiled:
            width = len(phrase_tokens)
            if width == 0 or width > len(tokens):
                continue
            for start in range(len(tokens) - width + 1):
                if consumed[start : start + width].any():
                    continue
                if tuple(tokens[start : start + width]) == phrase_tokens:
                    consumed[start : start + width] = True
                    hits.append(float(claims[phrase]))
        implied.append(float(np.mean(hits)) if hits else float("nan"))

    daily = pd.DataFrame(
        {
            "day": pd.to_datetime(frame[date_column]).dt.normalize().to_numpy(),
            "implied_yoy": implied,
        }
    )
    grouped = daily.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "n_documents": grouped["implied_yoy"].size(),
            "n_claims": grouped["implied_yoy"].count(),
            "implied_yoy": grouped["implied_yoy"].mean(),
        }
    )

    actual = pd.Series(actual_yoy).dropna()
    if actual.empty:
        out["actual_yoy"] = np.nan
        notes = ("no actual inflation observations supplied; calibration not computable",)
    else:
        actual = actual.copy()
        actual.index = pd.to_datetime(actual.index).to_period("M")
        months = out.index.to_period("M")
        out["actual_yoy"] = [
            float(actual.loc[month]) if month in actual.index else np.nan for month in months
        ]
        notes = ("actual YoY matched to each day by calendar month",)

    out["overstatement"] = out["implied_yoy"] - out["actual_yoy"]
    out.index.name = "date"

    scored = out["overstatement"].notna()
    return CalibrationReport(
        frame=out,
        days_scored=int(scored.sum()),
        days_unscored=int((~scored).sum()),
        mean_implied=float(out.loc[scored, "implied_yoy"].mean()) if scored.any() else float("nan"),
        mean_actual=float(out.loc[scored, "actual_yoy"].mean()) if scored.any() else float("nan"),
        mean_overstatement=(
            float(out.loc[scored, "overstatement"].mean()) if scored.any() else float("nan")
        ),
        claims_used=tuple(sorted(claims)),
        notes=notes,
    )


# --------------------------------------------------------------------------
# 6. FinBERT probe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FinbertProbeResult:
    """FinBERT's performance on the hand-built macro probe set.

    `available` is False when the weights are not on disk; `reason` then says
    so. That is a reportable outcome, not an error: the probe exists to produce
    a confusion matrix *if the model can be run*, and the rest of the pipeline
    has no dependency on it.

    If FinBERT performs well here, that is a finding too and the lexicon's role
    should shrink accordingly (BRAINSTORMING.md §4.2).
    """

    available: bool
    reason: str | None
    model: str
    n_sentences: int
    predictions: pd.DataFrame
    confusion: pd.DataFrame
    accuracy: float
    label_map: dict[str, str] = field(default_factory=lambda: dict(FINBERT_LABEL_MAP))
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if not self.available:
            return f"FinBERT probe not run: {self.reason}"
        return (
            f"FinBERT on {self.n_sentences} macro sentences: "
            f"accuracy {self.accuracy:.3f} against intended policy stance"
        )


def _empty_confusion() -> pd.DataFrame:
    return pd.DataFrame(
        0,
        index=pd.Index(PROBE_LABELS, name="intended"),
        columns=pd.Index(PROBE_LABELS, name="predicted"),
    )


def confusion_matrix(intended: Sequence[str], predicted: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix over `PROBE_LABELS`, intended (rows) x predicted (columns).

    Written out rather than taken from sklearn so that every label always appears
    even when the model never emits it — an all-zero column is the most
    informative cell pattern this probe can produce, and a matrix that silently
    omits the class the model never predicted would hide it.
    """
    if len(intended) != len(predicted):
        raise MediaError(f"{len(intended)} intended labels but {len(predicted)} predictions")
    table = _empty_confusion()
    for truth, guess in zip(intended, predicted, strict=True):
        if truth not in PROBE_LABELS:
            raise MediaError(f"unknown intended label {truth!r}")
        if guess in PROBE_LABELS:
            table.loc[truth, guess] += 1
    return table


def finbert_probe(
    probe: Sequence[tuple[str, str]] = FINBERT_PROBE,
    model: str = FINBERT_MODEL,
    allow_download: bool = False,
    batch_size: int = 8,
) -> FinbertProbeResult:
    """Score the macro probe set with FinBERT and return a confusion matrix.

    **Nothing is downloaded unless `allow_download=True`.** The default passes
    ``local_files_only=True``, so the probe runs only against weights already in
    the local HuggingFace cache and otherwise returns
    ``available=False`` with the reason. Tests rely on that: a test suite that
    pulls 440 MB of weights is not a test suite.

    FinBERT's labels are market sentiment; `FINBERT_LABEL_MAP` translates them
    to policy stance under the charitable reading. The mapping is reported in the
    result so the confusion matrix cannot be read without it.
    """
    sentences = [str(text) for text, _ in probe]
    intended = [str(label) for _, label in probe]
    for label in intended:
        if label not in PROBE_LABELS:
            raise MediaError(f"unknown intended label {label!r} in probe set")

    def unavailable(reason: str) -> FinbertProbeResult:
        return FinbertProbeResult(
            available=False,
            reason=reason,
            model=model,
            n_sentences=len(sentences),
            predictions=pd.DataFrame(
                {"sentence": sentences, "intended": intended, "predicted": pd.NA}
            ),
            confusion=_empty_confusion(),
            accuracy=float("nan"),
            notes=(
                (
                    "probe set and confusion scaffold are still returned so the "
                    "report can state what would have been tested"
                ),
            ),
        )

    try:
        # Imported lazily and with warnings muted: transformers is a heavy
        # import with its own deprecation noise, and this project runs pytest
        # with DeprecationWarning promoted to an error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model, local_files_only=not allow_download
            )
            classifier = AutoModelForSequenceClassification.from_pretrained(
                model, local_files_only=not allow_download
            )
    except Exception as exc:
        reason = f"FinBERT weights unavailable locally ({type(exc).__name__}: {exc})"
        if not allow_download:
            reason += "; call with allow_download=True to fetch ~440 MB"
        logger.info("FinBERT probe skipped: %s", reason)
        return unavailable(reason)

    try:
        classifier.eval()
        id2label = {int(key): str(value).lower() for key, value in classifier.config.id2label.items()}
        raw_labels: list[str] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                for start in range(0, len(sentences), batch_size):
                    batch = sentences[start : start + batch_size]
                    encoded = tokenizer(
                        batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
                    )
                    logits = classifier(**encoded).logits
                    for index in logits.argmax(dim=-1).tolist():
                        raw_labels.append(id2label.get(int(index), "unknown"))
    except Exception as exc:  # pragma: no cover - requires local weights
        return unavailable(f"FinBERT inference failed ({type(exc).__name__}: {exc})")

    mapped = [FINBERT_LABEL_MAP.get(label, "neutral") for label in raw_labels]
    predictions = pd.DataFrame(
        {
            "sentence": sentences,
            "intended": intended,
            "finbert_label": raw_labels,
            "predicted": mapped,
        }
    )
    table = confusion_matrix(intended, mapped)
    correct = int(np.trace(table.to_numpy()))
    unmapped = sorted({label for label in raw_labels if label not in FINBERT_LABEL_MAP})
    notes = [
        (
            "FinBERT market-sentiment labels mapped to policy stance via "
            "FINBERT_LABEL_MAP (positive->dovish, negative->hawkish)"
        ),
    ]
    if unmapped:
        notes.append(f"labels not in the map, scored as neutral: {', '.join(unmapped)}")

    return FinbertProbeResult(
        available=True,
        reason=None,
        model=model,
        n_sentences=len(sentences),
        predictions=predictions,
        confusion=table,
        accuracy=float(correct / len(sentences)) if sentences else float("nan"),
        notes=tuple(notes),
    )
