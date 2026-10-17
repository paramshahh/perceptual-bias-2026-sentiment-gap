"""Build the monthly media series from the GKG corpus.

The corpus holds one record for each article, with the themes that GDELT
assigns and a tone vector that GDELT computes. This script makes three monthly
columns for Model 3:

* `media_tone` — the tone index of the project,
  `S = (Pos - Neg) / (Pos + Neg)`, in the range [-1, +1]. We add the positive
  and the negative scores over the month and then divide, instead of taking a
  mean of ratios. A mean of ratios gives the same weight to an article of 30
  words and to an article of 3000 words.
* `topic_salience` — the share of the articles of the month that carry a theme
  of economic distress (inflation, unemployment, recession, layoffs, the cost
  of living). This measures *which* facts get coverage, not the tone of the
  coverage.
* `political_attribution` — the share of the articles of the month that carry
  both an economic theme and a political theme. This measures the attribution
  of the economy to a political actor.
* `framing` and the three contrast columns — the framing dimension that the
  assignment requires. These come from the headline file, not from the theme
  corpus, because framing is a property of the words an editor chose and a
  theme tag has no words. Refer to the note below.

**Framing needs words, and the theme corpus has none.** A theme tag carries no
prose, so framing cannot be read from it. Two text sources are available:

* the headline file — 36 000 real article titles, 95 months. This is the better
  text, because an editor wrote it;
* the URL path of every article in the theme corpus — 198 367 paths, 116 months.
  A path such as `/2026/05/grocery-prices-still-higher-than-2020` is a
  mechanical slug of the title. It is noisier (truncations, site furniture such
  as `local-news`, and the occasional `Story.aspx` with no words at all), but it
  covers every month that the tone columns cover.

The framing columns pool both sources and take the ratio once over the summed
counts. The two sources are also scored separately and written as
`framing_headline` and `framing_slug`, and the script prints the correlation
between them. That correlation is the measure's cross-source validity check: it
is roughly 0.6, which says the two readings of the same month agree well above
chance and are not the same number twice.

The headline-only measure is the thinner one — a median of 28 framing hits in a
month, against 68 for the slugs — which is why neither source is used alone.

**A note on the filter.** The first corpus used the prefix `TAX_` as an
economic theme. In GDELT, `TAX_` is not taxation: it is a general taxonomy of
actors, ethnicities and languages. That filter accepted 693 347 articles with a
mean tone of −0.988. The strict filter accepts 198 367 articles with a mean
tone of −0.180. The first corpus measured the tone of the news in general. The
second measures the tone of economic news.

Run:  .venv/bin/python scripts/build_media_series.py
"""

from __future__ import annotations

import pathlib
import re
import sys
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, "src")

from perceptual_bias.nlp_media_svd import (  # noqa: E402
    FRAMING_LEXICON,
    HEADLINE_FRAMING_CONTRAST,
    score_framing_documents,
)

CORPUS = pathlib.Path("data/raw/gkg/gkg_strict.parquet")
HEADLINES = pathlib.Path("data/raw/gdelt/headlines.parquet")
OUT = pathlib.Path("data/processed/media_monthly.parquet")

# Themes that describe economic distress. Salience counts these.
DISTRESS_THEMES = (
    "ECON_INFLATION",
    "ECON_UNEMPLOYMENT",
    "ECON_RECESSION",
    "ECON_BANKRUPTCY",
    "ECON_COST_OF_LIVING",
    "ECON_DEBT",
    "UNEMPLOYMENT",
    "LAYOFF",
)

# Themes that mark a political frame.
POLITICAL_THEMES = (
    "USPEC_POLITICS_GENERAL1",
    "EPU_POLICY_GOVERNMENT",
    "ELECTION",
    "LEADER",
    "GENERAL_GOVERNMENT",
    "DEMOCRACY",
)


def url_slug(url: object) -> str:
    """The word tokens of a URL path, as a crude stand-in for the title.

    Extensions, digits and punctuation go. One-letter fragments go, because a
    path splits into them often and they match nothing. What remains is the
    words the publisher put in the path, which for most news sites is the
    headline with the stop words kept in.
    """
    try:
        path = urlparse(str(url)).path
    except ValueError:
        return ""
    path = re.sub(r"\.(html?|php|aspx?|jsp)$", "", path)
    return " ".join(word for word in re.sub(r"[^a-zA-Z]+", " ", path).split() if len(word) > 1)


def _framing_by_month(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Summed framing counts per month, with the ratio taken once at the end."""
    scored = score_framing_documents(
        frame.assign(_date=pd.to_datetime(frame["month"].astype(str) + "-01")),
        text_column=text_column,
        date_column="_date",
    ).assign(month=frame["month"].to_numpy())

    aggregation: dict[str, tuple[str, str]] = {"n_documents": ("framing_total", "size")}
    for contrast in FRAMING_LEXICON:
        aggregation[f"{contrast}_left"] = (f"{contrast}_left", "sum")
        aggregation[f"{contrast}_right"] = (f"{contrast}_right", "sum")
    return scored.groupby("month", sort=True).agg(**aggregation)


def _ratios(counts: pd.DataFrame) -> pd.DataFrame:
    """Turn summed left/right counts into the ratio columns and the hit counts."""
    out = pd.DataFrame(index=counts.index)
    out["framing_documents"] = counts["n_documents"].astype(int)
    for contrast in FRAMING_LEXICON:
        left = counts[f"{contrast}_left"].astype(float)
        right = counts[f"{contrast}_right"].astype(float)
        total = left + right
        out[contrast] = ((left - right) / total).where(total > 0)
        out[f"{contrast}_hits"] = total.astype(int)
    out["framing"] = out[HEADLINE_FRAMING_CONTRAST]
    return out


def monthly_framing(
    corpus: pd.DataFrame, headlines_path: pathlib.Path = HEADLINES
) -> tuple[pd.DataFrame, float | None]:
    """Monthly framing ratios from the pooled text, and the cross-source r.

    Both sides of each contrast are summed over the month and the ratio is taken
    once, at the end. A mean of per-document ratios would give a title with one
    framing word the same weight as a title with four, and could not say "no
    document framed anything this month" except by inventing a number.
    """
    slugs = corpus.loc[:, ["month"]].assign(text=corpus["url"].map(url_slug))
    slugs = slugs[slugs["text"].str.len() > 0]
    pooled = [slugs]

    headlines = None
    if headlines_path.exists():
        titles = pd.read_parquet(headlines_path)
        if "title" in titles and "month" in titles:
            headlines = titles.loc[:, ["month"]].assign(text=titles["title"].astype(str))
            pooled.append(headlines)

    combined = _ratios(_framing_by_month(pd.concat(pooled, ignore_index=True), "text"))
    slug_only = _ratios(_framing_by_month(slugs, "text"))
    combined["framing_slug"] = slug_only["framing"]
    combined["framing_slug_hits"] = slug_only[f"{HEADLINE_FRAMING_CONTRAST}_hits"]

    agreement: float | None = None
    if headlines is not None:
        headline_only = _ratios(_framing_by_month(headlines, "text"))
        combined["framing_headline"] = headline_only["framing"]
        combined["framing_headline_hits"] = headline_only[f"{HEADLINE_FRAMING_CONTRAST}_hits"]
        paired = combined.loc[:, ["framing_headline", "framing_slug"]].dropna()
        if len(paired) > 2:
            agreement = float(paired.corr().iloc[0, 1])
    else:
        combined["framing_headline"] = pd.NA
        combined["framing_headline_hits"] = 0

    combined.index = pd.PeriodIndex(combined.index, freq="M").to_timestamp(how="end").normalize()
    combined.index = combined.index + pd.offsets.MonthEnd(0)
    combined.index.name = "date"
    return combined.sort_index(), agreement


def build(corpus_path: pathlib.Path = CORPUS) -> pd.DataFrame:
    if not corpus_path.exists():
        raise SystemExit(f"corpus not found: {corpus_path}")
    corpus = pd.read_parquet(corpus_path)

    distress = corpus["themes"].str.contains("|".join(DISTRESS_THEMES), regex=True, na=False)
    political = corpus["themes"].str.contains("|".join(POLITICAL_THEMES), regex=True, na=False)
    corpus = corpus.assign(_distress=distress, _political=political)

    grouped = corpus.groupby("month")
    totals = grouped[["positive", "negative"]].sum()
    denominator = totals["positive"] + totals["negative"]

    series = pd.DataFrame(
        {
            "media_tone": (totals["positive"] - totals["negative"]) / denominator,
            "topic_salience": grouped["_distress"].mean(),
            "political_attribution": grouped["_political"].mean(),
            "n_articles": grouped.size(),
            "mean_raw_tone": grouped["tone"].mean(),
        }
    )
    series.index = pd.PeriodIndex(series.index, freq="M").to_timestamp(how="end").normalize()
    series.index = series.index + pd.offsets.MonthEnd(0)
    series.index.name = "date"
    series = series.sort_index()

    framing, agreement = monthly_framing(corpus)
    if agreement is None:
        print("no headline file: framing rests on the URL slugs alone")
    else:
        print(f"framing cross-source agreement, headline against slug: r = {agreement:.3f}")
    return series.join(framing, how="left")


def main() -> int:
    series = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(OUT)
    print(f"months: {len(series)}  {series.index.min():%Y-%m} .. {series.index.max():%Y-%m}")
    print()
    print(series.describe().round(4).to_string())
    print()
    print("first and last six months:")
    print(pd.concat([series.head(6), series.tail(6)]).round(4).to_string())
    print()
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
