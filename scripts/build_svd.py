"""Run the latent semantic analysis over the GDELT theme corpus.

Each article becomes one document. The words of that document are the theme
tags that GDELT assigns to the article. A theme tag is a token, so the TF-IDF
matrix and the truncated SVD work in the normal way.

**Why theme tags and not headlines.** The GDELT DOC API returns headlines, but
it permits one request each five seconds and it blocked this project. The GKG
bulk files have no limit, and they carry the themes. A theme list is a cleaner
input than a headline: it holds no stop words, no punctuation and no
repetition of the same event in different words.

**The matrix is not centred.** Refer to BRAINSTORMING.md paragraph 3.2. An
article with no theme in the vocabulary must map to the zero vector, because
"no signal" is not "the average signal".

The script writes three files:

* `svd_spectrum.parquet` — the singular values and the cumulative energy, for
  the scree panel of the dashboard;
* `svd_topics.parquet` — the strongest terms for each topic, for Table 2;
* `svd_monthly.parquet` — the share of each topic in each month, which gives
  the salience of each topic over time.

Run:  .venv/bin/python scripts/build_svd.py [k]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from perceptual_bias.nlp_media_svd import (  # noqa: E402
    fit_lsa,
    fit_term_document,
    select_k,
    top_terms_per_topic,
)

CORPUS = pathlib.Path("data/raw/gkg/gkg_strict.parquet")
HEADLINES = pathlib.Path("data/raw/gdelt/headlines.parquet")

# Headlines that mention the economy. The GDELT query is wide, so the file also
# holds articles that match a term in the body only, plus some site names in
# place of a title.
HEADLINE_KEYWORDS = (
    "inflation|price|prices|economy|economic|recession|layoff|jobs|unemploy|"
    "mortgage|cost of living|grocer|gas price|wage|fed|interest rate"
)
MIN_TITLE_LENGTH = 25
OUT = pathlib.Path("data/processed")
DEFAULT_K = 12
MAX_DOCUMENTS = 60000  # a sample keeps the dense permutation step affordable


def theme_documents(corpus: pd.DataFrame) -> pd.Series:
    """Turn the theme list of each article into a document of tokens.

    GDELT separates themes with a semicolon and repeats a theme when it occurs
    more than once. A repeat is information: it says the article is more about
    that theme. So the repeats stay, and the TF-IDF weight uses them.
    """
    cleaned = (
        corpus["themes"]
        .fillna("")
        .str.replace(";", " ", regex=False)
        .str.replace(",", " ", regex=False)
        .str.strip()
    )
    return cleaned[cleaned.str.len() > 0]


def headline_documents() -> tuple[pd.Series, pd.Series] | None:
    """Return the economic headlines and their months, or None.

    **Headlines are preferred over theme tags.** A term loading over English
    words is readable. A term loading over tags such as
    `wb_696_public_sector_management` is not. The brief asks for term loadings,
    so the table must be readable.

    The headline pull finished after the theme corpus, because GDELT blocked
    the API for hours. The theme corpus stays for the monthly topic shares,
    which need every month.
    """
    if not HEADLINES.exists():
        return None
    frame = pd.read_parquet(HEADLINES)
    if "title" not in frame or "month" not in frame:
        return None
    economic = frame["title"].str.contains(HEADLINE_KEYWORDS, case=False, na=False)
    long_enough = frame["title"].str.len() > MIN_TITLE_LENGTH
    kept = frame[economic & long_enough]
    if len(kept) < 1000:
        return None
    return kept["title"], kept["month"]


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_K
    if not CORPUS.exists():
        raise SystemExit(f"corpus not found: {CORPUS}")
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(CORPUS)
    headlines = headline_documents()
    if headlines is not None:
        documents, months_of_document = headlines
        source = "headlines"
        print(f"economic headlines: {len(documents)}")
    else:
        documents = theme_documents(corpus)
        months_of_document = corpus.loc[documents.index, "month"]
        source = "theme tags"
        print(f"articles with themes: {len(documents)} (no headline file)")
    print(f"source for the decomposition: {source}")

    sample = documents
    if len(documents) > MAX_DOCUMENTS:
        sample = documents.sample(MAX_DOCUMENTS, random_state=20261015).sort_index()
        print(f"sampled {MAX_DOCUMENTS} documents for the decomposition")

    # A theme tag is already a token. One-word tokens only: a two-word token
    # would join two unrelated tags.
    if source == "headlines":
        tdm = fit_term_document(
            sample.to_list(), doc_ids=sample.index.to_list(),
            min_df=5, max_df=0.4, ngram_range=(1, 2),
        )
    else:
        tdm = fit_term_document(
            sample.to_list(), doc_ids=sample.index.to_list(),
            min_df=20, max_df=0.5, ngram_range=(1, 1),
        )
    print(f"matrix: {tdm.n_terms} terms x {tdm.n_docs} documents, density {tdm.density:.5f}")

    selection = select_k(tdm, k_max=min(40, tdm.n_terms - 1), n_permutations=5)
    print(
        f"k by energy {selection.k_energy}, by the noise floor {selection.k_noise}, "
        f"by parallel analysis {selection.k_parallel}"
    )
    for note in selection.notes:
        print(f"  note: {note}")

    basis = fit_lsa(tdm, k=k)
    print(f"fitted k={k}; the top {k} components hold {basis.cumulative_energy[-1]:.1%} of the energy")

    spectrum = selection.spectrum_frame()
    spectrum.to_parquet(OUT / "svd_spectrum.parquet")

    topics = top_terms_per_topic(basis, n_terms=10)
    topics.to_parquet(OUT / "svd_topics.parquet")

    # The share of each topic in each month. The coordinates of a document are
    # its position in topic space; the absolute value gives the weight.
    coordinates = pd.DataFrame(
        np.abs(basis.doc_coordinates),
        index=pd.Index(basis.doc_ids, name="doc"),
        columns=[f"topic_{i + 1}" for i in range(basis.k)],
    )
    months = months_of_document.loc[coordinates.index]
    monthly = coordinates.groupby(months).mean()
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp(how="end").normalize()
    monthly.index = monthly.index + pd.offsets.MonthEnd(0)
    monthly.index.name = "date"
    monthly.to_parquet(OUT / "svd_monthly.parquet")

    print()
    print("strongest terms for the first five topics:")
    for topic in sorted(topics["topic"].unique())[:5]:
        terms = topics[topics["topic"] == topic].head(5)
        labels = ", ".join(f"{row.term}" for row in terms.itertuples())
        print(f"  topic {topic}: {labels}")
    print()
    print(f"saved -> {OUT}/svd_spectrum.parquet, svd_topics.parquet, svd_monthly.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
