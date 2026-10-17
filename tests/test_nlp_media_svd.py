"""Tests for the media-text pipeline: corpus, LSA, lexicons, FinBERT probe.

Everything here runs **offline**. No test touches GDELT, no test touches
twitterapi.io, and no test downloads model weights — the FinBERT probe test
asserts the *graceful degradation* path and skips the inference path unless
weights already happen to be cached locally.

Several tests pin properties the module's correctness actually rests on rather
than happy paths: the non-demeaning guarantee (a document with no vocabulary
terms maps to the origin), the fit/apply split that prevents look-ahead, and
tone-index bounds.
"""

from __future__ import annotations

import pathlib
from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perceptual_bias.nlp_media_svd import (
    CALIBRATION_CLAIMS,
    FRAMING_LEXICON,
    HEADLINE_FRAMING_CONTRAST,
    FramingScore,
    framing_index,
    score_framing,
    score_framing_documents,
    ECONOMIC_TERMS,
    FINBERT_PROBE,
    LEXICON,
    PROBE_LABELS,
    GdeltSource,
    LexiconScore,
    MediaError,
    TermDocumentMatrix,
    apply_term_document,
    calibration_gap,
    confusion_matrix,
    daily_tone_index,
    daily_topic_salience,
    finbert_probe,
    fit_lsa,
    fit_term_document,
    project_documents,
    reconstruction_error,
    score_documents,
    score_text,
    select_k,
    tidy_articles,
    tokenize,
    top_terms_per_topic,
)

RNG = np.random.default_rng(20261015)


# --------------------------------------------------------------------------
# Synthetic corpora — the offline stand-in for GDELT
# --------------------------------------------------------------------------

INFLATION_DOCS = [
    "soaring inflation squeezes household budgets as grocery prices climb",
    "inflation keeps grocery prices high and squeezes family budgets",
    "household budgets squeezed again as inflation and grocery prices surge",
    "grocery prices climb while inflation squeezes budgets nationwide",
]
LABOUR_DOCS = [
    "layoffs mount as hiring slows across the manufacturing sector",
    "hiring slows and layoffs mount in manufacturing payrolls report",
    "manufacturing layoffs continue while hiring slows sharply",
    "payrolls report shows layoffs mounting and hiring slowing",
]
POLITICS_DOCS = [
    "democrats and republicans blame each other ahead of the midterms",
    "republicans blame the administration as midterms approach",
    "administration defends record as democrats blame republicans before midterms",
    "midterms campaign sees democrats and republicans trade blame",
]


def synthetic_corpus(repeats: int = 3) -> pd.DataFrame:
    """A tidy corpus with three deliberately separable topic blocks.

    Known structure is the point: the SVD should find three components clearly
    above a permutation noise floor, and `top_terms_per_topic` should surface
    the planted vocabulary.
    """
    texts: list[str] = []
    for _ in range(repeats):
        texts.extend(INFLATION_DOCS + LABOUR_DOCS + POLITICS_DOCS)
    dates = pd.date_range("2026-05-01", periods=len(texts), freq="h")
    return pd.DataFrame(
        {
            "date": dates.normalize(),
            "title": texts,
            "url": [f"https://example{i}.com/a{i}" for i in range(len(texts))],
            "domain": [f"example{i % 5}.com" for i in range(len(texts))],
            "text": texts,
            "language": "English",
            "sourcecountry": "United States",
        }
    )


def raw_gdelt_frame() -> pd.DataFrame:
    """A raw article list in GDELT's own column shape, including bad rows."""
    return pd.DataFrame(
        {
            "url": [
                "https://a.com/1",
                "https://a.com/1",  # duplicate
                "https://b.com/2",
                "",  # missing url
                "https://c.com/3",
                "https://d.com/4",
            ],
            "url_mobile": [""] * 6,
            "title": [
                "Inflation soars as grocery prices climb",
                "Inflation soars as grocery prices climb",
                "Layoffs mount in manufacturing",
                "Orphan headline",
                "   ",  # empty title
                "La inflacion sigue subiendo",
            ],
            "seendate": [
                "20260512T120000Z",
                "20260512T120000Z",
                "20260512T153000Z",
                "20260513T090000Z",
                "20260513T100000Z",
                "not-a-date",
            ],
            "socialimage": [""] * 6,
            "domain": ["a.com", "a.com", "b.com", "x.com", "c.com", "d.com"],
            "language": ["English", "English", "English", "English", "English", "Spanish"],
            "sourcecountry": ["United States"] * 6,
        }
    )


class StubGdeltClient:
    """Offline stand-in for `GdeltDoc`, recording the queries it was handed."""

    def __init__(self, frame: pd.DataFrame | None = None, fail: bool = False) -> None:
        self.frame = raw_gdelt_frame() if frame is None else frame
        self.fail = fail
        self.calls: list[object] = []

    def article_search(self, filters: object) -> pd.DataFrame:
        self.calls.append(filters)
        if self.fail:
            raise ConnectionError("simulated network failure")
        return self.frame


# --------------------------------------------------------------------------
# 1. Corpus ingestion
# --------------------------------------------------------------------------


class TestTidyArticles:
    def test_names_every_dropped_document_and_why(self) -> None:
        tidy, report = tidy_articles(raw_gdelt_frame(), terms=("inflation",))
        assert len(tidy) == report.kept_rows
        assert report.raw_rows == 6
        assert report.kept_rows == 2
        assert report.dropped_rows == 4
        # Each exclusion is named, not aggregated into a bare total.
        assert report.dropped_reasons == {
            "unparseable date": 1,
            "missing url": 1,
            "missing or empty title": 1,
            "duplicate url": 1,
        }
        assert report.kept_rows + report.dropped_rows == report.raw_rows

    def test_schema_and_text_column(self) -> None:
        tidy, _ = tidy_articles(raw_gdelt_frame())
        assert list(tidy.columns) == [
            "date",
            "title",
            "url",
            "domain",
            "text",
            "language",
            "sourcecountry",
        ]
        # `text` is the headline; GDELT serves no bodies.
        assert (tidy["text"] == tidy["title"]).all()
        assert tidy["date"].dt.normalize().equals(tidy["date"])

    def test_empty_input_is_not_an_error(self) -> None:
        tidy, report = tidy_articles(pd.DataFrame())
        assert tidy.empty
        assert report.kept_rows == 0
        assert report.start is None

    def test_missing_date_column_raises_mediaerror(self) -> None:
        with pytest.raises(MediaError, match="neither a `seendate` nor a `date`"):
            tidy_articles(pd.DataFrame({"title": ["x"], "url": ["https://a.com"]}))


class TestGdeltSource:
    def test_fetch_with_stub_client_caches_to_parquet(self, tmp_path) -> None:
        stub = StubGdeltClient()
        source = GdeltSource(cache_dir=tmp_path, client=stub)
        tidy, report = source.fetch_range("2026-05-12", "2026-05-13", terms=("inflation",))
        assert report.fetched_chunks == 2
        assert report.cached_chunks == 0
        assert len(stub.calls) == 2  # one chunk per day, one term
        assert sorted(p.suffix for p in tmp_path.iterdir()) == [".parquet", ".parquet"]
        assert not tidy.empty

    def test_second_run_is_served_entirely_from_cache(self, tmp_path) -> None:
        stub = StubGdeltClient()
        source = GdeltSource(cache_dir=tmp_path, client=stub)
        source.fetch_range("2026-05-12", "2026-05-13", terms=("inflation",))
        calls_after_first = len(stub.calls)

        tidy, report = source.fetch_range("2026-05-12", "2026-05-13", terms=("inflation",))
        assert len(stub.calls) == calls_after_first  # no new network traffic
        assert report.cached_chunks == 2
        assert report.fetched_chunks == 0
        assert report.offline is True
        assert not tidy.empty

    def test_network_failure_degrades_with_named_gaps(self, tmp_path) -> None:
        source = GdeltSource(cache_dir=tmp_path, client=StubGdeltClient(fail=True))
        tidy, report = source.fetch_range("2026-05-12", "2026-05-14", terms=("inflation",))
        assert tidy.empty
        assert len(report.failed_chunks) == 3
        assert all("fetch failed" in chunk for chunk in report.failed_chunks)
        assert any("gaps" in note for note in report.notes)

    def test_no_client_available_is_reported_not_raised(self, tmp_path) -> None:
        source = GdeltSource(cache_dir=tmp_path, allow_network=False)
        assert source.available is False
        assert "network disabled" in (source.unavailable_reason or "")
        tidy, report = source.fetch_range("2026-05-12", "2026-05-12", terms=("inflation",))
        assert tidy.empty
        assert report.network_available is False
        assert len(report.failed_chunks) == 1

    def test_cache_survives_a_client_that_later_dies(self, tmp_path) -> None:
        """A corpus already on disk must remain usable with no client at all."""
        GdeltSource(cache_dir=tmp_path, client=StubGdeltClient()).fetch_range(
            "2026-05-12", "2026-05-12", terms=("inflation",)
        )
        offline = GdeltSource(cache_dir=tmp_path, allow_network=False)
        tidy, report = offline.fetch_range("2026-05-12", "2026-05-12", terms=("inflation",))
        assert report.cached_chunks == 1
        assert not tidy.empty

    def test_bad_arguments_raise_mediaerror(self, tmp_path) -> None:
        source = GdeltSource(cache_dir=tmp_path, client=StubGdeltClient())
        with pytest.raises(MediaError, match="precedes start"):
            source.fetch_range("2026-05-13", "2026-05-12")
        with pytest.raises(MediaError, match="chunk_days"):
            source.fetch_range("2026-05-12", "2026-05-13", chunk_days=0)
        with pytest.raises(MediaError, match="no query terms"):
            source.fetch_range("2026-05-12", "2026-05-13", terms=())

    def test_default_terms_are_economic_and_nonempty(self) -> None:
        assert ECONOMIC_TERMS
        assert "inflation" in ECONOMIC_TERMS


# --------------------------------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------------------------------


class TestTermDocumentMatrix:
    def test_orientation_is_terms_by_documents(self) -> None:
        corpus = synthetic_corpus()
        tdm = fit_term_document(corpus["text"], min_df=1)
        assert tdm.matrix.shape == (len(tdm.terms), len(corpus))
        assert tdm.n_docs == len(corpus)
        assert sp.issparse(tdm.matrix)

    def test_matrix_stays_sparse_and_sparsity_is_real(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        assert tdm.density < 0.5
        assert tdm.frobenius_sq > 0

    def test_stopwords_are_removed(self) -> None:
        tdm = fit_term_document(
            [
                "the economy is the story of the year",
                "the economy and the jobs market",
                "the widgets of the warehouse",
            ],
            min_df=1,
            ngram_range=(1, 1),
        )
        assert "the" not in tdm.terms
        assert "of" not in tdm.terms
        assert "economy" in tdm.terms

    def test_max_df_prunes_a_term_present_in_every_document(self) -> None:
        """A term in all documents discriminates nothing and would dominate sigma_1."""
        tdm = fit_term_document(
            ["inflation rises", "inflation falls", "inflation steady"],
            min_df=1,
            max_df=0.8,
            ngram_range=(1, 1),
        )
        assert "inflation" not in tdm.terms

    def test_document_with_no_vocabulary_terms_is_kept_and_named(self) -> None:
        """Never silently drop documents: an off-topic doc is a zero column."""
        texts = list(INFLATION_DOCS) + ["zzz qqq"]
        tdm = fit_term_document(texts, doc_ids=[f"d{i}" for i in range(len(texts))], min_df=2)
        assert tdm.n_docs == len(texts)  # nothing dropped
        assert tdm.empty_doc_ids == ("d4",)
        assert any("kept, not dropped" in note for note in tdm.notes)

    def test_mismatched_doc_ids_raise(self) -> None:
        with pytest.raises(MediaError, match="document ids"):
            fit_term_document(["a b", "c d"], doc_ids=["only-one"], min_df=1)

    def test_empty_vocabulary_raises_mediaerror_not_valueerror(self) -> None:
        with pytest.raises(MediaError, match="vocabulary came out empty"):
            fit_term_document(["the and of", "a an the"], min_df=1)

    def test_no_documents_raises(self) -> None:
        with pytest.raises(MediaError, match="no documents supplied"):
            fit_term_document([])

    def test_shape_mismatch_is_caught_on_construction(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        with pytest.raises(MediaError, match="does not match"):
            TermDocumentMatrix(
                matrix=tdm.matrix,
                terms=tdm.terms[:-1],
                doc_ids=tdm.doc_ids,
                vectorizer=tdm.vectorizer,
                fitted=True,
            )


class TestNoLookAhead:
    def test_apply_does_not_refit_the_vocabulary(self) -> None:
        train = fit_term_document(INFLATION_DOCS + LABOUR_DOCS, min_df=1)
        holdout = apply_term_document(
            train, ["quantum entanglement of flavour polymers", *INFLATION_DOCS]
        )
        # Identical vocabulary: the holdout's own words never enter the basis.
        assert holdout.terms == train.terms
        assert holdout.fitted is False
        assert holdout.n_docs == 5

    def test_holdout_only_vocabulary_maps_to_the_origin(self) -> None:
        train = fit_term_document(INFLATION_DOCS, min_df=1)
        holdout = apply_term_document(train, ["zzz qqq wibble"], doc_ids=["h0"])
        assert holdout.matrix.nnz == 0
        assert holdout.empty_doc_ids == ("h0",)

    def test_cannot_apply_an_unfitted_matrix(self) -> None:
        train = fit_term_document(INFLATION_DOCS, min_df=1)
        holdout = apply_term_document(train, INFLATION_DOCS)
        with pytest.raises(MediaError, match="never fitted"):
            apply_term_document(holdout, INFLATION_DOCS)


# --------------------------------------------------------------------------
# 3. Truncated SVD / LSA
# --------------------------------------------------------------------------


class TestFitLsa:
    def test_shapes_and_descending_singular_values(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(tdm, k=4)
        assert basis.term_loadings.shape == (tdm.n_terms, 4)
        assert basis.doc_coordinates.shape == (tdm.n_docs, 4)
        assert basis.singular_values.shape == (4,)
        assert np.all(np.diff(basis.singular_values) <= 1e-10)
        assert basis.k == 4

    def test_is_not_demeaned(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(tdm, k=3)
        assert basis.demeaned is False
        assert any("non-demeaned" in note for note in basis.notes)

    def test_reconstruction_error_decreases_in_k(self) -> None:
        tdm = fit_term_document(synthetic_corpus(repeats=4)["text"], min_df=1)
        errors = [reconstruction_error(tdm, fit_lsa(tdm, k=k)) for k in (1, 2, 3, 5, 8)]
        assert all(later <= earlier + 1e-12 for earlier, later in pairwise(errors))
        assert errors[0] > errors[-1]
        assert 0.0 <= errors[-1] <= 1.0

    def test_truncation_matches_eckart_young_against_dense_svd(self) -> None:
        """Closed-form error must equal the dense residual it stands in for."""
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        dense = tdm.matrix.toarray()
        for k in (1, 3, 5):
            basis = fit_lsa(tdm, k=k)
            approximation = (
                basis.term_loadings * basis.singular_values
            ) @ basis.doc_coordinates.T
            direct = np.linalg.norm(dense - approximation) / np.linalg.norm(dense)
            assert reconstruction_error(tdm, basis) == pytest.approx(direct, abs=1e-8)

    def test_bad_k_raises_mediaerror(self) -> None:
        tdm = fit_term_document(INFLATION_DOCS, min_df=1)
        with pytest.raises(MediaError, match="at least 1"):
            fit_lsa(tdm, k=0)
        with pytest.raises(MediaError, match="strictly less than"):
            fit_lsa(tdm, k=tdm.n_docs)

    def test_all_zero_matrix_raises(self) -> None:
        train = fit_term_document(INFLATION_DOCS, min_df=1)
        holdout = apply_term_document(train, ["zzz qqq", "wibble wobble"])
        with pytest.raises(MediaError, match="entirely zero"):
            fit_lsa(holdout, k=1)

    def test_signs_are_deterministic_across_runs(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        first = fit_lsa(tdm, k=3, random_state=0)
        second = fit_lsa(tdm, k=3, random_state=7)
        for column in range(3):
            pivot = int(np.argmax(np.abs(first.term_loadings[:, column])))
            assert first.term_loadings[pivot, column] > 0
            assert second.term_loadings[int(np.argmax(np.abs(second.term_loadings[:, column]))), column] > 0


class TestProjectDocuments:
    def test_projection_reproduces_training_coordinates(self) -> None:
        """Applying the basis to the training matrix must return V_k itself."""
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(tdm, k=3)
        projected = project_documents(basis, tdm).to_numpy()
        assert np.allclose(projected, basis.doc_coordinates, atol=1e-8)

    def test_zero_prior_drift_document_maps_to_zero_vector(self) -> None:
        """The non-demeaning guarantee of BRAINSTORMING.md section 3.2.

        A document containing none of the dictionary terms must land exactly on
        the origin. Under a demeaned decomposition it would acquire minus the
        mean vector and be assigned a position it never earned.
        """
        train = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(train, k=3)
        holdout = apply_term_document(
            train,
            ["zzz qqq wibble", INFLATION_DOCS[0]],
            doc_ids=["off_topic", "on_topic"],
        )
        coordinates = project_documents(basis, holdout)
        assert np.allclose(coordinates.loc["off_topic"].to_numpy(), 0.0)
        # And the on-topic document is emphatically not at the origin, so the
        # test cannot pass by the projection being degenerate.
        assert np.linalg.norm(coordinates.loc["on_topic"].to_numpy()) > 1e-6

    def test_vocabulary_mismatch_raises(self) -> None:
        train = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(train, k=2)
        other = fit_term_document(POLITICS_DOCS, min_df=1)
        with pytest.raises(MediaError, match="does not match the basis"):
            project_documents(basis, other)


class TestTopTerms:
    def test_returns_n_terms_per_topic_with_signed_loadings(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(tdm, k=3)
        table = top_terms_per_topic(basis, n_terms=5)
        assert len(table) == 15
        assert set(table.columns) == {
            "topic",
            "rank",
            "term",
            "loading",
            "singular_value",
            "energy_share",
        }
        assert table.groupby("topic").size().eq(5).all()
        # Ranked by absolute loading within each topic.
        for _, group in table.groupby("topic"):
            magnitudes = group["loading"].abs().to_numpy()
            assert np.all(np.diff(magnitudes) <= 1e-12)

    def test_surfaces_the_planted_vocabulary(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1, ngram_range=(1, 1))
        basis = fit_lsa(tdm, k=3)
        table = top_terms_per_topic(basis, n_terms=8)
        surfaced = set(table["term"])
        # Each planted block must appear somewhere in the top terms.
        assert surfaced & {"inflation", "grocery", "prices"}
        assert surfaced & {"layoffs", "hiring", "manufacturing"}
        assert surfaced & {"democrats", "republicans", "midterms"}

    def test_bad_n_terms_raises(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        with pytest.raises(MediaError, match="n_terms"):
            top_terms_per_topic(fit_lsa(tdm, k=2), n_terms=0)


# --------------------------------------------------------------------------
# 4. k-selection diagnostics
# --------------------------------------------------------------------------


TOPIC_CORES: dict[str, list[str]] = {
    "inflation": ["inflation", "grocery", "prices", "soaring", "households", "budgets"],
    "labour": ["layoffs", "hiring", "payrolls", "manufacturing", "jobless", "claims"],
    "politics": [
        "democrats",
        "republicans",
        "midterms",
        "campaign",
        "blame",
        "administration",
    ],
}


def topic_corpus(docs_per_topic: int = 16, seed: int = 7) -> list[str]:
    """A corpus with exactly three planted topics at realistic TF-IDF sparsity.

    Each document mixes five of its topic's six core terms with three unique
    filler tokens. The filler enlarges the vocabulary without adding shared
    structure, which is what makes the matrix sparse enough for the permutation
    null to be informative — see `KSelection` on why a *dense* non-negative
    matrix defeats the noise floor.
    """
    rng = np.random.default_rng(seed)
    documents: list[str] = []
    counter = 0
    for core in TOPIC_CORES.values():
        for _ in range(docs_per_topic):
            filler = [f"zzz{counter}{position}" for position in range(3)]
            counter += 1
            words = list(rng.choice(core, size=5, replace=False)) + filler
            documents.append(" ".join(words))
    return documents


class TestSelectK:
    def test_noise_floor_recovers_the_known_topic_count(self) -> None:
        """Three planted topics in a sparse corpus: both readings must find 3."""
        tdm = fit_term_document(topic_corpus(), min_df=1, ngram_range=(1, 1))
        assert tdm.density < 0.1  # the regime where the null is informative
        selection = select_k(tdm, k_max=10, n_permutations=10, random_state=1)
        assert selection.k_noise == 3
        assert selection.k_parallel == 3
        assert selection.singular_values[2] > selection.noise_floor
        assert selection.singular_values[3] < selection.noise_floor

    def test_k_noise_never_exceeds_the_componentwise_reading(self) -> None:
        """The spec's max-based floor is the conservative one, by construction."""
        for corpus in (topic_corpus(), synthetic_corpus(repeats=4)["text"].tolist()):
            tdm = fit_term_document(corpus, min_df=1)
            selection = select_k(tdm, k_max=8, n_permutations=6, random_state=4)
            assert selection.k_noise <= selection.k_parallel
            if selection.k_noise != selection.k_parallel:
                assert any("disagree" in note for note in selection.notes)

    def test_cumulative_energy_is_monotone_and_bounded(self) -> None:
        tdm = fit_term_document(synthetic_corpus(repeats=4)["text"], min_df=1)
        selection = select_k(tdm, k_max=10, n_permutations=4)
        cumulative = selection.cumulative_energy
        assert np.all(np.diff(cumulative) >= -1e-12)
        assert 0.0 < cumulative[0] <= cumulative[-1] <= 1.0 + 1e-9

    def test_k_energy_is_the_first_k_reaching_the_target(self) -> None:
        tdm = fit_term_document(synthetic_corpus(repeats=4)["text"], min_df=1)
        selection = select_k(tdm, k_max=10, energy_target=0.5, n_permutations=3)
        cumulative = selection.cumulative_energy
        if cumulative[-1] >= 0.5:
            assert cumulative[selection.k_energy - 1] >= 0.5
            if selection.k_energy > 1:
                assert cumulative[selection.k_energy - 2] < 0.5

    def test_permutation_preserves_total_energy(self) -> None:
        """Within-column shuffling must not change ||X||_F, or the floor lies."""
        from perceptual_bias.nlp_media_svd import _permute_within_columns

        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        source = sp.csc_matrix(tdm.matrix)
        shuffled = _permute_within_columns(source, np.random.default_rng(3))
        assert shuffled.shape == source.shape
        assert shuffled.nnz == source.nnz
        assert np.square(shuffled.data).sum() == pytest.approx(tdm.frobenius_sq)
        # Column norms are preserved one by one, not merely in total.
        assert np.allclose(
            np.asarray(source.multiply(source).sum(axis=0)).ravel(),
            np.asarray(shuffled.multiply(shuffled).sum(axis=0)).ravel(),
        )

    def test_spectrum_frame_is_tidy_data_not_a_plot(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        selection = select_k(tdm, k_max=6, n_permutations=3)
        frame = selection.spectrum_frame()
        assert list(frame.columns) == [
            "component",
            "singular_value",
            "cumulative_energy",
            "permuted_max",
            "above_noise_floor",
            "above_permuted_component",
        ]
        assert len(frame) == 6
        assert frame["above_noise_floor"].sum() == selection.k_noise
        assert selection.permuted_spectrum.shape == (3, 6)

    def test_structureless_matrix_yields_no_retained_components(self) -> None:
        """A matrix with no co-occurrence structure must not fabricate topics."""
        dense = (RNG.random((40, 60)) < 0.05) * RNG.uniform(0.5, 1.0, (40, 60))

        class _V:
            pass

        tdm = TermDocumentMatrix(
            matrix=sp.csr_matrix(dense),
            terms=tuple(f"t{i}" for i in range(40)),
            doc_ids=tuple(range(60)),
            vectorizer=_V(),  # type: ignore[arg-type]
            fitted=True,
        )
        selection = select_k(tdm, k_max=8, n_permutations=10, random_state=2)
        assert selection.k_noise <= 1
        if selection.k_noise == 0:
            assert any("no latent structure" in note for note in selection.notes)

    def test_bad_arguments_raise(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        with pytest.raises(MediaError, match="energy_target"):
            select_k(tdm, energy_target=1.5)
        with pytest.raises(MediaError, match="n_permutations"):
            select_k(tdm, n_permutations=0)


# --------------------------------------------------------------------------
# 5. Lexicon scoring, negation, tone index
# --------------------------------------------------------------------------


class TestLexiconStructure:
    def test_lexicon_covers_the_four_dimensions(self) -> None:
        assert set(LEXICON) == {
            "negative",
            "positive",
            "selection_salience",
            "factual_calibration",
            "political_attribution",
        }
        for dimension, phrases in LEXICON.items():
            assert phrases, dimension
            assert len(set(phrases)) == len(phrases), f"duplicate phrase in {dimension}"
            assert all(phrase == phrase.lower().strip() for phrase in phrases)

    def test_polar_dimensions_do_not_overlap(self) -> None:
        assert not set(LEXICON["negative"]) & set(LEXICON["positive"])

    def test_calibration_claims_are_all_in_the_lexicon(self) -> None:
        missing = set(CALIBRATION_CLAIMS) - set(LEXICON["factual_calibration"])
        assert not missing, missing


class TestTokenize:
    def test_strips_contractions_and_punctuation(self) -> None:
        assert tokenize("Prices aren't falling -- they're soaring!") == [
            "prices",
            "arent",
            "falling",
            "theyre",
            "soaring",
        ]

    def test_handles_none_and_empty(self) -> None:
        assert tokenize("") == []
        assert tokenize(None) == []


class TestScoreText:
    def test_counts_negative_and_positive_phrases(self) -> None:
        score = score_text("Recession fears grow even as hiring and wage gains continue")
        assert score.negative >= 2  # recession, fears
        assert score.positive >= 2  # hiring, wage gains
        assert score.negated_hits == 0

    def test_negation_flips_a_positive_to_negative(self) -> None:
        plain = score_text("The economy is improving")
        negated = score_text("The economy is not improving")
        assert plain.positive == 1 and plain.negative == 0
        assert negated.positive == 0 and negated.negative == 1
        assert negated.negated_hits == 1
        assert negated.tone == -1.0

    def test_negation_flips_a_negative_to_positive(self) -> None:
        score = score_text("This is not a recession")
        assert score.positive == 1
        assert score.negative == 0
        assert score.negated_hits == 1

    def test_negation_respects_the_window(self) -> None:
        """A negator far from the phrase must not reach it."""
        near = score_text("inflation is not easing but improving")
        far = score_text("not one of the many analysts we surveyed this week said improving")
        assert near.negated_hits >= 1
        assert far.negated_hits == 0
        assert far.positive == 1

    def test_multiword_phrase_beats_its_substring(self) -> None:
        score = score_text("The cost of living crisis deepens")
        # "cost of living crisis" consumes its tokens, so "crisis" cannot
        # double-count as a second negative hit.
        assert score.negative == 1

    def test_salience_is_not_flipped_by_negation(self) -> None:
        """'Not a crisis' still makes crisis the frame; salience stays counted."""
        score = score_text("Officials insist this is not a crisis")
        assert score.selection_salience == 1

    def test_political_attribution_distinguishes_reporting(self) -> None:
        score = score_text("Critics warn the economy is deteriorating, democrats say")
        assert score.political_attribution >= 2
        assert score.negative >= 1

    def test_off_topic_document_scores_all_zeros(self) -> None:
        score = score_text("The cat sat on the mat")
        assert score.positive == score.negative == 0
        assert score.polar_total == 0
        assert score.tone == 0.0
        assert score.salience_density == 0.0

    def test_empty_text_is_safe(self) -> None:
        score = score_text("")
        assert score.n_tokens == 0
        assert score.tone == 0.0
        assert score.salience_density == 0.0


class TestToneBounds:
    @pytest.mark.parametrize(
        "text",
        [
            "soaring inflation recession layoffs crisis collapse hardship",
            "recovery growth strong resilient improving optimism hiring",
            "recession recovery layoffs hiring weak strong",
            "the cat sat on the mat",
            "not improving, not recovering, no relief in sight",
            "",
        ],
    )
    def test_tone_is_bounded(self, text: str) -> None:
        assert -1.0 <= score_text(text).tone <= 1.0

    def test_extremes_are_attainable(self) -> None:
        assert score_text("recession layoffs collapse").tone == -1.0
        assert score_text("recovery hiring optimism").tone == 1.0

    def test_random_token_soup_stays_bounded(self) -> None:
        vocabulary = (
            list(LEXICON["negative"])
            + list(LEXICON["positive"])
            + list(NEGATORS_FOR_FUZZ)
            + ["the", "and", "widget"]
        )
        for _ in range(200):
            words = RNG.choice(vocabulary, size=12, replace=True)
            tone = score_text(" ".join(words)).tone
            assert -1.0 <= tone <= 1.0


NEGATORS_FOR_FUZZ = ("not", "no", "never", "hardly", "without")


class TestScoreDocuments:
    def test_one_row_in_one_row_out(self) -> None:
        corpus = synthetic_corpus()
        scored = score_documents(corpus)
        assert len(scored) == len(corpus)
        assert scored.index.equals(corpus.index)

    def test_missing_columns_raise_mediaerror(self) -> None:
        with pytest.raises(MediaError, match="missing required columns"):
            score_documents(pd.DataFrame({"headline": ["x"]}))

    def test_nan_text_does_not_explode(self) -> None:
        frame = pd.DataFrame({"date": [pd.Timestamp("2026-05-01")], "text": [np.nan]})
        scored = score_documents(frame)
        assert scored.loc[0, "n_tokens"] == 0


class TestDailyToneIndex:
    def test_bounded_and_daily(self) -> None:
        daily = daily_tone_index(synthetic_corpus())
        assert not daily.empty
        finite = daily["S_media"].dropna()
        assert ((finite >= -1.0) & (finite <= 1.0)).all()
        assert daily.index.name == "date"
        assert daily.index.is_monotonic_increasing

    def test_aggregates_over_summed_counts_not_averaged_tones(self) -> None:
        """A day of off-topic documents must not drag the index towards zero."""
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-01")] * 4,
                "text": [
                    "recession layoffs collapse",
                    "the cat sat on the mat",
                    "the dog sat on the mat",
                    "the bird sat on the mat",
                ],
            }
        )
        daily = daily_tone_index(frame)
        assert daily.loc[pd.Timestamp("2026-05-01"), "S_media"] == -1.0
        assert daily.loc[pd.Timestamp("2026-05-01"), "n_documents"] == 4

    def test_day_with_no_polar_language_is_nan_not_zero(self) -> None:
        """NaN is not a measured neutral: the baseline_model convention."""
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-02")] * 2,
                "text": ["the cat sat on the mat", "a quiet day for widgets"],
            }
        )
        daily = daily_tone_index(frame)
        assert pd.isna(daily.loc[pd.Timestamp("2026-05-02"), "S_media"])
        assert daily.loc[pd.Timestamp("2026-05-02"), "polar_total"] == 0

    def test_known_counts_give_the_exact_index(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-03")] * 2,
                "text": ["recession layoffs collapse weak", "hiring recovery"],
            }
        )
        daily = daily_tone_index(frame)
        row = daily.loc[pd.Timestamp("2026-05-03")]
        assert row["negative"] == 4
        assert row["positive"] == 2
        assert row["S_media"] == pytest.approx((2 - 4) / 6)

    def test_empty_corpus_returns_an_empty_frame(self) -> None:
        daily = daily_tone_index(pd.DataFrame({"date": [], "text": []}))
        assert daily.empty
        assert "S_media" in daily.columns


class TestDailyTopicSalience:
    def test_shares_sum_to_one_per_day(self) -> None:
        corpus = synthetic_corpus()
        tdm = fit_term_document(corpus["text"], min_df=1)
        basis = fit_lsa(tdm, k=3)
        daily = daily_topic_salience(basis, tdm, corpus["date"])
        assert daily.shape[1] == 3
        totals = daily.sum(axis=1).dropna()
        assert np.allclose(totals.to_numpy(), 1.0)

    def test_day_of_origin_documents_is_nan_not_uniform(self) -> None:
        """Zero-prior drift again: no terms means no salience, not equal shares."""
        corpus = synthetic_corpus()
        train = fit_term_document(corpus["text"], min_df=1)
        basis = fit_lsa(train, k=3)
        holdout = apply_term_document(train, ["zzz qqq", "wibble wobble"])
        daily = daily_topic_salience(
            basis, holdout, [pd.Timestamp("2026-06-01")] * 2
        )
        assert daily.isna().all(axis=None)

    def test_date_length_mismatch_raises(self) -> None:
        tdm = fit_term_document(synthetic_corpus()["text"], min_df=1)
        basis = fit_lsa(tdm, k=2)
        with pytest.raises(MediaError, match="documents but"):
            daily_topic_salience(basis, tdm, [pd.Timestamp("2026-05-01")])


class TestCalibrationGap:
    def test_overstatement_is_implied_minus_actual(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-04")] * 2,
                "text": ["soaring inflation hits households", "prices are soaring again"],
            }
        )
        actual = pd.Series({pd.Timestamp("2026-05-31"): 3.0})
        report = calibration_gap(frame, actual)
        row = report.frame.loc[pd.Timestamp("2026-05-04")]
        assert row["implied_yoy"] == pytest.approx(CALIBRATION_CLAIMS["soaring"])
        assert row["actual_yoy"] == pytest.approx(3.0)
        assert row["overstatement"] == pytest.approx(CALIBRATION_CLAIMS["soaring"] - 3.0)
        assert report.days_scored == 1

    def test_documents_without_a_magnitude_claim_are_unscored_not_imputed(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-05")] * 2,
                "text": ["the cat sat on the mat", "a quiet day for widgets"],
            }
        )
        report = calibration_gap(frame, pd.Series({pd.Timestamp("2026-05-31"): 3.0}))
        row = report.frame.loc[pd.Timestamp("2026-05-05")]
        assert row["n_documents"] == 2
        assert row["n_claims"] == 0
        assert pd.isna(row["implied_yoy"])
        assert report.days_scored == 0
        assert report.days_unscored == 1

    def test_accurate_coverage_shows_no_overstatement(self) -> None:
        """Hypothesis B's own falsification path must be reachable."""
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-06")],
                "text": ["inflation is cooling, officials say"],
            }
        )
        report = calibration_gap(frame, pd.Series({pd.Timestamp("2026-05-31"): 2.0}))
        assert report.mean_overstatement == pytest.approx(0.0)

    def test_missing_actual_series_is_reported_not_raised(self) -> None:
        frame = pd.DataFrame(
            {"date": [pd.Timestamp("2026-05-07")], "text": ["soaring inflation"]}
        )
        report = calibration_gap(frame, pd.Series(dtype=float))
        assert report.days_scored == 0
        assert any("not computable" in note for note in report.notes)

    def test_custom_claims_dict_is_honoured(self) -> None:
        frame = pd.DataFrame(
            {"date": [pd.Timestamp("2026-05-08")], "text": ["soaring prices"]}
        )
        report = calibration_gap(
            frame, pd.Series({pd.Timestamp("2026-05-31"): 3.0}), claims={"soaring": 4.0}
        )
        assert report.mean_implied == pytest.approx(4.0)


# --------------------------------------------------------------------------
# 6. FinBERT probe
# --------------------------------------------------------------------------


class TestFinbertProbeSet:
    def test_probe_set_size_and_labels(self) -> None:
        assert len(FINBERT_PROBE) >= 30
        labels = [label for _, label in FINBERT_PROBE]
        assert set(labels) == set(PROBE_LABELS)
        for label in PROBE_LABELS:
            assert labels.count(label) >= 8  # roughly balanced

    def test_sentences_are_unique_and_nonempty(self) -> None:
        sentences = [text for text, _ in FINBERT_PROBE]
        assert len(set(sentences)) == len(sentences)
        assert all(sentence.strip() for sentence in sentences)

    def test_canonical_failure_sentence_is_in_the_probe(self) -> None:
        """BRAINSTORMING.md section 4.1's example must actually be tested."""
        assert any(
            "Strong productivity growth" in text and label == "hawkish"
            for text, label in FINBERT_PROBE
        )


class TestConfusionMatrix:
    def test_all_labels_always_appear(self) -> None:
        table = confusion_matrix(["hawkish", "hawkish"], ["dovish", "dovish"])
        assert list(table.index) == list(PROBE_LABELS)
        assert list(table.columns) == list(PROBE_LABELS)
        assert table.loc["hawkish", "dovish"] == 2
        assert table.to_numpy().sum() == 2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(MediaError, match="intended labels"):
            confusion_matrix(["hawkish"], ["dovish", "neutral"])

    def test_unknown_intended_label_raises(self) -> None:
        with pytest.raises(MediaError, match="unknown intended label"):
            confusion_matrix(["bullish"], ["dovish"])


class TestFinbertProbe:
    def test_degrades_gracefully_when_weights_are_absent(self) -> None:
        """Must never download during tests, and must say so when it cannot run."""
        result = finbert_probe(model="definitely-not-a-real-model/finbert")
        assert result.available is False
        assert result.reason is not None
        assert "allow_download=True" in result.reason
        # The scaffold still comes back so the report can state what was planned.
        assert len(result.predictions) == len(FINBERT_PROBE)
        assert result.confusion.to_numpy().sum() == 0
        assert np.isnan(result.accuracy)
        assert "not run" in result.summary()

    def test_rejects_an_unknown_label_in_a_custom_probe(self) -> None:
        with pytest.raises(MediaError, match="unknown intended label"):
            finbert_probe(probe=(("Rates will rise.", "bullish"),))

    def test_real_weights_path_if_and_only_if_cached(self) -> None:
        """Runs the model only when weights are already on disk; never fetches."""
        result = finbert_probe()
        if not result.available:
            pytest.skip(f"FinBERT weights not cached locally: {result.reason}")
        assert len(result.predictions) == len(FINBERT_PROBE)
        assert result.confusion.to_numpy().sum() == len(FINBERT_PROBE)
        assert 0.0 <= result.accuracy <= 1.0
        assert set(result.predictions["predicted"]) <= set(PROBE_LABELS)


# --------------------------------------------------------------------------
# End-to-end, still offline
# --------------------------------------------------------------------------


class TestPipelineOffline:
    def test_train_fit_holdout_apply_end_to_end(self, tmp_path) -> None:
        """The whole module, cached corpus to daily indices, with no network."""
        GdeltSource(cache_dir=tmp_path, client=StubGdeltClient()).fetch_range(
            "2026-05-12", "2026-05-13", terms=("inflation",)
        )
        offline = GdeltSource(cache_dir=tmp_path, allow_network=False)
        corpus, report = offline.fetch_range("2026-05-12", "2026-05-13", terms=("inflation",))
        assert report.offline and report.cached_chunks == 2

        full = pd.concat([synthetic_corpus(), corpus], ignore_index=True)
        train = full.iloc[: len(full) // 2]
        holdout = full.iloc[len(full) // 2 :]

        tdm = fit_term_document(train["text"], min_df=1)
        selection = select_k(tdm, k_max=8, n_permutations=4)
        basis = fit_lsa(tdm, k=max(selection.k_noise, 2))

        holdout_tdm = apply_term_document(tdm, holdout["text"])
        coordinates = project_documents(basis, holdout_tdm)
        assert coordinates.shape == (len(holdout), basis.k)

        tone = daily_tone_index(full)
        assert ((tone["S_media"].dropna() >= -1) & (tone["S_media"].dropna() <= 1)).all()
        salience = daily_topic_salience(basis, holdout_tdm, holdout["date"])
        assert len(salience) == holdout["date"].nunique()
        table = top_terms_per_topic(basis, n_terms=5)
        assert len(table) == 5 * basis.k

    def test_lexicon_score_dataclass_is_frozen(self) -> None:
        score = score_text("recession")
        with pytest.raises(FrozenInstanceError):
            score.negative = 99  # type: ignore[misc]
        assert isinstance(score, LexiconScore)


class TestFramingLexiconStructure:
    """The framing lexicon is a measurement instrument, so its shape matters."""

    def test_every_contrast_has_two_non_empty_sides(self) -> None:
        """A one-sided contrast is a count, and a count is not a framing measure."""
        assert FRAMING_LEXICON
        for contrast, sides in FRAMING_LEXICON.items():
            assert len(sides) == 2, contrast
            left, right = sides
            assert left, f"{contrast} has no left side"
            assert right, f"{contrast} has no right side"

    def test_the_brief_s_three_contrasts_are_all_present(self) -> None:
        assert set(FRAMING_LEXICON) == {
            "level_vs_rate",
            "household_vs_macro",
            "current_vs_forecast",
        }

    def test_no_phrase_appears_on_both_sides_of_one_contrast(self) -> None:
        """A phrase on both sides would cancel itself and measure nothing."""
        for contrast, (left, right) in FRAMING_LEXICON.items():
            overlap = {p.lower() for p in left} & {p.lower() for p in right}
            assert not overlap, f"{contrast}: {overlap}"

    def test_the_headline_contrast_is_the_one_with_a_predicted_sign(self) -> None:
        assert HEADLINE_FRAMING_CONTRAST == "level_vs_rate"
        assert HEADLINE_FRAMING_CONTRAST in FRAMING_LEXICON


class TestScoreFraming:
    def test_level_language_scores_positive(self) -> None:
        score = score_framing("Grocery prices are still higher than in 2020")
        assert score.ratio("level_vs_rate") > 0

    def test_rate_language_scores_negative(self) -> None:
        score = score_framing("Inflation eased to a 3 percent annual rate")
        assert score.ratio("level_vs_rate") < 0

    def test_a_silent_document_is_nan_and_never_zero(self) -> None:
        """The whole point: no framing is not balanced framing."""
        score = score_framing("the weather was pleasant on tuesday")
        assert np.isnan(score.ratio("level_vs_rate"))
        assert score.framing_total == 0

    def test_balanced_framing_is_zero_and_is_distinguishable_from_silence(self) -> None:
        score = score_framing("prices rose while the inflation rate cooled")
        left, right = score.counts["level_vs_rate"]
        assert left > 0 and right > 0
        assert score.ratio("level_vs_rate") == pytest.approx(
            (left - right) / (left + right)
        )

    def test_the_ratio_stays_inside_the_unit_interval(self) -> None:
        for text in (
            "prices prices prices cost cost of living grocery bill",
            "inflation inflation cpi disinflation annual rate",
            "households at the checkout say the grocery bill hurts",
            "gross domestic product shows the national economy grew",
        ):
            for contrast in FRAMING_LEXICON:
                value = score_framing(text).ratio(contrast)
                if not np.isnan(value):
                    assert -1.0 <= value <= 1.0

    def test_contrasts_are_scored_independently(self) -> None:
        """One sentence may be both level-framed and household-framed."""
        score = score_framing("Households say the grocery bill is still higher")
        assert score.counts["level_vs_rate"][0] > 0
        assert score.counts["household_vs_macro"][0] > 0

    def test_matching_inside_one_side_does_not_double_count(self) -> None:
        """"cost of living" is one hit, not a hit for "cost" plus the phrase."""
        once = score_framing("cost of living")
        assert sum(once.counts["level_vs_rate"]) == 1

    def test_negation_is_not_applied_to_framing(self) -> None:
        """A negator flips a valence, and framing has no valence to flip.

        "Prices are not higher than in 2020" still frames the question as one
        about the price level, which is exactly what the measure detects.
        """
        plain = score_framing("prices are higher than in 2020")
        negated = score_framing("prices are not higher than in 2020")
        assert negated.counts["level_vs_rate"] == plain.counts["level_vs_rate"]

    def test_the_score_is_frozen(self) -> None:
        score = score_framing("prices")
        with pytest.raises(FrozenInstanceError):
            score.n_tokens = 99  # type: ignore[misc]
        assert isinstance(score, FramingScore)


class TestScoreFramingDocuments:
    def test_one_row_in_one_row_out_and_nothing_dropped(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2026-05-01", "2026-05-02", "2026-05-03"],
                "text": ["prices are higher", "nothing here", "inflation eased"],
            }
        )
        scored = score_framing_documents(frame)
        assert len(scored) == len(frame)
        assert list(scored.index) == list(frame.index)

    def test_the_framing_column_repeats_the_headline_contrast(self) -> None:
        frame = pd.DataFrame(
            {"date": ["2026-05-01"], "text": ["the grocery bill is higher"]}
        )
        scored = score_framing_documents(frame)
        assert scored["framing"].iloc[0] == scored[HEADLINE_FRAMING_CONTRAST].iloc[0]

    def test_an_empty_frame_gives_the_full_column_set(self) -> None:
        empty = score_framing_documents(pd.DataFrame(columns=["date", "text"]))
        assert empty.empty
        for contrast in FRAMING_LEXICON:
            assert contrast in empty.columns
            assert f"{contrast}_left" in empty.columns
        assert "framing" in empty.columns


class TestFramingIndex:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [
                    "2026-05-01",
                    "2026-05-01",
                    "2026-05-02",
                ],
                "text": [
                    "prices are higher than in 2020 and the grocery bill hurts",
                    "inflation eased to a 3 percent annual rate",
                    "no economic words at all here",
                ],
            }
        )

    def test_a_day_with_no_framing_word_is_nan(self) -> None:
        daily = framing_index(self._frame())
        assert np.isnan(daily.loc["2026-05-02", "level_vs_rate"])

    def test_aggregation_sums_the_sides_rather_than_averaging_ratios(self) -> None:
        """Summed counts, so a densely framed document is not out-voted."""
        daily = framing_index(self._frame())
        row = daily.loc["2026-05-01"]
        left = row["level_vs_rate_left"]
        right = row["level_vs_rate_right"]
        assert row["level_vs_rate"] == pytest.approx((left - right) / (left + right))

    def test_the_index_stays_inside_the_unit_interval(self) -> None:
        daily = framing_index(self._frame())
        for contrast in FRAMING_LEXICON:
            values = daily[contrast].dropna()
            assert ((values >= -1.0) & (values <= 1.0)).all()

    def test_document_counts_are_reported_beside_the_ratio(self) -> None:
        """A ratio of +1 from one hit and from forty hits are different evidence."""
        daily = framing_index(self._frame())
        assert daily.loc["2026-05-01", "n_documents"] == 2
        assert daily.loc["2026-05-02", "framing_total"] == 0

    def test_an_empty_corpus_gives_an_empty_frame_and_not_an_error(self) -> None:
        daily = framing_index(pd.DataFrame(columns=["date", "text"]))
        assert daily.empty
        assert daily.index.name == "date"


class TestFramingAgainstTheBuiltSeries:
    """The built monthly series must keep the properties the module promises."""

    PATH = pathlib.Path("data/processed/media_monthly.parquet")

    @pytest.mark.skipif(not PATH.exists(), reason="the media series is not built")
    def test_the_framing_columns_are_present(self) -> None:
        series = pd.read_parquet(self.PATH)
        for column in ("framing", *FRAMING_LEXICON, "framing_headline", "framing_slug"):
            assert column in series.columns, column

    @pytest.mark.skipif(not PATH.exists(), reason="the media series is not built")
    def test_every_ratio_is_inside_the_unit_interval(self) -> None:
        series = pd.read_parquet(self.PATH)
        for contrast in FRAMING_LEXICON:
            values = series[contrast].dropna()
            assert ((values >= -1.0) & (values <= 1.0)).all(), contrast

    @pytest.mark.skipif(not PATH.exists(), reason="the media series is not built")
    def test_a_month_with_no_framing_hit_has_no_framing_value(self) -> None:
        """A missing month must be missing, never a zero."""
        series = pd.read_parquet(self.PATH)
        key = f"{HEADLINE_FRAMING_CONTRAST}_hits"
        silent = series[series[key] == 0]
        assert silent[HEADLINE_FRAMING_CONTRAST].isna().all()

    @pytest.mark.skipif(not PATH.exists(), reason="the media series is not built")
    def test_the_two_text_sources_agree_well_above_chance(self) -> None:
        """The cross-source validity check that the report quotes."""
        series = pd.read_parquet(self.PATH)
        paired = series.loc[:, ["framing_headline", "framing_slug"]].dropna()
        assert len(paired) > 30
        correlation = float(paired.corr().iloc[0, 1])
        assert 0.4 < correlation < 0.95, correlation
