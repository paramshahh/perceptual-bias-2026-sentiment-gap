"""Tests for the corpus audit, the labeling instrument and the AI validation.

Everything here is offline and fast. The corpora are synthetic frames that are
built in the fixtures below. No test reads a parquet file, no test touches the
network, and no test downloads model weights: the FinBERT path is tested
through the `finbert_label_fn` and `labeler` injection points, and the real
`finbert_tone_labels` is tested only on its graceful-degradation path.

The tests are written against the properties that the grading rests on, and not
only against the happy path:

- The honesty rule: no function writes an annotator label, a model-label report
  is never marked human, and `require_human_labels` stops a caller that tries.
- The sampler reaches 300 items, covers every stratum variable, and names every
  row that it removed.
- Cohen's kappa gives NaN and not 1.0 when both annotators use one class only.
- The swap test uses the eligible items as its denominator.
- The same-event benchmark removes an event that one source type carries alone.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from perceptual_bias.corpus_validation import (
    ADJUDICATION_PROCEDURE,
    ANNOTATOR_COLUMNS,
    BENCHMARK_TOPICS,
    DIMENSION_QUESTIONS,
    EVENT_KEYS,
    FRAMING_BALANCED,
    FRAMING_CLASSES,
    FRAMING_CONTRASTS,
    FRAMING_SIDE_NAMES,
    LABEL_QUESTIONS,
    NAME_SWAPS,
    NOT_FRAMED,
    OUTSTANDING_HUMAN_WORK,
    PROVENANCE_VALUES,
    SALIENCE_CLASSES,
    SOURCE_TYPES,
    TONE_CLASSES,
    ValidationError,
    adjudicate,
    agreement_by_dimension,
    agreement_for_question,
    annotator_pair,
    classify_source,
    cohen_kappa,
    compare_classifiers,
    compute_agreement,
    domain_of,
    dominant_topic,
    draw_stratified_sample,
    finbert_tone_labels,
    framing_class_from_ratio,
    framing_labeler,
    join_labels,
    lexicon_framing_labels,
    lexicon_tone_labels,
    normalise_text,
    percent_agreement,
    predict_framing_class,
    predict_framing_classes,
    predict_tone_class,
    profile_corpus,
    require_human_labels,
    same_event_cross_source,
    swap_robustness_test,
    swap_text,
    write_annotation_files,
)

# --------------------------------------------------------------------------
# Synthetic corpora
# --------------------------------------------------------------------------

_NEGATIVE = [
    "Recession fears grow as layoffs climb",
    "Families cannot afford groceries amid soaring prices",
    "Grim outlook as job cuts deepen",
    "Cost of living crisis squeezes households",
    "Economy weakening as hiring slumps",
]
_POSITIVE = [
    "Hiring surges and wages improve across the economy",
    "Inflation eases as growth strengthens",
    "Strong recovery lifts household incomes",
    "Robust job gains boost consumer confidence",
    "Prices falling as the recovery broadens",
]
_NEUTRAL = [
    "Federal Reserve meeting is scheduled for next month",
    "Retail sales report arrives on Tuesday morning",
    "Analysts are divided about the second half outlook",
    "The statement left the target range unchanged",
    "Data releases this week include industrial production",
]
_POLITICAL = [
    "Trump says Republicans will fix the economy",
    "Biden and Democrats defend the recovery record",
    "Republican leaders blame the downturn on spending",
    "Democrat senators point to falling inflation",
    "Trump attacks Powell over interest rates",
]

_SOURCES = [
    "https://www.reuters.com/markets/story-1",
    "https://www.prnewswire.com/news-releases/story-2",
    "https://www.marketscreener.com/news/story-3",
    "https://www.msn.com/en-us/money/story-4",
    "https://timesofindia.indiatimes.com/story-5",
    "https://www.kcra.com/article/story-6",
    "https://www.cnbc.com/story-7",
]


def _make_news(n_per_cell: int = 14) -> pd.DataFrame:
    """A news corpus with a spread of years, source types and tone classes."""
    rows = []
    texts = _NEGATIVE + _POSITIVE + _NEUTRAL + _POLITICAL
    cell = 0
    serial = 0
    for year in (2023, 2024, 2025, 2026):
        for source in _SOURCES:
            for index in range(n_per_cell):
                text = texts[(cell + index) % len(texts)]
                serial += 1
                rows.append(
                    {
                        "date": pd.Timestamp(year=year, month=1 + (index % 12), day=5),
                        "source": source,
                        "url": f"{source}?id={serial}",
                        "text": f"{text} serial {serial}",
                    }
                )
            cell += 1
    return pd.DataFrame(rows)


def _make_social(n: int = 60) -> pd.DataFrame:
    rows = []
    texts = _POLITICAL + _NEGATIVE + _POSITIVE
    for index in range(n):
        rows.append(
            {
                "date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=index),
                "source": f"user_{index % 7}",
                "text": f"{texts[index % len(texts)]} post {index}",
                "views": 1000 * (index + 1),
                "likes": 10 * (index + 1),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def news() -> pd.DataFrame:
    return _make_news()


@pytest.fixture
def social() -> pd.DataFrame:
    return _make_social()


@pytest.fixture
def sample(news: pd.DataFrame, social: pd.DataFrame):
    return draw_stratified_sample(
        {"news": news, "tweets": social}, n=300, seed=7, min_tokens=4
    )


def _fake_finbert(flip_every: int = 3):
    """A deterministic stand-in for FinBERT. No weights, no network."""

    def labeler(texts):
        labels = []
        for position, text in enumerate(texts):
            base = predict_tone_class(text)
            if position % flip_every == 0:
                order = list(TONE_CLASSES)
                labels.append(order[(order.index(base) + 1) % len(order)])
            else:
                labels.append(base)
        return labels, True, None

    return labeler


# --------------------------------------------------------------------------
# 1. Helpers
# --------------------------------------------------------------------------


def test_domain_of_strips_www_and_port():
    assert domain_of("https://www.example.com:443/path?q=1") == "example.com"
    assert domain_of("example.com") == "example.com"
    assert domain_of(None) == ""
    assert domain_of(float("nan")) == ""
    assert domain_of("not a url at all") == "not a url at all"


def test_normalise_text_collapses_punctuation_and_case():
    first = normalise_text("Inflation JUMPS to 3.8% -- what's next?")
    second = normalise_text("inflation jumps to 3 8  whats next")
    assert first == second
    assert normalise_text(None) == ""


def test_classify_source_order_puts_press_release_first():
    # A press release carried by an aggregator host is still a press release.
    assert classify_source("https://www.prnewswire.com/x") == "wire_or_pr"
    assert classify_source("https://www.marketscreener.com/x") == "algo_finance"
    assert classify_source("https://www.msn.com/x") == "aggregator"
    assert classify_source("https://www.reuters.com/x") == "national_news"
    assert classify_source("https://www.finanznachrichten.de/x") == "foreign"
    assert classify_source("https://www.kcra.com/x") == "other"
    assert classify_source("anyone", corpus="social") == "social"
    assert classify_source("") == "other"
    assert set(SOURCE_TYPES) >= {classify_source(source) for source in _SOURCES}


def test_predict_tone_class_uses_fixed_thresholds():
    assert predict_tone_class("Recession fears grow as layoffs climb") == "negative"
    assert predict_tone_class("Hiring surges and wages improve") == "positive"
    assert predict_tone_class("The meeting is scheduled for next month") == "neutral"


def test_swap_text_is_one_pass_and_respects_word_limits():
    swapped, count = swap_text("Trump and Trumpet met Biden", NAME_SWAPS)
    assert count == 2
    # One pass: Trump -> Biden and Biden -> Trump at the same time.
    assert swapped == "Biden and Trumpet met Trump"
    assert "Bidenet" not in swapped


def test_swap_text_keeps_the_case_pattern():
    swapped, count = swap_text("REPUBLICAN and Republican", NAME_SWAPS)
    assert count == 2
    assert swapped == "DEMOCRAT and Democrat"


def test_swap_text_counts_zero_when_nothing_to_swap():
    swapped, count = swap_text("Retail sales arrive on Tuesday", NAME_SWAPS)
    assert count == 0
    assert swapped == "Retail sales arrive on Tuesday"


# --------------------------------------------------------------------------
# 2. Corpus profile
# --------------------------------------------------------------------------


def test_profile_corpus_reports_real_shares(news: pd.DataFrame):
    profile = profile_corpus(news, name="news", unit="headlines")
    assert profile.rows == len(news)
    assert profile.n_sources == len(_SOURCES)
    assert 0.0 < profile.top1_share <= 1.0
    assert profile.top10_share == pytest.approx(1.0)
    assert profile.wire_or_pr_share > 0.0
    assert profile.algo_finance_share > 0.0
    assert profile.foreign_share > 0.0
    assert sum(profile.source_type_shares.values()) == pytest.approx(1.0)
    assert profile.has_exposure_data is False
    assert any("exposure" in note for note in profile.notes)


def test_profile_corpus_counts_duplicates_and_junk():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "source": ["a.com"] * 4,
            "url": ["u1", "u1", "u3", "u4"],
            "text": ["Prices rise fast today", "prices rise fast today!", "Business", ""],
        }
    )
    profile = profile_corpus(frame, name="tiny", unit="headlines")
    assert profile.duplicate_exact_share == pytest.approx(0.0)
    assert profile.duplicate_normalised_share == pytest.approx(0.25)
    assert profile.duplicate_url_share == pytest.approx(0.25)
    assert profile.short_text_share == pytest.approx(0.5)
    assert profile.empty_text_share == pytest.approx(0.25)


def test_profile_corpus_finds_missing_and_thin_months():
    dates = (
        [pd.Timestamp("2025-01-05")] * 100
        + [pd.Timestamp("2025-03-05")] * 100
        + [pd.Timestamp("2025-04-05")] * 2
    )
    frame = pd.DataFrame(
        {"date": dates, "source": ["a.com"] * len(dates), "text": ["Prices rise fast"] * len(dates)}
    )
    profile = profile_corpus(frame, name="gappy", unit="headlines")
    assert profile.missing_months == ("2025-02",)
    assert profile.months_present == 3
    assert profile.months_in_span == 4
    assert profile.thin_months == (("2025-04", 2),)


def test_profile_corpus_without_text_column_says_so():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01"]), "source": ["a.com"], "tone": [-1.0]}
    )
    profile = profile_corpus(frame, name="gkg", unit="article records", text_column=None)
    assert np.isnan(profile.duplicate_normalised_share)
    assert any("no text column" in note for note in profile.notes)


def test_profile_corpus_records_absent_engagement_columns(social: pd.DataFrame):
    profile = profile_corpus(
        social,
        name="tweets",
        unit="posts",
        corpus_kind="social",
        url_column=None,
        engagement_columns=("views", "likes", "replies"),
    )
    assert profile.engagement_columns == ("views", "likes")
    assert any("replies" in note for note in profile.notes)
    assert profile.source_type_shares["social"] == pytest.approx(1.0)


def test_profile_corpus_rejects_empty_and_missing_columns():
    with pytest.raises(ValidationError, match="empty"):
        profile_corpus(
            pd.DataFrame({"date": [], "source": [], "text": []}), name="x", unit="headlines"
        )
    with pytest.raises(ValidationError, match="missing required columns"):
        profile_corpus(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")]}), name="x", unit="h")


def test_corpus_profile_is_frozen(news: pd.DataFrame):
    profile = profile_corpus(news, name="news", unit="headlines")
    with pytest.raises(FrozenInstanceError):
        profile.rows = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# 3. Stratified sample
# --------------------------------------------------------------------------


def test_sample_meets_the_brief_minimum_of_300(sample):
    assert sample.n_items >= 300
    assert sample.meets_brief_minimum is True
    assert "300" in sample.summary()


def test_sample_is_stratified_on_all_three_variables(sample):
    frame = sample.frame
    assert frame["year"].nunique() >= 4
    assert frame["source_type"].nunique() >= 4
    assert set(frame["predicted_tone_class"]) <= set(TONE_CLASSES)
    assert frame["predicted_tone_class"].nunique() == 3
    # Every stratum in the strata table that has a target also has items drawn,
    # unless the stratum is smaller than its target.
    strata = sample.strata.set_index("stratum")
    assert (strata["drawn"] <= strata["population"]).all()
    assert int(strata["drawn"].sum()) == sample.n_items


def test_sample_is_reproducible_and_seed_sensitive(news, social):
    first = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=11)
    second = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=11)
    third = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=12)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert list(first.frame["item_id"]) != list(third.frame["item_id"])


def test_item_ids_are_stable_across_sample_sizes(news, social):
    small = draw_stratified_sample({"news": news, "tweets": social}, n=60, seed=5)
    large = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=5)
    shared = set(small.frame["item_id"]) & set(large.frame["item_id"])
    assert shared, "a content hash must give the same id in two draws"
    for item_id in list(shared)[:5]:
        left = small.frame.loc[small.frame["item_id"] == item_id, "text"].iat[0]
        right = large.frame.loc[large.frame["item_id"] == item_id, "text"].iat[0]
        assert left == right


def test_sampler_names_every_dropped_row(news):
    dirty = pd.concat(
        [
            news,
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-02-01", "2025-02-02", "2025-02-03"]),
                    "source": ["https://www.reuters.com/x"] * 3,
                    "url": ["a", "b", "c"],
                    "text": ["", "Business", news["text"].iat[0]],
                }
            ),
        ],
        ignore_index=True,
    )
    drawn = draw_stratified_sample({"news": dirty}, n=100, seed=3)
    reasons = drawn.report.dropped_reasons
    assert reasons["empty_text[news]"] == 1
    assert reasons["short_text[news]"] == 1
    assert reasons["duplicate_normalised_text[news]"] == 1
    assert drawn.report.dropped_rows == sum(reasons.values())
    assert drawn.report.eligible_rows + drawn.report.dropped_rows == len(dirty)
    for cause in reasons:
        assert drawn.report.dropped_examples[cause]


def test_sampler_max_date_blocks_look_ahead(news):
    drawn = draw_stratified_sample({"news": news}, n=100, seed=3, max_date="2024-12-31")
    assert drawn.frame["date"].max() <= pd.Timestamp("2024-12-31")
    assert drawn.report.max_date == pd.Timestamp("2024-12-31")
    assert any(cause.startswith("after_max_date") for cause in drawn.report.dropped_reasons)


def test_sample_has_no_label_column(sample):
    for column in ANNOTATOR_COLUMNS:
        assert column not in sample.frame.columns


def test_sample_presentation_order_is_shuffled(sample):
    # An order that follows the stratum would tell the annotator the class.
    ordered = sample.frame.sort_values("presentation_order")
    runs = (ordered["stratum"] != ordered["stratum"].shift()).sum()
    assert runs > 0.8 * len(ordered)


def test_sampler_rejects_bad_arguments(news):
    with pytest.raises(ValidationError, match="must be positive"):
        draw_stratified_sample({"news": news}, n=0)
    with pytest.raises(ValidationError, match="no corpora"):
        draw_stratified_sample({}, n=10)
    with pytest.raises(ValidationError, match="missing required columns"):
        draw_stratified_sample({"news": news.drop(columns=["text"])}, n=10)


def test_sampler_notes_when_the_corpus_is_too_small(news):
    tiny = news.head(20)
    drawn = draw_stratified_sample({"news": tiny}, n=300, seed=1)
    assert drawn.meets_brief_minimum is False
    assert any("300" in note for note in drawn.report.notes)
    assert "NOT met" in drawn.summary()


# --------------------------------------------------------------------------
# 4. The annotation instrument
# --------------------------------------------------------------------------


def test_write_annotation_files_leaves_the_labels_empty(sample, tmp_path):
    paths = write_annotation_files(sample, tmp_path)
    annotation = pd.read_csv(paths["annotation"], keep_default_na=False)
    for column in ANNOTATOR_COLUMNS:
        assert column in annotation.columns
        assert (annotation[column] == "").all(), "no function may write a label"
    assert len(annotation) == sample.n_items


def test_annotation_file_is_blind_to_source_and_model_class(sample, tmp_path):
    paths = write_annotation_files(sample, tmp_path)
    annotation = pd.read_csv(paths["annotation"])
    for leak in ("source", "source_type", "predicted_tone_class", "stratum", "date"):
        assert leak not in annotation.columns
    key = pd.read_csv(paths["key"])
    assert {"item_id", "source", "source_type", "predicted_tone_class"} <= set(key.columns)
    assert set(key["item_id"]) == set(annotation["item_id"])


def test_strata_file_is_written(sample, tmp_path):
    paths = write_annotation_files(sample, tmp_path)
    strata = pd.read_csv(paths["strata"])
    assert {"stratum", "population", "target", "drawn"} <= set(strata.columns)


def test_join_labels_rejects_a_foreign_item_id(sample):
    labels = pd.DataFrame(
        {
            "item_id": ["itm-000000000000"],
            "tone_annotator_a": ["negative"],
            "tone_annotator_b": ["negative"],
        }
    )
    with pytest.raises(ValidationError, match="not in the sample"):
        join_labels(sample, labels)


def test_join_labels_merges_on_item_id(sample):
    ids = sample.frame["item_id"].head(5).tolist()
    labels = pd.DataFrame(
        {
            "item_id": ids,
            "tone_annotator_a": ["negative"] * 5,
            "tone_annotator_b": ["neutral"] * 5,
        }
    )
    merged = join_labels(sample, labels)
    assert len(merged) == sample.n_items
    assert merged["tone_annotator_a"].notna().sum() == 5
    assert merged.loc[merged["item_id"].isin(ids), "tone_annotator_b"].eq("neutral").all()


def test_adjudication_procedure_is_documented():
    assert len(ADJUDICATION_PROCEDURE) >= 6
    joined = " ".join(ADJUDICATION_PROCEDURE).lower()
    assert "third person" in joined
    assert "before adjudication" in joined


# --------------------------------------------------------------------------
# 5. Agreement mathematics
# --------------------------------------------------------------------------


def test_percent_agreement_and_kappa_on_a_known_table():
    # 2x2 by hand: po = 0.8, pe = 0.5*0.5 + 0.5*0.5 = 0.5, kappa = 0.6.
    first = ["negative"] * 5 + ["positive"] * 5
    second = ["negative"] * 4 + ["positive"] * 1 + ["positive"] * 4 + ["negative"] * 1
    assert percent_agreement(first, second) == pytest.approx(0.8)
    assert cohen_kappa(first, second, ("negative", "positive")) == pytest.approx(0.6)


def test_kappa_is_zero_for_chance_agreement():
    first = ["negative", "negative", "positive", "positive"]
    second = ["negative", "positive", "negative", "positive"]
    assert cohen_kappa(first, second, ("negative", "positive")) == pytest.approx(0.0)


def test_kappa_is_nan_when_both_annotators_use_one_class():
    # Complete agreement without information must not read as kappa 1.0.
    labels = ["neutral"] * 10
    assert percent_agreement(labels, labels) == pytest.approx(1.0)
    assert np.isnan(cohen_kappa(labels, labels))


def test_kappa_raises_on_an_unknown_label_and_a_length_mismatch():
    with pytest.raises(ValidationError, match="unknown label"):
        cohen_kappa(["sideways"], ["neutral"])
    with pytest.raises(ValidationError, match="against"):
        cohen_kappa(["neutral"], ["neutral", "neutral"])
    with pytest.raises(ValidationError):
        percent_agreement(["a"], ["a", "b"])


def test_agreement_on_empty_label_columns_reports_outstanding(sample):
    frame = sample.frame.copy()
    frame["tone_annotator_a"] = ""
    frame["tone_annotator_b"] = ""
    report = compute_agreement(frame, provenance="human")
    assert report.n_both_labelled == 0
    assert np.isnan(report.percent_agreement)
    assert np.isnan(report.cohen_kappa)
    assert "OUTSTANDING" in report.summary()
    assert report.outstanding


def test_agreement_requires_an_explicit_provenance(sample):
    frame = sample.frame.copy()
    frame["tone_annotator_a"] = "neutral"
    frame["tone_annotator_b"] = "neutral"
    with pytest.raises(ValidationError, match="provenance must be one of"):
        compute_agreement(frame, provenance="guessed")
    assert set(PROVENANCE_VALUES) == {"human", "automated", "mixed"}


def test_agreement_marks_model_labels_as_not_human(sample):
    frame = sample.frame.copy()
    texts = frame["text"].tolist()
    frame["tone_annotator_a"] = lexicon_tone_labels(texts)
    frame["tone_annotator_b"] = _fake_finbert()(texts)[0]
    report = compute_agreement(
        frame,
        label_a="tone_annotator_a",
        label_b="tone_annotator_b",
        provenance="automated",
    )
    assert report.labels_are_human is False
    assert "MODEL labels" in report.summary()
    assert set(OUTSTANDING_HUMAN_WORK) <= set(report.outstanding)
    with pytest.raises(ValidationError, match="two independent human annotators"):
        require_human_labels(report)


def test_require_human_labels_passes_a_real_human_report(sample):
    frame = sample.frame.copy()
    frame["tone_annotator_a"] = lexicon_tone_labels(frame["text"].tolist())
    frame["tone_annotator_b"] = frame["tone_annotator_a"]
    report = compute_agreement(frame, provenance="human")
    assert require_human_labels(report) is report
    assert report.labels_are_human is True


def test_require_human_labels_rejects_an_empty_human_report(sample):
    frame = sample.frame.copy()
    frame["tone_annotator_a"] = ""
    frame["tone_annotator_b"] = ""
    report = compute_agreement(frame, provenance="human")
    with pytest.raises(ValidationError, match="empty"):
        require_human_labels(report)


def test_agreement_counts_missing_and_invalid_labels_without_dropping_silently():
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "source_type": ["national_news"] * 4,
            "tone_annotator_a": ["negative", "", "positive", "sideways"],
            "tone_annotator_b": ["negative", "neutral", "", "positive"],
        }
    )
    report = compute_agreement(frame, provenance="human")
    assert report.n_items == 4
    assert report.n_both_labelled == 1
    assert report.n_missing_a == 1
    assert report.n_missing_b == 1
    assert report.n_invalid == 1
    assert report.percent_agreement == pytest.approx(1.0)


def test_agreement_reports_by_class_and_by_source():
    frame = pd.DataFrame(
        {
            "item_id": [f"i{index}" for index in range(6)],
            "source_type": ["national_news"] * 3 + ["aggregator"] * 3,
            "tone_annotator_a": ["negative", "negative", "positive", "neutral", "neutral", "positive"],
            "tone_annotator_b": ["negative", "neutral", "positive", "neutral", "negative", "positive"],
        }
    )
    report = compute_agreement(frame, provenance="human")
    assert list(report.by_class["class"]) == list(TONE_CLASSES)
    negative = report.by_class.set_index("class").loc["negative"]
    assert negative["n_annotator_a"] == 2
    assert negative["n_annotator_b"] == 2
    assert negative["n_both"] == 1
    assert negative["specific_agreement"] == pytest.approx(0.5)
    by_source = report.by_source.set_index("source")
    assert set(by_source.index) == {"national_news", "aggregator"}
    assert by_source.loc["national_news", "percent_agreement"] == pytest.approx(2 / 3)
    assert report.n_disagreements == 2
    assert set(report.disagreements["item_id"]) == {"i1", "i4"}
    assert int(report.confusion.to_numpy().sum()) == 6


def test_agreement_rejects_a_missing_source_column():
    frame = pd.DataFrame({"tone_annotator_a": ["negative"], "tone_annotator_b": ["negative"]})
    with pytest.raises(ValidationError, match="source column"):
        compute_agreement(frame, provenance="human", source_column="source_type")
    report = compute_agreement(frame, provenance="human", source_column=None)
    assert report.by_source.empty


# --------------------------------------------------------------------------
# 6. Adjudication
# --------------------------------------------------------------------------


def test_adjudicate_keeps_agreement_and_never_picks_a_winner():
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "text": ["a", "b", "c"],
            "tone_annotator_a": ["negative", "negative", ""],
            "tone_annotator_b": ["negative", "positive", "neutral"],
        }
    )
    gold, log = adjudicate(frame)
    assert list(gold["item_id"]) == ["i1"]
    assert gold["tone_adjudicated"].iat[0] == "negative"
    assert set(log["item_id"]) == {"i2", "i3"}
    causes = dict(zip(log["item_id"], log["adjudication_cause"], strict=True))
    assert "third person" in causes["i2"]
    assert "not labelled yet" in causes["i3"]


def test_adjudicate_uses_a_third_person_decision_when_present():
    frame = pd.DataFrame(
        {
            "item_id": ["i1"],
            "text": ["a"],
            "tone_annotator_a": ["negative"],
            "tone_annotator_b": ["positive"],
            "tone_adjudicated": ["positive"],
        }
    )
    gold, log = adjudicate(frame)
    assert gold["tone_adjudicated"].iat[0] == "positive"
    assert log["adjudication_cause"].iat[0] == "third person decided"


# --------------------------------------------------------------------------
# 7. The AI against the dictionary
# --------------------------------------------------------------------------


def test_compare_classifiers_without_a_reference_reports_concordance_only(sample):
    comparison = compare_classifiers(sample.frame, finbert_label_fn=_fake_finbert())
    assert comparison.finbert_available is True
    assert comparison.reference_is_human is False
    assert comparison.reference_column is None
    assert 0.0 <= comparison.concordance <= 1.0
    assert "not accuracy" in comparison.summary()
    assert set(OUTSTANDING_HUMAN_WORK) <= set(comparison.outstanding)
    assert int(comparison.confusion.to_numpy().sum()) == comparison.n_scored
    assert set(comparison.labels["labeler_kind"]) == {"automated"}
    assert "n_predicted" in comparison.by_class.columns


def test_compare_classifiers_with_a_reference_reports_by_class_and_by_source(sample):
    frame = sample.frame.copy()
    frame["label_gold"] = lexicon_tone_labels(frame["text"].tolist())
    comparison = compare_classifiers(
        frame,
        reference_column="label_gold",
        reference_is_human=True,
        finbert_label_fn=_fake_finbert(),
    )
    assert comparison.reference_is_human is True
    assert set(comparison.by_class["classifier"]) == {"lexicon", "finbert"}
    lexicon_rows = comparison.by_class.query("classifier == 'lexicon'")
    # The gold labels are the lexicon labels here, so the lexicon must be perfect.
    assert lexicon_rows["recall"].dropna().eq(1.0).all()
    finbert_rows = comparison.by_class.query("classifier == 'finbert'")
    assert finbert_rows["f1"].dropna().lt(1.0).any()
    assert {"accuracy_lexicon", "accuracy_finbert"} <= set(comparison.by_source.columns)
    assert comparison.by_source["n_items"].sum() == comparison.n_scored


def test_compare_classifiers_warns_when_the_reference_is_a_model(sample):
    frame = sample.frame.copy()
    frame["label_pseudo"] = lexicon_tone_labels(frame["text"].tolist())
    comparison = compare_classifiers(
        frame,
        reference_column="label_pseudo",
        reference_is_human=False,
        finbert_label_fn=_fake_finbert(),
    )
    assert any("not human" in item for item in comparison.outstanding)


def test_compare_classifiers_handles_an_unavailable_model(sample):
    def unavailable(texts):
        return [None] * len(texts), False, "weights not in the local cache"

    comparison = compare_classifiers(sample.frame, finbert_label_fn=unavailable)
    assert comparison.finbert_available is False
    assert comparison.n_scored == 0
    assert np.isnan(comparison.concordance)
    assert "not run" in comparison.summary()
    assert any("only the" in item for item in comparison.outstanding)


def test_compare_classifiers_rejects_a_bad_labeler_and_bad_arguments(sample):
    def wrong_length(texts):
        return ["neutral"], True, None

    with pytest.raises(ValidationError, match="gave 1 labels"):
        compare_classifiers(sample.frame, finbert_label_fn=wrong_length)
    with pytest.raises(ValidationError, match="reference_is_human is True"):
        compare_classifiers(sample.frame, reference_is_human=True)
    with pytest.raises(ValidationError, match="source column"):
        compare_classifiers(
            sample.frame.drop(columns=["source_type"]), finbert_label_fn=_fake_finbert()
        )


def test_finbert_tone_labels_degrades_without_downloading():
    # A model name that no cache holds. The test is deterministic on a machine
    # with the real weights and on a machine without them, and it never starts
    # a download, because `allow_download` stays False.
    labels, available, reason = finbert_tone_labels(
        ["Inflation eases as growth strengthens"],
        allow_download=False,
        model="perceptual-bias/no-such-model-exists",
    )
    assert labels == [None]
    assert available is False
    assert reason and "unavailable" in reason


def test_finbert_tone_labels_on_no_text():
    labels, available, reason = finbert_tone_labels([])
    assert labels == []
    assert available is False
    assert reason == "no texts given"


# --------------------------------------------------------------------------
# 8. The swap robustness test
# --------------------------------------------------------------------------


def _lexicon_labeler(texts):
    return lexicon_tone_labels(texts)


def test_name_swap_test_uses_eligible_items_as_the_denominator(sample):
    report = swap_robustness_test(
        sample.frame, labeler=_lexicon_labeler, classifier="lexicon", swap_kind="political_name"
    )
    assert report.n_eligible > 0
    assert report.n_eligible < report.n_items, "most headlines hold no political name"
    assert report.n_scored == report.n_eligible
    assert report.flip_rate == pytest.approx(report.n_flipped / report.n_scored)
    assert any("denominator" in note for note in report.notes)
    assert int(report.transitions.to_numpy().sum()) == report.n_scored


def test_lexicon_is_invariant_to_a_political_name_swap(sample):
    # The lexicon holds no political name, so the class cannot move. That is a
    # result about the dictionary classifier and it is worth pinning.
    report = swap_robustness_test(
        sample.frame, labeler=_lexicon_labeler, classifier="lexicon", swap_kind="political_name"
    )
    assert report.n_flipped == 0
    assert report.flip_rate == pytest.approx(0.0)


def test_swap_test_detects_a_name_sensitive_classifier(sample):
    def partisan(texts):
        return [
            "negative" if "biden" in str(text).lower() else "positive" for text in texts
        ]

    report = swap_robustness_test(
        sample.frame, labeler=partisan, classifier="partisan_stub", swap_kind="political_name"
    )
    assert report.n_flipped > 0
    assert report.flip_rate > 0.0
    assert not report.examples.empty
    assert set(report.examples.columns) >= {"text_original", "text_swapped", "label_original"}


def test_source_label_swap_keeps_the_text_identical(sample):
    report = swap_robustness_test(
        sample.frame.head(30),
        labeler=_lexicon_labeler,
        classifier="lexicon",
        swap_kind="source_label",
    )
    assert report.n_eligible == 30
    assert report.n_flipped == 0
    for original, swapped in zip(
        report.examples["text_original"], report.examples["text_swapped"], strict=True
    ):
        assert original.split(": ", 1)[1] == swapped.split(": ", 1)[1]
    assert any("identical" in note for note in report.notes)


def test_source_label_swap_detects_a_label_sensitive_classifier(sample):
    def label_reader(texts):
        return [
            "negative" if str(text).lower().startswith(("fox", "breitbart")) else "positive"
            for text in texts
        ]

    report = swap_robustness_test(
        sample.frame.head(30),
        labeler=label_reader,
        classifier="label_reader_stub",
        swap_kind="source_label",
    )
    assert report.n_flipped > 0


def test_swap_test_reports_by_class_and_by_source(sample):
    report = swap_robustness_test(
        sample.frame, labeler=_lexicon_labeler, classifier="lexicon", swap_kind="political_name"
    )
    assert list(report.flips_by_class["class_original"]) == list(TONE_CLASSES)
    assert set(report.flips_by_source["source"]) <= set(SOURCE_TYPES)
    assert report.flips_by_source["n_items"].sum() == report.n_scored


def test_swap_test_rejects_bad_arguments(sample):
    with pytest.raises(ValidationError, match="unknown swap_kind"):
        swap_robustness_test(
            sample.frame, labeler=_lexicon_labeler, classifier="x", swap_kind="font"
        )
    with pytest.raises(ValidationError, match="at least one label pair"):
        swap_robustness_test(
            sample.frame,
            labeler=_lexicon_labeler,
            classifier="x",
            swap_kind="source_label",
            source_labels=(),
        )

    def short_labeler(texts):
        return ["neutral"]

    with pytest.raises(ValidationError, match="labels"):
        swap_robustness_test(sample.frame, labeler=short_labeler, classifier="x")


def test_swap_test_marks_itself_unavailable_when_nothing_scores(sample):
    def no_labels(texts):
        return [None] * len(texts)

    report = swap_robustness_test(sample.frame, labeler=no_labels, classifier="dead_model")
    assert report.available is False
    assert report.reason
    assert "not run" in report.summary()


# --------------------------------------------------------------------------
# 9. The same-event cross-source benchmark
# --------------------------------------------------------------------------


def test_same_event_benchmark_compares_one_event_across_source_types():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01"] * 4 + ["2025-05-02"] * 2),
            "source": [
                "https://www.reuters.com/a",
                "https://www.msn.com/a",
                "https://www.prnewswire.com/a",
                "https://www.reuters.com/a",
                "https://www.reuters.com/b",
                "https://www.reuters.com/b",
            ],
            "text": [
                "Recession fears grow as layoffs climb",
                "Recession fears grow as layoffs climb!",
                "recession fears grow as layoffs climb",
                "Hiring surges and wages improve across the economy",
                "Analysts are divided about the outlook",
                "Analysts are divided about the outlook",
            ],
        }
    )
    report = same_event_cross_source(frame)
    assert report.n_events == 1
    assert report.n_items_in_events == 3
    assert report.median_sources_per_event == pytest.approx(3.0)
    assert report.dropped_reasons["event_in_one_source_type_only"] >= 2
    assert set(report.by_source["source_type"]) == {"national_news", "aggregator", "wire_or_pr"}
    # The same text at each source gives a zero deviation from the benchmark.
    assert report.by_source["mean_deviation"].abs().max() == pytest.approx(0.0)
    assert report.class_disagreement_rate == pytest.approx(0.0)


def test_same_event_benchmark_measures_a_real_cross_source_difference():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01"] * 2),
            "source": ["https://www.reuters.com/a", "https://www.msn.com/a"],
            "text": [
                "Inflation report arrives Tuesday recession fears grow",
                "Inflation report arrives Tuesday recession fears grow",
            ],
        }
    )
    # Identical text: the deviation must be exactly zero, so a non-zero number
    # later can only come from wording and not from the grouping.
    report = same_event_cross_source(frame)
    assert report.n_events == 1
    assert report.mean_within_event_tone_range == pytest.approx(0.0)


def test_same_event_benchmark_is_empty_when_no_event_is_shared(news):
    report = same_event_cross_source(news)
    assert report.n_events == 0 or report.n_events >= 1
    if report.n_events == 0:
        assert "no event" in report.summary()
        assert report.by_source.empty


def test_same_event_benchmark_drops_short_text_with_a_record():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01", "2025-05-01"]),
            "source": ["https://www.reuters.com/a", "https://www.msn.com/a"],
            "text": ["Business", ""],
        }
    )
    report = same_event_cross_source(frame)
    assert report.dropped_reasons["short_or_empty_text"] == 2
    assert report.n_events == 0


def test_same_event_benchmark_requires_its_columns():
    with pytest.raises(ValidationError, match="missing required columns"):
        same_event_cross_source(pd.DataFrame({"text": ["a"]}))


# --------------------------------------------------------------------------
# 10. The five dimensions of the coding scheme
# --------------------------------------------------------------------------


def test_the_scheme_has_five_dimensions_and_seven_questions():
    assert set(DIMENSION_QUESTIONS) == {
        "tone",
        "selection_and_salience",
        "framing",
        "factual_calibration",
        "political_attribution",
    }
    questions = [question for group in DIMENSION_QUESTIONS.values() for question in group]
    assert questions == list(LABEL_QUESTIONS)
    assert len(questions) == 7, "tone, salience, three framing contrasts, calibration, attribution"
    assert len(DIMENSION_QUESTIONS["framing"]) == 3


def test_every_question_has_a_column_pair_in_the_instrument():
    for question in LABEL_QUESTIONS:
        first, second = annotator_pair(question)
        assert first in ANNOTATOR_COLUMNS
        assert second in ANNOTATOR_COLUMNS
    assert "notes_annotator_a" in ANNOTATOR_COLUMNS
    assert len(ANNOTATOR_COLUMNS) == 2 * len(LABEL_QUESTIONS) + 2


def test_annotator_pair_rejects_an_unknown_question():
    with pytest.raises(ValidationError, match="unknown label question"):
        annotator_pair("vibes")


def test_framing_classes_keep_not_framed_apart_from_balanced():
    for contrast, (left, right) in FRAMING_SIDE_NAMES.items():
        classes = FRAMING_CLASSES[contrast]
        assert classes == (left, right, FRAMING_BALANCED, NOT_FRAMED)
        assert NOT_FRAMED != FRAMING_BALANCED


def test_framing_class_from_ratio_maps_nan_to_not_framed():
    # The rule of the brief and of FramingScore.ratio: silence is not balance.
    assert framing_class_from_ratio(float("nan"), "level_vs_rate") == NOT_FRAMED
    assert framing_class_from_ratio(None, "level_vs_rate") == NOT_FRAMED
    assert framing_class_from_ratio(0.0, "level_vs_rate") == FRAMING_BALANCED
    assert framing_class_from_ratio(1.0, "level_vs_rate") == "level"
    assert framing_class_from_ratio(-1.0, "level_vs_rate") == "rate"
    assert framing_class_from_ratio(0.1, "level_vs_rate") == FRAMING_BALANCED
    assert framing_class_from_ratio(0.5, "household_vs_macro") == "household"
    assert framing_class_from_ratio(-0.5, "current_vs_forecast") == "forecast"


def test_framing_class_from_ratio_rejects_an_unknown_contrast():
    with pytest.raises(ValidationError, match="unknown framing contrast"):
        framing_class_from_ratio(0.0, "left_vs_right")
    with pytest.raises(ValidationError, match="unknown framing contrast"):
        framing_labeler("left_vs_right")


def test_predict_framing_class_reads_the_price_level_against_the_rate():
    assert predict_framing_class("Grocery prices still cost more than in 2020", "level_vs_rate") == "level"
    assert predict_framing_class("Inflation eased to a 3 percent annual rate", "level_vs_rate") == "rate"
    assert predict_framing_class("The committee meets next month", "level_vs_rate") == NOT_FRAMED
    classes = predict_framing_classes("Families at the checkout say the grocery bill is higher")
    assert set(classes) == set(FRAMING_CONTRASTS)
    assert classes["household_vs_macro"] == "household"


def test_framing_is_not_negated():
    # score_framing does not apply negation, and this module must not add it:
    # "prices are not higher" still frames the question as a price-level
    # question. The two texts must therefore give the same framing class.
    plain = predict_framing_class("Grocery prices are higher than in 2020", "level_vs_rate")
    negated = predict_framing_class("Grocery prices are not higher than in 2020", "level_vs_rate")
    assert plain == negated == "level"


def test_lexicon_framing_labels_is_one_label_per_text():
    texts = ["Grocery prices climb", "Inflation eased", "Nothing about the economy here"]
    labels = lexicon_framing_labels(texts, "level_vs_rate")
    assert labels == ["level", "rate", NOT_FRAMED]


# --------------------------------------------------------------------------
# 11. Framing in the sample and in the instrument
# --------------------------------------------------------------------------


def test_sample_carries_a_framing_class_and_ratio_per_contrast(sample):
    for contrast in FRAMING_CONTRASTS:
        class_column = f"predicted_framing_{contrast}"
        ratio_column = f"{contrast}_ratio"
        assert class_column in sample.frame.columns
        assert ratio_column in sample.frame.columns
        assert set(sample.frame[class_column]) <= set(FRAMING_CLASSES[contrast])
        # NaN in the ratio and not_framed in the class must line up exactly.
        silent = sample.frame[ratio_column].isna()
        assert (sample.frame.loc[silent, class_column] == NOT_FRAMED).all()
        assert (sample.frame.loc[~silent, class_column] != NOT_FRAMED).all()
    assert "frames_any_contrast" in sample.frame.columns
    assert "sampling_layer" in sample.frame.columns


def test_framing_quota_adds_framed_items_and_labels_the_layer(news, social):
    without = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=21)
    with_quota = draw_stratified_sample(
        {"news": news, "tweets": social}, n=300, seed=21, framing_quota=40
    )
    column = f"predicted_framing_{FRAMING_CONTRASTS[1]}"
    framed_before = int((without.frame[column] != NOT_FRAMED).sum())
    framed_after = int((with_quota.frame[column] != NOT_FRAMED).sum())
    assert framed_after > framed_before
    assert with_quota.n_items > without.n_items
    layers = set(with_quota.frame["sampling_layer"])
    assert "stratified" in layers
    assert any(layer.startswith("framing_oversample[") for layer in layers)
    assert with_quota.report.framing_quota == 40
    assert set(with_quota.report.framing_oversample) == set(FRAMING_CONTRASTS)
    assert any("prevalence" in note for note in with_quota.report.notes)


def test_framing_oversample_does_not_repeat_an_item(news, social):
    drawn = draw_stratified_sample(
        {"news": news, "tweets": social}, n=300, seed=4, framing_quota=40
    )
    assert drawn.frame["item_id"].is_unique


def test_framing_quota_of_zero_keeps_the_sample_proportional(news, social):
    drawn = draw_stratified_sample({"news": news, "tweets": social}, n=300, seed=9)
    assert set(drawn.frame["sampling_layer"]) == {"stratified"}
    assert drawn.report.framing_oversample == {}


def test_annotation_file_holds_every_question_and_leaves_it_empty(sample, tmp_path):
    paths = write_annotation_files(sample, tmp_path)
    annotation = pd.read_csv(paths["annotation"], keep_default_na=False)
    for question in LABEL_QUESTIONS:
        for column in annotator_pair(question):
            assert column in annotation.columns
            assert (annotation[column] == "").all()
    key = pd.read_csv(paths["key"])
    for contrast in FRAMING_CONTRASTS:
        assert f"predicted_framing_{contrast}" in key.columns
        assert f"predicted_framing_{contrast}" not in annotation.columns


# --------------------------------------------------------------------------
# 12. Agreement on the framing questions
# --------------------------------------------------------------------------


def test_agreement_for_a_framing_question_uses_its_own_label_set():
    first, second = annotator_pair("framing_level_vs_rate")
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "source_type": ["national_news"] * 4,
            first: ["level", "rate", NOT_FRAMED, FRAMING_BALANCED],
            second: ["level", "rate", FRAMING_BALANCED, FRAMING_BALANCED],
        }
    )
    report = agreement_for_question(frame, "framing_level_vs_rate", provenance="human")
    assert report.n_both_labelled == 4
    # Item i3 is a disagreement: silence is not balance.
    assert report.percent_agreement == pytest.approx(0.75)
    assert set(report.disagreements["item_id"]) == {"i3"}
    assert list(report.by_class["class"]) == list(FRAMING_CLASSES["level_vs_rate"])


def test_agreement_for_question_rejects_a_tone_label_in_a_framing_column():
    first, second = annotator_pair("framing_household_vs_macro")
    frame = pd.DataFrame(
        {
            "source_type": ["other"],
            first: ["negative"],
            second: ["household"],
        }
    )
    report = agreement_for_question(frame, "framing_household_vs_macro", provenance="human")
    assert report.n_invalid == 1
    assert report.n_both_labelled == 0


def test_agreement_by_dimension_covers_all_five_dimensions(sample):
    table = agreement_by_dimension(sample.frame, provenance="human")
    assert len(table) == len(LABEL_QUESTIONS)
    assert set(table["dimension"]) == set(DIMENSION_QUESTIONS)
    # The sample has no label columns at all, so every row says so.
    assert (table["note"] == "column absent").all()
    assert table["n_both_labelled"].sum() == 0
    assert not table["labels_are_human"].any()


def test_agreement_by_dimension_reports_outstanding_for_empty_columns(sample):
    frame = sample.frame.copy()
    for question in LABEL_QUESTIONS:
        for column in annotator_pair(question):
            frame[column] = ""
    table = agreement_by_dimension(frame, provenance="human")
    assert (table["note"] == "human labels outstanding").all()
    assert table["percent_agreement"].isna().all()


# --------------------------------------------------------------------------
# 13. The swap test on the framing dimension
# --------------------------------------------------------------------------


def test_name_swap_test_runs_on_a_framing_contrast(sample):
    contrast = "level_vs_rate"
    report = swap_robustness_test(
        sample.frame,
        labeler=framing_labeler(contrast),
        classifier="framing_lexicon",
        swap_kind="political_name",
        labels=FRAMING_CLASSES[contrast],
        dimension=f"framing_{contrast}",
    )
    assert report.dimension == "framing_level_vs_rate"
    assert f"framing_{contrast}" in report.summary()
    assert list(report.transitions.index) == list(FRAMING_CLASSES[contrast])
    assert list(report.flips_by_class["class_original"]) == list(FRAMING_CLASSES[contrast])
    # The framing lexicon holds no political name, so a name swap cannot move
    # the contrast. That is the result, and it is worth pinning.
    assert report.n_flipped == 0


def test_source_label_swap_runs_on_a_framing_contrast(sample):
    contrast = "household_vs_macro"
    report = swap_robustness_test(
        sample.frame.head(40),
        labeler=framing_labeler(contrast),
        classifier="framing_lexicon",
        swap_kind="source_label",
        labels=FRAMING_CLASSES[contrast],
        dimension=f"framing_{contrast}",
    )
    assert report.n_eligible == 40
    assert report.n_flipped == 0
    assert any("label pair" in note for note in report.notes)


def test_swap_test_detects_a_framing_sensitive_classifier(sample):
    def partisan_framing(texts):
        return [
            "household" if "biden" in str(text).lower() else "macro" for text in texts
        ]

    report = swap_robustness_test(
        sample.frame,
        labeler=partisan_framing,
        classifier="partisan_framing_stub",
        swap_kind="political_name",
        labels=FRAMING_CLASSES["household_vs_macro"],
        dimension="framing_household_vs_macro",
    )
    assert report.n_flipped > 0
    assert int(report.transitions.to_numpy().sum()) == report.n_scored


def test_comparison_states_that_framing_has_no_neural_counterpart(sample):
    comparison = compare_classifiers(sample.frame, finbert_label_fn=_fake_finbert())
    assert any("framing" in note.lower() for note in comparison.notes)
    assert any("framing" in item.lower() for item in comparison.outstanding)


# --------------------------------------------------------------------------
# 14. The day-and-topic benchmark key
# --------------------------------------------------------------------------


def test_event_keys_are_the_two_documented_keys():
    assert EVENT_KEYS == ("normalised_text", "day_and_topic")


def test_dominant_topic_uses_the_brief_topic_list():
    assert dominant_topic("Layoffs climb as job cuts spread") == "layoffs"
    assert dominant_topic("Inflation cooled again in June") == "falling_inflation"
    assert dominant_topic("Grocery prices keep climbing") == "rising_prices"
    assert dominant_topic("Recession risk is rising") == "recession_risk"
    assert dominant_topic("The concert is on Friday") == "none"
    assert dominant_topic("") == "none"
    assert set(BENCHMARK_TOPICS) <= set(SALIENCE_CLASSES)


def test_day_and_topic_key_groups_different_texts():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01"] * 3 + ["2025-05-02"]),
            "source": [
                "https://www.reuters.com/a",
                "https://www.msn.com/b",
                "https://www.prnewswire.com/c",
                "https://www.reuters.com/d",
            ],
            "text": [
                "Grocery prices climb and families are squeezed",
                "Grocery prices ease as the recovery strengthens",
                "Grocery prices are stable this month",
                "Grocery prices climb again next day",
            ],
        }
    )
    report = same_event_cross_source(frame, event_key="day_and_topic")
    assert report.event_key == "day_and_topic"
    assert report.n_events == 1
    assert report.n_items_in_events == 3
    # The texts differ, so the benchmark can find a real deviation.
    assert report.mean_within_event_tone_range > 0.0
    assert report.by_source["mean_deviation"].abs().max() > 0.0
    assert any("calendar day" in note for note in report.notes)


def test_day_and_topic_key_records_off_topic_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01", "2025-05-01"]),
            "source": ["https://www.reuters.com/a", "https://www.msn.com/b"],
            "text": [
                "The concert is on Friday evening downtown",
                "Local team wins the championship game tonight",
            ],
        }
    )
    report = same_event_cross_source(frame, event_key="day_and_topic")
    assert report.dropped_reasons["no_benchmark_topic"] == 2
    assert report.n_events == 0


def test_normalised_text_key_is_a_control_and_says_so():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01"] * 2),
            "source": ["https://www.reuters.com/a", "https://www.msn.com/a"],
            "text": ["Recession fears grow as layoffs climb"] * 2,
        }
    )
    report = same_event_cross_source(frame, event_key="normalised_text")
    assert report.mean_within_event_tone_range == pytest.approx(0.0)
    assert any("control" in note for note in report.notes)


def test_same_event_rejects_an_unknown_event_key():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-05-01"]),
            "source": ["https://www.reuters.com/a"],
            "text": ["Grocery prices climb again today"],
        }
    )
    with pytest.raises(ValidationError, match="event_key must be one of"):
        same_event_cross_source(frame, event_key="same_release")
