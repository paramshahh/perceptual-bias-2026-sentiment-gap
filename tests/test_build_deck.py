"""Tests for the presentation.

The brief asks for "a 6-8-minute presentation explaining one practical
implication for financial forecasting or risk management". Two of those words
are testable and are easy to break:

* **6 to 8 minutes.** A deck grows while nobody measures it. The real length of
  a talk is the speaker notes, not the slide count, so the notes are counted and
  held inside the window.
* **one practical implication.** A deck that drifts back into a project summary
  fails the brief even when every number in it is correct. So the argument's
  own vocabulary is asserted.

Nothing here builds the deck twice: the module-scope fixture builds it once.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pptx = pytest.importorskip("pptx")

import build_deck as bd  # noqa: E402

WORDS_PER_MINUTE = 130
MIN_MINUTES = 6.0
MAX_MINUTES = 8.0

CSV_PRESENT = bd.CSV.exists()


@pytest.fixture(scope="module")
def deck() -> pptx.presentation.Presentation:
    if not CSV_PRESENT:
        pytest.skip("the issued CSV is not present")
    path = bd.build()
    return pptx.Presentation(path)


@pytest.fixture(scope="module")
def notes(deck) -> list[str]:
    return [
        slide.notes_slide.notes_text_frame.text
        for slide in deck.slides
        if slide.has_notes_slide
    ]


def slide_text(slide) -> str:
    parts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
    return "\n".join(parts)


class TestTheTimeBudget:
    def test_the_talk_fits_the_six_to_eight_minute_window(self, notes) -> None:
        words = sum(len(text.split()) for text in notes)
        minutes = words / WORDS_PER_MINUTE
        assert MIN_MINUTES <= minutes <= MAX_MINUTES, (
            f"{words} words of notes is about {minutes:.1f} minutes; the brief "
            f"asks for {MIN_MINUTES:.0f} to {MAX_MINUTES:.0f}"
        )

    def test_the_slide_count_suits_the_window(self, deck) -> None:
        """Seven slides: a title, the five questions, and a close."""
        assert 6 <= len(deck.slides._sldIdLst) <= 8

    def test_every_slide_carries_speaker_notes(self, deck) -> None:
        for index, slide in enumerate(deck.slides, start=1):
            assert slide.has_notes_slide, f"slide {index} has no notes"
            assert slide.notes_slide.notes_text_frame.text.strip(), index

    def test_no_single_slide_runs_past_about_ninety_seconds(self, notes) -> None:
        for index, text in enumerate(notes, start=1):
            minutes = len(text.split()) / WORDS_PER_MINUTE
            assert minutes <= 1.6, f"slide {index} runs {minutes:.1f} minutes"


class TestTheArgument:
    def test_the_title_frames_the_talk_and_names_its_audience(self, deck) -> None:
        """Slide 1 must say what the talk is for, not just what it is about.

        The deck is five questions and a practical implication, so the title
        slide names the implication's audience. The decision rule itself is
        asserted across the deck by the next test.
        """
        first = slide_text(list(deck.slides)[0]).lower()
        assert "five questions" in first
        assert "forecasting" in first or "risk management" in first
        assert "sentiment gap" in first

    def test_the_deck_names_the_decision_rule(self, deck) -> None:
        """The implication the brief asks for must be on a slide, in words."""
        text = " ".join(slide_text(s) for s in deck.slides).lower()
        for phrase in ("one-sided", "prediction interval", "rebuild", "recalibrat"):
            assert phrase in text, phrase

    def test_the_five_questions_are_each_on_their_own_slide(self, deck) -> None:
        """The deck's shape is the five questions, so the shape is asserted."""
        headings = [slide_text(s) for s in deck.slides]
        numbered = [h for h in headings if h.strip().startswith("Q")]
        assert len(numbered) == 5
        for index, heading in enumerate(numbered, start=1):
            assert heading.strip().startswith(f"Q{index}"), heading[:40]
            assert "?" in heading, heading[:60]

    def test_each_question_slide_answers_itself_in_one_line(self, deck) -> None:
        """A listener who reads only two lines still leaves with the answer."""
        for slide in deck.slides:
            text = slide_text(slide)
            if not text.strip().startswith("Q"):
                continue
            lines = [line for line in text.split("\n") if line.strip()]
            assert len(lines) >= 3, text[:80]

    def test_the_charts_are_on_the_slides(self, deck) -> None:
        """Two charts were asked for, so their absence must fail the build."""
        pictures = sum(
            1
            for slide in deck.slides
            for shape in slide.shapes
            if shape.shape_type == 13  # MSO_SHAPE_TYPE.PICTURE
        )
        assert pictures >= 2, f"expected two charts, found {pictures}"

    def test_the_three_fixes_are_distinguished(self, deck) -> None:
        """The implication is useless without the diagnosis that picks the fix."""
        text = " ".join(slide_text(s) for s in deck.slides).lower()
        for phrase in ("level variables", "re-base", "text feed"):
            assert phrase in text, phrase

    def test_the_limits_are_on_a_slide_and_not_only_in_the_notes(self, deck) -> None:
        text = " ".join(slide_text(s) for s in deck.slides).lower()
        assert "exposure" in text
        assert "no causal claim" in text

    def test_the_cutoff_is_printed(self, deck) -> None:
        text = " ".join(slide_text(s) for s in deck.slides)
        assert "2026-10-16" in text
        assert "post-election" in text.lower()


class TestHonestDegradation:
    def test_a_p_value_that_does_not_exist_is_not_invented(self) -> None:
        assert bd._p(None) == "not available"
        assert bd._p(float("nan")) == "not available"
        assert bd._p("rubbish") == "not available"

    def test_a_small_p_value_is_not_printed_as_zero(self) -> None:
        assert bd._p(1e-9) == "p < 0.001"
        assert bd._p(0.0304) == "p = 0.030"

    @pytest.mark.skipif(not CSV_PRESENT, reason="the issued CSV is not present")
    def test_the_deck_states_no_number_it_could_not_read(self, deck) -> None:
        """A failed module must produce a sentence, never a NaN on a slide."""
        text = " ".join(slide_text(s) for s in deck.slides)
        for forbidden in ("nan", "NaN", "None", "inf"):
            assert forbidden not in text.split(), forbidden


@pytest.mark.skipif(not CSV_PRESENT, reason="the issued CSV is not present")
class TestTheNumbersAreLive:
    def test_the_gap_numbers_come_from_the_live_fit(self) -> None:
        data = bd.gather()
        assert data["tested"] >= 19
        assert data["outside"] <= data["tested"]
        assert data["worst_gap"] < -20
        assert 0.6 < data["r2"] < 0.9

    def test_no_superseded_number_appears_on_any_slide(self, deck) -> None:
        """These are the contemporaneous-panel numbers. They are void."""
        text = " ".join(slide_text(s) for s in deck.slides)
        for stale in ("0.766", "-25.24", "25.24", "33.13", "77.93", "-0.159"):
            assert stale not in text, stale
