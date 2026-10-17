# The 2026 Sentiment Gap

**AI, Economic Perceptions and Consumer Sentiment Before the 2026 US Midterm
Elections.** A course project for AI Applications in Finance.

This repository documents the whole project in ASD-STE100 Simplified Technical
English.

## The question

US consumer sentiment sits near record lows while unemployment and inflation
look ordinary. Either the indicators miss something real, or people see a
distorted picture, or the measurement changed. The project tells those three
apart.

## The result

A model fitted on 2010–2024 explains **76%** of consumer sentiment from
unemployment, inflation and real GDP growth. Asked about months it has never
seen, it misses by a mean of **−25.3 index points**, and **19 of 20 months fall
outside its own 95% prediction interval, every one of them downward**. A correct
interval is breached about one month in twenty.

| Explanation | Verdict |
|---|---|
| **A.** Economic pressure the indicators miss | **Supported in part.** Adding the cumulative price level and the mortgage rate absorbs **27.6%** of the gap and cuts the one-step forecast error from 15.58 to 6.98. The mortgage rate does the measurable work; the price level is not separately identified. |
| **C.** Measurement, timing, model | **Supported in part, smaller.** A real break at the survey's move to web collection, but an independent survey is gloomy too. |
| **B.** Media selection, framing, negativity | **Fails,** on all four conditions fixed before the corpus existed. |

Hypothesis B was the attractive story. Of **12 declared tests**, four reject
before correction and **none survives** Holm or Benjamini-Hochberg; the media
block does not improve an out-of-sample forecast; **no model beats a free
last-observed-value guess**; both associations die in all 24 cells of a
specification grid once household and political controls enter; and the only
precedence that reaches 5% runs **backwards** — the mood moves first and the
coverage follows.

The assignment states that a rigorous finding that media bias does *not* explain
the gap can earn full credit. This is that finding.

## The finding that outranks the three hypotheses

**FinBERT changes its tone verdict on 5.8% of byte-identical texts when only the
outlet's name changes.** A plain word list changes on 0%, and so does the
framing lexicon on all three contrasts. For a project that measures political
media bias, a classifier that reads the outlet label would have manufactured the
bias the project went looking for.

## The practical implication

A one-sided breach of a prediction interval is a **specification failure**, not a
run of bad luck. Rebuild the model rather than recalibrating it: a re-fit on the
new level will track the new level and keep exactly the blindness that caused the
breach. Then diagnose before fixing, because the three explanations need three
different fixes.

## What is here

| Path | What it is |
|---|---|
| `perceptual_bias_report.pdf` | The 12-page report |
| `perceptual_bias_deck.pptx` | A 7-slide, 6–8 minute presentation with speaker notes |
| `app.py` | The monitoring dashboard, 13 panels (`streamlit run app.py`) |
| `BRAINSTORMING.md` | The reasoning record: every measurement decision and why |
| `PROGRESS.md` | Status; **paragraphs 1–10 are superseded, see the banner** |
| `docs/` | The brief, corpus documentation, labelling guide, ethics and limitations, method notes, update plan, provenance register |
| `src/perceptual_bias/` | The modules: data, baseline, NLP and SVD, econometrics, corpus validation, survey diagnostics, tables |
| `outputs/labels/` | The validation artifacts, including the **empty** labelling instrument |
| `tests/` | 671 tests |

## Reproducing it

```bash
uv sync                                   # or: pip install -e .
cp .env.example .env                      # add a free FRED API key
.venv/bin/python -m pytest -q             # 671 tests
.venv/bin/python build_report.py          # the PDF
.venv/bin/python build_deck.py            # the deck
.venv/bin/streamlit run app.py            # the dashboard
```

The processed media series are included, so the report and the dashboard rebuild
without a network pull. The raw corpora are not: 198,367 GDELT GKG records and
36,000 article titles stay out of the repository. `scripts/pull_gkg.py` and
`scripts/build_media_series.py` rebuild them, and `docs/update_plan.md` gives the
order.

## What this project does not claim

It does not claim that pessimism is irrational, that negative coverage is biased
coverage, or that media exposure causes declining sentiment. **There is no
exposure data at all**, so no causal claim about readers is available, and the
causal design is observational with a written proposal for stronger
identification.

There is also **no inter-annotator agreement statistic**, because the project has
one operator and no second annotator. The 359-item instrument is built,
stratified and blind, and its 16 annotator columns are deliberately empty. The
code refuses to mark model labels as human.

Individual social-media accounts in the published artifacts are pseudonymised;
see `docs/ethics_and_limitations.md`.

## Honesty notes

`docs/ethics_and_limitations.md` lists the eight errors this project made in its
own work, with before-and-after numbers for the two that changed a conclusion.
The largest: obeying the brief's real-time timing rule turned a significant GDP
coefficient (−0.159, p = 0.039) into −0.030, p = 0.524. The finding was an
artefact of look-ahead.

---

Built with [Claude Code](https://claude.com/claude-code).
