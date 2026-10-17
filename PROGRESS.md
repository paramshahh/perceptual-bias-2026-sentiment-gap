# PROGRESS — perceptual-bias

This document gives the status of the project. The team updates this document
when a stage is complete. A new reader can start work from this document.

This document uses ASD-STE100 Simplified Technical English.

Date of the last change: 2026-10-17

---

> ## ⚠ READ THIS FIRST — much of this document is superseded
>
> Paragraphs 1 to 10 were written between 2026-10-14 and 2026-10-15, against a
> **reconstruction** of the assignment and a panel that broke two of the real
> brief's timing rules. Their numbers are **void**. They are kept because the
> size of the correction is itself evidence, and because a record that quietly
> deletes its own history is not a record.
>
> **The live numbers are in paragraph 12. The final results of all three
> hypotheses are in paragraph 13.** If a number appears in both places, the
> later paragraph is the true one. Do not copy a coefficient, an R², a p-value
> or a gap figure out of paragraphs 1 to 10.
>
> `BRAINSTORMING.md` is version 2 and is current throughout.

## 1. Status of the deliverable items

| No. | Item | Status |
|---|---|---|
| 1 | `BRAINSTORMING.md` | Complete. The numbers are old. Refresh them. |
| 2 | `src/perceptual_bias/data_loader.py` | Complete. The CSV data and the FRED data are available. |
| 3 | `src/perceptual_bias/baseline_model.py` | Complete. |
| — | `src/perceptual_bias/twitter_source.py` | Complete. The module has a credit limit. |
| 4 | `src/perceptual_bias/nlp_media_svd.py` | Complete. |
| 5 | `src/perceptual_bias/causal_econometrics.py` | An agent writes this module now. |
| — | `src/perceptual_bias/survey_diagnostics.py` | An agent writes this module now. |
| 6 | `src/perceptual_bias/table_generator.py` | An agent writes this module now. |
| 7 | `app.py` | Not started. It needs items 4, 5 and 6. |
| 8 | `build_report.py` | Not started. It needs items 4, 5 and 6. |
| — | `AI_USE.md` | Not started. |
| — | The presentation | Not started. Do this task last. |

The repository has no commits. The files are in the staging area. There is no
remote repository.

---

## 2. Results

You can get these results again. Use the command `pytest`. Refer to the folder
`tests/`.

### 2.1 Model 1. The baseline macro model

The model uses OLS. The data is from 2010-01 to 2024-12. The sample has 180
months. The model uses HAC standard errors with 12 lags.

```
sentiment = 127.56 - 4.28*unemployment - 9.77*pce_yoy - 0.16*real_gdp_qoq
R2=0.766  adjR2=0.762  RMSE=6.20  AIC=1175.5
```

All three regressors are significant. The GDP coefficient has a p-value of
0.039. But the sign of the GDP coefficient is negative. A negative sign is not
correct for this variable. This is a warning about the specification. It is not
a result.

### 2.2 The Sentiment Gap

The gap uses data from 2025-01. The model does not see this data.

| Measurement | Value |
|---|---|
| Months with a gap value | 19 of 22 |
| Mean gap | −25.24 |
| Maximum gap | −33.13 at 2026-05 |
| Observed value at 2026-05 | 44.8 |
| Predicted value at 2026-05 | 77.93 |
| Months out of the 95% prediction interval | 18 of 19 |

These values agree with the values in the project brief. The brief gives 77.8
and −33.0.

The system removes three months. The data for these months is not complete:

- 2025-10. The unemployment value is not available.
- 2026-09. The PCE YoY value is not available.
- 2026-10. The unemployment value, the PCE YoY value and the GDP value are not
  available.

The class `PredictionReport` gives the names of the months that the system
removes. The system does not remove a month without a report.

### 2.3 Hypothesis A. Economic pressure that the baseline does not measure

The cumulative PCE price level is 25.7% more than the mean level of 2020. This
value comes from the FRED series `PCEPI`.

| Model | R² | AIC | Mean gap | Gap at 2026-05 |
|---|---|---|---|---|
| Model 1 | 0.766 | 1175.5 | −25.24 | −33.13 |
| Model 1 and the price level | 0.793 | 1155.5 | −17.90 | −25.17 |
| Model 1, the price level and the mortgage rate | 0.835 | 1117.4 | −17.40 | −25.58 |
| Model 2 (all variables) | 0.837 | 1116.2 | −15.21 | −22.88 |

The coefficients of Model 2 use HAC standard errors:

| Variable | Coefficient | p-value |
|---|---|---|
| `pce_level` | −0.360 | 0.038 |
| `mortgage_30y` | −1.835 | 0.030 |
| `debt_service` | −1.604 | 0.256 |

Hypothesis A explains 40% of the gap. A gap of −15.21 stays. Hypothesis B and
Hypothesis C must explain this remainder.

The variable `debt_service` is not significant. The FRED series `TDSP` stops at
2026-04. Thus this variable also removes two months from the out-of-sample
data. Make a decision about this variable before the report.

---

## 3. The data sources

| Source | Status | Remarks |
|---|---|---|
| The supplied CSV file | The primary source | The file is in `data/raw/`. The size is 30 kB. The repository keeps this file. |
| FRED | Available | The series are `PCEPI`, `MORTGAGE30US` and `TDSP`. The key is in `.env`. Git ignores `.env`. |
| GDELT | Not used yet | GDELT is free. GDELT does not need a key. GDELT is the primary media source. |
| twitterapi.io | Limited | Refer to paragraph 3.1. |

### 3.1 The twitterapi.io credit record

One request costs 300 credits. One page gives 20 tweets. The team measured
these values. The team did not estimate them.

| Event | Credits | Result |
|---|---|---|
| The probe request | 300 | Correct. The probe showed the structure of the data. |
| The pull of two months (14 requests) | 4200 | Not usable. Refer to the failure below. |
| The diagnostic request with `Top` | 300 | Correct. This request showed the solution. |
| Total credits used | 4800 | The initial quantity was 21 000. |
| Credits available | 16 200 | This is near to 54 requests. |

**The failure.** The parameter `queryType: Latest` with a window of one month
gives the newest tweets at the end of the window. Thus 140 tweets covered only
14 minutes. The tweets also have a date after the end of the requested month.
The quality filters are not the cause. The filters accepted 187 tweets of 280.

**The solution.** Use the parameter `queryType: Top`. Use a window of one day.
This method gives 16 tweets of 20 in the correct day. All 20 tweets obey the
quality filters. The text is about the economy.

The system keeps each page in a cache. The system does not request a page two
times. Thus a second run of the pipeline costs no credits.

---

## 4. Decisions

| Subject | Decision |
|---|---|
| The primary data | The supplied CSV file. FRED gives the additional series for Hypothesis A. |
| The media data | GDELT is the primary source. twitterapi.io is a small source with a limit. |
| Quarterly data to monthly data | Put the value of a quarter in the three months of the same quarter. |
| The estimation window | Use a long window for the headline result. Show the 2025–26 data separately. |
| Negative results | Report negative results as results. |
| FinBERT | Run FinBERT on a test set. Show the failures. |
| The presentation | Make the presentation last. |

---

## 5. Rules for new modules

Obey these rules.

1. Make a module that does a complete task. A public function must return an
   object. The object must contain the results and also a record of the data
   that the function removed or approximated. Refer to `PanelReport`,
   `PredictionReport` and `BaselineFit`.
2. Do not remove a row without a record. Give the name of each row that you
   remove. Give the cause.
3. A value that is not available is not a negative result. Do not count a
   missing value as a correct value. The field `outside_pi` uses a nullable
   boolean for this reason.
4. Do not use future data. Calculate each transform on the training data only.
   For cross-validation, calculate the transform in each fold.
5. Use HAC standard errors for each time-series regression. Use 12 lags for
   monthly data.
6. Make an exception class for each module. Raise your exception before pandas
   raises a `KeyError`.
7. Write a test for each defect that a review finds. Do not test only the
   correct conditions.
8. In a docstring, give the cause of a decision. This is important when a
   different decision is possible.
9. Write all documents in ASD-STE100 Simplified Technical English.

To run the tests, use this command:

```
.venv/bin/python -m pytest -q
```

---

## 6. Known problems

- The numbers in `BRAINSTORMING.md` are old. The document gives 24% and a
  p-value of 0.161 for Hypothesis A. The correct values are 40% and 0.038.
  Correct this document before you write the report.
- The file `04b-SVD-Appl.pdf` is not on the disk. The SVD mathematics in
  `BRAINSTORMING.md` comes from first principles.
- The variable `debt_service` is not significant. It also removes two months of
  out-of-sample data.
- The repository has no commits and no remote repository.


---

## 7. Handoff — continue from here tonight

### 7.1 What runs now

A background job pulls the GDELT headline corpus:

- Script: `scripts/pull_gdelt.py`, started by `scripts/wait_and_pull.sh`.
- Target: 232 days, two days in each month, 2017-01 to 2026-08.
- Log: `/tmp/gdelt_wait.log`. Cache: `data/raw/gdelt/*.csv`, one file for each day.
- The job caches each day. If it stops, start it again and it continues.

To see the progress:

```
ls data/raw/gdelt/*.csv | wc -l        # of 232
tail -5 /tmp/gdelt_wait.log
```

To start it again:

```
nohup sh scripts/wait_and_pull.sh > /tmp/gdelt_wait.log 2>&1 &
```

### 7.2 The GDELT rules

GDELT permits one request each five seconds. It answers a faster client with
HTTP 429. The library `gdeltdoc` converts that answer into an exception with an
empty message, so a rate limit looks the same as a bad query. The request
pattern follows Ken Blake, "GDELT Headline Fetch" (rpubs.com/drkblake/1007551):
`mode=artlist`, `maxrecords=250`, `sort=datedesc`, `format=CSV`, and the start
and the end on the same calendar day. The archive starts in January 2017.

### 7.3 The next tasks, in order

1. When the corpus is complete, build the monthly media series. Use
   `nlp_media_svd.py`: TF-IDF, then the SVD, then the tone index. Give the
   columns the names `media_tone`, `topic_salience` and `political_attribution`.
2. Join these columns to the panel. Then `compare_models` gives Model 3.
3. Run `granger_media_gap` and `rigobon_sack_iv` on the long window.
4. Calculate the Hypothesis C numbers again through a second path. Refer to
   paragraph 7.4.
5. Make the four tables with `table_generator.py`.
6. Write `app.py`, then `build_report.py`, then `AI_USE.md`.
7. Make the presentation last.

### 7.4 Open problems

- ~~Verify the Hypothesis C numbers.~~ **DONE 2026-10-15.** An independent
  path with `scipy` gives: Chow F = 83.7 against the module's 81.3 (a
  difference of 3%, with the same conclusion); the level shift is −35.27
  against −35.50; the supremum-Wald break is 1990-08 against 1990-07 (a
  convention for the break date); the cross-survey ratio is **0.258** against
  0.26. The position of the 2024-07 transition at 95.7% of the sample is
  confirmed.

  **A new caution from this check.** The verdict for Hypothesis C rests on
  0.008. The ratio is 0.258 and the limit for "supported" is 0.25. A different
  comparison series, or a different scaling rule, changes the verdict from
  "partially supported" to "supported". The report must give the ratio and the
  limit together. It must not give the word alone.
- **The price level is not separately identified.** Its p-value moves from
  0.038 to 0.310 with a change of specification. Do not give it a causal
  reading.
- **The GDP coefficient is negative and significant.** Give this as a warning
  about the specification.
- **The gap has 19 months.** The modules refuse inference at this size. Use the
  long window for the media models.

### 7.5 The budget

| Item | Used | Available |
|---|---|---|
| twitterapi.io credits | 16 800 | 4 200 |
| FRED | no limit | — |
| GDELT | no money | one request each 5 s |

The tweet corpus is complete: 365 tweets, 20 months, in
`data/raw/tweets/tweets_daily.parquet`.


---

## 8. Cleanup of the data after the push

The raw corpus is large and Git ignores it. Do these steps in this order.

### 8.1 First commit the derived data

Git must hold the small files that the analysis needs, or the work is not
reproducible without a download of several gigabytes:

- the monthly media series (tone, salience, attribution) as CSV or parquet;
- the SVD term loadings for Table 2;
- the tweet corpus `data/raw/tweets/tweets_daily.parquet` (365 rows, small);
- the issued CSV file, which Git already holds.

Add an exception to `.gitignore` for each of these paths. Then push.

### 8.2 Then remove the raw corpus

These paths are safe to remove, because the scripts can get them again:

```
data/raw/gkg/*.parquet          # approximately 2 to 3 GB of cached GKG files
data/raw/gdelt/*.csv            # cached headline files
```

Do not remove:

```
data/raw/Copy_of_Perceptual_Bias_Data.csv    # the issued data
data/raw/tweets/                             # 16 800 credits paid for this
data/raw/gkg/gkg_economic.parquet            # the combined corpus, if it is small
```

**The tweet corpus is not replaceable.** The twitterapi.io balance is 4 200
credits. A new collection of 365 tweets costs approximately 6 000 credits.
Keep this file and commit it.

### 8.3 How to get the corpus again

```
.venv/bin/python scripts/pull_gkg.py 2017-01 2026-08
```

The script caches each file. It takes approximately 15 minutes.


---

## 9. Hypothesis B — the result

The corpus is complete: **198 367 economic articles over 116 months**, from
2017-01 to 2026-08, from the GDELT GKG bulk files.

**Hypothesis B fails each of its own tests.**

| Test | Result | Decision |
|---|---|---|
| `media_tone` in Model 3 | coefficient +6.79, p = 0.570 | Not significant |
| `topic_salience` in Model 3 | coefficient −11.56, p = 0.679 | Not significant |
| Granger, tone to gap | F = 0.441, p = 0.779 | No precedence |
| Granger, gap to tone | F = 0.501, p = 0.735 | No precedence |
| Rigobon-Sack IV | Not identified. No variance change. | No estimate |

The falsification condition of paragraph 2.2 says: if the tone index is not
significant in Model 3, Hypothesis B fails. The condition is met.

**One result stays.** `political_attribution` is significant: the coefficient
is −34.30 with p = 0.030. In a month when economic coverage carries a political
frame, the sentiment is lower. But this variable also fails the Granger test in
both directions. Thus it is an association. It is not predictive precedence,
and it is not an identified cause.

**The model comparison on the window 2017 to 2026:**

| Model | R² | adjusted R² | AIC |
|---|---|---|---|
| Model 1 | 0.518 | 0.505 | 899.3 |
| Model 2 | 0.904 | 0.900 | 717.2 |
| Model 3 | 0.912 | 0.905 | 713.9 |

Model 3 adds 0.008 of R². The economic variables do the work.

**A defect in the first corpus, and its correction.** The first filter used the
prefix `TAX_` as an economic theme. In GDELT, `TAX_` is a general taxonomy of
actors, ethnicities and languages, not taxation. That filter accepted 693 347
articles with a mean tone of −0.988. The strict filter accepts 198 367 articles
with a mean tone of −0.180. The first corpus measured the tone of the news in
general. The second measures the tone of economic news. **The error would have
given a strong false result for Hypothesis B.**

### 9.1 What this means for the report

The project now has a clear answer, and it is a negative answer. The tone of
economic coverage does not explain the Sentiment Gap. The economic variables of
Hypothesis A explain most of what any model explains. Hypothesis C explains a
small part. A large part of the gap stays unexplained.

The report must give this result as a result. It must not reduce it to a
footnote.


---

## 10. The project is complete

Date: 2026-10-16.

| No. | Artifact | State |
|---|---|---|
| 1 | `BRAINSTORMING.md` | Complete. The numbers are current. |
| 2 | `src/perceptual_bias/data_loader.py` | Complete. |
| 3 | `src/perceptual_bias/baseline_model.py` | Complete. |
| 4 | `src/perceptual_bias/nlp_media_svd.py` | Complete. |
| 5 | `src/perceptual_bias/causal_econometrics.py` | Complete. |
| 6 | `src/perceptual_bias/table_generator.py` | Complete. |
| 7 | `app.py` | Complete. Six panels, all with data. |
| 8 | `build_report.py` | Complete. The PDF has 12 pages and four tables. |
| — | `notebooks/perceptual_bias_analysis.ipynb` | Complete. 40 cells, no error. |
| — | `AI_USE.md` | Complete. |
| — | `build_deck.py` | Complete. 14 slides. |

Tests: 381 pass.

### 10.1 The result

| Hypothesis | Verdict | Evidence |
|---|---|---|
| A. Economic pressure | Supported in part | 31% of the gap. The mortgage rate is significant. The price level is not separately identified. |
| B. Media tone | **Fails** | Tone p = 0.570. Salience p = 0.679. No Granger precedence. The IV is not identified. |
| C. The survey method | Supported in part | The cross-survey ratio is 0.258 against a threshold of 0.25. |

One result stays inside Hypothesis B: the political attribution has a
coefficient of −34.30 with a p-value of 0.030. It fails the Granger test, so it
is an association.

A large part of the gap stays unexplained. The report gives this.

### 10.2 The corpora

| Corpus | Size | Use |
|---|---|---|
| GKG themes and tone | 198 367 articles, 116 months | The monthly tone, the salience, the attribution |
| Headlines | 36 000, of which 5131 are economic, 95 months | The SVD terms of Table 2 |
| Tweets | 365, 20 months | A comparison. **Not replaceable: it cost 16 800 credits.** |

### 10.3 What a next session can still do

1. Re-run the SVD over the full headline corpus when more days are cached.
2. Test the political attribution with an instrument. It is the one media
   variable that holds a signal.
3. Find a consumer survey that is independent of Michigan and covers 2025 and
   2026. No free series does this now.


---

## 11. The real assignment arrived (2026-10-16). The project is NOT complete.

Until this date the work followed a pasted reconstruction of the brief. The
true brief is now in the repository:

- `docs/Final_Project_Applications_in_Finance_Oct_2026.pdf`
- `docs/assignment_brief.txt` (the text, for grep)

**The two documents differ in important ways. Read the PDF, not the memory of
the pasted text.**

### 11.1 The grading weights

| Criterion | Weight |
|---|---|
| Economic measurement, timing, and alternative explanations | 20% |
| Corpus design and validated AI measurement | 25% |
| Statistical analysis and causal reasoning | 30% |
| Reproducibility and dashboard quality | 15% |
| Communication, limitations, and research ethics | 10% |

The brief also says: "A rigorous finding that media bias does not explain the
sentiment gap can earn full credit. The quality of the evidence and reasoning
matters more than the direction of the conclusion." **Our negative result for
Hypothesis B is therefore an asset and not a problem.**

### 11.2 Two direct violations in the work before this date

1. **"Do not interpolate quarterly GDP into artificial monthly observations."**
   The function `expand_quarterly` does this.
2. **"Use only information available by each prediction date."** The panel uses
   a contemporaneous alignment, which is look-ahead. The brief wants real-time
   vintages with publication lags.

### 11.3 What the brief requires and the work did not have

| Weight | Requirement | State on 2026-10-16 |
|---|---|---|
| 25% | Human-label 300+ items, two independent annotators, agreement and adjudication | Not started |
| 25% | Compare the AI with a dictionary classifier on that sample | Partial: a probe of 30 sentences |
| 25% | Test whether a change of political name or source label changes a classification | Not started |
| 25% | Corpus documentation: terms, dates, duplicates, bots, missingness, exposure | Not started |
| 20% | Separate the headline index from Current Conditions and Expectations | Not done |
| 30% | Model 3 must add household or political controls | Model 3 holds media only |
| 30% | Multiple-testing correction | Not done |
| 30% | Rolling forecasts, coefficient stability, specification sensitivity | Not done |
| 30% | A detailed causal-design proposal | In prose only |
| 15% | Dashboard: framing, factual calibration, events, source composition, missingness, model performance | Not done |
| 10% | Research ethics, the cutoff date, a prospective update plan | Not done |
| — | A fifth dimension, **framing** | Added 2026-10-16 as `FRAMING_LEXICON` |
| — | A 6 to 8 minute presentation on ONE practical implication for financial forecasting or risk management | The deck has no such slide |

### 11.4 The honesty rule for the labelling work

The brief asks for **human** annotators. A model cannot produce human labels.
The repository therefore builds the instrument, the sample and the agreement
mathematics, and it **states that the human labels are outstanding**. Any
automated label is marked as automated. Do not present an automated label as a
human one. The brief rewards rigour, and a false claim here would be the worst
possible trade.

### 11.5 The agents that run now

Each agent owns two files and must not touch another agent's files.

| Block | Files |
|---|---|
| Economic measurement and timing | `data_loader.py`, `tests/test_data_loader.py` |
| Corpus validation | `corpus_validation.py`, `tests/test_corpus_validation.py`, `docs/labeling_guide.md`, `docs/corpus_documentation.md` |
| Dashboard | `app.py`, `tests/test_app.py` |
| Statistics and causality | `causal_econometrics.py`, `tests/test_causal_econometrics.py` |
| The main session | `nlp_media_svd.py`, `build_report.py`, `build_deck.py`, the documents |

**The GDP alignment change will move every headline coefficient.** When the
first agent reports, recalculate each number in BRAINSTORMING.md, PROGRESS.md,
the report and the deck. Do not copy an old number forward.

---

## 12. State at 2026-10-16, 23:30

### 12.1 Done since paragraph 11

| Block | What was done |
|---|---|
| 20% | **Both violations are gone.** `build_monthly_panel` defaults to `alignment="real_time"`. `PUBLICATION_LAGS` records each release calendar with a source. Quarterly GDP is a step function of published advance estimates. A provenance frame records the observation period, the publication date and the age of every value. Two tests assert the brief's rules on the real data. |
| 20% | The Michigan sub-index requirement was **probed and is not available**. FRED carries `UMCSENT` and `MICH` only. `UMCSENT1` is the discontinued pre-1978 headline series and is a trap. ICC and ICE are not on FRED. The panel report states this on every build. |
| 25% | The **framing** dimension exists end to end: `score_framing`, `score_framing_documents`, `framing_index`, and the monthly columns in `data/processed/media_monthly.parquet`. Two text sources are pooled and they agree at **r = 0.609**. |
| 10% | `docs/update_plan.md` and `docs/ethics_and_limitations.md` are written. The report has new sections for the research-question index, the practical implication, the cutoff and the update plan, and ethics and limitations. |
| 10% | `AI_USE.md` is updated with three new errors, the model versions, and the prompts-and-evidence register that the brief requires. |
| — | `BRAINSTORMING.md` is **version 2**, written against the real brief. Version 1 is kept at `docs/BRAINSTORMING_v1_pre_real_brief.md`. |

### 12.2 The headline numbers, verified on the real-time panel

Every number below was recomputed on 2026-10-16 and verified by a direct run,
not copied from an agent's report.

| Quantity | Value |
|---|---|
| Equation | `sentiment = 126.52 − 4.13*unemployment − 9.81*pce_yoy − 0.03*real_gdp_qoq` |
| Training window | 2010-01 to 2024-12, n = 180, no row dropped |
| R² / adjusted R² | 0.7632 / 0.7591 |
| In-sample RMSE / AIC | 6.24 / 1177.95 |
| Unemployment | −4.1269 (HAC se 0.597, p < 0.001) |
| PCE YoY | −9.8118 (HAC se 0.770, p < 0.001) |
| Real GDP QoQ | **−0.0304 (HAC se 0.048, p = 0.524)** |
| Ridge penalty | 0.01 |
| Gap window | 2025-01 to 2026-09, 20 months tested |
| Mean / median gap | −25.31 / −25.80 |
| Worst gap | −33.64 in 2026-05 (observed 44.8, predicted 78.44, interval 65.77 to 91.12) |
| Months outside the 95% interval | **19 of 20** |
| Months with no complete feature set | 2025-11, 2026-10 |

**The finding that the correction removed.** The significant negative GDP
coefficient (−0.1585, p = 0.039) was an artefact of look-ahead. The honest
statement is that published real GDP growth adds nothing to a monthly sentiment
model once the model may use only the data that existed on the prediction date.

### 12.3 What is still open

| Weight | Item | Owner |
|---|---|---|
| 25% | `corpus_validation.py`, the labelling guide, the corpus documentation, the AI-against-dictionary comparison, the name-swap test | agent `a39ef43eae74bbcca` |
| 30% | Model 3 controls, the multiple-testing correction, rolling-origin forecasts, the causal-design proposal | agent `acea06a7aa23647a8` |
| 15% | The dashboard panels: framing, calibration, events, source composition, missingness, model performance, the demographics-unavailable notice | agent `affe58b0bfe7011b9` |
| — | The report's corpus-and-measurement section, which needs the validation results | the main session |
| — | The deck: rebuild as a 6 to 8 minute presentation around **one practical implication**. The current deck is a project summary and it is too long. | the main session |
| — | **The page budget.** The report is 15 pages and the brief allows 10 to 12. The overflow is prose detail that belongs in the reproducible package. Trim after the analysis lands. | the main session |
| — | Re-run the code review, **after** the rework | the person asked for this order |

### 12.4 A note on the session limit

Three agents stopped at 22:50 on a session rate limit and were resumed at 23:05.
Their work to that point was saved. If an agent stops again, resume it with
`SendMessage` instead of starting a new one: a new agent would start cold and
would repeat the file reads.


---

## 13. The final results, all three hypotheses

Every number here was verified by a direct run on the real-time panel on
2026-10-16 or 2026-10-17. Nothing here is copied from an agent's report.

### 13.1 The quantity to explain

| Quantity | Value |
|---|---|
| Baseline | `sentiment = 126.52 − 4.13*unemployment − 9.81*pce_yoy − 0.03*real_gdp_qoq` |
| Training window | 2010-01 to 2024-12, n = 180, no row dropped |
| R² / adjusted R² | 0.7632 / 0.7591 |
| Months tested out of sample | 20, 2025-01 to 2026-09 |
| Mean / median gap | −25.31 / −25.80 |
| Worst gap | −33.64 at 2026-05 (observed 44.8, predicted 78.44) |
| Months outside the 95% prediction interval | **19 of 20** |

**The one-sided breach is the result, not the size.** A correct 95% interval is
wrong about one month in twenty. This model is wrong in nineteen, always
downward.

### 13.2 Hypothesis A — supported in part, and the largest of the three

| | Model 1 | Model 2 |
|---|---|---|
| R² | 0.7632 | **0.8099** |
| AIC | 1177.95 | **1142.4** |
| Mean gap, same 20 months | −25.31 | **−18.33** |
| Share of the gap absorbed | — | **27.6%** |
| Rolling-origin forecast RMSE | 15.58 | **6.98** |

The mortgage rate is −2.443, p < 0.001. The price level is **not** separately
identified (p = 0.351, and 0.233 with debt service added; VIF 3.68). So the
measurable finding is the **cost of money**, not the price level, and the report
claims only the first. The earlier figure of 31% was computed on the
non-compliant panel.

### 13.3 Hypothesis B — fails, on all four pre-registered conditions

The ladder, 2017-01 to 2026-08, n = 114, HAC(12):

| Model | R² | AIC | Rolling RMSE |
|---|---|---|---|
| 1. Three indicators | 0.535 | 886.0 | 15.58 |
| 2. Plus price level, mortgage | 0.897 | 718.6 | 6.98 |
| 2b. Media only | 0.437 | 909.8 | 17.06 |
| 3. Economic plus media | 0.914 | 706.0 | 6.61 |
| **4. Plus household, political** | **0.941** | **671.5** | **6.12** |
| Naive, last observed value | — | — | **4.45** |

1. **Significance after correction: fails.** Political attribution −41.30
   (p = 0.0128) and framing −4.08 (p = 0.0054) reject raw. Declared family of
   **12 tests**; 4 reject raw, **0 survive BH, 0 survive Holm**.
2. **Granger precedence: fails.** No media-to-gap precedence. The only
   directions reaching 5% run backwards (gap → salience p = 0.0217,
   gap → framing p = 0.0103), and neither survives correction.
3. **Out-of-sample improvement: fails.** Model 3 does not beat Model 2
   (Diebold-Mariano p = 0.322), and **no model beats the naive benchmark**.
4. **Factual calibration: no systematic mismatch.** Holding day and topic
   constant, the largest source-type deviation is 0.044 on [−1, +1].

Both associations are fragile: unstable over rolling windows, significant in
100% and 58% of a 24-cell grid with no controls, and in **0%** of cells with
the household and political controls. Policy uncertainty alone (−0.040,
p ≈ 5 × 10⁻²⁰) absorbs them.

**Framing carried a sign predicted in code before estimation** and came out as
predicted. It still did not survive. That is what a fair test looks like.

### 13.4 Hypothesis C — supported in part, and smaller

Chow F at 2024-07 is 84.97 (p < 0.001), level shift −35.47. Cross-survey ratio
0.258 against a 0.25 threshold, so **the margin is 0.008** and the ratio is
always printed beside the threshold. The supremum-Wald break is 1990, but
2024-07 sits at 96% of the sample and no usual trim reaches it, so that is not
evidence against C.

### 13.5 The measurement finding that outranks all three

FinBERT changes its tone class on **5.8%** of byte-identical texts when only
the outlet label changes (21 of 359), and on **7.3%** when a political name is
swapped. The lexicon changes on **0%**, and so does the framing lexicon on all
three contrasts. A classifier that reads the outlet label would have
manufactured the bias this project went looking for.

### 13.6 What remains unavailable, permanently

| Item | Why |
|---|---|
| Inter-annotator agreement | One operator, no second annotator. The instrument is built and empty. Code refuses to mark model labels as human. |
| Michigan sub-indices (ICC, ICE) | Not on FRED. Michigan licenses them. |
| Demographic and partisan splits | Michigan does not publish them free of charge. |
| Any exposure measure | Nothing in this project observes what a respondent read. This is why no causal claim about readers is made. |
| News volume or attention | The title pull is capped at 250 records a day, so the corpus measures composition only. |
