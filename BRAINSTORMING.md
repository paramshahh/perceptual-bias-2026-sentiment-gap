# BRAINSTORMING — AI, Economic Perceptions and Consumer Sentiment Before the 2026 Midterm Elections

**Status.** This is version 2. It replaces the version that was written against a
reconstruction of the assignment. The real assignment is
`docs/Final_Project_Applications_in_Finance_Oct_2026.pdf`, and its extracted text
is `docs/assignment_brief.txt`. Version 1 is kept at
`docs/BRAINSTORMING_v1_pre_real_brief.md`, because two of its decisions were
wrong against the real brief and a reader must be able to see what changed.

**What this document is.** It is the reasoning record. It states what we measure,
why we measure it that way, what each measurement can and cannot support, and
what result would prove us wrong. It is not the report. The report is
`perceptual_bias_report.pdf`.

**The rule that governs every sentence here.** The brief says: *"Do not assume
that pessimism is irrational, that negative coverage is biased, or that media
exposure causes declining sentiment. These are hypotheses to test."* It also
says: *"A rigorous finding that media bias does not explain the sentiment gap
can earn full credit."* So a negative result is a result. We report the
direction the data gives.

**Language.** This document uses ASD-STE100 Simplified Technical English: short
sentences, the simple present tense, one idea in each sentence, and no stacked
noun groups.

---

## 0. Where each required research question is answered

The brief lists six questions. This table is the index. A grader must not have
to search for an answer.

| # | The question | Where it is answered |
|---|---|---|
| 1 | How can we measure bias in news coverage or social media? | §4 (the corpus), §5 (the five dimensions and their benchmarks) |
| 2 | How can we determine the impact of that bias on the Michigan index? | §7 (association against causality), §7.4 (the causal design), §7.6 (transport to the national index) |
| 3 | How much of the discrepancy do unemployment, inflation and real GDP explain? | §3.5 and §3.6. The answer is a number: the economic-only model explains 76.3% of the variation in the training window and misses 2026 by 25 to 34 points. |
| 4 | Which perceptual mechanisms can the data actually measure? | §5.8. The answer is: framing and selection, yes; negativity in the text, yes; confirmation bias and availability bias, **no**. |
| 5 | Does media analysis improve predictions beyond economic indicators? | §7.1 and §7.2 (Model 1 against Model 2 against Model 3, on rolling-origin forecasts) |
| 6 | What evidence would contradict a media-bias explanation? | §8, which was written **before** the corpus existed, and §9 |

---

## 1. The two violations that version 1 contained

The real brief has two rules about timing. Version 1 broke both. We state this
plainly, because the correction changes a headline number.

**Rule 1.** *"Do not interpolate quarterly GDP into artificial monthly
observations."* Version 1 expanded each quarterly GDP value across its three
months. That creates three monthly observations from one quarterly measurement.

**Rule 2.** *"Use only information available by each prediction date."* Version 1
put the unemployment rate for month *t* in the row for month *t*. The Bureau of
Labor Statistics publishes that rate in month *t+1*. So the model used a number
that did not exist on the prediction date.

**The correction.** `data_loader.build_monthly_panel` now defaults to
`alignment="real_time"`:

* Each row holds only values that were published by the end of that month. The
  publication lag of each series comes from its own release calendar and is
  recorded in `PUBLICATION_LAGS` with a source for each entry.
* Quarterly GDP enters as a **step function of published quarterly estimates**.
  April to June carry the Q1 advance estimate. No monthly GDP value is invented.
* A row with no release available stays empty. Nothing is carried forward
  silently. The panel report names each column and counts the empty rows.
* The old behaviour is still available as `alignment="contemporaneous"`, and its
  own report text says that it is not compliant. It exists so that a reader can
  reproduce the difference.

**What the correction did to the result.** It removed a finding.

| | Real time (compliant) | Contemporaneous (version 1) |
|---|---|---|
| `real_gdp_qoq` coefficient | **−0.0304** | −0.1585 |
| its p-value (HAC, 12 lags) | **0.524** | 0.0387 |
| R² / adjusted R² | 0.7632 / 0.7591 | 0.7663 / 0.7623 |

Version 1 reported a negative and significant GDP coefficient and discussed it
as a specification warning. It was an artefact of look-ahead. The honest
statement is now: **published real GDP growth adds nothing to a monthly
sentiment model once the model may use only the data that existed on the
prediction date.** Unemployment and inflation keep their signs, their sizes and
their significance.

This is the clearest example in the project of a methodological rule that
changes a conclusion rather than decorating it.

---

## 2. The economic and survey baseline: data and provenance

### 2.1 The sources

| Measure | Source | Series | Treatment |
|---|---|---|---|
| Michigan Index of Consumer Sentiment | The issued CSV, cross-checked against FRED `UMCSENT` | `sentiment` | The headline index. The two sources agree month for month. |
| U-3 unemployment rate | BLS, through FRED | `UNRATE` | Seasonally adjusted. Lagged by 7 days (the Employment Situation comes out on the first Friday of the next month). |
| Headline PCE inflation | BEA, through FRED | `PCEPI` | YoY computed as 100 × (index / index 12 months earlier − 1). Lagged by 28 days. |
| Real GDP growth | BEA, through FRED | `GDPC1` | Quarter-over-quarter annualised, as a step function. The advance estimate is lagged by 30 days. A YoY series is also built, by compounding four published annualised rates. |
| Cumulative price level | FRED `PCEPI`, rebased to 2020 = 100 | `pce_level` | When FRED is not available, an approximation is accumulated from the published YoY rates in logs, and it is labelled as an approximation everywhere it appears. |
| 30-year mortgage rate | Freddie Mac PMMS, through FRED | `MORTGAGE30US` | Weekly, collapsed to a monthly mean. |
| Household debt service ratio | Federal Reserve, through FRED | `TDSP` | Quarterly. Lagged by 90 days. It stops at 2026-04. |
| 1-year inflation expectation | Michigan, through FRED | `MICH` | An expectations measure. It is **not** the Index of Consumer Expectations. |

### 2.2 A limitation that we state and do not hide

The brief asks us to *"separate the headline index, Current Economic Conditions,
and Consumer Expectations where available."*

We probed FRED for both sub-indices. **They are not there.** `UMCSENT1` is the
discontinued pre-1978 headline series, not a sub-index, and it is a trap for
anybody who guesses series names. `UMCSENT2`, `UMCSENT3`, `UMCCURR`, `UMCEXP`,
`ICC` and `ICE` all return HTTP 400. A FRED search for "Current Economic
Conditions Michigan" returns zero series. Michigan distributes both sub-indices
on `data.sca.isr.umich.edu` under its own terms.

So the headline / conditions / expectations split is not available from this
pipeline. The panel report says so on every build, whether or not FRED answers.
We did not quietly drop the requirement, and we did not substitute `MICH` for
the expectations index and hope that nobody checks.

### 2.3 Timing, publication dates and revisions

The panel carries a provenance frame. For each month and each column it records
the observation period that the value belongs to, the date the value was
published, and the age of the value in days. Two tests assert the rules on the
real data:

* no value published after month *t* ever appears in the row for month *t*;
* every monthly GDP value in the panel is a published quarterly value, not an
  interpolation.

A survey series is never carried forward. A survey answer for month *t* is
valid for month *t* and for nothing else.

The ragged edge moves when the alignment changes. Under the real-time alignment
the months with no complete feature set are **2025-11** (no unemployment release
in time) and **2026-10** (no PCE release in time).

---

## 3. The economic-only model and the sentiment gap

### 3.1 The specification

$$\text{Sentiment}_t = \beta_0 + \beta_1 \text{Unemployment}_t + \beta_2 \text{PCE\_YoY}_t + \beta_3 \text{RealGDP}_t + \varepsilon_t$$

The training window is 2010-01 to 2024-12, which is 180 complete months. The
standard errors are Newey-West HAC with 12 lags, because monthly sentiment is
strongly autocorrelated and a classical standard error would be too small. A
Ridge model is fitted beside the OLS model, with the penalty chosen by
forward-chaining cross-validation. The scaler sits inside the pipeline, inside
each fold, so no fold sees the statistics of its own validation data. The chosen
penalty is 0.01, which is close to OLS; the data do not ask for shrinkage at
three regressors.

### 3.2 What the gap is, and what it is not

$$\text{Gap}_t = \text{Observed Sentiment}_t - \text{Predicted Sentiment}_t$$

The prediction is out of sample: the model never sees a month after 2024-12.

The brief states the honesty rule for this quantity: *"A negative residual is
unexplained pessimism under your model—not proof of perceptual bias."* So the
gap is named a gap and not a bias. A gap becomes evidence of a broken model only
when the observed value falls outside the model's own **prediction** interval,
not its confidence interval. A confidence interval asks where the mean sits. We
ask whether one month's reading is consistent with the model, so the residual
variance must be inside the interval.

Where sentiment itself is missing, the test result is NA and not False. False
would read as "the model explained this month" when no test ran.

### 3.3 The result

| Term | Coefficient | HAC std. error | p |
|---|---|---|---|
| Constant | 126.52 | 4.804 | <0.001 |
| Unemployment rate | **−4.13** | 0.597 | <0.001 |
| PCE inflation, YoY | **−9.81** | 0.770 | <0.001 |
| Real GDP growth, QoQ annualised | −0.03 | 0.048 | 0.524 |

R² = 0.7632. Adjusted R² = 0.7591. In-sample RMSE = 6.24 points. AIC = 1177.95.
No training row was dropped.

**The answer to research question 3.** Unemployment, inflation and real GDP
growth explain about 76% of the variation in consumer sentiment from 2010 to
2024. One percentage point more unemployment costs about 4.1 index points. One
percentage point more inflation costs about 9.8 index points. So inflation is
worth about 2.4 times unemployment to this survey. Published GDP growth is worth
nothing that we can measure.

Then the same three variables fail in 2025 and 2026.

### 3.4 The out-of-sample failure

The holdout is 2025-01 to 2026-09. Twenty months have a complete feature set and
an observed value.

| Quantity | Value |
|---|---|
| Months tested | 20 |
| Mean gap | **−25.31** points |
| Median gap | −25.80 points |
| Worst gap | **−33.64** points, in 2026-05 |
| 2026-05 observed | 44.8 |
| 2026-05 predicted | 78.44, interval [65.77, 91.12] |
| Months outside the 95% prediction interval | **19 of 20 (95%)** |

The single month inside the interval is 2025-01, the first holdout month.

**This is a statistical result and not a story.** A correct 95% prediction
interval is wrong about one month in twenty. This model is wrong in nineteen
months out of twenty, and always in the same direction. The probability of that
pattern under the model is not a quantity that needs calculation to make the
point. The model is broken, and it is broken downward.

The quantity to explain is therefore about 25 points on average and 34 points at
its worst.

---

## 4. The media corpus

### 4.1 What we collected

| Source | What it is | Count | Window |
|---|---|---|---|
| GDELT GKG 2.0 bulk files | Article records with theme tags and a tone vector. National and local US sources. | **198,367 articles** | 2017-01 to 2026-08, 116 months |
| GDELT DOC API | Article titles | **36,000 titles** | 2017-01 to 2026-08, 95 months |
| twitterapi.io | Tweets, the social-media platform the brief requires | **365 tweets** | A narrow window |

### 4.2 Search terms, selection and inclusion criteria

The GKG filter keeps an article when its theme list contains an economic theme:
`ECON_INFLATION`, `ECON_UNEMPLOYMENT`, `ECON_RECESSION`, `ECON_BANKRUPTCY`,
`ECON_COST_OF_LIVING`, `ECON_DEBT`, `UNEMPLOYMENT` or `LAYOFF`. The political
frame uses `USPEC_POLITICS_GENERAL1`, `EPU_POLICY_GOVERNMENT`, `ELECTION`,
`LEADER`, `GENERAL_GOVERNMENT` and `DEMOCRACY`.

The sampling is two 15-minute GKG files for each sampled day, at 12:00 and at
20:00 UTC. This is a sample and not a census. The bulk archive is far too large
to download in full for a six-day project, and the sample is the honest
description of what we have.

### 4.3 A filter defect that we found in our own work

The first filter accepted any theme with the prefix `TAX_`. We read that prefix
as taxation. In GDELT, `TAX_` is not taxation. It is a general taxonomy of
actors, ethnicities and languages, and it matches most news articles.

| Filter | Articles | Mean tone |
|---|---|---|
| The first filter, with `TAX_` | 693,347 | **−0.988** |
| The strict filter | 198,367 | **−0.180** |

The first corpus measured the tone of the news in general. The second measures
the tone of economic news. A mean tone of −0.988 across 693,347 articles would
have given Hypothesis B a false positive with a large sample behind it. This is
the most dangerous error that the project made, because it was wrong in the
direction that confirmed the interesting hypothesis.

### 4.4 Unit of observation, duplicates and automated activity

**The unit.** The GKG records are one row for each article, and the text we score
is the title or the URL slug. They are **not full article text**. The DOC API
records are headlines. The tweets are posts.

This limit matters for Hypothesis B and we state it in the report. A headline
measures emphasis. It cannot measure a qualification inside a paragraph. The
headline "Inflation jumps to 3.8%" and an article that explains that 3.8% is
close to normal are the same text to this method.

**Duplicates.** GDELT indexes syndicated copies of one story under several
domains. The corpus de-duplicates on URL. It does not de-duplicate on near
identical titles, so a syndicated story can still count more than once. We
report the count of distinct source domains for each month in the dashboard, so
a month that is carried by one syndicator is visible.

**Advertisements and automated activity.** The GKG does not label either. The
tweet filter excludes retweets, accounts with fewer than 10 followers and posts
with fewer than 100 views. These are blunt rules. They are stated as rules and
not as a guarantee.

### 4.5 Supply, engagement and exposure are three different things

The brief is explicit: *"Likes and reposts are imperfect attention measures;
they do not establish what survey respondents encountered."*

We measure **supply**: what was published. We have weak **engagement** data for
the tweets only. We have **no exposure data at all**: nothing in this project
tells us what a Michigan respondent read.

Every causal claim in §7 is limited by that sentence. This is not a caveat at
the end of the report. It is the reason the causal design in §7.4 is the third
option in the brief's list and not the first.

### 4.6 Missingness and access limits

* The GDELT DOC API permits one request in five seconds and answers a faster
  client with HTTP 429. The `gdeltdoc` library turns that answer into an
  exception with an **empty message**, so a rate limit looks exactly like a bad
  query. This cost three cycles of wrong diagnosis. The bulk GKG files have no
  such limit, which is why they became the primary corpus.
* The twitterapi.io budget is almost spent: 365 tweets. A query with
  `queryType: Latest` and a month-long window returns about 14 minutes of
  tweets, because the endpoint returns the newest page and stops. The working
  pattern is `queryType: Top` with single-day windows. We learned this after
  spending credits, and the brief's instruction applies: *"If historical
  platform access is limited, narrow your claims accordingly."* We make no
  claim about social media as a population. The tweets are an illustration.
* Framing is measured from text, so it exists only where text exists: 2017-01 to
  2026-08. Months outside that window have no framing value, and the value is
  missing rather than zero.

---

## 5. Measurement with AI: the coding scheme

The coding scheme was fixed before the measurement ran. The brief requires that
order, and it also protects the result: a scheme written after seeing the data
can be tuned until it agrees.

### 5.1 The five dimensions

| Dimension | The question it answers | How it is measured | The benchmark it is measured against |
|---|---|---|---|
| **Tone** | How does the text describe the economy? | A domain lexicon with a negation window of 3 tokens. `S = (Pos − Neg) / (Pos + Neg)` in [−1, +1]. | Coverage of comparable releases, and the same event across sources. |
| **Selection and salience** | Which facts get coverage? | The share of articles in the month that carry a distress theme, and the daily topic shares from the SVD. | The relative frequency of the underlying events. |
| **Framing** | Price levels or inflation rates? Households or aggregates? Now or the forecast? | Three two-sided contrasts; see §5.2. | The other side of the same contrast, inside the same text. |
| **Factual calibration** | Does a numerical claim match the data that existed when it was published? | A magnitude claim ("soaring", "cooling") is mapped to an implied YoY rate and compared with the realised series. | The official number available at publication. |
| **Political attribution** | Who receives credit or blame? | Attribution patterns near partisan words. | Symmetry: criticism of a party in office is not by itself bias. |

The brief says: *"Define the benchmark against which each proposed bias is
measured."* That is the last column. Without it, "negative coverage" is not a
measurement of anything, because the economy may really be bad.

### 5.2 Framing, and why it is a ratio and not a count

The brief defines framing as the *"emphasis on price levels versus inflation
rates, national aggregates versus household experiences, and current conditions
versus forecasts."* Each is a **pair**. So each measure is a ratio between the
two sides, and not a count of one side. A count of price words would only
measure how much an outlet writes about the economy.

Each contrast returns a value in [−1, +1]:

| Contrast | +1 means | −1 means |
|---|---|---|
| `level_vs_rate` | all price-**level** language | all inflation-**rate** language |
| `household_vs_macro` | all **household** language | all **aggregate** language |
| `current_vs_forecast` | all **current** language | all **forecast** language |

The two sides are summed over the month, and the ratio is taken once at the end.
A mean of per-document ratios would give a title with one framing word the same
weight as a title with four, and it could not say "no document framed anything
this month" except by inventing a number. A month with no framing word gets a
missing value and never 0.0.

**`level_vs_rate` carries a prediction that was made in advance.** Hypothesis A
says that a household anchors to the price it pays and not to the rate of
change. So coverage that frames the question by the price **level** is coverage
that is consistent with the anchoring account, and coverage that frames it by
the falling inflation **rate** contradicts that account. This is why the single
`framing` column reports this contrast. The other two contrasts are reported
because the brief requires them, and no sign was predicted for them in advance.
We do not claim one now.

Negation is deliberately **not** applied to framing. A negator flips a valence,
and framing has no valence to flip. "Prices are not higher than in 2020" still
frames the question as one about the price level, which is exactly what the
measure claims to detect.

**Two text sources, and a validity check between them.** Framing needs words,
and a GDELT theme tag has none. Two text sources exist: the 36,000 real article
titles, and the URL path of each of the 198,367 GKG articles, which for most
news sites is a mechanical slug of the title. The slug source is noisier and
covers all 116 months; the title source is cleaner and covers 95. The measure
pools both and takes the ratio once. Both are also scored alone, and the two
monthly readings of the same month correlate at **r = 0.609**. That number is
reported, because it is the cross-source validity evidence for the measure: the
two readings agree well above chance, and they are not the same number twice.

The title-only measure has a median of 28 framing hits in a month, against 68
for the slugs. That is why neither source is used alone.

### 5.3 Why we do not trust an off-the-shelf classifier here

FinBERT learns from market text, where the question is "is this good for the
asset?". Macro policy text asks a different question, and the two answers often
have opposite signs. *"Strong productivity growth and robust labour market
conditions continued through the quarter"* is strongly positive to FinBERT. In
policy language it is hawkish, because strong growth justifies a higher interest
rate. The model reads the words correctly and answers a question nobody asked.

We did not quote this failure from a paper. We measured it. A probe set of 30
macro sentences, 10 hawkish, 10 dovish and 10 neutral:

| Measurement | Value |
|---|---|
| FinBERT accuracy | **0.300** |
| Accuracy of a random guess | 0.333 |
| Hawkish sentences called dovish | 6 of 10 |
| Dovish sentences called hawkish | 7 of 10 |

FinBERT is worse than chance on this text and it inverts the two classes that
matter. So the domain lexicon is the primary instrument and FinBERT is reported
as a measured failure mode.

### 5.4 Validation against human labels

The brief requires a stratified sample of at least 300 items, two independent
annotators, reported agreement, an adjudication procedure, AI performance by
class and by source, a comparison against a simple dictionary or conventional
classifier, and a test that changes political names and source labels on
otherwise identical text.

**The instrument exists and is deliberately empty.** 359 items, above the
brief's minimum of 300, drawn across 161 strata of year × source type ×
predicted tone class, seed 20261016, from 24 483 eligible rows of 36 365. The
items dropped are named with their cause: 11 080 duplicate texts, 799 short
texts, 3 empty. The sheet is **blind** — it carries only an item id, a
presentation order and the text, with source, date, stratum and model class in
a separate key file, and the order is shuffled. The seven questions cover all
five dimensions, and the framing questions use the same side names as the
lexicon so a human label sits on the same scale as a lexicon ratio.

A framing oversample was unavoidable: only 9.2%, 2.5% and 2.2% of titles frame
the three contrasts, so a proportional 300 would give about ten framed items for
two of them. A quota of 40 per contrast adds 59 items, tagged by sampling
layer. The recorded cost is that any prevalence estimate must use the 300
stratified rows only.

**What is not available, and will not be faked.** This project has one operator
and no second annotator, so **there is no inter-annotator agreement statistic
and none is reported**. The agreement mathematics, the ten-step adjudication
procedure, and the per-class and per-source breakdowns are written and tested,
and they are waiting for labels. The honesty rule is enforced in code and not
only in prose: `provenance` is a required keyword with no default,
`labels_are_human` is true only for human provenance, `require_human_labels`
raises on a model-label report, no function ever writes into an annotator
column, and a test asserts that all 16 annotator columns of the written sheet
are empty.

**AI against a dictionary: accuracy is unknown, robustness is settled.**
Accuracy needs the human labels that do not exist, so the only honest headline
is concordance between the two instruments on the 359 items: percent agreement
0.557, Cohen's kappa 0.224, which is poor. The class mixes differ sharply —
FinBERT calls 34.0% of items negative where the lexicon calls 12.8%. By source,
the two agree on formulaic press-release copy (wire or PR, 0.850) and disagree
on national news (0.432) and foreign coverage (0.400).

**On robustness the dictionary wins outright, and this is the project's
sharpest finding.** The source-label test holds the text byte-identical and
changes only the outlet name in front of it:

| Dimension | Classifier | Swap | Eligible | Flipped |
|---|---|---|---|---|
| Tone | lexicon | political name | 55 | **0 (0.0%)** |
| Tone | FinBERT | political name | 55 | **4 (7.3%)** |
| Tone | lexicon | source label | 359 | **0 (0.0%)** |
| Tone | FinBERT | source label | 359 | **21 (5.8%)** |
| All three framing contrasts | framing lexicon | both | 55 / 359 | **0 (0.0%)** |

**For a project that measures political media bias, this is decisive.** A
classifier that reads the outlet label would manufacture the bias it reports.
The caveat belongs here too: robust and wrong is possible, and this is not
accuracy. Framing also has no neural counterpart at all — FinBERT has one
market-sentiment head and cannot emit a framing contrast.

### 5.5 What text can and cannot establish

The brief states the limit: *"Text can establish framing or selective coverage;
it cannot, by itself, establish a reader's cognitive bias. Claims about
confirmation or availability bias require corresponding exposure, recall, belief,
or experimental evidence."*

**The answer to research question 4** follows directly:

| Mechanism | Can this project measure it? | Why |
|---|---|---|
| Negativity in the supply of coverage | **Yes** | It is a property of the text, and it has a benchmark in §5.1. |
| Selective coverage, or salience | **Yes** | It is a property of the text against event frequency. |
| Framing, including anchoring to past prices in the coverage | **Yes** | §5.2, and it has a predicted direction. |
| Factual miscalibration | **Yes** | The claim is compared with the number that existed. |
| **Anchoring in the reader's head** | **No** | It needs belief data. We have the Michigan 1-year inflation expectation as weak evidence and nothing else. |
| **Availability bias** | **No** | It needs recall data. We have no exposure measurement at all (§4.5). |
| **Confirmation bias** | **No** | It needs prior beliefs for each respondent, and partisan sub-index data that Michigan does not release publicly. |

Three of the mechanisms in the brief's question are **not measurable with these
data**, and the correct answer to the question is to say which three and why.

---

## 6. Latent semantic analysis with a truncated SVD

### 6.1 The method

Let $X \in \mathbb{R}^{m \times n}$ be the TF-IDF term-document matrix with $m$
terms and $n$ documents. The truncated SVD keeps the $k$ largest singular
values:

$$X \approx U_k S_k V_k^\top$$

The columns of $U_k$ are the latent topics in term space. The rows of $V_k$ give
the position of each document. $S_k$ gives the size of each topic. We use
`scipy.sparse.linalg.svds`, which needs only sparse matrix-vector products.

### 6.2 Why we do not remove the mean

**Sparsity.** A TF-IDF matrix is more than 99% zeros. `svds` costs
$O(\text{nnz})$ for each step. If you subtract a mean, every zero becomes a
non-zero value, and the storage grows from $O(\text{nnz})$ to $O(mn)$. At this
size the decomposition is then not possible.

**The zero-prior property, which matters more.** In a non-centred
decomposition, a document with no vocabulary term maps to the zero vector. It
contributes nothing. That is the correct result, because no signal is not the
same as an average signal. If you subtract the mean, that document receives the
negative of the mean vector, so it gets a position it did not earn. Then a daily
sum moves with the number of off-topic documents on that day. The origin must
mean "no terms are present". Tests hold this property.

**It is a different decomposition and not a worse one.** A centred SVD is PCA.
It uses the covariance matrix and finds directions of maximum variation about
the mean. A non-centred SVD uses the second-moment matrix $X^\top X$ and finds
directions of maximum co-occurrence. Latent semantic analysis wants
co-occurrence, measured from zero.

### 6.3 How we choose $k$

Three diagnostics are reported together: the singular value spectrum and its
elbow; the cumulative energy
$\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$; and a permutation noise floor.

**A correction to the permutation method.** The floor is too strict on a dense
non-negative matrix. $X$ is non-negative and not centred, so a large part of
$\sigma_1$ is the mean direction, and a permutation inside a column keeps that
direction. So the permuted $\sigma_1$ can exceed the true $\sigma_1$, and the
method returns $k = 0$ when real topics are present. We measured it: a matrix
with 3 true topics at 33% density gives real values of 11.66, 11.60 and 11.57
against a permuted maximum of 11.93. At true TF-IDF sparsity (below 10%) the
effect disappears. We keep the permutation floor and add a component-wise
parallel analysis, and the module prints a note when the two disagree.

### 6.4 Signal selection with the SVD, and the trap inside it

Let $s \in \mathbb{R}^{T \times K}$ hold $K$ text signals over $T$ periods and
let $\delta y$ be the forward change in the target. The cross-covariance is

$$C = \frac{1}{T}\, s^\top \delta y$$

and its decomposition $C = U_k S_k V_k^\top$ gives the signal combinations with
the most covariance with the target. The truncation removes text noise.

**The trap.** $C$ contains $\delta y$, which is the quantity to predict. $C$ must
be computed on the training window only. If you compute it on the full sample
and then report a holdout result, the transform already contains the answer.
This project made the same class of error in its own code: a scaler was fitted
before the cross-validation, so each fold saw the statistics of its own
validation data. It is fixed, and the fix is a `Pipeline` inside each fold.

---

## 7. Association and causality, kept separate

### 7.1 The three models

| Model | Regressors |
|---|---|
| **1. Economic only** | unemployment, PCE YoY, real GDP QoQ |
| **2. Economic plus media** | Model 1 plus media tone, salience, framing, factual calibration and political attribution |
| **3. Economic plus media plus household and political controls** | Model 2 plus the cumulative price level, the 30-year mortgage rate, the debt-service burden and a policy-uncertainty control |

The brief's ordering is used, so that the media block is added **before** the
household controls. That ordering is the harder test for the media hypothesis:
the media variables get their chance before the alternative explanation enters.

**The ladder as estimated**, with a media-only rung added because a media block
that cannot stand alone is itself a result. The household and political
controls are policy uncertainty (`USEPUINDXD`), real disposable income per head
as a year-over-year change, labour underutilisation (`U6RATE`), and election
proximity computed from the statutory calendar so it needs no network and
carries no look-ahead.

The result is in paragraph 8.2. In one line: **Model 4 dominates, and every
media coefficient collapses inside it.** Policy uncertainty alone absorbs them.

### 7.2 How the models are compared

A higher R² on the same sample is not evidence that media analysis helps. Model 3
has more regressors and must fit better. So the comparison is **out of sample**,
on rolling-origin forecasts: fit to month *t*, predict month *t+1*, step forward,
and repeat. We report the forecast error, the stability of each coefficient
across origins, the uncertainty, and the sensitivity to the specification.

This is the answer to research question 5, and it is a prediction question, so it
gets a prediction answer.

### 7.3 Autocorrelation, seasonality, lags and multiple testing

* **Autocorrelation.** Monthly sentiment is strongly autocorrelated. All standard
  errors are Newey-West HAC with 12 lags.
* **Seasonality.** The survey series and the macro series are seasonally
  adjusted at source where adjustment exists. The media series is not, so the
  month is available as a control.
* **Lag selection.** VAR lag order is chosen by information criterion, and the
  chosen order is reported with the result and not hidden inside a function.
* **Multiple testing.** We test five media dimensions in several models. Five
  dimensions at the 5% level give about a 23% chance of one false positive. So
  the media coefficients carry a Benjamini-Hochberg false-discovery-rate
  correction, and both the raw and the corrected p-values appear in the table.
  A finding that survives correction and a finding that does not are different
  findings.

### 7.4 The causal design: option three, honestly

The brief gives three options. We take the third: **observational analysis with
a causal-design proposal.**

We take it because of §4.5. A randomized framing experiment needs consenting
participants and institutional approval, and this is a six-day project with no
such approval. A quasi-experiment needs plausibly external variation in media
**exposure**, and we have no exposure data at all. Claiming either design would
be the most serious kind of overstatement available here.

What we do instead:

1. Estimate the associations, and label them as associations.
2. Run Granger tests, and label them as **predictive precedence** and not as
   causation. A common cause, such as the election cycle, can move coverage and
   sentiment together, and then earlier coverage predicts the later gap with no
   causal link.
3. Run the Rigobon-Sack heteroskedasticity instrument where its conditions hold,
   and report where they do not (§7.5).
4. Write the design that would actually identify the effect, in enough detail
   that somebody could run it: the sampling frame, the randomisation, the
   balanced factual summaries, the belief and sentiment instruments, the power
   calculation, and the approvals needed.

### 7.5 What the Rigobon-Sack instrument can and cannot do here

**The order condition in the brief is not usable for this system.** The brief
gives $N^2 - N - 2K > 0$. With $N = 2$ this is $2 - 2K$, which holds only when
$K = 0$ and fails with any exogenous regressor. The module reports the value so
that the report can discuss it.

**The system is under-identified by one restriction.** Two regimes give six
second moments against six parameters, which looks sufficient and is not. All the
identifying information sits in $\Omega_H - \Omega_L$, which gives three
equations in four unknowns. So we estimate the direction media → gap only, under
the Rigobon-Sack assumption that the regime change moves the variance of the
media shock alone. **The reverse direction is not identified and we do not
report it.** That assumption is the weakest part of the method, and two
covariance matrices cannot test it.

**The rank condition in the brief disagrees with the assumption.** Under that
assumption, $\Omega_H - \Omega_L$ has rank 1 in the population, so its
determinant is exactly zero. A determinant near zero is therefore **consistent
with** identification and is not a failure. We compute the determinant and its
bootstrap distribution because the brief asks for them, but the operative test
is different: the shift in the media variance must be different from zero in the
bootstrap. On synthetic data with a true coefficient of 0.60 the method returns
0.598, and with no variance change it returns no estimate.

**A defect we found in our own code.** The first regime rule used a rolling
variance window that included the current month. That rule sorts months by their
own deviation, so the rank condition "passes" on pure noise. The window now
stops at the previous month, and the homoskedastic control case now fails as it
should.

**The power limit.** The monitoring window has 20 monthly gap values. A VAR with
lags and a two-regime covariance estimator on 20 values produce output that looks
like a result and is not one. The modules refuse to compute these statistics at
that sample size and return a warning instead of a number. The media models are
therefore estimated on the long window, 2017 to 2026, about 116 months, and the
2025–2026 window is shown separately with a statement about its power.

### 7.6 Reverse causality, and transport to the national index

**Reverse causality is a real possibility and not a footnote.** Pessimistic
consumers may seek negative content. Publishers respond to audience demand. An
economic shock may move coverage and sentiment at the same time. The Granger
tests run in both directions and both directions are reported. The IV identifies
one direction only, and §7.5 says which.

**Transport.** Even a clean effect in our sample would not be an effect on the
national Michigan index. To connect the two we would need: an exposure estimate
for the respondent population; a representative sample or weights that map our
corpus to what respondents actually encountered; and a transportability
assumption that the effect size does not depend on the characteristics that
differ between the samples. We do not have any of the three. We say so.

---

## 8. The three hypotheses, and what would falsify each

These conditions were fixed before the corpus existed. That is what makes a
negative result usable.

### 8.1 Hypothesis A. Economic pressure that the headline indicators miss

**The claim.** Households respond to the cumulative price level, to housing
affordability and to debt service, and the headline three do not carry those.

**The intuition.** Inflation fell to about 3%. That is a statement about the
*rate*. A household pays the *level*. Prices are still far above where they were
in 2020. A slower rate of increase on a level that is already high is not
relief, and a household does not celebrate a smaller second derivative.

**What would falsify it.** Adding the price level, the mortgage rate and the
debt-service burden to Model 1 leaves the gap unchanged, and none of the new
regressors is significant.

**The result: supported in part, and it is the largest of the three.** Verified
on the real-time panel on 2026-10-16.

| Quantity | Model 1 | Model 2 (adds the price level and the mortgage rate) |
|---|---|---|
| R² on 2010-01 to 2024-12 | 0.7632 | **0.8099** |
| AIC | 1177.95 | **1142.4** |
| Mean out-of-sample gap, same 20 months | −25.31 | **−18.33** |
| Share of the gap absorbed | — | **27.6%** |
| Out-of-sample forecast RMSE, 54 rolling origins | 15.58 | **6.98** |

The forecast number is the one that matters, because it is the only one that
cannot be bought with extra regressors. **Adding the two level variables cuts
the one-step forecast error by more than half.** No other block in this project
does anything comparable.

**Which variable does the work, and which does not.** The mortgage rate is
−2.443 with p < 0.001. The cumulative price level is **not** separately
identified: p = 0.351, and it stays insignificant at 0.233 when the
debt-service burden enters. Its variance inflation factor is 3.68 against a
usual limit of 5, so collinearity alone does not explain it. The more likely
reason is that the mortgage rate carries the same information about the cost of
living and carries it better.

**So the honest version of Hypothesis A is narrower than the hypothesis.** The
*cost of money* is measurable and large. The *price level* is the mechanism the
hypothesis describes, and these data cannot separate it from the mortgage rate.
We report the first and we claim nothing about the second.

Note that the earlier figure for this block was 31%, computed on the
non-compliant panel. It is 27.6% on the real-time panel.

### 8.2 Hypothesis B. Selection, framing and media bias

**The claim.** Coverage is more negative, more selective, or differently framed
than the economic facts justify, and that difference is associated with the gap.

**What would falsify it**, fixed in advance:

1. The media tone is not significant in Model 2, after the FDR correction.
2. Lagged media measures do not Granger-cause the gap.
3. The out-of-sample forecast does not improve when the media block is added.
4. The factual calibration shows no systematic mismatch between the claims and
   the data available at publication.

**The result: Hypothesis B fails, on all four of its own pre-registered
conditions.** The test is now complete, because the framing dimension exists,
and the framing measure carried a sign predicted in code before estimation.

**The model ladder**, 2017-01 to 2026-08, n = 114, HAC(12):

| Model | R² | adj R² | AIC |
|---|---|---|---|
| 1. The three headline indicators | 0.535 | 0.523 | 886.0 |
| 2. Plus the price level and the mortgage rate | 0.897 | 0.892 | 718.6 |
| **2b. The media measures alone** | **0.437** | 0.417 | 909.8 |
| 3. Economic plus media | 0.914 | 0.906 | 706.0 |
| **4. Plus household and political controls** | **0.941** | 0.933 | **671.5** |

**Condition 1, significance after correction: fails.** Two of the four media
coefficients reject before any correction — political attribution at −41.30
(p = 0.0128) and framing at −4.08 (p = 0.0054, the predicted sign). The
declared family is **12 tests**: the four media coefficients of Model 3 only,
plus both Granger directions for each of the four measures. **Four reject raw.
Zero survive Benjamini-Hochberg. Zero survive Holm.**

The family decides this result, so the family is declared rather than chosen.
Coefficients come from one model, so a nested model cannot inflate the count,
and every direction tested is counted, not only the interesting ones. In a
narrower family of the four coefficients alone, both would survive. We report
that fact too, because a reader is entitled to know how sensitive the verdict
is to a choice we made.

**Condition 2, Granger precedence: fails, and it fails in an informative
direction.** No measure shows precedence from media to gap. The only
directions that reach 5% before correction run the *other* way: the gap
precedes the salience (p = 0.0217) and the gap precedes the framing
(p = 0.0103). Neither survives correction.

**Condition 3, out-of-sample improvement: fails.** On 54 rolling origins Model
3 does not beat Model 2 (Diebold-Mariano p = 0.322), and **no model beats a
free last-observed-value benchmark**: the naive RMSE is 4.45 against 6.98 for
Model 2 and 6.61 for Model 3. Model 2b, the media measures with no economic
variable, is worse than Model 1.

**Condition 4, factual calibration: no systematic mismatch.** Holding the
calendar day and the dominant topic constant, no source type is reliably more
negative; the largest systematic deviation is 0.044 on a scale from −1 to +1.
The large *unconditional* spread between source types is mostly **which events
each type covers**, not how it describes them.

**And both surviving associations are fragile.** Over 48-month rolling windows
every Model 3 coefficient is unstable: political attribution ranges from −66.7
to +5.3 with four sign changes and is significant in 30% of windows; framing
changes sign six times and is significant in 4%. In a documented 24-cell
specification grid, political attribution is significant in 100% of cells with
no controls and **0% with the household and political controls**, and framing
in 58% and then **0%**. In Model 4 no media coefficient is significant even
before correction.

**The one-line verdict.** The media measures are an artefact of leaving
household and political conditions out of the model. Policy uncertainty alone
(−0.040, p ≈ 5 × 10⁻²⁰) absorbs them.

### 8.3 Hypothesis C. Measurement, timing and an incomplete model

**The claim.** Part of the gap is a property of the measuring instrument and of
our own specification, and not of the consumer.

**Three separate parts, and they must stay separate:**

1. **Survey method.** Michigan moved to web collection over 2024 to 2026. A mode
   change can move a level. A gap that begins at the mode change is suspicious.
2. **Timing.** This is the part that version 1 got wrong, and §1 shows that one
   timing rule removed one of our findings. That is direct evidence that the
   specification mattered.
3. **Our model.** Three regressors and a linear form. The Michigan survey asks
   about personal finances, business conditions and buying conditions. It is not
   a rating of three macro statistics, and the brief says so.

**What would falsify it.** An independent sentiment series with no method change
shows the same gap over the same months.

**The result: supported in part, and smaller than Hypothesis A.** The Chow test
at the specified 2024-07 transition rejects stability with F = 84.97 (p < 0.001)
and a level shift of −35.47. Two cautions limit it, and they are not footnotes.
A shift of 35 index points is too large for a change of collection method
alone, and the test cannot separate mid-2024 method change from any other
mid-2024 event. The unrestricted supremum-Wald search puts its break in 1990,
not at the transition — but the 2024 date sits at about 96% of the sample, so no
usual trim can reach it. "No break found there" is therefore not evidence
against Hypothesis C, because that date was never a candidate.

The cross-survey ratio is 0.258 against a threshold of 0.25 for "supported".
**The margin is 0.008**, so the report always prints the ratio beside the
threshold and never prints the word alone.

> **A note on the earlier work.** Under the earlier work an
> independent instrument placed about one quarter of the Michigan gap in this
> block. The comparison series had to be a **business** survey, because no free
> consumer series was available, and that is a weakness we name rather than
> hide. An earlier version of this comparison used a series that was partly
> derived from Michigan itself; that is circular and it was removed before it
> reached the report.

### 8.4 How the three are separated

They are not mutually exclusive, and the correct answer may be "all three, in
these proportions". The separation is by **what each hypothesis predicts that
the others do not**:

| | Hypothesis A | Hypothesis B | Hypothesis C |
|---|---|---|---|
| New economic controls shrink the gap | **yes** | no | no |
| Media measures help out of sample | no | **yes** | no |
| Lagged media predicts the gap | no | **yes** | no |
| An independent survey shows the same gap | yes | yes | **no** |
| The gap starts at the survey mode change | no | no | **yes** |

---

## 9. The answer to research question 6: what would contradict the media story

The brief asks this question directly, so it gets its own section.

Evidence that would contradict a media-bias explanation of the 2026 gap:

1. **The gap appears in a survey with no method change and no plausible media
   channel.** Then the gap is about the economy or the instrument.
2. **The media measures add nothing out of sample.** A regressor that cannot
   forecast is hard to defend as a cause.
3. **The coverage is factually calibrated.** If the claims match the numbers that
   existed when they were published, then "the coverage describes a bad
   situation" explains the data as well as "the coverage is biased", and the
   simpler explanation wins.
4. **The direction of precedence runs the other way.** If the gap predicts
   coverage better than coverage predicts the gap, the demand-side account is
   stronger than the supply-side account.
5. **The gap is fully absorbed by household economic controls.** Then Hypothesis
   A explains it and the media block is decoration.
6. **The gap begins exactly at the survey mode change.** Then the instrument is
   the better suspect.

We have already found that several of these hold. That is a finding, and the
brief states that it can earn full credit.

---

## 10. The pre-election cutoff and the prospective update plan

The brief requires two things here: *"Do not attribute a change to election
proximity merely because it occurred during the campaign"*, and *"If your
project finishes before election day, state the cutoff and provide a prospective
update plan."*

**On attribution to the election.** Several of our series move in 2026. The
campaign also happens in 2026. Those two facts together are not evidence. A
claim that the election proximity is the cause needs a counterfactual: the same
months in a non-election year, or a comparison between places with different
campaign intensity. We do not have that comparison, so we do not make the claim.
The election is the reason for the monitoring window, and not an explanation
inside it.

**The cutoff.** The analysis cutoff is the data available to **2026-10-16**. The
last complete macro month is 2026-09. The media corpus ends 2026-08. Election
day is after the cutoff, so **no post-election data is in this project at all.**

**The prospective update plan** is in `docs/update_plan.md`: what to re-run, in
what order, which numbers must move when, and which pre-registered conditions
decide each hypothesis when the remaining months arrive.

---

## 11. Ethics, licensing and privacy

* **Data.** FRED, BLS, BEA and GDELT are public. twitterapi.io data is used under
  its terms. No paywalled article text was scraped; the corpus holds titles, URLs
  and GDELT-derived metadata.
* **No human subjects.** No experiment was run, no participant was recruited, and
  no institutional approval was needed. The randomized design in §7.4 is a
  **proposal**, and it says what approval it would require.
* **Privacy.** Tweets are from public accounts. No individual is named in any
  output, and no author-level data is published. The corpus is aggregated to
  monthly series before it reaches any artifact.
* **AI use.** `AI_USE.md` records what the AI did, with model versions and
  prompts. Every prompt, model version and evidence excerpt is preserved, as the
  brief requires.
* **The honesty rules we hold ourselves to**, in one list, because they are the
  easiest thing to break under time pressure:
  1. A label is described as what it is. No second human annotator is claimed.
  2. A derived series is never compared against itself. The circular comparison
     in §8.3 was caught and removed.
  3. A missing value is missing. It is never a zero and never carried forward.
  4. A number that was computed on a superseded panel is void, and it is marked
     void and not quietly reused.
  5. A negative result is reported with the same emphasis as a positive one.

---

## 12. Decisions, and what they cost

| Subject | The decision | The cost |
|---|---|---|
| Alignment | Real time by default. Nothing is used before it was published. | One finding disappeared (§1). R² falls by 0.003. |
| Quarterly GDP | A step function of published estimates. | No monthly GDP variation inside a quarter. |
| Michigan sub-indices | State that FRED does not have them. | The brief's split is not delivered, and the reason is recorded. |
| The primary data | The issued CSV, cross-checked against FRED. | The numbers agree with the graded file. |
| The media corpus | GDELT GKG is primary. The DOC API titles are the text source. twitterapi.io is an illustration. | Titles and slugs only. No article bodies (§4.4). |
| The framing measure | A ratio between two sides, from two pooled text sources. | It exists only for 2017-01 to 2026-08. |
| Tone classifier | A domain lexicon. FinBERT is reported as a measured failure. | No transformer-grade language understanding. |
| The causal design | Observational, with a full design proposal. | No causal effect is claimed. |
| Multiple testing | Benjamini-Hochberg across the media block. | Some associations will not survive, and that is the point. |
| Negative results | Report them. | None. |

---

## 13. What this document commits us to

1. The quantity to explain is a gap of **−25.31 points on average** and
   **−33.64 points at its worst (2026-05)**, with **19 of 20 months outside the
   95% prediction interval**. These numbers stand on a real-time panel.
2. The economic-only model explains about **76%** of sentiment variation from
   2010 to 2024. Inflation is worth about **2.4 times** unemployment to this
   survey. Published GDP growth is worth nothing measurable.
3. Every number that was computed on the contemporaneous panel is void. Each one
   is recomputed before it appears in the report, the dashboard or the deck.
4. Each hypothesis is decided on the conditions in §8, which were fixed before
   the corpus existed.
5. Each causal statement names the strength of its identification: association,
   predictive precedence, or instrumented. No statement exceeds its evidence.
6. Three of the perceptual mechanisms the brief names cannot be measured with
   these data, and we say which three (§5.5).
