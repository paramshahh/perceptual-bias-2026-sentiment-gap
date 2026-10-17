# CORPUS DOCUMENTATION

This document describes the media corpora of the project `perceptual-bias`. It
answers each item that section 2 of the assignment brief asks for.

This document uses ASD-STE100 Simplified Technical English.

Date of the measurements: 2026-10-16.
Code module: `src/perceptual_bias/corpus_validation.py`.
Function that gives these numbers: `profile_corpus`.

Each number in this document comes from the corpus files. No number is an
estimate. You can get each number again with `profile_corpus`.

---

## 1. The three corpora

| Corpus | File | Rows | What one row is | Period | Months |
|---|---|---|---|---|---|
| GKG strict | `data/raw/gkg/gkg_strict.parquet` | 198 367 | One article record: a URL, a theme list and a tone vector. **No text.** | 2017-01 to 2026-08 | 116 of 116 |
| Headlines | `data/raw/gdelt/headlines.parquet` | 36 000 | One headline (the title of the article) | 2017-01 to 2026-08 | 95 of 116 |
| Tweets | `data/raw/tweets/tweets_daily.parquet` | 365 | One social-media post | 2025-01 to 2026-08 | 20 of 20 |

### 1.1 What the observations are, and what they are not

The brief asks whether an observation is a headline, a full article, a post or
a comment. The answer is important, and it limits each result of the project.

- **The headline corpus holds headlines only.** The GDELT Doc 2.0 API serves
  metadata and not article bodies. The project does not read the article pages.
  Thus a headline and an article that qualifies that headline are the same text
  for this project. The headline "Inflation jumps to 3.8%" and an article that
  says that 3.8 percent is near to normal give one measurement, and that
  measurement is the headline.
- **The GKG corpus holds no words at all.** It holds a URL, a GDELT theme list
  and a tone vector that GDELT calculated. The project uses the URL slug as a
  text proxy for the framing measure. A URL slug is a short and incomplete
  version of the headline. This corpus therefore cannot enter the human-label
  sample, and `profile_corpus` records this with the note "no text column".
- **The tweet corpus holds posts and not comments.** No reply and no quote-post
  is in the corpus.
- **No corpus holds a comment, a broadcast transcript or a video.**

---

## 2. Search terms, dates and inclusion criteria

### 2.1 The news corpora

The query terms are in `nlp_media_svd.ECONOMIC_TERMS`:

```
inflation, cost of living, grocery prices, gas prices, recession,
layoffs, consumer sentiment, economy
```

The GDELT Doc API joins a list of keywords with AND and it gives no OR for a
plain keyword list. Thus the pipeline sends each term as its own request and
joins the results. Each extra term is one more HTTP request, and GDELT permits
one request each five seconds. The term list is short for that cause.

The GKG corpus uses a different and stricter rule. A record enters the corpus
when its theme list holds an economic theme. The measured theme coverage of the
corpus:

| GDELT theme | Share of the 198 367 records |
|---|---|
| `EPU_ECONOMY_HISTORIC` | 84.8 % |
| `EPU_POLICY` | 56.5 % |
| `ECON_STOCKMARKET` | 18.9 % |
| `ECON_INFLATION` | 6.6 % |
| `UNEMPLOYMENT` | 4.5 % |

Inclusion criteria for a news record:

1. The record is in the period 2017-01 to 2026-08.
2. The record holds an economic query term, or an economic GDELT theme.
3. The record has a URL and a date.
4. The language field is English, where GDELT gives that field.

Exclusion criteria that the pipeline applies:

1. A record with no title or with a title of three words or fewer. These are
   navigational strings such as "Business" or "Latest Articles". They are 2.2
   percent of the headline corpus.
2. A record whose normalised text is the same as the text of an earlier record.
   Refer to paragraph 4.

### 2.2 The sampling frame is a day sample, not a census

**This is the largest sampling limitation of the news corpora, and it is not
obvious from the row count.**

The headline corpus holds 36 000 rows. These rows come from **144 sampled
days** and not from 3 500 days. Each sampled day gives exactly 250 rows, which
is the hard record cap of the GDELT Doc API. The corpus is therefore:

- A **cluster sample of days**, with a census-like pull inside each sampled day.
- **Truncated at 250 records for every day.** A day with 2 000 economic
  articles and a day with 250 economic articles both give 250 rows. The corpus
  cannot measure the **volume** of coverage. It can measure only the
  **composition** of coverage on a sampled day.

The GKG corpus is also a day sample. Its monthly row counts range from 368 to
3 816, with a median of 1 614.5. That range is a property of the sampling and
of the API cap, and it is not a measurement of news volume.

**Consequence for the analysis.** A monthly media series from these corpora is
a series of composition measures (tone, framing, salience shares). It is not a
series of attention or of volume. A claim about "more coverage of inflation"
cannot come from this corpus.

### 2.3 The tweet corpus

The source is twitterapi.io. The pull method is in `PROGRESS.md` paragraph 3.1.

| Property | Value |
|---|---|
| Rows | 365 |
| Distinct days | 39 |
| Months | 20 (2025-01 to 2026-08) |
| Query type | `Top` with a window of one day |
| Language | English only (365 of 365) |
| Distinct accounts | 243 |
| Largest account share | 5.2 % (19 posts) |

The parameter `queryType: Latest` with a window of one month failed. It gave
the newest tweets at the end of the window, thus 140 tweets covered 14 minutes.
The pipeline uses `Top` with a window of one day for that cause.

**`Top` is a selection on popularity.** The platform decides which posts are
"top". The corpus is therefore a sample of **popular** economic posts and not a
sample of economic posts. A tone measure from this corpus is the tone of
popular posts. The project budget permitted 54 more requests, which is not
sufficient for a random sample.

---

## 3. Geographic relevance and audience information

### 3.1 Geography

The project is about the US consumer. GDELT collects worldwide coverage. The
measured foreign share:

| Corpus | Foreign share | Method |
|---|---|---|
| GKG strict | **22.9 %** | A source with a country top-level domain, or a source in a list of known foreign outlets |
| Headlines | 0.9 % | The same method |
| Tweets | 0.0 % | The accounts are US accounts |

The two news numbers are very different, and the cause is the classifier and
not the corpora. The headline corpus holds mostly `.com` hosts, and a `.com`
host of a foreign outlet (for example `indiatimes.com`) is caught only by the
name list. The 0.9 percent for the headline corpus is therefore a **lower
bound** and the 22.9 percent for the GKG corpus is the better estimate. The
project must not use "0.9 percent" as the foreign share of the news corpus.

The corpora hold **no geographic field for the audience**. GDELT gives a source
country for the publisher and not a region for the reader. The project
therefore cannot do the analysis by geography that section 5 of the brief asks
for. This is a missing-data limitation and not a decision.

### 3.2 Audience information

| Corpus | Audience data available |
|---|---|
| GKG strict | None |
| Headlines | None |
| Tweets | Views, likes, retweets, and the follower count of the account |

The tweet engagement values:

| Measure | Mean | Median |
|---|---|---|
| Views | 578 632 | 163 391 |
| Likes | 8 403 | 3 952 |
| Followers of the account | 1 223 429 | 446 941 |

No value is zero in the corpus, which is another effect of the `Top` selection.

---

## 4. Duplicates, reposts, advertisements and automated activity

### 4.1 Duplicate stories and reposts

| Corpus | Exact duplicate text | Duplicate after normalisation | Duplicate URL |
|---|---|---|---|
| Headlines | 31.2 % | **31.8 %** | 0.0 % |
| Tweets | 0.0 % | 0.0 % | not applicable |
| GKG strict | not measurable (no text) | not measurable | 0.0 % |

**Approximately one third of the headline corpus is a repeat of another
headline.** The URLs are all different, so a URL filter does not find these
rows. They are syndicated copy: one story that many outlets carry with the same
title. The largest examples:

| Normalised headline | Rows |
|---|---|
| latest articles | 65 |
| spielberg west side story debuts weakly with 10 5m | 57 |
| even in the amazon era black friday shows stores are alive | 51 |
| health care for older immigrants sees momentum among states | 51 |
| trump threatens canada with 100 tariffs over its new trade deal with china | 40 |

**Consequence.** A tone average over the raw headline corpus weights a
syndicated story by the number of outlets that carried it. That is a correct
weight for a question about supply, and it is a wrong weight for a question
about how many distinct stories were written. The stratified sample removes the
duplicates (11 080 rows), because an annotator must not label the same headline
50 times.

### 4.2 Advertisements, press releases and template content

The classifier `classify_source` separates these source types:

| Source type | GKG strict | Headlines |
|---|---|---|
| `wire_or_pr` (press releases: PR Newswire, GlobeNewswire, Business Wire and similar) | 1.6 % | 0.7 % |
| `algo_finance` (template market copy: MarketScreener, 4-traders, wkrb13, dailypolitical and similar) | **4.4 %** | 0.7 % |
| `aggregator` (MSN, Yahoo, Flipboard and similar) | 3.0 % | 6.4 % |
| `national_news` | 4.7 % | 7.9 % |
| `foreign` | 22.9 % | 0.9 % |
| `other` (local outlets and the long tail) | 63.5 % | 83.4 % |

**Suspected automated activity.** The 4.4 percent `algo_finance` share of the
GKG corpus is the clearest case. These outlets produce one article for each
price movement from a template. The text is written by a program. Such an
article is in the corpus because it holds an economic theme, and it has no
reader. The project keeps these rows and reports the tone by source type
separately, so a reader can see what the group contributes.

The project did **not** find a way to detect an advertisement inside a news
page, because the corpus holds no page body.

### 4.3 Source concentration, and a correction

The brief asks about source selection. An early note in the project said that
`iheart.com` dominates the headline set. **That statement is not correct.** The
measurements:

| Measurement | GKG strict | Headlines (host) | Headlines (registrable domain) |
|---|---|---|---|
| Distinct sources | 13 728 | 5 096 | 4 635 |
| Largest source share | 1.8 % (`yahoo.com`) | 2.0 % (`msn.com`) | 3.3 % (`yahoo.com`) |
| Top ten share | 9.0 % | 8.4 % | 11.0 % |
| Herfindahl index | 0.0016 | 0.0016 | — |
| `iheart.com` | 2 836 rows, **1.4 %**, rank 2 | 493 rows, 1.4 % | 493 rows, rank 3 |

No source dominates either news corpus. The Herfindahl index of 0.0016 is the
index of a very unconcentrated set. The long tail is large: 1 723 of the 5 096
headline hosts give exactly one row.

The iheart case is still real, and it is a lesson about the unit of counting.
In the GKG corpus, iHeartMedia appears as one source, `iheart.com`, and it is
the second largest. In the headline corpus, the same network appears as
hundreds of station hosts, such as `960weli.iheart.com` and `wspd.iheart.com`.
A count by host hides the network; a count by registrable domain shows it. This
document reports both.

---

## 5. Missing data, access restrictions and sampling limitations

### 5.1 Missing months

The headline corpus holds **95 of the 116 months** of the period. The 21 months
that have no headline are:

```
2017-06  2017-07  2017-08  2017-09  2018-04  2018-06  2018-12
2019-08  2019-11  2021-02  2021-05  2021-08  2022-01  2022-03
2022-05  2022-09  2022-11  2023-09  2023-12  2024-04  2024-08
```

The GKG corpus holds all 116 months. Thus the GKG corpus is the better source
for a continuous monthly series, and the headline corpus is the better source
for words. A monthly media series that pools the two sources covers all 116
months. The two sources agree at r = 0.609 on the framing measure, which is
positive and far from 1.0. The difference comes from the text that each one
gives: a full headline against a URL slug.

### 5.2 Thin months

A month is thin when its row count is in the lowest quarter of the months and
also below half of the median month. The GKG corpus has two thin months:

| Month | Rows | Median month |
|---|---|---|
| 2021-12 | 790 | 1 614.5 |
| 2025-06 | **368** | 1 614.5 |

**2025-06 is inside the monitoring window of the project.** A month with 368
records gives a less reliable monthly measure than a month with 1 614. Each
monthly series must carry the row count of the month, so that a reader can see
the precision of each point.

### 5.3 Access restrictions

| Restriction | Effect |
|---|---|
| GDELT serves metadata and not article bodies | No full text. The project measures headlines. |
| GDELT caps a query at 250 records | The corpus cannot measure volume. Refer to paragraph 2.2. |
| GDELT permits one request each five seconds | A larger term list or a day census is not practical. |
| The `gdeltdoc` library converts HTTP 429 into an exception with no message | A rate limit looks the same as a bad query. This cost three cycles of wrong diagnosis in the project. |
| twitterapi.io costs 300 credits for each request of 20 posts | 365 posts for 4 800 credits. 16 200 credits remain. |
| No historical platform access for other platforms | No Reddit, no Facebook, no TikTok. |

### 5.4 Other quality measures

| Measure | Headlines | Tweets |
|---|---|---|
| Rows with an empty title | 3 (0.008 %) | 0 |
| Rows with three words or fewer | 799 (2.2 %) | 0 |
| Rows with a mobile URL | 33.5 % | not applicable |

No row was removed from the corpus files. The sampler removes rows, and
`SampleReport` names the cause and the count of each removed row.

---

## 6. Content supply, engagement and exposure

The brief gives this instruction: "Distinguish content supply, engagement, and
actual exposure. Likes and reposts are imperfect attention measures; they do
not establish what survey respondents encountered."

The project has three different levels of evidence, and they must not be mixed:

| Level | What it means | What the project has |
|---|---|---|
| **Supply** | What publishers produced | **Partial.** The composition of coverage on 144 sampled days, with a cap of 250 records for each day. The project has no volume measure. |
| **Engagement** | What an audience did with the content | **Partial and for one corpus only.** Views, likes and retweets for the 365 tweets. Nothing for the 234 367 news records. |
| **Exposure** | What a survey respondent read, saw or heard | **None. Zero.** |

`CorpusProfile.has_exposure_data` is False for each corpus, and the field is a
constant False and not a calculation. Each profile also carries the note "no
exposure data: the project has no record of what a respondent read".

### 6.1 What the project must not claim

1. The project cannot say that a Michigan survey respondent read any item in
   these corpora. There is no link between the corpora and the survey
   respondents.
2. The project cannot use 578 632 mean views as an exposure measure. A view on
   the platform is not a view by a survey respondent, and the `Top` selection
   makes the views of the corpus unrepresentative.
3. The project cannot measure availability bias or confirmation bias from text.
   The brief states this: "Claims about confirmation or availability bias
   require corresponding exposure, recall, belief, or experimental evidence."
   The project has none of the four.
4. The project can measure the **framing** and the **selective coverage** of
   the supply side. That is what a text corpus can support.

### 6.2 What exposure data would be needed

To connect a media measure to the national Michigan index, the project would
need:

1. A measure of which outlets a respondent uses, by month.
2. Audience sizes by outlet and by region, so that supply can be weighted into
   exposure.
3. A sampling weight that makes the respondents representative of the US adult
   population.
4. An assumption of transportability from the sample to the population.

The project has none of the four. The analysis therefore reports associations
and predictive precedence, and it marks each causal statement with the strength
of its identification.

---

## 7. The stratified validation sample

The sampler is `draw_stratified_sample`. The instrument is in
`docs/labeling_guide.md`.

| Property | Value |
|---|---|
| Items | **359** (the brief asks for at least 300) |
| Source corpora | Headlines (345 items) and tweets (14 items) |
| Period of the items | 2017 to 2026 |
| Strata | 161, from year x source type x predicted tone class |
| Seed | 20261016 |
| Population rows | 36 365 |
| Eligible rows after filtering | 24 483 |

Rows that the sampler removed, with the cause:

| Cause | Rows |
|---|---|
| `duplicate_normalised_text[headlines]` | 11 080 |
| `short_text[headlines]` | 799 |
| `empty_text[headlines]` | 3 |

The composition of the sample:

| Source type | Items | | Predicted tone | Items |
|---|---|---|---|---|
| `other` | 209 | | `neutral` | 253 |
| `national_news` | 37 | | `positive` | 60 |
| `aggregator` | 34 | | `negative` | 46 |
| `foreign` | 25 | | | |
| `algo_finance` | 20 | | | |
| `wire_or_pr` | 20 | | | |
| `social` | 14 | | | |

### 7.1 The framing oversample

The framing lexicon is silent on most headlines. The measured coverage of the
corpus: 9.2 percent of headlines frame the price level against the inflation
rate, 2.5 percent frame the household against the aggregate, and 2.2 percent
frame the current condition against the forecast.

A proportional sample of 300 items therefore gives about 10 framed items for
two of the three contrasts, which cannot support an agreement number. The
sampler adds a second layer with a quota of 40 framed items for each contrast.
The layer balances the two sides of each contrast, so the added items cannot
make the framing of the corpus look one-sided.

| Layer | Items |
|---|---|
| `stratified` | 300 |
| `framing_oversample[household_vs_macro]` | 26 |
| `framing_oversample[current_vs_forecast]` | 25 |
| `framing_oversample[level_vs_rate]` | 8 |

Framed items in the sample after the oversample:

| Contrast | Left side | Right side | Balanced | Not framed |
|---|---|---|---|---|
| `level_vs_rate` | 30 `level` | 13 `rate` | 6 | 310 |
| `household_vs_macro` | 17 `household` | 14 `macro` | 4 | 324 |
| `current_vs_forecast` | 10 `current` | 23 `forecast` | 8 | 318 |

**The cost of the oversample, and it is recorded.** The sample is no longer a
proportional sample of the corpus. A prevalence estimate must use the 300 rows
where `sampling_layer` is `stratified`. An agreement number and a performance
number may use all 359 rows, because they condition on the item.

### 7.2 Files of the instrument

| File | Content |
|---|---|
| `outputs/labels/coding_sample_annotation.csv` | What an annotator receives: the item id, the order, the text and 16 empty columns |
| `outputs/labels/coding_sample_key.csv` | The analyst copy: the item id with the source, the date, the stratum, the layer and the model classes |
| `outputs/labels/coding_sample_strata.csv` | The population, the target and the drawn count of each stratum |

The annotation file does not hold the source, the date or the model class. The
cause is in the labeling guide, paragraph 1.1.

---

## 8. Validation results that exist now

### 8.1 Two automated labelers (not human labels)

| Measurement | Value |
|---|---|
| Items with two labels | 359 of 359 |
| Percent agreement (lexicon against FinBERT) | **0.557** |
| Cohen's kappa | **0.224** |

Agreement by source type:

| Source type | Items | Percent agreement | Kappa |
|---|---|---|---|
| `wire_or_pr` | 20 | 0.850 | 0.726 |
| `other` | 209 | 0.608 | 0.135 |
| `algo_finance` | 20 | 0.500 | 0.200 |
| `national_news` | 37 | 0.432 | 0.131 |
| `social` | 14 | 0.429 | 0.111 |
| `aggregator` | 34 | 0.412 | 0.124 |
| `foreign` | 25 | 0.400 | 0.118 |

The two instruments agree about press-release copy and disagree about national
news. Press-release copy uses a small and repeated vocabulary, and national
news does not.

**These are model labels. They are not the two human annotators of the brief.**
Each report marks them as automated.

### 8.2 The name-swap and source-label robustness test

The test takes the real items, exchanges the political names (Trump against
Biden, Republican against Democrat and four more pairs), and scores the items
again. The source-label test puts an outlet name in front of the text and
exchanges the name. **The text is identical in the two source-label
conditions.**

| Dimension | Classifier | Swap | Eligible items | Classifications that change |
|---|---|---|---|---|
| Tone | Lexicon | Political name | 55 | **0 (0.0 %)** |
| Tone | FinBERT | Political name | 55 | **4 (7.3 %)** |
| Tone | Lexicon | Source label | 359 | **0 (0.0 %)** |
| Tone | FinBERT | Source label | 359 | **21 (5.8 %)** |
| Framing: level/rate | Framing lexicon | Political name | 55 | 0 (0.0 %) |
| Framing: household/macro | Framing lexicon | Political name | 55 | 0 (0.0 %) |
| Framing: current/forecast | Framing lexicon | Political name | 55 | 0 (0.0 %) |
| Framing: all three | Framing lexicon | Source label | 359 | 0 (0.0 %) |

Examples of a FinBERT flip. The text is the same in each pair, and only the
name in front of it is different:

| Text | Class | Class after the swap |
|---|---|---|
| `Breitbart:` Idaho inflation rate hit 7.7 percent in January | positive | negative |
| `Breitbart:` Whitmer seeks to make more workers eligible for overtime | positive | neutral |
| Trump threatens 100 % tariff on Canada if it makes a deal with China | neutral | negative |

**The dictionary wins this comparison.** The lexicon is invariant to a
political name and to a source label, because it holds no political name and no
outlet name. FinBERT changes the tone class of about 6 percent of identical
texts when the outlet label changes. For a project that measures political
media bias, a classifier that reads the outlet label is not usable: it would
find bias that it brought with it.

This result is a measurement of **robustness** and not of **accuracy**. Neither
instrument can be scored for accuracy until the human labels exist. A
classifier can be robust and wrong.

### 8.3 The benchmark: the same event across sources

The brief asks for the benchmark against which each proposed bias is measured.
The benchmark here is the **mean tone of all items about one event**. The bias
of a source type is the mean of its deviation from that benchmark, over the
events that it covered. The construction holds the event constant, so a source
that covers worse events cannot look negative for that cause.

Two event keys were run.

**Key 1: the same normalised headline. This is a control.**

| Measurement | Value |
|---|---|
| Events with two or more source types | 661 |
| Items in those events | 3 642 |
| Mean tone range inside an event | **0.000** |
| Events where the source types disagree about the class | **0.0 %** |

The text is identical inside an event, so a deviation of zero is the correct
result. A number other than zero would show a defect in the scorer. This key
finds syndicated copy and it cannot find a difference of wording.

**Key 2: the same calendar day and the same dominant topic. This is the
measurement.**

| Measurement | Value |
|---|---|
| Events with two or more source types | 280 |
| Items in those events | 2 169 |
| Median items for one event | 5 |
| Mean tone range inside an event | **0.493** |
| Events where the source types disagree about the class | **45.0 %** |

Deviation from the benchmark, by source type:

| Source type | Events | Items | Mean deviation | Mean tone |
|---|---|---|---|---|
| `wire_or_pr` | 13 | 20 | −0.030 | +0.550 |
| `national_news` | 168 | 247 | −0.022 | −0.073 |
| `other` | 265 | 1 647 | +0.002 | −0.042 |
| `aggregator` | 143 | 198 | +0.005 | +0.086 |
| `foreign` | 23 | 26 | +0.040 | +0.269 |
| `algo_finance` | 23 | 31 | +0.044 | +0.065 |

**The result is a near-null, and it is a result.** The tone of the corpus
varies a great deal inside one day and one topic: the mean tone range is 0.493
on a scale from −1 to +1, and the source types disagree about the class in 45
percent of the events. But the **systematic** difference between source types
is very small. The largest mean deviation is 0.044 on that scale. When the day
and the topic are held constant, a source type is not reliably more negative
than another source type. The large unconditional difference in the last column
(national news −0.073 against foreign +0.269) comes mostly from **which events
each source type covers** and not from how it describes them.

This is evidence against a simple source-level negativity bias in this corpus.
It agrees with the conclusion of `BRAINSTORMING.md` paragraph 8 item 4.

**The limits of this key.** Two items on one day about prices need not be about
one release. The key is weaker than a release-level key. A release-level
benchmark needs the release calendar joined to the corpus, and that work is not
done.

---

## 9. What is still outstanding

| No. | Item of the brief | Status |
|---|---|---|
| 1 | Human-label a stratified sample of at least 300 items | **Not done.** The 359-item instrument is complete and empty. |
| 2 | Two independent annotators | **Not done.** No person is assigned. |
| 3 | Report agreement | **Not done.** The code exists and it is tested. The numbers need the labels. |
| 4 | Adjudication procedure | Written and coded. The log needs a third person. |
| 5 | AI performance by class and by source | **Not done.** This needs a human standard. The table that exists is concordance between two models. |
| 6 | Compare the AI with a dictionary | Partly done. Robustness is measured and the dictionary wins. Accuracy needs the labels. |
| 7 | Validate the framing measure | **Not done, and it has no model counterpart.** FinBERT gives market sentiment from one output head and it cannot give a framing contrast. The only validation of the framing lexicon is the human labels. |
| 8 | Name-swap and source-label test | **Done.** Refer to paragraph 8.2. |
| 9 | Define the benchmark | **Done.** Refer to paragraph 8.3. |
| 10 | Volume or attention measure for news | **Not possible** with this corpus. Refer to paragraph 2.2. |
| 11 | Exposure data | **None.** Refer to paragraph 6. |
| 12 | Analysis by geography | **Not possible.** The corpora hold no audience region. |

The report must state item 1, item 5 and item 7 as limitations. The project
must not present a model label as a human label. The function
`require_human_labels` raises an exception if a caller tries.
