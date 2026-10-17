# Research ethics, limitations and what we do not claim

**Why this document exists.** The assignment weights *"Communication,
limitations, and research ethics"* at 10%, and it sets conditions: *"Use public
or appropriately licensed data, protect participant privacy, and obtain required
institutional approval before human-subject research."*

**Language.** ASD-STE100 Simplified Technical English.

---

## 1. Ethics

### 1.1 Human subjects

**No human-subject research was done.** No participant was recruited. No person
was surveyed, interviewed or assigned to a condition. So no institutional
approval was required, and none was obtained.

The randomized framing experiment in BRAINSTORMING.md §7.4 is a **written
proposal and nothing more**. It is included because the brief asks for a causal
design, and it states the approval it would need before anybody ran it:
institutional review, informed consent, a debrief that corrects any framing
effect the study created, and a rule that every summary shown to a participant
is factually true. The proposal uses *alternative truthful framings* of the same
information, which is what the brief specifies. It does not use false
information, and a design that misinforms a participant about the economy would
be refused.

### 1.2 Privacy

* The social-media records come from public accounts.
* **No individual is named in any output.** The report, the dashboard and the
  deck contain no author name, no handle, no user id and no author-level
  statistic.
* **The published label artifacts are pseudonymised.** The stratified sample
  drew 14 tweets, and the key file originally carried each account's handle
  beside a political statement. Several of those accounts are private
  individuals, so each handle became a stable, non-reversible pseudonym
  (`account_<hash>`) and a mention inside an item's text became `@[handle]`.
  The handle-to-pseudonym mapping is written to `outputs/labels_private/`,
  which git ignores. Nothing in the analysis needs the handle: agreement,
  concordance and the benchmark are all computed by **source type**. The step
  is `scripts/redact_label_artifacts.py`, so it is auditable and repeatable.
  This was found while preparing the repository to be made public, which is the
  point at which a privacy claim stops being theoretical.
* The corpus is aggregated to monthly series before it enters any artifact.
* No attempt was made to identify, profile or locate any author.
* No personal data was sent to any third-party service.

### 1.3 Data licensing and provenance

| Source | Status | How it is used |
|---|---|---|
| FRED (BLS, BEA, Federal Reserve, Freddie Mac series) | Public | Series are fetched by their official ids and the publication lags are recorded. |
| GDELT GKG 2.0 and DOC API | Public | Theme tags, tone vectors, titles and URLs. |
| twitterapi.io | Paid, used under its terms | 365 tweets. |
| The issued course CSV | Given for the assignment | The primary sentiment and macro series. |

**No paywalled article text was scraped.** The corpus holds titles, URLs and
GDELT-derived metadata. Where a measure needs words and only a URL exists, the
URL path is used, and the report says so rather than implying that article
bodies were read.

### 1.4 AI use

`AI_USE.md` records what the AI did and what the operator did. Model versions,
prompts and evidence excerpts are preserved, as the brief requires. The AI wrote
code, ran analysis and drafted documents. It also made the errors listed in §3,
and those are recorded with the same detail as the results.

---

## 2. Limitations, ordered by how much they would change a conclusion

### 2.1 No exposure data at all

Nothing in this project measures what a Michigan respondent read. We measure the
**supply** of coverage. We have weak engagement data for the tweets and none for
the news.

**What this costs.** Every statement about the effect of coverage on a reader is
unavailable. This is the reason the causal design is observational, and it is the
reason three of the perceptual mechanisms in the brief cannot be measured at all.

### 2.2 One annotator, not two

The brief asks for two independent annotators on at least 300 items. This project
has one operator.

**What this costs.** A true inter-annotator agreement statistic is not available.
Whatever second label track exists is described as exactly what it is. **It is
never described as a second human annotator.** A fabricated agreement figure
would be the most serious failure available in this project, and it is ruled out
in advance rather than resisted in the moment.

### 2.3 Titles and URL slugs, not article bodies

A title measures emphasis. It cannot measure a qualification inside a paragraph.
The headline "Inflation jumps to 3.8%" and an article that explains that 3.8% is
close to normal are the same text to this method.

**What this costs.** The factual-calibration dimension is weaker than it should
be, because a correction that appears in paragraph four is invisible here.

### 2.4 The Michigan sub-indices are not available

FRED does not carry the Index of Current Economic Conditions or the Index of
Consumer Expectations. Both are distributed by Michigan under its own terms.

**What this costs.** We cannot separate "how things are" from "where things are
going". That split would discriminate between the hypotheses directly, because a
media effect should appear first in expectations.

### 2.5 Power in the monitoring window

The holdout has 20 monthly gap values. A VAR with lags, and a two-regime
covariance estimator, consume degrees of freedom quickly. The modules refuse to
compute these statistics at that sample size and return a warning instead of a
number.

**What this costs.** The media models are estimated on the long window, 2017 to
2026, and the pre-election window is shown separately with its power stated. A
reader who wants a causal estimate from the pre-election window alone cannot have
one from these data.

### 2.6 The corpus is a sample, not a census

Two 15-minute GKG files for each sampled day. 36,000 titles. 365 tweets.

**What this costs.** A month carried by one syndicator can look like broad
coverage. The dashboard reports the source composition for each month so that
such a month is visible.

### 2.7 No partisan, income, age or geographic breakdown

The brief asks us to examine differences by income, age, geography and political
affiliation *"where data permit"*. Michigan does not publish those breakdowns
free of charge.

**What this costs.** Partisanship is a documented moderator of economic
perception, and it is also a plausible confounder. We can neither control for it
nor use it. The dashboard says that this panel is unavailable, and why. It does
not show an empty chart and it does not substitute a proxy.

### 2.8 One lexicon, one operator, one language

The lexicon is ours. It was written before the measurement, which protects
against tuning, but it was written by one person and it is in English only.

---

## 3. Errors we made, and what each one cost

A limitations section that lists only external constraints is incomplete. These
are our own errors. Each one was found, each one is fixed, and two of them
changed a conclusion.

| The error | What it did | How it was found | Status |
|---|---|---|---|
| **The `TAX_` theme filter.** We read the GDELT prefix `TAX_` as taxation. It is a general taxonomy of actors and languages. | 693,347 articles with a mean tone of −0.988 instead of 198,367 with −0.180. It would have given Hypothesis B a **false positive** with a large sample behind it. | A check of what the filter actually matched. | Fixed. The strict filter is in `scripts/build_media_series.py` with the numbers in its docstring. |
| **Look-ahead in the panel.** Month *t* carried values published in month *t+1*. | A negative and **significant** GDP coefficient (−0.159, p=0.039) that collapses to −0.030, p=0.524 under the compliant alignment. A finding that was an artefact. | The real brief's timing rule. | Fixed. Real-time alignment is the default. |
| **Interpolated quarterly GDP.** One quarterly value was spread across three months. | Three observations from one measurement. | The real brief's explicit prohibition. | Fixed. A step function of published estimates. |
| **A scaler fitted outside the cross-validation folds.** | Each fold saw the statistics of its own validation data. | A code review. | Fixed with a `Pipeline` inside each fold. |
| **A circular comparison series.** The independent series for Hypothesis C was partly derived from Michigan itself. | It would have compared Michigan against Michigan and found agreement. | Caught before it reached the report. | Removed. |
| **A regime rule that included the current month.** The rolling variance window contained the month being classified. | The rank condition "passed" on pure noise. | A synthetic homoskedastic control case that should have failed and did not. | Fixed. The window stops at the previous month. |
| **Reporting two dashboard panels as live when they were not.** We checked the helper that decides availability instead of the page a reader opens. | A false statement about the state of the work. | A reviewer checked the render path. | Fixed, and drift tests now compare the claim against the rendered page. |
| **A quarterly alignment error in the first panel.** Forward-fill put Q1 on March to May. | It changed the GDP p-value from 0.518 to 0.039 and therefore changed the written conclusion. | A code review. | Fixed, and then superseded by the real-time alignment. |

**The pattern behind three of these.** The check was run against the thing that
decides a claim, and not against the thing a reader sees. That is now a rule: a
claim about an artifact is verified on the artifact.

---

## 4. What we do not claim

This list is the short version of the whole document.

1. We do not claim that consumer pessimism is irrational.
2. We do not claim that negative coverage is biased coverage. Coverage can be
   negative because the situation is bad, and the factual-calibration dimension
   exists to separate the two.
3. We do not claim that media exposure causes declining sentiment. We have no
   exposure data.
4. We do not claim a causal effect on the national Michigan index from any
   sample result. The transport assumptions in BRAINSTORMING.md §7.6 are not
   met.
5. We do not claim that the election proximity caused any change. A change that
   happens during a campaign is not evidence about the campaign without a
   counterfactual.
6. We do not claim to measure confirmation bias, availability bias, or anchoring
   inside a reader's head. Those need belief, recall or experimental data.
7. We do not claim an inter-annotator agreement statistic.
8. We do not claim that a Granger test shows causation. It shows predictive
   precedence.
9. We do not claim identification of the reverse direction in the
   heteroskedasticity instrument. It is not identified, and it is not reported.
