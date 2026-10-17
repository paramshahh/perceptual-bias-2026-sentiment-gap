> # ⚠ ARCHIVED — EVERY NUMBER IN THIS FILE IS SUPERSEDED
>
> This is version 1, written 2026-10-14 to 2026-10-15 against a **reconstruction**
> of the assignment. The real brief arrived on 2026-10-16 and two of the
> decisions here break its timing rules, so the panel behind these numbers used
> data that did not exist on the prediction date.
>
> **The current document is `BRAINSTORMING.md` in the repository root.** This
> file is kept only so that a reader can see what changed and how much the
> correction moved. Do not quote a number from it.
>
> The largest single change: the GDP coefficient here is −0.159 with p = 0.039
> and is discussed as a real effect. Under the compliant alignment it is −0.030
> with p = 0.524. The finding was an artefact of look-ahead.

# BRAINSTORMING — AI, Economic Perceptions and Consumer Sentiment Before the 2026 Midterm Elections

This is Step 0 of the project. We state the hypotheses before we build the
pipeline.

This document uses ASD-STE100 Simplified Technical English.

Every number in this document comes from the issued data. The modules
`data_loader.py`, `baseline_model.py`, `causal_econometrics.py` and
`survey_diagnostics.py` calculate these numbers. The command `pytest` gives
the numbers again. When a number is an approximation, this document says so in
the text.

Date of the last change: 2026-10-15

---

## 1. The baseline model and the Sentiment Gap

### 1.1 The specification

The baseline is the first model that an economist selects:

$$\text{Sentiment}_t = \beta_0 + \beta_1\,\text{Unemployment}_t + \beta_2\,\text{PCE\_YoY}_t + \beta_3\,\text{RealGDP}_t + \varepsilon_t$$

The model uses 180 monthly values from 2010-01 to 2024-12. The model uses
Newey-West HAC standard errors with 12 lags. Monthly macro data has strong
autocorrelation. Classical standard errors are too small for this data. Each
t-value is then too large.

$$\widehat{\text{Sentiment}} = 127.56 - 4.28\,\text{Unemployment} - 9.77\,\text{PCE\_YoY} - 0.16\,\text{RealGDP}$$

| Statistic | Value |
|---|---|
| R² | 0.766 |
| Adjusted R² | 0.762 |
| RMSE | 6.20 |
| AIC | 1175.5 |

| Regressor | Coefficient | p-value |
|---|---|---|
| Unemployment | −4.28 | <0.001 |
| PCE YoY | −9.77 | <0.001 |
| Real GDP QoQ | −0.159 | 0.039 |

**Make a note of two conditions before you use this model.**

The GDP coefficient is negative and significant. A negative sign is not
correct. More growth must not give less sentiment. The probable cause is this:
in the window 2010 to 2024, high growth occurs after a shock (in 2010 and in
2021). The sentiment is still low at these times. Thus the coefficient
measures the shock and not the growth. This is a warning about the
specification. It is not a result.

The GDP coefficient changes with the alignment of the data. The CSV file gives
the GDP value at the end of the quarter. You must put this value in the three
months of the same quarter. If you use a forward fill, the value of quarter 1
moves to March, April and May. Then January and February hold the value of the
previous quarter. With that error the same coefficient is −0.037 with a p-value
of 0.518. The same data and the same specification then give the opposite
conclusion. Refer to the function `expand_quarterly` and to its tests.

### 1.2 The Sentiment Gap

$$\text{Gap}_t = \text{Observed Sentiment}_t - \widehat{\text{Sentiment}}_t$$

The model does not see the data after 2024-12.

| Measurement | Value |
|---|---|
| Months with a gap value | 19 |
| Mean gap | **−25.24** |
| Median gap | −25.50 |
| Maximum gap | **−33.13** at 2026-05 |
| Observed value at 2026-05 | **44.8** |
| Predicted value at 2026-05 | **77.93** |
| Months out of the 95% prediction interval | **18 of 19** |

These values agree with the project brief. The brief gives 77.8 and −33.0.

### 1.3 The model failure is a statistical result

A difference of 33 points is only a failure if it is large for this model. It
is large. 18 months of 19 months are out of the 95% prediction interval. This
interval contains the residual variance. It is not only the uncertainty of the
mean.

The direction is also important. All 19 errors are negative. The sentiment is
not more variable than the model expects. The sentiment is lower each month.
Thus an explanation is necessary.

### 1.4 Three months have no gap value

The CSV file stops at a different date for each series:

- 2025-10. The unemployment value is not available.
- 2026-09. The PCE YoY value is not available.
- 2026-10. Three values are not available.

The gap series stops at 2026-08. The sentiment column continues to 2026-10 with
the values 48.1 and 46.3. A report with the title "the 2026 Sentiment Gap" must
not stop in August without a statement. The class `PredictionReport` gives the
name of each month that the system removes, and the cause.

---

## 2. The three hypotheses

The gap is the quantity to explain. Each hypothesis below has a **test** and a
**falsification condition**. We write the falsification condition before we see
the result. Thus we cannot move the conclusion after the analysis.

### 2.1 Hypothesis A. Economic pressure that the baseline does not measure

**The statement.** The baseline measures the **rate of change** of the economy.
A household feels the **level**. An inflation rate of 3% does not decrease a
price. A household that remembers the prices of 2020 feels more damage, not
less damage.

**The measurement.** The cumulative PCE price level is **25.7% more** than the
mean level of 2020. The 30-year mortgage rate is **7.36%**. The premise of
Hypothesis A is correct in this data.

**The result.** Add the price level and the mortgage rate to the baseline:

| Model | R² | AIC | Mean gap | Gap at 2026-05 |
|---|---|---|---|---|
| Model 1 | 0.766 | 1175.5 | −25.24 | −33.13 |
| Model 2 | **0.835** | **1117.4** | **−17.40** | −25.58 |

**Hypothesis A explains 31% of the gap.** A gap of −17.40 stays.

The coefficients of Model 2 are important, and they are not simple:

| Regressor | Coefficient | p-value |
|---|---|---|
| Unemployment | −4.687 | <0.001 |
| PCE YoY | −7.942 | <0.001 |
| Real GDP QoQ | −0.210 | <0.001 |
| Price level | −0.144 | **0.310** |
| Mortgage rate | **−2.936** | **<0.001** |

The mortgage rate is strong. The price level is **not significant**.

Collinearity is one cause, but it is not a complete explanation. The price
level and the PCE YoY rate measure the same process at different speeds. Their
correlation in the training window is **0.758**. The price level also
correlates 0.632 with the mortgage rate and −0.624 with the unemployment rate.

But the variance inflation factor of the price level is only **3.73**. The
usual limit for a problem is 5 or 10. Thus collinearity alone does not explain
a p-value of 0.310. A second cause is probable: the mortgage rate carries the
same information about the cost of living, and it carries it better.

The significance of the price level also changes with the specification. With
the debt service ratio in the model, the price level has a p-value of 0.038.
**A coefficient that moves from 0.038 to 0.310 when you add or remove a third
variable is not a stable estimate. Do not give it a causal reading.**

**The honest statement is this.** The variables of Hypothesis A improve the
model and decrease the gap by approximately one third. The mortgage rate does
the measurable work. This data cannot show the separate effect of the price
level.

**A decision about the debt service ratio.** The FRED series `TDSP` stops at
2026-04. The quarterly expansion extends it to 2026-06, so the holdout window
has 17 months instead of 19. The variable is not significant (p = 0.256).

We first compared 40% against 31% across **different samples**, which is not a
correct comparison. On the same 17 months the result is **39% against 31%**.
The advantage of the debt service ratio is 2.1 points of gap. It is real, and
it is smaller than the first comparison showed.

We removed the variable from Model 2. We keep it as a robustness test. The
report must show both results and must state the sample of each.

**The falsification condition.** If the extended model does not decrease the
gap, and if no added regressor is significant, then Hypothesis A fails.
**Result: Hypothesis A is supported, but only in part.**

### 2.2 Hypothesis B. Selection, framing and media bias

**The statement.** A household does not read a database. A household reads the
news. Sacerdote et al. (2020) show that coverage of the economy has a negative
bias, because negative text gets attention. Before an election, partisan text
increases this effect.

**The measurement.** We use a corpus of news from GDELT. We score the text
across four dimensions. We use a truncated SVD to find the latent topics. The
tone index is:

$$S_{\text{media},t} = \frac{\text{Pos}_t - \text{Neg}_t}{\text{Total}_t} \in [-1, +1]$$

**The tests.**
1. The tone index is significant in Model 3, and Model 3 decreases the gap more
   than Model 2.
2. The tone of an earlier month gives information about the gap of a later
   month (a Granger test).
3. The Rigobon-Sack IV gives a coefficient with the same sign as the OLS
   coefficient.

**The falsification conditions.** Any one of these conditions rejects
Hypothesis B:
- The tone index is not significant in Model 3.
- The Granger test shows that the gap gives information about the tone, and not
  the opposite. This is the condition for reverse causality.
- The IV coefficient changes sign, or it decreases to zero.
- **The calibration test shows that the coverage was correct.** This condition
  is the most important. If the press described a 3% inflation economy as a 3%
  inflation economy, then there is no bias, and the narrative fails. The tone
  alone cannot divide "the coverage is too negative" from "the economy is bad".
  The calibration can.

**Status: tested. The hypothesis fails.**

The corpus holds 198 367 economic articles over 116 months, from 2017-01 to
2026-08. The three tests of this paragraph give:

| Test | Result | Decision |
|---|---|---|
| `media_tone` in Model 3 | coefficient +6.79, p = 0.570 | Not significant |
| `topic_salience` in Model 3 | coefficient −11.56, p = 0.679 | Not significant |
| Granger, tone to gap | F = 0.441, p = 0.779 | No precedence |
| Granger, gap to tone | F = 0.501, p = 0.735 | No precedence |
| Rigobon-Sack IV | The regime split gives a variance ratio of 2.05, which the bootstrap cannot separate from no change | Not identified. No estimate. |

The first falsification condition of this paragraph says: if the tone index is
not significant in Model 3, Hypothesis B fails. The condition is met.

**One result stays.** `political_attribution` has a coefficient of −34.30 with
a p-value of 0.030. A month in which economic coverage carries a political
frame has a lower sentiment. That variable also fails the Granger test in both
directions, so it is an association. It is not predictive precedence, and it
is not an identified cause.

Model 3 adds 0.008 of R-squared over Model 2 on the same window. The economic
variables do the work.

### 2.3 Hypothesis C. The survey method changed

**The statement.** A part of the gap is an error of the instrument. The
University of Michigan survey moved to the internet between 2024 and 2026. A
change of collection method changes the answers.

**Why we must test this hypothesis first.** Hypothesis A and Hypothesis B
explain a true quantity. Hypothesis C says that the quantity is too large. If
Hypothesis C is strong, then A and B compete to explain a number that is not
correct.

**The results.**

| Test | Result |
|---|---|
| Chow test at the specified date 2024-07 | F(1,581) = 81.3, p < 1e-16, level shift −35.50 |
| Supremum-Wald test, unknown date | Break at **1990-07**, supF = 1234, bootstrap p = 0.001 |
| Cross-survey comparison, 19 common months | Michigan −1.963 against comparison −0.507, **ratio 0.258** |
| Shape of the gap | The level part is 4.7 times the month-to-month part |

**Verdict: partially supported.** The threshold for the verdict "supported"
is 0.25 and the ratio is 0.258, so the verdict rests on 0.008. Always give
the ratio with the threshold. Never give the word alone.

**A trap in the comparison, and how we avoided it.** The obvious comparison
series is the OECD consumer confidence indicator for the United States, FRED
`USACSCICP02STSAM`. **This series correlates 1.0000 with the Michigan survey
after 2010, in levels and in differences.** The cause: the OECD consumer input
for the United States **is** the Michigan survey. A test against this series
gives a gap ratio of 1.00 and "rejects" Hypothesis C with Michigan's own
numbers. This looks like a clean result and it is circular. An earlier version
of this document gave a series from the same family as the plan. The module
`survey_diagnostics.py` now screens each candidate series automatically.

The series `CSCICP03USM665S` stops at 2024-01. It covers no month of the
holdout window.

We used `BSCICP02USM460S` instead. This is an OECD business survey for United
States manufacturing. Its correlation with Michigan is 0.36. It covers the
complete window. Firms answer this survey, so a change in a *consumer* survey
cannot affect it. **The cost: it measures business confidence, not consumer
confidence. This is a different quantity.**

**There is no free, independent consumer confidence series for the United
States that covers 2025 and 2026.** The Conference Board index has a licence.
The report must give this limitation.

**Our conclusion.** Hypothesis C is true but secondary. The ratio of 0.258 shows
that an independent instrument also finds pessimism in 2025 and 2026, at
approximately one quarter of the size. The gap is not mostly an artefact. Two
cautions:

1. The Chow test cannot separate a change of method in 2024-07 from any other
   event of mid-2024, for example the election cycle. A level shift of −35.5
   points is also too large for a change of method alone.
2. The supremum-Wald test finds its break in 1990-07. The 2024 transition is at
   95% of the sample, so no usual trim can examine it. "No break found there"
   is therefore not evidence against Hypothesis C.

### 2.4 How we separate the three hypotheses

The three hypotheses are not exclusive. The probable result is a division of
the gap.

| Evidence | Supports |
|---|---|
| The price level, the mortgage rate and the debt data decrease the gap | A |
| The tone is significant, gives earlier information, and survives the IV | B |
| A break occurs at the change of method, and an independent survey shows no gap | C |
| The gap stays after all three sets of controls | None. The specification fails. Report this. |

The last row is a possible result. We will report it if it occurs.

---

## 3. LSA with a truncated SVD, and why we keep the mean

### 3.1 The method

Let $X \in \mathbb{R}^{m \times n}$ be the TF-IDF term-document matrix. It has
$m$ terms and $n$ documents. The truncated SVD keeps the $k$ largest singular
values:

$$X \approx U_k S_k V_k^\top$$

The columns of $U_k$ are the latent topics in term space. The rows of $V_k$
give the position of each document. The matrix $S_k$ gives the size of each
topic. We use `scipy.sparse.linalg.svds`, which needs only sparse
matrix-vector products.

### 3.2 Why we do not remove the mean

**Sparsity.** A TF-IDF matrix has more than 99% zero values. The function
`svds` costs $O(\text{nnz})$ for each step. If you remove a mean, each zero
becomes a non-zero value. The storage increases from $O(\text{nnz})$ to
$O(mn)$. Then the decomposition is not possible at this size.

**The zero-prior property.** This is the more important cause. In a
non-centred decomposition, a document with no dictionary terms maps to the zero
vector. Its projection is zero. It adds nothing. This is the correct result: no
signal is not the same as an average signal. If you remove the mean, this
document gets the negative of the mean vector. The document then has a position
that it did not earn. If you add the daily values, the sum moves with the
quantity of off-topic documents on that day. The origin must mean "no terms
are present". The tests hold this property.

**It is a different decomposition, not a worse one.** A centred SVD is PCA. It
uses the covariance matrix and finds the directions of maximum variation about
the mean. A non-centred SVD uses the second-moment matrix $X^\top X$ and finds
the directions of maximum co-occurrence. Latent semantic analysis wants
co-occurrence. We measure co-occurrence from zero.

### 3.3 How we select $k$

We report three diagnostics together:

1. The spectrum of the singular values, and its elbow.
2. The cumulative energy $\sum_{i\le k} \sigma_i^2 / \sum_i \sigma_i^2$.
3. A noise floor from permutation. Permute the values in each column, do the
   decomposition again, and keep the components above the permuted maximum.

**A correction to this paragraph.** The permutation floor is too strict on a
dense non-negative matrix. The matrix $X$ is non-negative and not centred, so a
large part of $\sigma_1$ is the mean direction. A permutation inside a column
keeps that direction. Thus the permuted $\sigma_1$ can be larger than the true
$\sigma_1$, and the method gives $k = 0$ with real topics present. A test
measured this: a matrix with 3 true topics at 33% density gives real values of
11.66, 11.60 and 11.57 against a permuted maximum of 11.93. At the sparsity of
a true TF-IDF matrix (less than 10%) the effect is not present.

We therefore keep the permutation floor and we add a component-wise parallel
analysis. The module gives a note when the two methods do not agree.

### 3.4 Selection of signals with the SVD

Let $s \in \mathbb{R}^{T \times K}$ be a matrix of $K$ text signals over $T$
periods. Let $\delta y$ be the forward change of the target. The
cross-covariance is:

$$C = \frac{1}{T}\, s^\top \delta y$$

Its decomposition $C = U_k S_k V_k^\top$ gives the combinations of signals with
the most covariance with the target. The truncation removes the directions of
text noise.

**A trap.** The matrix $C$ contains $\delta y$, which is the quantity to
predict. You must calculate $C$ on the training window only. If you use the
complete sample and then report the holdout result, the transform contains the
answer. This project found the same class of error in its own code: a scaler
was fitted before the cross-validation, so each fold saw the statistics of its
own validation data.

---

## 4. Domain lexicons against FinBERT

### 4.1 Why FinBERT fails on macro policy text

FinBERT learns from market text. The question in that text is "is this good for
the asset?". Macro policy text asks a different question. The two answers often
have opposite signs.

The example: *"Strong productivity growth and robust labour market conditions
continued through the quarter."* FinBERT gives a strong positive score. In
policy language, strong growth and a tight labour market **give a higher
interest rate**. The sentence is hawkish. The model reads the words correctly.
The model answers a question that nobody asked.

### 4.2 We measured the failure

We did not quote the failure. We made a probe set of 30 macro sentences with
10 hawkish, 10 dovish and 10 neutral sentences. We scored them with FinBERT.

| Measurement | Value |
|---|---|
| Accuracy | **0.300** |
| Accuracy of a random guess | 0.333 |
| Hawkish sentences that FinBERT calls dovish | 6 of 10 |
| Dovish sentences that FinBERT calls hawkish | 7 of 10 |

**FinBERT is worse than a random guess on this text, and it inverts the two
important classes.** This is the inversion that paragraph 4.1 predicts.

### 4.3 The four dimensions of the lexicon

| Dimension | Question | Method |
|---|---|---|
| Negative and positive language | How does the text describe the economy? | A domain lexicon with negation control |
| Selection and salience | Which facts get coverage? | Daily topic shares from $V_k$ |
| Factual calibration | Is the statement correct? | Compare the statement with the macro data |
| Political attribution | Who gets the blame or the credit? | Attribution patterns near partisan words |

The calibration dimension makes Hypothesis B falsifiable. The tone alone cannot
divide "the coverage has a negative bias" from "the coverage describes a bad
condition". The calibration can.

---

## 5. Identification and statistical power

**This paragraph gives the weakest part of the project.**

**The gap has 19 monthly values.** The Rigobon-Sack method divides the sample
into two regimes and estimates a covariance matrix in each regime. A VAR with
lags uses many degrees of freedom. These methods on 19 values give output that
looks like a result and is not one. The modules refuse to calculate these
statistics at that sample size. They give a warning instead of a number.

**The correction.** We estimate the media models on a long window. The GDELT
corpus covers 2017 to 2026, which gives approximately 116 months. We then show
the window 2025 to 2026 separately, with a statement about its power.

### 5.1 The conditions of the Rigobon-Sack method

**The order condition in the brief is not useful for this system.** The brief
gives $N^2 - N - 2K > 0$. With $N = 2$ this becomes $2 - 2K$. The condition is
satisfied only when $K = 0$, and it fails with any exogenous regressor. The
module reports the value of this formula, so that the report can discuss it.

**The system is under-identified by one restriction.** Two regimes give six
second moments against six parameters. This looks sufficient and it is not.
All the identifying information is in $\Omega_H - \Omega_L$, which gives three
equations with four unknown values. We therefore estimate the direction
media → gap only. We use the Rigobon-Sack assumption: the regime change moves
the variance of the media shock only. **The reverse direction is not
identified. We do not report it.** This assumption is the weakest part of the
method, and two covariance matrices cannot test it.

**The rank condition in the brief does not agree with the assumption.** Under
the assumption above, the matrix $\Omega_H - \Omega_L$ has rank 1 in the
population. Its determinant is then **exactly zero**. A determinant near zero
is therefore in agreement with identification, and it is not a failure. We
calculate the determinant and its bootstrap distribution, because the brief
asks for them. But the operative test is different: the shift of the media
variance must be different from zero in the bootstrap. A test on synthetic data
confirms this design. With a true coefficient of 0.60, the method gives 0.598.
With no change of variance, the method gives no estimate.

**A defect that we found and corrected.** The first regime rule used a rolling
variance window that contains the current month. That rule sorts the months by
their own deviation. The rank condition then "passes" on pure noise. The window
now stops at the previous month. The homoskedastic test now fails correctly.
The cost is a smaller measured variance ratio, because the lag puts some months
in the wrong regime.

### 5.2 Granger tests

A Granger test measures **predictive precedence**. It does not measure
causation. The modules give this statement with each result. A common cause,
for example the election cycle, can move the coverage and the sentiment
together. Then the earlier coverage gives information about the later gap with
no causal connection. This is the reason for the IV.

---

## 6. The data sources and their limits

| Source | Use | Limit |
|---|---|---|
| The issued CSV file | The primary data | Three months have no complete data. |
| FRED | The Hypothesis A variables | `TDSP` stops at 2026-04. |
| GDELT | The news corpus | **The API gives headlines, not article text.** |
| twitterapi.io | A comparison corpus | 365 tweets. The budget is almost used. |

**The GDELT limit is important for Hypothesis B.** A corpus of headlines
measures emphasis. It cannot measure a qualification inside a paragraph. The
headline "Inflation jumps to 3.8%" and an article that explains that 3.8% is
near to normal are the same text for this method. The report must give this
limitation.

**An operational note.** GDELT permits one request each five seconds. It
answers a faster client with HTTP 429. The library `gdeltdoc` converts that
answer into an exception with an empty message. Thus a rate limit looks the
same as an incorrect query. This cost three cycles of incorrect diagnosis in
this project.

---

## 7. Decisions

| Subject | Decision | Cost |
|---|---|---|
| The primary data | The issued CSV. FRED gives the additional series. | The numbers agree with the graded file. |
| The media corpus | GDELT is primary. twitterapi.io is a small comparison. | GDELT gives headlines only. |
| Quarterly to monthly | Put each quarterly value in its own quarter. | This is contemporaneous, not real-time. |
| The debt service ratio | Removed from Model 2. Kept as a robustness test. | 31% of the gap, not 40%. |
| The estimation window | A long window for the main result. | The window must be stable. |
| Negative results | Report them. | — |
| FinBERT | Measure the failure. | — |

---

## 8. What this document commits us to

1. The gap is **−25.24** on average and **−33.13** at its maximum. 18 months of
   19 are out of the 95% prediction interval. This is the quantity to explain.
2. Hypothesis A explains **31%** of the gap. The mortgage rate does this work.
   The price level is not separately identified.
3. Hypothesis C is **partially supported**. An independent instrument finds
   approximately one quarter of the Michigan gap. The best comparison series is
   a business survey, because no free consumer series is available.
4. Hypothesis B **fails**. The tone of coverage is not significant, it gives
   no information about a later gap, and the IV cannot identify a coefficient.
   The conditions above were fixed before the corpus existed, and they are met.
   Only `political_attribution` keeps a signal, as an association.
5. Each causal statement will give the strength of its identification:
   association, predictive precedence, or instrumented.
