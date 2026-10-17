# Method notes

**Why this document exists.** The report must fit 10 to 12 pages. These three
findings are about method, not about the economy, and a reader needs only the
short form to read the results correctly. The long form is here, because a
finding that cost this project real time should not be lost to a page limit.

**Language.** ASD-STE100 Simplified Technical English.

---

## 1. A comparison survey that contains the Michigan survey

The obvious comparison series for Hypothesis C is the OECD consumer confidence
indicator for the United States, FRED `USACSCICP02STSAM`. That series
correlates **0.758** with the Michigan survey after 2010, in levels and in
differences.

The cause is simple: **the OECD consumer input for the United States IS the
Michigan survey.** A test against it gives a gap ratio of about 1.00 and
"rejects" Hypothesis C with Michigan's own numbers. The result looks clean and
it is circular.

**This is a finding and not an anecdote.** The module `survey_diagnostics` now
screens each candidate series. It rejects a candidate whose correlation with
Michigan is above the level threshold **and** above the difference threshold.

**The difference threshold is the looser one on purpose.** Publication rounding
drags the impostor's difference correlation down to 0.976 over the full history.
A tight screen on both legs therefore lets it through. Two genuinely independent
household surveys do not come near 0.90 in differences. Fannie Mae against
Michigan gives **0.34**.

**The cost of the screen is real.** No free and independent consumer confidence
series for the United States covers 2025 and 2026:

| Candidate | Why it cannot be used |
|---|---|
| OECD `USACSCICP02STSAM` | It contains Michigan. |
| One candidate | It ends 2024-01. |
| One candidate | It ends 2025-09. |
| The series that covers the window | It is a **business** survey, so it measures a different quantity. |
| Conference Board | It has a licence. |

So the Hypothesis C test is weaker than it should be, and the report says so
rather than presenting the business survey as an equivalent instrument.

---

## 2. FinBERT inverts hawkish and dovish on macro text

FinBERT learns from market text. That text asks "is this good for the asset?".
Macro policy text asks a different question, and the two answers often have
opposite signs.

Take this sentence:

> Strong productivity growth and robust labour market conditions continued
> through the quarter.

FinBERT gives it a strong positive score. In policy language, strong growth and
a tight labour market **give a higher interest rate**, so the sentence is
hawkish. The model reads the words correctly. The model answers a question that
nobody asked.

**The project measured this failure. It did not quote it.** A probe set holds 30
macro sentences: 10 hawkish, 10 dovish and 10 neutral.

| Measurement | Value |
|---|---|
| FinBERT accuracy | **0.300** |
| Accuracy of a random guess | 0.333 |
| Hawkish sentences called dovish | 6 of 10 |
| Dovish sentences called hawkish | 7 of 10 |

FinBERT is worse than a random guess on this text, and it inverts the two
classes that matter. The probe set is in `FINBERT_PROBE` in
`src/perceptual_bias/nlp_media_svd.py`, so a reader can disagree with any
sentence in it.

The study therefore uses a domain lexicon with negation control, across the five
dimensions the brief requires: tone, selection and salience, framing, factual
calibration, and political attribution.

**A note on what this does and does not show.** It does not show that
transformer models are useless for text measurement. It shows that an
off-the-shelf model carries the question its training data asked, and that the
question matters more than the architecture. A model fine-tuned on central-bank
language would be a fair instrument. This project did not have one, and it did
not pretend that FinBERT was one.

---

## 3. A correction to the brief on the Rigobon-Sack method

### 3.1 The order condition is uninformative here

The brief gives an order condition of

$$N^2 - N - 2K > 0$$

With $N = 2$ this becomes $2 - 2K$. The condition holds only when $K = 0$, and
it fails with any exogenous regressor at all. The formula is uninformative for a
two-variable system. The module reports its value so that the report can say
what happened to it.

### 3.2 The rank condition disagrees with the identifying assumption

The brief also gives a rank condition: the determinant of $\Omega_H - \Omega_L$
must not equal zero.

Under the identifying assumption that **one** shock's variance shifts between
regimes, that matrix has **rank 1** in the population. Its determinant is then
**exactly zero**. A determinant near zero therefore *agrees* with
identification. It is not a failure.

The module computes the determinant and its bootstrap distribution, because the
brief asks for them. The operative test is different: the media series' own
variance must differ across the regimes, and that difference must be different
from zero in the bootstrap.

**The design was checked on synthetic data.** With a true coefficient of 0.60
the method returns 0.598. With no change in variance it returns no estimate.

### 3.3 The system is under-identified by one restriction

Two regimes give six second moments against six parameters. That looks
sufficient and it is not. All the identifying information sits in
$\Omega_H - \Omega_L$, which gives three equations with four unknown values.

So the study estimates the direction media → gap only, under the Rigobon-Sack
assumption that the regime change moves the variance of the media shock alone.
**The reverse direction is not identified, and the report does not give it.**
Two covariance matrices cannot test the assumption that the estimate rests on.

### 3.4 A defect in the first regime rule

The first rule used a rolling variance window that contained the current month.
That rule sorts the months by their own deviation, so the rank condition
"passed" on pure noise. The window now stops at the previous month, and the
synthetic homoskedastic control case now fails as it should. The cost is a
smaller measured variance ratio, because the lag puts some months in the wrong
regime.

---

## 4. The pattern behind these three

Each one is a case where a result looked clean and was not:

1. a comparison that agreed with itself;
2. a classifier that answered a different question with high confidence;
3. a condition that "passed" because the test was built on the quantity it was
   meant to test.

The general rule that came out of them: **when a result looks clean, find the
way it could be clean for the wrong reason, and test that.**
