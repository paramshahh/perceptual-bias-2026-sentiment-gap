# AI_USE — How an AI model was used in this project

This document uses ASD-STE100 Simplified Technical English.

The project brief requires a record of the use of artificial intelligence. This
document gives that record. It includes the errors that the model made, because
a record that gives only the successes is not a true record.

Model: Claude Opus 5, in the Claude Code terminal application. Subagents ran
on the same model. Where a second model gave advice inside the same session it
was Claude Opus 5.5, and that advice was always checked against the code.
Dates: 2026-10-14 to 2026-10-16.

**A change of direction on 2026-10-16.** The first two days of work used a
reconstruction of the assignment, which the person had pasted from memory. On
2026-10-16 the person supplied the real assignment PDF. The real brief contains
two timing rules that the reconstruction did not, and the project broke both.
The record of that correction is paragraph 3.9, and it is the most important
entry in this document, because the correction removed one of the project's
findings.

---

## 1. What the model did

The model wrote all of the code in this repository. The model also:

- read the issued CSV file and found its structure;
- fitted the models and calculated the results;
- wrote the tests, including tests for its own defects;
- wrote `BRAINSTORMING.md`, `PROGRESS.md` and this document;
- ran the data collection from FRED, GDELT and twitterapi.io.

The model worked in parallel for some tasks. It started separate agents for the
NLP module, the econometrics module, the Hypothesis C module, the table
module, the dashboard and the report. Each agent received the house rules and
wrote only its own two files.

## 2. What the person decided

The model did not make these decisions:

| Decision | Who |
|---|---|
| The hypotheses and the project scope | The brief |
| To spend 12 000 twitterapi.io credits | The person |
| To remove the debt service ratio from Model 2 | The person |
| To make a private repository and to push to it | The person |
| To use GDELT and not more tweets | The person, after the model gave the cost |
| Which fetch practice to follow for GDELT | The person gave the reference |

The model asked before each action that spends money or that sends data out of
the machine. The model did not create an account for the person, because an
account needs the identity of the person.

## 3. Errors that the model made

This paragraph is the important part of this document.

### 3.1 The model made an incorrect price level series

The first version of the price level put the value 100 in all twelve months of
2020. This makes twelve separate chains. The result has a false increase in
each January, and the year 2020 is flat. A regression on this series measures
the defect, not the economy.

**How we found it.** A code review of the module.

**The correction.** Accumulate the logarithm of the rate. A test now holds the
property: with a constant rate, each month must have the same increase.

### 3.2 The model put the GDP data in the wrong months

The CSV file gives the GDP value at the end of a quarter. The first version
used a forward fill. Thus the value of quarter 1 moved to March, April and May.
January and February held the value of the previous quarter.

**The effect.** With the error, the GDP coefficient is −0.037 with a p-value of
0.518. With the correction, it is −0.159 with a p-value of 0.039. **The same
data gives the opposite conclusion.**

**How we found it.** A code review.

### 3.3 The model let future data into a cross-validation

The code fitted a scaler on the complete training window, and then did a
forward-chaining cross-validation. Each fold then used statistics from its own
validation period. The docstring of that same function warned against this
error.

**How we found it.** A code review.

### 3.4 The model compared two models on different samples

The model reported "40% against 31%" for two versions of Model 2. The two
versions use different samples, 19 months and 17 months, because one FRED
series stops earlier. On the same 17 months the correct result is 39% against
31%.

**How we found it.** A validation pass.

### 3.5 The model gave a wrong correlation

The document gave 0.700 for a correlation. The true value is 0.758. The old
value came from the approximate series, and the model did not calculate it
again after the true series became available.

### 3.6 The model gave a cause that it did not test

The model said that collinearity explains an insignificant coefficient. The
variance inflation factor is 3.73, and the usual limit is 5. Thus collinearity
alone does not explain it.

### 3.7 The model used 4200 paid credits and got no usable data

The model requested one month of tweets for each query. The service returns the
newest tweets at the end of the window. Thus 140 tweets covered 14 minutes, on
the day after the requested month.

**The warning was present and the model did not use it.** The first test
request returned a tweet with a date after the end of the requested window. The
model recorded this as "the boundary is loose" and continued.

**The correction.** One day for each query, with the parameter `Top`. A test
request confirmed the correction before more credits were used.

### 3.8 The model made an incorrect diagnosis three times

GDELT answers a fast client with HTTP 429. The library converts that answer
into an exception with an empty message. Thus a rate limit looks the same as a
bad query. The model first concluded that the dates were not valid, then that
the archive was short. A direct request to the API showed the true cause in one
step.

### 3.9 The model used data that did not exist on the prediction date

The panel put the unemployment rate for month *t* in the row for month *t*. The
Bureau of Labor Statistics publishes that rate in month *t+1*. The panel also
spread each quarterly GDP value across three months, which makes three monthly
observations from one measurement.

**The effect.** With the error, the GDP coefficient is −0.1585 with a p-value of
0.039, and the document discussed it as a specification warning. With the
correction, it is −0.0304 with a p-value of 0.524. **The finding was an
artefact of the error.** R² falls from 0.7663 to 0.7632, and the gap conclusion
does not change: the mean gap moves from −25.24 to −25.31 and the count of
months outside the prediction interval moves from 18 of 19 to 19 of 20.

**How we found it.** The real assignment PDF states both rules in plain words.
The reconstruction of the brief did not contain them.

**The correction.** `data_loader.build_monthly_panel` now defaults to a
real-time alignment. Each row holds only values published by the end of that
month. The publication lag of each series is recorded with its release
calendar. Quarterly GDP enters as a step function of published estimates. The
old behaviour is still available, and its own report text says that it is not
compliant.

### 3.10 The model built a corpus with a filter that it had not checked

The GDELT theme filter accepted any theme with the prefix `TAX_`. The model read
that prefix as taxation. In GDELT, `TAX_` is a general taxonomy of actors,
ethnicities and languages, and it matches most news articles.

**The effect.** 693 347 articles with a mean tone of −0.988, in place of 198 367
articles with a mean tone of −0.180. The first corpus measured the tone of the
news in general. **It would have given Hypothesis B a false positive with a
large sample behind it.**

**Why this is the most dangerous error in the project.** It was wrong in the
direction that confirms the interesting hypothesis. A result that agrees with
the expected story gets less scrutiny than one that does not.

**How we found it.** A check of what the filter actually matched.

### 3.11 The model reported two dashboard panels as live when they were not

The model checked the helper function that decides whether a panel has data. It
did not check the render path that a reader sees. The helper said yes and the
page said "not available".

**How we found it.** A subagent checked the rendered page. The model then
verified the failure and corrected its own statement to the person.

**The correction.** Drift tests now compare the claim against the rendered page.
The rule that came out of it: **a claim about an artifact is verified on the
artifact, not on the function that decides it.** The same pattern caused three
other errors in this project.

## 4. How we checked the work of the model

1. **An independent calculation.** The headline numbers were calculated again
   with `numpy` and `scipy`, with no project code. On the earlier
   contemporaneous panel all agreed: R² = 0.766, mean gap = −25.24, maximum gap
   = −33.13, and 18 months of 19 out of the prediction interval. The real-time
   panel gives R² = 0.7632, mean gap = −25.31, maximum gap = −33.64 in 2026-05,
   and 19 months of 20 out of the prediction interval. The person reading this
   should note that the second set is the live one, and the first set is kept
   only to show the size of the correction in paragraph 3.9.
2. **Tests.** The repository has 623 tests. Several tests hold a defect that a
   review found, so the defect cannot return. Two tests assert the brief's
   timing rules directly on the real data: no value published after month *t*
   appears in the row for month *t*, and every monthly GDP value is a published
   quarterly value.
3. **Code review.** A review of the first two modules gave 12 findings. The
   model corrected all of them.
4. **A circularity screen.** The model nearly used a comparison survey that
   contains the Michigan survey. A correlation check found this before the
   result entered the report.

## 5. What the model did not do

- The model did not invent a number. Where a result does not exist, the
  document and the report say so.
- The model did not create an account or a key for the person.
- The model did not remove a file of the person without permission.
- The model did not hide an error. Each error above is also in the commit
  history.

## 6. Model versions, prompts and evidence

The brief requires that model versions, prompts and evidence excerpts are
preserved.

| Item | Where it is |
|---|---|
| Model version for every run | This document, paragraph 0, and the commit messages |
| The task given to each subagent | The commit history and `PROGRESS.md` paragraph 11, which holds the file-ownership table |
| The lexicons, which are the measurement instrument | `src/perceptual_bias/nlp_media_svd.py`, at module level, so a reviewer can argue with each phrase |
| The implied inflation rate behind each magnitude claim | `CALIBRATION_CLAIMS` in the same module. These are judgements, and they are deliberately generous to the press: "soaring" is read as 8%, not 20% |
| The FinBERT probe set and its result | `FINBERT_PROBE` in the same module, with the measured accuracy of 0.300 against a chance level of 0.333 |
| The human-labelling instrument | The labelling guide in the reproducible package |
| Evidence excerpts | Each lexicon match is returned with the phrase that matched it, so a score can be traced to the words that produced it |

**No prompt in this project asked a model to classify text for the headline
result.** The tone, salience, framing, calibration and attribution measures are
lexicon-based and deterministic. That is a deliberate choice: a lexicon can be
read, argued with and reproduced exactly, and a reviewer can see why a document
scored as it did. The transformer model in this project, FinBERT, appears as a
**measured failure mode** and not as an instrument.

## 7. The judgement of the author

The model is good at code, at tests and at finding its own defects when a
review asks it to look. The model is weak at the moment when a small warning
appears and the task continues. Two of the errors above, 3.7 and 3.8, had a
visible warning that the model saw and did not act on.

The correct use of the model is therefore: let it write and test the code, and
then make it prove each number through a second path.

**One more judgement, after the correction of 2026-10-16.** The model built two
days of work against a reconstruction of the assignment instead of asking for
the original document. The work was not wasted, and most of it survived, but two
of its decisions broke rules that were in the real brief all along. The lesson
is about the source and not about the model: a specification that is quoted
from memory is not a specification.
