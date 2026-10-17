# HUMAN LABELING GUIDE — ECONOMIC MEDIA CODING SCHEME

This document is the coding scheme for the project `perceptual-bias`. Two
persons use this document to label the same 359 items. The two persons work
alone. A third person then adjudicates the items where the two persons do not
agree.

This document uses ASD-STE100 Simplified Technical English.

Version: 1.0. Date: 2026-10-16.
Code module: `src/perceptual_bias/corpus_validation.py`.

> **STATUS. The human labels do not exist.** This document and the annotation
> file are complete. No person labeled the items. Each report in the analysis
> gives this status. Refer to paragraph 9.

---

## 1. What you must do

1. Open the file `outputs/labels/coding_sample_annotation.csv`.
2. Read this document one time completely. Do not start before you read it.
3. Do the practice items in paragraph 8. Compare your answers with the
   answers in that paragraph.
4. Label the items in the order of the column `presentation_order`.
5. Put one answer in each label column. Use the exact words of this document.
6. Write a short remark in your note column if an item is difficult.
7. Do not discuss an item with the other person. Do not look at the other
   file.
8. Save the file with your letter in the name. Example:
   `coding_sample_annotation_A.csv`.

### 1.1 What you see, and why

You see the item id, the order number and the text. **You do not see the name
of the outlet, the date or the result of the computer model.** This is not an
accident:

- The name of the outlet makes a label unreliable. A person who sees a
  political name attached to a text can label the name and not the text. The
  project measures this effect with the swap test, and the effect is real.
- The result of the model is a strong anchor. A person who sees the model
  answer agrees with the model more frequently. The labels are the standard
  against which the project measures the model. Thus the labels must be
  independent of the model.

The analyst holds the file with the outlet and the date. The analyst joins the
two files after you give your labels.

### 1.2 What one item is

One item is one of two things:

- One **news headline**. The corpus holds headlines and not full articles. You
  can see only the words in front of you. Do not guess what the article says.
- One **social-media post**. A post can be longer than a headline.

Label the text that you see. If a headline is not complete, this is a
limitation of the corpus. Use the rule in paragraph 7.3.

---

## 2. The five dimensions

The brief asks for five dimensions. The instrument has seven questions,
because the framing dimension has three contrasts.

| No. | Dimension | Question or questions | Column prefix |
|---|---|---|---|
| 1 | Tone | 1 | `tone_` |
| 2 | Selection and salience | 1 | `salience_` |
| 3 | Framing | 3 | `framing_level_vs_rate_`, `framing_household_vs_macro_`, `framing_current_vs_forecast_` |
| 4 | Factual calibration | 1 | `calibration_` |
| 5 | Political attribution | 1 | `attribution_` |

Each column has the suffix `_annotator_a` or `_annotator_b`. Person A uses the
columns with `_annotator_a`. Person B uses the columns with `_annotator_b`.

The dimensions are independent. An item can be negative in tone and also
correct in its facts. Answer each question separately. Do not make the answers
agree with one another.

---

## 3. Dimension 1. Tone

**Question: how does the text describe the economy?**

| Answer | Rule |
|---|---|
| `negative` | The text describes the economy, or a part of it, as bad, worse or at risk. |
| `positive` | The text describes the economy, or a part of it, as good, better or improved. |
| `neutral` | The text gives a fact without an evaluation. Or the text is not about the economy. Or the positive part and the negative part have equal weight. |

### 3.1 Decision rules

1. **Separate the fact from the wording.** The brief asks you to separate
   factual description from emotional wording. A bad number with plain words
   is still negative, because the text describes a bad condition. Use the rule
   below for the wording.
2. **The condition decides the class, not the word count.** One strong word is
   sufficient. Do not count words.
3. **A negator changes the class.** "Inflation is not falling" is negative.
   "Prices are not out of control" is positive or neutral.
4. **A quotation is part of the text.** A headline that quotes a critic is
   negative if the quotation is negative. You label the headline, not the
   critic.
5. **A market report is about the economy.** "Stocks fall on growth fears" is
   negative.
6. **A company result is not the economy** if the text gives no connection to
   the economy. "Company X reports a loss" is `neutral`. "Layoffs at Company X
   show a weak labor market" is `negative`.
7. **Two sides with equal weight are `neutral`.** "Jobs grow but prices rise"
   is `neutral`.
8. **A forecast of a bad condition is negative.** The tone question does not
   ask about the time. The time is dimension 3.

### 3.2 Examples

| Text | Answer | Cause |
|---|---|---|
| Recession fears grow as layoffs climb | `negative` | A bad condition and a risk. |
| Inflation eased to 3 percent in June | `positive` | An improvement. |
| Consumer prices rose 0.3 percent in the month | `neutral` | A fact without an evaluation. |
| Families cannot afford groceries | `negative` | A bad condition for households. |
| The Federal Reserve meets next month | `neutral` | Not an evaluation. |
| Hiring is strong but households feel the squeeze | `neutral` | Two sides, equal weight. |
| Analysts say the recovery is not real | `negative` | A negator and a negative claim. |

---

## 4. Dimension 2. Selection and salience

**Question: which economic fact does the text put first?**

Select **one** answer. If the text names two facts, select the fact in the
first part of the text.

| Answer | Rule |
|---|---|
| `layoffs` | Job losses, job cuts, a plant closure, a rise in unemployment. |
| `hiring` | New jobs, a fall in unemployment, a tight labor market. |
| `rising_prices` | Prices, costs or the cost of living that go up. |
| `falling_inflation` | Inflation or price growth that goes down. |
| `growth` | GDP, output, sales or the economy that expands. |
| `recession_risk` | A recession, a contraction or a downturn that can come. |
| `other_economic` | The text is about the economy, but about none of the six subjects. Examples: interest rates, trade, the stock market, taxes. |
| `not_economic` | The text is not about the economy. |

### 4.1 Decision rules

1. `rising_prices` is about the **amount** that a household pays.
   `falling_inflation` is about the **rate of change**. A text can name both.
   Select the one in the first part.
2. A recession that is in progress is `growth` with a negative tone, or
   `recession_risk` if the text speaks about the future. A fall that happens
   now is not a risk.
3. Interest rates are `other_economic`, and not `rising_prices`. An interest
   rate is a price of credit and not a consumer price.
4. `not_economic` is a correct and frequent answer. Approximately one third of
   the corpus is not about the economy, because the search terms are wide.

---

## 5. Dimension 3. Framing (three questions)

Framing is the **choice between two correct ways to tell the same fact**. Each
question gives two sides. You select one side, or `balanced`, or `not_framed`.

### 5.1 The rule that applies to all three questions

> **`not_framed` and `balanced` are different answers. Do not mix them.**
>
> - `not_framed`: the text says nothing about either side. The text is silent.
> - `balanced`: the text says something about both sides, with equal weight.
>
> The computer measure gives "no value" (NaN) for the silent text and 0.0 for
> the balanced text. Your two answers must keep the same difference. If you
> make the two answers one answer, the comparison between you and the computer
> counts silence as balance. The measure is then not valid.

`not_framed` is the most frequent answer. In the corpus, 91 percent of the
headlines are silent about the first contrast, and 97 to 98 percent are silent
about the other two. Do not try to find a frame. If the text does not frame,
write `not_framed`.

### 5.2 Question 3a. The price level against the inflation rate

Column: `framing_level_vs_rate_annotator_*`

| Answer | Rule |
|---|---|
| `level` | The text speaks about the **amount** that a person pays: a price, a cost, a bill, a rent, a fare, affordability. |
| `rate` | The text speaks about the **rate of change**: inflation, CPI, disinflation, a percent per year, a change of pace. |
| `balanced` | The text speaks about both with equal weight. |
| `not_framed` | The text speaks about neither. |

**Why this question is important.** Hypothesis A of the project says that a
household anchors to the price that it pays, and not to the rate of change.
Thus a positive value of this contrast connects the media measure to the
economic measure. The project made this prediction before it made the measure.

Examples:

| Text | Answer |
|---|---|
| Groceries still cost more than before the pandemic | `level` |
| Inflation slowed to a 3 percent annual rate | `rate` |
| Inflation fell, but the grocery bill is still high | `balanced` |
| The jobs report comes on Friday | `not_framed` |
| Prices are not higher than in 2020 | `level` |

The last example is important. A negator does not change a framing answer. The
text still makes the price level the subject. The computer measure also does
not apply a negator to framing, and the cause is the same.

### 5.3 Question 3b. The household against the national aggregate

Column: `framing_household_vs_macro_annotator_*`

| Answer | Rule |
|---|---|
| `household` | The text speaks about a person, a family, a shopper, a paycheck, a budget, or a place where a person buys. |
| `macro` | The text speaks about GDP, the national economy, an aggregate or a headline figure. |
| `balanced` | The text speaks about both with equal weight. |
| `not_framed` | The text speaks about neither. |

Examples:

| Text | Answer |
|---|---|
| Families at the checkout see a higher bill | `household` |
| GDP expanded 2.8 percent in the quarter | `macro` |
| The economy grew, but households do not feel it | `balanced` |
| Stocks closed lower | `not_framed` |

### 5.4 Question 3c. The current condition against the forecast

Column: `framing_current_vs_forecast_annotator_*`

| Answer | Rule |
|---|---|
| `current` | The text speaks about a condition now or in the past: "today", "this month", "the latest reading", "data showed". |
| `forecast` | The text speaks about a condition in the future: "forecast", "expected to", "outlook", "could reach", "warns of". |
| `balanced` | The text speaks about both with equal weight. |
| `not_framed` | The text gives no time. |

Examples:

| Text | Answer |
|---|---|
| Retail sales fell last month | `current` |
| Analysts expect a slower second half | `forecast` |
| Sales fell last month and could fall again | `balanced` |
| Economy | `not_framed` |

---

## 6. Dimension 4. Factual calibration

**Question: does a number in the text agree with the official data that was
available on the day of publication?**

| Answer | Rule |
|---|---|
| `no_numeric_claim` | The text gives no number and no quantity word. |
| `matches_official_data` | The number agrees with the official data. |
| `overstates` | The number or the quantity word is larger than the official data. |
| `understates` | The number or the quantity word is smaller than the official data. |
| `cannot_check` | The text gives a number, but you cannot find the official value, or the text does not say which measure it uses. |

### 6.1 Decision rules

1. `no_numeric_claim` is **not** the same as `matches_official_data`. A text
   with no number makes no claim. Do not read a correct claim into silence.
2. A quantity word is a claim. "Inflation is soaring" is a claim about a
   magnitude. The project reads "soaring" as approximately 8 percent per year.
   The list is in `nlp_media_svd.CALIBRATION_CLAIMS`. Use that list.
3. Use only the data that was available on the day of publication. The analyst
   gives you a table with the official values by month. Do not use a later
   revision. A text cannot be wrong about a number that did not exist.
4. A difference of less than 0.2 percentage points is `matches_official_data`.
5. If the text does not say which measure it uses, and two measures give
   different answers, write `cannot_check`. Do not select the measure that
   makes the text correct, and do not select the measure that makes the text
   wrong.
6. A number about a company, a market or a place is `cannot_check` unless the
   official table holds that number.

---

## 7. Dimension 5. Political attribution

**Question: which actor gets the credit or the blame for an economic
condition?**

| Answer | Rule |
|---|---|
| `credit_administration` | The text gives the government or a political party the credit for a good condition. |
| `blame_administration` | The text gives the government or a political party the blame for a bad condition. |
| `credit_federal_reserve` | The text gives the central bank the credit. |
| `blame_federal_reserve` | The text gives the central bank the blame. |
| `credit_other_actor` | The text gives the credit to a company, a union, a foreign country or another actor. |
| `blame_other_actor` | The text gives the blame to another actor. |
| `no_attribution` | The text names no actor, or it names an actor without credit and without blame. |

### 7.1 Decision rules

1. **Criticism is not bias.** You record the direction of the attribution. You
   do not make a judgement about whether the attribution is correct or fair.
   The brief gives this instruction.
2. A name alone is not an attribution. "Trump speaks about the economy" is
   `no_attribution`. "Trump policies caused the price rise" is
   `blame_administration`.
3. A quoted attribution is an attribution. The text carries it.
4. If the text gives credit to one actor and blame to another actor, select the
   one in the first part of the text. Write the second one in your note
   column.

### 7.2 A difficult example

"Critics say the president is responsible for high prices" is
`blame_administration`. The text reports the attribution of other persons, but
the headline carries it to the reader.

### 7.3 Text that you cannot label

Some items are navigational text or are not complete. Examples: "Latest
Articles", "Business Highlights". The sampler removed the items with three
words or fewer. Some items of four or five words are still not usable.

For such an item:

1. Write `neutral`, `not_economic`, `not_framed` (three times),
   `no_numeric_claim` and `no_attribution`.
2. Write `UNUSABLE` in your note column.

Do not leave a cell empty. An empty cell and the answer `UNUSABLE` are
different signals. The agreement code counts an empty cell as work that is not
complete.

---

## 8. Practice items

Label these six items before you start. Then compare your answers.

| No. | Text |
|---|---|
| P1 | Grocery prices climb again as families cut back |
| P2 | Fed holds rates steady; analysts expect two cuts next year |
| P3 | Inflation cooled to 2.9 percent in May, the lowest in three years |
| P4 | Biden says Republicans caused the cost of living crisis |
| P5 | Business Highlights |
| P6 | GDP grew 3.1 percent, but paychecks did not keep up |

The correct answers:

| No. | Tone | Salience | Level/rate | Household/macro | Current/forecast | Calibration | Attribution |
|---|---|---|---|---|---|---|---|
| P1 | `negative` | `rising_prices` | `level` | `household` | `not_framed` | `no_numeric_claim` | `no_attribution` |
| P2 | `neutral` | `other_economic` | `rate` | `not_framed` | `forecast` | `no_numeric_claim` | `no_attribution` |
| P3 | `positive` | `falling_inflation` | `rate` | `not_framed` | `current` | `matches_official_data` or `cannot_check` | `no_attribution` |
| P4 | `negative` | `rising_prices` | `level` | `not_framed` | `not_framed` | `no_numeric_claim` | `blame_administration` |
| P5 | `neutral` | `not_economic` | `not_framed` | `not_framed` | `not_framed` | `no_numeric_claim` | `no_attribution` |
| P6 | `neutral` | `growth` | `not_framed` | `balanced` | `current` | `cannot_check` | `no_attribution` |

Remarks on the practice items:

- P2: "Fed holds rates steady" is a fact and not an evaluation, thus the tone
  is `neutral`. The interest rate is `other_economic`.
- P3: The calibration answer depends on the official table. If the table holds
  the CPI value for May, check it. If the table holds only the PCE value, the
  text does not say which measure it uses, thus write `cannot_check`.
- P4: A name and a causal claim together make an attribution. The subject is
  the cost of living, thus the level side.
- P6: Two sides with equal weight give `neutral` and `balanced`.

If your answers do not agree with four or more of these answers, read
paragraph 5.1 and paragraph 6.1 again before you start.

---

## 9. Agreement and adjudication

The code in `corpus_validation.py` makes these measurements:

| Measurement | Function |
|---|---|
| Percent agreement | `percent_agreement` |
| Cohen's kappa | `cohen_kappa` |
| Agreement by class, with specific agreement | `compute_agreement` |
| Agreement by source type | `compute_agreement` |
| One row for each question of the five dimensions | `agreement_by_dimension` |
| The gold labels and the adjudication log | `adjudicate` |

`cohen_kappa` gives "no value" and not 1.0 when both persons use one class for
every item. Agreement is then complete and also without information.

### 9.1 The adjudication procedure

The steps are in `ADJUDICATION_PROCEDURE` in the code module. The summary:

1. Each person labels all items alone.
2. The analyst calculates the agreement **before** adjudication. This is the
   measurement. The agreement after adjudication is always 100 percent, and it
   is not a measurement.
3. An item with two equal labels keeps that label. It becomes the gold label.
4. An item with two different labels goes to a third person. The third person
   reads the item and the two notes. The third person selects one of the two
   labels, or the answer `unusable`.
5. An item that the third person calls `unusable` leaves the gold set. The
   analyst records the cause. **No model label replaces a human label.**
6. The analyst records each decision in `outputs/labels/adjudication_log.csv`.
7. Each question gets its own adjudication. Two persons can agree about tone
   and disagree about framing.
8. For a framing question, the third person must keep `not_framed` and
   `balanced` apart. Refer to paragraph 5.1.
9. **If the two persons disagree on more than 20 percent of the items, stop.**
   Make the rules in this document more exact. Then label a new sample. Do not
   adjudicate a bad instrument.

### 9.2 Status of the human work

No person labeled the sample. The function `require_human_labels` raises an
exception if a caller tries to use model labels as human labels. The list
`OUTSTANDING_HUMAN_WORK` gives what the brief asks for and what is not done:

1. The two independent human annotators.
2. The agreement before adjudication.
3. The performance of the model by class and by source. This needs a human
   standard.
4. The adjudication log, which needs a third person.
5. The framing validation. Refer to paragraph 9.3.

### 9.3 The framing dimension has no model to compare against

FinBERT gives market sentiment from one output head. It cannot give a framing
contrast. No model in this project gives one. Thus the framing measure has
only one validation: the human labels against the framing lexicon. Until the
human labels exist, the framing measure is not validated. The constant
`FRAMING_AI_LIMITATION` holds this statement in the code.

---

## 10. The automated stand-in, and what it is not

The project ran two **automated** labelers on the same 359 items. The purpose
was to test the agreement code with real data. The two labelers are the project
lexicon and FinBERT.

| Measurement | Value |
|---|---|
| Items with two labels | 359 of 359 |
| Percent agreement | 0.557 |
| Cohen's kappa | 0.224 |

**These are not human labels.** Each report marks them as automated. The field
`AgreementReport.labels_are_human` is False. The brief asks for two human
annotators, and two models are not two annotators. The numbers show that the
agreement code works on real data. They do not answer the question of the
brief.

The same run also measured the robustness of the two instruments. The lexicon
does not change a tone class when a political name or an outlet label changes.
FinBERT changes the tone class of 5.8 percent of identical texts when the
outlet label in front of the text changes. Refer to
`docs/corpus_documentation.md` paragraph 8.2. This is a result about
robustness. It is not a result about accuracy, and accuracy needs your
labels.
