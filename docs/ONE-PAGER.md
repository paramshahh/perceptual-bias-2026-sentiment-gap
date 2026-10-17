# ONE PAGE SUMMARY — The 2026 Sentiment Gap

This document uses ASD-STE100 Simplified Technical English.
Date: 2026-10-17. Every number here is from the real-time panel.

---

## 1. The problem

The economy looks acceptable in the data. The people feel bad. The project
measures the difference and finds the cause.

A standard model predicts consumer sentiment from three numbers: the
unemployment rate, the inflation rate and the GDP growth rate. The model learns
from 15 years of data (2010 to 2024). Then the model predicts the years 2025
and 2026.

The model fails. In May 2026 the model predicts a sentiment of 78.4. The true
sentiment is 44.8. The difference is 33.6 points. We call this difference the
Sentiment Gap.

## 2. Why this is not an error in the model

A model can make a bad prediction by chance. This model does not.

The model gives a prediction interval for each month. The interval shows the
normal range of error. In 19 months of 20 months, the true sentiment is below
this interval. All 20 errors have the same direction.

Thus the model does not have more noise than usual. The model is wrong in one
direction, every month. Something that the model does not measure is present.

## 3. The intuition

### 3.1 A rate is not a level

The model uses the inflation rate. A household uses the price.

The inflation rate decreased to 3%. This means that the prices increase more
slowly. It does not mean that the prices decrease. The price level is now 25.7%
more than the price level of 2020.

Compare two households. The model sees a household that gets good news: "the
inflation rate is low". The true household sees the price of food. That price
is one quarter more than the price six years ago. The household does not feel
relief. The household feels a permanent loss.

The model measures the speed of the damage. The household measures the total
damage.

### 3.2 The cost of money

The model does not use the mortgage rate. The mortgage rate is near 7%.

A person who wants a house cannot buy the house. A person who has a house with
a low rate cannot move. The unemployment rate does not show this condition. The
person has a job. The person also has no access to a house.

### 3.3 What the people hear

The people do not read the FRED database. The people read the news and the
social media. If the reports about the economy are more negative than the data,
the perception becomes more negative than the condition.

This effect is possible. It is not yet proved. We must measure it.

### 3.4 A different possibility

The survey changed its method. The University of Michigan moved the survey to
the internet between 2024 and 2026. A change of method can change the result.

If this is the cause, a part of the gap is an error of measurement. Then the
gap that Hypothesis A and Hypothesis B must explain is smaller.

## 4. What we found

Add the price level and the mortgage rate to the model. The mean gap decreases
from −25.3 to −18.3. The model explains 27.6% of the gap.

| Measurement | Value |
|---|---|
| Mean gap, standard model | −25.31 |
| Mean gap, model with the price level and the mortgage rate | −18.33 |
| Part of the gap that economic pressure explains | 27.6% |
| Part of the gap that stays | 72.4% |
| One-step forecast error, standard model | 15.58 |
| One-step forecast error, model with the level variables | 6.98 |

**The mortgage rate does the measurable work** (−2.443, p < 0.001). The price
level is NOT separately identified (p = 0.351, and 0.233 when the debt service
ratio enters). So the finding is about the **cost of money**, and this project
claims nothing about the price level by itself.

**A gap of −18.3 points stays. The economy does not explain this remainder.**

## 4.1 The media explanation fails

Four conditions were fixed before the corpus existed, and all four hold:

1. Two of four media measures reject before correction. Across the **12 tests**
   this project ran, **zero** survive Holm or Benjamini-Hochberg.
2. No media measure precedes the gap. The only directions that reach 5% run
   BACKWARDS: the gap precedes the coverage.
3. Adding the media block does not improve a forecast of an unseen month, and
   **no model beats a free last-observed-value guess** (4.45 against 6.61).
4. Holding the calendar day and the topic constant, no source type is reliably
   more negative. The largest deviation is 0.044 on a scale from −1 to +1.

Both media associations die in all 24 specification cells once household and
political controls enter. Policy uncertainty alone absorbs them.

## 4.2 The finding that outranks all three hypotheses

FinBERT changes its tone verdict on **5.8%** of byte-identical texts when only
the outlet label changes. The word list changes on **0%**. A classifier that
reads the label would have manufactured the bias this project went looking for.

## 5. What we needed

| Item | Status | Cause |
|---|---|---|
| The supplied CSV file | Available | It gives the sentiment data and the macro data. |
| A FRED key | Available | It gives the price level, the mortgage rate and the debt data. |
| A media data source | GDELT | GDELT is free. GDELT gives the news data. |
| Tweets | Limited | The budget is small. Refer to paragraph 6. |
| The SVD lecture file | Not available | The mathematics comes from first principles. |

## 6. The cost control

One request to twitterapi.io costs 300 credits. The initial quantity was
21 000 credits.

The first method was not correct. The method requested one month of tweets. The
service gave the newest tweets only. Thus 140 tweets covered 14 minutes. This
error used 4200 credits.

The second method is correct. The method requests one day of tweets and uses
the parameter `Top`. The data is correct and is about the economy.

Credits used: 4800. Credits available: 16 200.

## 7. The next work

1. Measure the tone of the news with GDELT. Find the latent topics with the
   SVD method.
2. Test if the tone of the news causes the gap. Use the VAR method and the
   instrumental variable method.
3. Test the survey method change. Compare with a different survey.
4. Build the dashboard and the report.

## 8. The decisions that we need

1. **The tweet budget.** Use 12 000 credits to get a monthly series of tweets?
   Or stop and use GDELT only?
2. **The backup.** Make a private repository and save the work? The repository
   has no commits now.
3. **The debt service variable.** Keep it or remove it? It is not significant.
   It also removes two months of data.
