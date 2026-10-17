# Provenance register

**Why this document exists.** The report has a page budget of 10 to 12 pages,
and the brief asks for a separate reproducible analytical package. This register
belongs to the package.

**How to read it.** Almost every number in the report is computed live at build
time: `build_report.py` runs `data_loader`, `baseline_model`,
`causal_econometrics`, `corpus_validation` and `survey_diagnostics` and puts
their output into the tables. A small set of numbers cannot be recomputed in
every environment, because FRED needs a key and a network, or because a second
tool performed the check. Each of those carries a tag, and the report prints the
tag beside the value so that a reader can always separate a recorded number from
a live one.

**A warning about the superseded values.** The seven Hypothesis A constants were
recomputed on 2026-10-16 when the panel moved to the real-time alignment. Each
one records the value it replaced, because the size of the change is itself
evidence about how much the timing rule mattered.

| Tag | Quantity | Value | Source | Note |
|---|---|---|---|---|
| C1 | Share of the gap that Hypothesis A absorbs | 27.6% | recomputed 2026-10-16 on the real-time panel | Model 2 leaves a mean gap of -18.33 against -25.31 over the same 20 months. The superseded value on the contemporaneous panel was 31%. |
| C2 | Model 2 R-squared | 0.8099 | recomputed 2026-10-16 on the real-time panel | Model 1, the price level and the mortgage rate. FRED supplied both. The superseded value was 0.835. |
| C3 | Model 2 AIC | 1142.4 | recomputed 2026-10-16 on the real-time panel | The superseded value was 1117.4. |
| C4 | Model 2 mean out-of-sample gap | -18.33 | recomputed 2026-10-16 on the real-time panel | Over the 20 months both models can predict. The superseded value was -17.40. |
| C5 | Mortgage rate coefficient in Model 2 | -2.443, p < 0.001 | recomputed 2026-10-16 on the real-time panel | The mortgage rate does the measurable work in Hypothesis A. The superseded value was -2.936. |
| C6 | Price level p-value in Model 2 | 0.351 | recomputed 2026-10-16 on the real-time panel | The same coefficient has p = 0.233 with the debt service ratio added, so it is not identified either way. The superseded values were 0.310 and 0.038. |
| C7 | Variance inflation factor of the price level | 3.68 | recomputed 2026-10-16 on the real-time panel | The usual limit is 5 or 10. Collinearity alone does not explain 0.351. |
| C8 | Chow F at the 2024-07 transition | 81.3 | BRAINSTORMING.md paragraph 2.3, recorded run | An independent path with scipy gives 83.7. The difference is 3%. |
| C9 | Level shift at the 2024-07 transition | -35.27 | PROGRESS.md paragraph 7.4, independent recomputation | The module's recorded value is -35.50. |
| C10 | Supremum-Wald break date | 1990-07 (module) / 1990-08 (independent path) | PROGRESS.md paragraph 7.4 | The date is NOT the 2024-07 survey transition. |
| C11 | Cross-survey gap ratio | 0.258 | PROGRESS.md paragraph 7.4, independent recomputation | The threshold for 'supported' is 0.25. The margin is 0.008. |
| C12 | Correlation of FRED USACSCICP02STSAM with Michigan after 2010 | 1.0000 | BRAINSTORMING.md paragraph 2.3 | The OECD United States consumer input IS the Michigan survey. |
| C13 | FinBERT accuracy on the 30-sentence macro probe | 0.300 against 0.333 for chance | BRAINSTORMING.md paragraph 4.2, measured | 6 of 10 hawkish sentences read as dovish. 7 of 10 dovish read as hawkish. |

---

## How to reproduce every number

```
.venv/bin/python -m pytest -q            # the module tests
.venv/bin/python build_report.py         # this report
.venv/bin/python build_deck.py           # the presentation
.venv/bin/streamlit run app.py           # the dashboard
```

The notebook `notebooks/perceptual_bias_analysis.ipynb` repeats the same
pipeline step by step and prints the same values.

## The strength of each claim

| Class | What it covers |
|---|---|
| Association | Every model coefficient, and the gap itself. No identification is claimed. |
| Predictive precedence | The Granger tests. A rejection would not license a causal reading. |
| Instrumented | The Rigobon-Sack IV, and only in the direction media to gap, under the assumption stated in the method notes. |
| Measured failure | The FinBERT probe and the swap tests. These measure an instrument, not the economy. |
| Not available | A result the project specified and does not have. The report names each one. |
