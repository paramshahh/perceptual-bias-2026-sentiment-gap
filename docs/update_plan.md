# Prospective update plan

**Why this document exists.** The assignment says: *"If your project finishes
before election day, state the cutoff and provide a prospective update plan."*
This is that plan.

**Language.** ASD-STE100 Simplified Technical English.

---

## 1. The cutoff

| Item | Value |
|---|---|
| Analysis cutoff | **2026-10-16** |
| Last complete macro month in the panel | 2026-09 |
| Last month of the media corpus | 2026-08 |
| Months in the monitoring holdout | 20 (2025-01 to 2026-09) |
| Post-election data in this project | **none** |

Election day is after the cutoff. So no result in this project can be a
statement about what happened at the election. Each artifact prints the cutoff
date, so a reader cannot mistake an old build for a current one.

---

## 2. What arrives, and when

| Release | Who publishes it | When | What it changes |
|---|---|---|---|
| Employment Situation (U-3) | BLS | First Friday of the next month | One new row of `unemployment` |
| Personal Income and Outlays (PCE) | BEA | About 28 days after the month | One new row of `pce_yoy` and `pce_level` |
| GDP advance estimate | BEA | About 30 days after the quarter | One new quarterly step |
| Surveys of Consumers | University of Michigan | Preliminary near day 15, final on the last Friday of the month | One new `sentiment` value, which is one new gap value |
| PMMS mortgage rate | Freddie Mac | Weekly | The monthly mean |
| Household debt service ratio | Federal Reserve | Quarterly, about 90 days late | One new quarterly value |
| GDELT GKG files | GDELT | Every 15 minutes | New media rows |

**The binding release is the Michigan survey.** A new gap value needs it. The
other series only move the prediction.

---

## 3. The order to re-run things

Run these in order. Each step depends on the step before it.

```
1.  .venv/bin/python scripts/pull_gkg.py            # new media days
2.  .venv/bin/python scripts/build_media_series.py  # monthly media + framing
3.  .venv/bin/python scripts/build_svd.py           # topics and the spectrum
4.  .venv/bin/python build_report.py                # panel, models, tables, PDF
5.  .venv/bin/python build_deck.py                  # the presentation
6.  .venv/bin/streamlit run app.py                  # check the dashboard
```

The panel, the baseline, the gap and all three models are rebuilt inside step 4.
The panel is always rebuilt from the sources, so a new release enters
automatically. Nothing is cached across a build.

---

## 4. What must move, and what must not

This is the check that catches a broken update.

**Must move when a new month arrives:**

* the number of months tested in the gap summary;
* the last date printed on every artifact;
* the media month count, if new GKG days were pulled.

**Must not move:**

* the training window, which stays 2010-01 to 2024-12. The holdout must stay out
  of sample. If the training window moves, the out-of-sample claim is void.
* the baseline coefficients, unless a source **revision** changes the history.
  A revision is a legitimate reason and it must be stated as the reason.
* the pre-registered conditions in BRAINSTORMING.md §8. They were fixed before
  the corpus existed. Changing them after seeing a result destroys the test.

**A warning about revisions.** BEA revises GDP and PCE. BLS revises
unemployment. A changed history changes the training fit. The real-time panel
keeps the publication date of each value, so a revision is visible rather than
silent. If a coefficient moves and no new month arrived, look for a revision
first and do not look for a bug.

---

## 5. How each hypothesis gets decided by the new data

The conditions are from BRAINSTORMING.md §8. They are repeated here so that the
update needs no interpretation.

| Hypothesis | It gains support when | It loses support when |
|---|---|---|
| **A.** Unmeasured economic pressure | The household controls keep shrinking the gap as months arrive, and they stay significant. | New months widen the gap while the household controls stay flat. |
| **B.** Media selection and framing | The media block improves the rolling-origin forecast, survives the FDR correction, and `level_vs_rate` keeps its predicted sign. | The media block stays insignificant after correction, or the forecast does not improve. |
| **C.** Measurement and timing | The independent comparison series stays flat while Michigan falls. | The comparison series falls with Michigan. |

**One decision rule in advance.** If the gap narrows sharply in the final
pre-election months and the media tone does not change, that is evidence against
B and for A or C. We write that rule down now, before those months exist.

---

## 6. What we would add if more time or budget existed

In priority order. Each entry says what it would buy.

1. **A second human annotator.** The brief asks for two independent annotators.
   One operator cannot produce an honest inter-annotator agreement statistic.
   This is the largest single gap in the project.
2. **Michigan's own sub-indices**, from `data.sca.isr.umich.edu`. They would give
   the Current Conditions and Expectations split that the brief asks for, and
   they would separate "how things are" from "where things are going". That split
   would discriminate between Hypothesis A and Hypothesis B directly.
3. **Full article text** instead of titles and URL slugs. A title measures
   emphasis. A body measures whether the emphasis was qualified.
4. **Exposure data.** Without it, no causal claim about readers is available at
   all (BRAINSTORMING.md §4.5).
5. **A larger social-media sample.** 365 tweets support an illustration and not a
   population claim.
6. **The randomized framing experiment** in BRAINSTORMING.md §7.4, with the
   institutional approval that it needs.

---

## 7. After election day

Three things become possible, and one trap opens.

**Possible:** the post-election months test whether the gap was about the
campaign. A gap that closes after the election, with no change in the macro
data, is evidence that the campaign period mattered. A gap that persists is
evidence that it did not.

**The trap.** That test is only valid if the conditions were written down first.
They are, in §5 above. A condition invented after seeing the post-election data
proves nothing.
