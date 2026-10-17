"""Baseline macro model and the 2026 Sentiment Gap.

The baseline is deliberately the *naive* model a macroeconomist would reach for:
sentiment on unemployment, YoY inflation and real GDP growth. Its purpose is not
to fit well out of sample — it is to establish how badly headline macro data
fails to explain 2025-26 sentiment, which is the quantity the rest of the
project tries to explain.

Two estimators are fitted on the same design:

* **OLS with HAC (Newey-West) standard errors.** Monthly macro series are
  strongly autocorrelated; classical OLS standard errors would be badly
  understated. The point estimates are identical to plain OLS — only inference
  changes.
* **Ridge**, standardised inside each CV fold, with alpha chosen by
  forward-chaining cross-validation. Unemployment, inflation and growth co-move,
  so the OLS coefficients are individually unstable even when the fit is decent.
  Ridge is reported as a robustness check on the *predictions*, not as the
  headline model.

The Sentiment Gap is defined as ``observed - predicted`` on data the model never
saw, so a negative gap means sentiment is worse than macro fundamentals imply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASELINE_FEATURES: tuple[str, ...] = ("unemployment", "pce_yoy", "real_gdp_qoq")
TARGET = "sentiment"

# Newey-West lag truncation. 12 months covers a full seasonal cycle, the
# conventional choice for monthly macro data at this sample length.
HAC_LAGS = 12

RIDGE_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
RIDGE_CV_SPLITS = 5


class ModelError(Exception):
    """Raised when a model cannot be fitted or applied as specified."""


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    """Fail with `ModelError` before pandas can raise a bare `KeyError`.

    Callers catch `ModelError`; they should not have to also catch pandas'
    internal exception types to find out a feature was missing.
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ModelError(f"frame is missing required columns: {missing}")


@dataclass
class BaselineFit:
    """A fitted baseline, with everything the report needs to quote it."""

    features: tuple[str, ...]
    target: str
    n_train: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    ols_result: sm.regression.linear_model.RegressionResultsWrapper
    ridge: Pipeline
    ridge_alpha: float
    ridge_cv_ran: bool
    dropped_train_dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def params(self) -> pd.Series:
        return self.ols_result.params

    @property
    def r_squared(self) -> float:
        return float(self.ols_result.rsquared)

    @property
    def adj_r_squared(self) -> float:
        return float(self.ols_result.rsquared_adj)

    @property
    def aic(self) -> float:
        return float(self.ols_result.aic)

    @property
    def rmse_in_sample(self) -> float:
        resid = np.asarray(self.ols_result.resid, dtype=float)
        return float(np.sqrt(np.mean(resid**2)))

    def equation(self, precision: int = 2) -> str:
        """Human-readable fitted equation, for the report's prose."""
        params = self.params
        terms = [f"{params['const']:.{precision}f}"]
        for name in self.features:
            coefficient = params[name]
            sign = "+" if coefficient >= 0 else "-"
            terms.append(f"{sign} {abs(coefficient):.{precision}f}*{name}")
        return f"{self.target} = " + " ".join(terms)


@dataclass(frozen=True)
class PredictionReport:
    """Which rows were predicted, and which were dropped for missing inputs.

    The dropped dates matter editorially: the two freshest months of the source
    data lack a published PCE figure, so a report headlined "the 2026 Sentiment
    Gap" must not silently stop in August without saying why.
    """

    predicted_rows: int
    dropped_dates: tuple[pd.Timestamp, ...]
    dropped_reasons: tuple[str, ...]

    def summary(self) -> str:
        if not self.dropped_dates:
            return f"{self.predicted_rows} rows predicted, none dropped"
        dropped = ", ".join(stamp.strftime("%Y-%m") for stamp in self.dropped_dates)
        return f"{self.predicted_rows} rows predicted; dropped {dropped}"


def _design(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    _require_columns(frame, features)
    return sm.add_constant(frame.loc[:, list(features)], has_constant="add")


def _choose_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    splits: int = RIDGE_CV_SPLITS,
) -> tuple[float, bool]:
    """Pick Ridge alpha by forward-chaining CV, standardising inside each fold.

    `TimeSeriesSplit` never trains on observations later than those it validates
    on. Standardisation must happen *inside* the fold for that to mean anything:
    scaling on the full sample first would leak each validation period's own
    mean and variance back into its training data.

    Returns the chosen alpha and whether CV actually ran — with too few rows to
    split, it falls back to 1.0, and callers must be able to say so.
    """
    if len(y) <= splits + 1:
        logger.warning(
            "only %d rows available; skipping ridge CV and defaulting alpha=1.0", len(y)
        )
        return 1.0, False

    splitter = TimeSeriesSplit(n_splits=splits)
    best_alpha, best_error = 1.0, np.inf
    for alpha in alphas:
        errors: list[float] = []
        for train_idx, test_idx in splitter.split(x):
            model = Pipeline(
                [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
            ).fit(x[train_idx], y[train_idx])
            predicted = model.predict(x[test_idx])
            errors.append(float(np.mean((y[test_idx] - predicted) ** 2)))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_alpha, best_error = alpha, mean_error
    return best_alpha, True


def fit_baseline(
    train: pd.DataFrame,
    features: tuple[str, ...] = BASELINE_FEATURES,
    target: str = TARGET,
    hac_lags: int = HAC_LAGS,
    cv_splits: int = RIDGE_CV_SPLITS,
) -> BaselineFit:
    """Fit OLS (HAC errors) and Ridge on the training window."""
    columns = list(features) + [target]
    _require_columns(train, columns)
    subset = train.loc[:, columns]
    usable = subset.dropna()
    if usable.empty:
        raise ModelError("no complete rows available to fit the baseline")

    dropped_dates = tuple(subset.index.difference(usable.index))
    notes: list[str] = []
    if dropped_dates:
        shown = ", ".join(stamp.strftime("%Y-%m") for stamp in dropped_dates[:6])
        suffix = "" if len(dropped_dates) <= 6 else f" (+{len(dropped_dates) - 6} more)"
        notes.append(f"dropped {len(dropped_dates)} incomplete training rows: {shown}{suffix}")
        logger.info("baseline training dropped %d incomplete rows", len(dropped_dates))

    x = _design(usable, features)
    y = usable[target]
    ols_result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    raw_x = usable.loc[:, list(features)].to_numpy(dtype=float)
    raw_y = y.to_numpy(dtype=float)
    alpha, cv_ran = _choose_ridge_alpha(raw_x, raw_y, splits=cv_splits)
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(
        raw_x, raw_y
    )

    if cv_ran:
        notes.append(f"ridge alpha={alpha} chosen by {cv_splits}-fold forward-chaining CV")
    else:
        notes.append(f"ridge alpha={alpha} is a fallback default; CV did not run (too few rows)")
    notes.append(f"OLS inference uses Newey-West HAC errors, maxlags={hac_lags}")

    return BaselineFit(
        features=tuple(features),
        target=target,
        n_train=len(usable),
        train_start=usable.index.min(),
        train_end=usable.index.max(),
        ols_result=ols_result,
        ridge=ridge,
        ridge_alpha=alpha,
        ridge_cv_ran=cv_ran,
        dropped_train_dates=dropped_dates,
        notes=tuple(notes),
    )


def predict(
    fit: BaselineFit, frame: pd.DataFrame, alpha: float = 0.05
) -> tuple[pd.DataFrame, PredictionReport]:
    """Predict sentiment with OLS prediction intervals and a Ridge comparison.

    The interval is a *prediction* interval, not a confidence interval on the
    mean: the question is whether an individual month's observed sentiment is
    consistent with the model, so residual variance must be included.

    Rows missing any feature cannot be predicted. They are dropped, but the
    dropped dates and the reason are returned rather than discarded.
    """
    _require_columns(frame, fit.features)
    subset = frame.loc[:, list(fit.features)]
    usable = subset.dropna()
    if usable.empty:
        raise ModelError("no complete rows available to predict on")

    dropped_dates = tuple(subset.index.difference(usable.index))
    reasons: list[str] = []
    for stamp in dropped_dates:
        missing = [name for name in fit.features if pd.isna(subset.loc[stamp, name])]
        reasons.append(f"{pd.Timestamp(stamp):%Y-%m}: missing {', '.join(missing)}")
    if dropped_dates:
        logger.info("prediction dropped %d incomplete rows: %s", len(dropped_dates), "; ".join(reasons))

    x = _design(usable, fit.features)
    prediction = fit.ols_result.get_prediction(x)
    summary = prediction.summary_frame(alpha=alpha)
    ridge_predicted = fit.ridge.predict(usable.to_numpy(dtype=float))

    out = pd.DataFrame(
        {
            "predicted": summary["mean"].to_numpy(),
            "pi_lower": summary["obs_ci_lower"].to_numpy(),
            "pi_upper": summary["obs_ci_upper"].to_numpy(),
            "predicted_ridge": ridge_predicted,
        },
        index=usable.index,
    )
    out.index.name = frame.index.name or "date"
    report = PredictionReport(
        predicted_rows=len(out),
        dropped_dates=dropped_dates,
        dropped_reasons=tuple(reasons),
    )
    return out, report


def sentiment_gap(
    fit: BaselineFit, panel: pd.DataFrame, alpha: float = 0.05
) -> tuple[pd.DataFrame, PredictionReport]:
    """Observed minus predicted sentiment, with whether the miss is significant.

    ``outside_pi`` is the honest test of "the macro model broke": a gap is only
    evidence of structural breakdown if observed sentiment falls outside the
    model's own prediction interval. Where sentiment itself is missing,
    ``outside_pi`` is NA — *not* False, which would otherwise read as "the model
    explained this month fine" when nothing was tested.
    """
    _require_columns(panel, (fit.target,))
    predictions, report = predict(fit, panel, alpha=alpha)
    observed = panel.loc[predictions.index, fit.target]

    frame = predictions.copy()
    frame.insert(0, "observed", observed)
    frame["gap"] = frame["observed"] - frame["predicted"]
    frame["gap_ridge"] = frame["observed"] - frame["predicted_ridge"]
    outside = (frame["observed"] < frame["pi_lower"]) | (frame["observed"] > frame["pi_upper"])
    frame["outside_pi"] = outside.where(frame["observed"].notna(), other=pd.NA).astype("boolean")
    return frame, report


def gap_summary(gaps: pd.DataFrame) -> pd.Series:
    """Headline numbers the report and dashboard both quote."""
    if gaps.empty:
        raise ModelError("cannot summarise an empty gap frame")
    _require_columns(gaps, ("gap", "observed", "predicted", "outside_pi"))
    if not gaps["gap"].notna().any():
        raise ModelError("every gap is missing; nothing to summarise")

    worst = gaps["gap"].idxmin()
    worst_label = (
        worst.strftime("%Y-%m") if isinstance(worst, pd.Timestamp) else str(worst)
    )
    tested = gaps["outside_pi"].notna()
    outside = gaps.loc[tested, "outside_pi"].astype(bool)

    return pd.Series(
        {
            "n_months": float(len(gaps)),
            "n_tested": float(int(tested.sum())),
            "mean_gap": float(gaps["gap"].mean()),
            "median_gap": float(gaps["gap"].median()),
            "worst_gap": float(gaps.loc[worst, "gap"]),
            "worst_month": worst_label,
            "worst_observed": float(gaps.loc[worst, "observed"]),
            "worst_predicted": float(gaps.loc[worst, "predicted"]),
            "months_outside_pi": float(int(outside.sum())),
            "share_outside_pi": float(outside.mean()) if len(outside) else float("nan"),
        }
    )
