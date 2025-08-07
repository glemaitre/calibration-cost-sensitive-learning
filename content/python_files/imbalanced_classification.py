# %% [markdown]
#
# # Classification with imbalanced datasets
#
# # TOC:
#
# - Discuss the problem of imbalanced datasets in classification settings
# - Relate to the meaning of rare events and thus low probability of occurrence
# - Make the distinction between probability, well-calibrated probability, and
#   low statistical decision metric by default
# - Show the issue with resampling on the calibration of the model
# - Show the effect of changing cut-off values on the decision metric
# - Finally, show the how to tune the cut-off value to maximize a metric or a
#   metric under constraints
#
# # TODO: describe what we call imbalanced datasets with classification settings

# %%
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_classification

# X, y = make_classification(
#     n_samples=1_000_000,
#     n_classes=2,
#     weights=[0.99, 0.01],
#     random_state=0,
# )

rng = np.random.default_rng(0)
n_samples = 1_000_000
n_features = 5

true_coef = rng.normal(size=n_features)
X = rng.normal(size=(n_samples, n_features))
Z = X @ true_coef
y = rng.binomial(n=1, p=expit(Z * 1 - 4))

X, y = pd.DataFrame(X), pd.Series(y)

# %%
y.value_counts(normalize=True) * 100

# %% [markdown]
#
# Looking at the true target distribution, we therefore observe that the probability
# for a sample to be the positive class with label 1 is rare (~1%).

# %%
# TODO: Warn about the small number of samples of the minority class with high imbalance
# ratio even with a large total number of samples.

y.value_counts()

# %%
# TODO: point out the difference between the true and estimated probability.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)
y_proba = model.predict_proba(X)
y_proba = pd.DataFrame(y_proba, columns=["p_hat(y=0 | X)", "p_hat(y=1 | X)"])

# %%
y_proba

# %%
# TODO: rename columns the columns `p_hat(y=0)` and `p_hat(y=1)`
# TODO: to look "balanced property" (maybe definition)
y_proba.mean()

# %%
y_proba.plot.hist(bins=100, figsize=(10, 5), subplots=True, layout=(1, 2), sharey=True)

# %%
from sklearn.calibration import CalibrationDisplay

disp = CalibrationDisplay.from_estimator(model, X, y, n_bins=10, strategy="quantile")

# %%
disp.plot()

axis_lim = (
    min(disp.prob_true.min(), disp.prob_pred.min()) * 0.9,
    max(disp.prob_true.max(), disp.prob_pred.max()) * 1.1,
)
disp.ax_.set(xlim=axis_lim, ylim=axis_lim)

# %% [markdown]
# TODO: introduce decision making

# %%
from sklearn.metrics import classification_report

print(classification_report(y, model.predict(X)))

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
(y_proba.iloc[:, 1] > 0.5).value_counts()

# %% [markdown]
#
# # What people naively do and why you should not do it

# %%
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

model = make_pipeline(
    RandomUnderSampler(sampling_strategy=0.7, random_state=0), LogisticRegression()
)
model.fit(X, y)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

# %%
from sklearn.calibration import CalibratedClassifierCV

model = CalibratedClassifierCV(
    make_pipeline(
        RandomUnderSampler(sampling_strategy=0.1, random_state=0), LogisticRegression()
    ),
    method="isotonic",
    ensemble=False,
)
model.fit(X, y)

# %%
disp = CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

axis_lim = (
    min(disp.prob_true.min(), disp.prob_pred.min()) * 0.9,
    max(disp.prob_true.max(), disp.prob_pred.max()) * 1.1,
)
_ = disp.ax_.set(xlim=axis_lim, ylim=axis_lim)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %% [markdown]
#
# # What you should do instead

# %%
from sklearn.model_selection import FixedThresholdClassifier

model = FixedThresholdClassifier(
    LogisticRegression(), threshold=y.value_counts(normalize=True)[1]
)
model.fit(X, y)

# %%
disp = CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

axis_lim = (
    min(disp.prob_true.min(), disp.prob_pred.min()) * 0.9,
    max(disp.prob_true.max(), disp.prob_pred.max()) * 1.1,
)
_ = disp.ax_.set(xlim=axis_lim, ylim=axis_lim)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _CurveScorer

thresholds = np.linspace(0, 1, 15)
precision_curve_scorer = _CurveScorer.from_scorer(
    get_scorer("precision"), response_method="predict_proba", thresholds=thresholds
)
recall_curve_scorer = _CurveScorer.from_scorer(
    get_scorer("recall"), response_method="predict_proba", thresholds=thresholds
)

precision_scores, precision_thresholds = precision_curve_scorer(model, X, y)
recall_scores, recall_thresholds = recall_curve_scorer(model, X, y)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(precision_thresholds, precision_scores, marker="+", label="Precision")
ax.plot(recall_thresholds, recall_scores, marker="+", label="Recall")

# TODO: maybe plot precision recall curve + annotation
# # Annotate threshold values on markers
# for i, (threshold, score) in enumerate(zip(precision_thresholds, precision_scores)):
#     ax.annotate(
#         f"{threshold:.2f}",
#         (threshold, score),
#         textcoords="offset points",
#         xytext=(5, 0),
#         ha="left",
#         fontsize=8,
#     )

# for i, (threshold, score) in enumerate(zip(recall_thresholds, recall_scores)):
#     ax.annotate(
#         f"{threshold:.2f}",
#         (threshold, score),
#         textcoords="offset points",
#         xytext=(5, 0),
#         ha="left",
#         fontsize=8,
#     )

ax.set(xlabel="Threshold", ylabel="Score")
ax.legend()

# %%
# Interactive version with plotly
import plotly.graph_objects as go

fig_plotly = go.Figure()
fig_plotly.add_trace(
    go.Scatter(
        x=precision_thresholds,
        y=precision_scores,
        mode="lines+markers",
        name="Precision",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.3f}",
    )
)
fig_plotly.add_trace(
    go.Scatter(
        x=recall_thresholds,
        y=recall_scores,
        mode="lines+markers",
        name="Recall",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.3f}",
    )
)
fig_plotly.update_layout(
    xaxis_title="Threshold",
    yaxis_title="Score",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest",
    width=600,
    height=500,
)
fig_plotly.show()

# %%
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(model, X, y)

# %%
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import TunedThresholdClassifierCV


def maximize_precision_under_constrained_recall(y_true, y_pred, recall_level):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if recall < recall_level:
        return -np.inf
    return precision


model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(),
    scoring=make_scorer(maximize_precision_under_constrained_recall, recall_level=0.3),
    n_jobs=-1,
).fit(X, y)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
model.best_threshold_

# %%
# TODO: go and tune a business metric and link to the scikit-learn example.
