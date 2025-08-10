# %% [markdown]
#
# # Classification with imbalanced datasets: effects and solutions
#
# Classification with imbalanced datasets refers to the problem of training and
# predicting on a dataset where the classes frequencies in the target variable are
# not balanced. Such datasets are common in practice, for example in fraud detections or
# medical diagnosis since the event of interest is rare. It should also be noted that
# the difference in class frequencies can vary from being mild (1 event of interest per
# 10 samples) to extreme (1 event of interest per 1,000 samples).
#
# Recommendations in the scientific literature boiled down to resampling the training
# dataset in order to optimize a thresholded classification metric. However, it is only
# recently that the problem has been looked at from a different perspective. Indeed,
# resampling has been shown to have a detrimental effect on the calibration of the
# probabilities of classifiers.
#
# This notebook explores the problem of imbalanced datasets and in particular aspects
# related to the calibration of the classifier probabilities, the tuning of decision
# thresholds and its impact on the classification metrics.

# %% [markdown]
# - Discuss the problem of imbalanced datasets in classification settings
# - Relate to the meaning of rare events and thus low probability of occurrence
# - Make the distinction between probability, well-calibrated probability, and
#   low statistical decision metric by default
# - Show the issue with resampling on the calibration of the model
# - Show the effect of changing cut-off values on the decision metric
# - Finally, show the how to tune the cut-off value to maximize a metric or a
#   metric under constraints


# %% [markdown]
#
# ## What do we call imbalanced datasets in classification settings?
#
# In real-world applications, it commonly happens that we are interested in predicting
# rare events, e.g. frauds, rare diseases, rare climatic events, etc. Simplifying this
# problem to a binary outcome, it means that the probability for this rare event to
# happen is rather low in comparison to the probability of the rare event not to happen.
#
# To later cover the implications of class imbalance, we first generate a synthetic
# dataset for which we control the success rate of the positive class. The generative
# process below is defined as follows:
#
# - We generate a vector of coefficients `true_coef` of shape `(n_features,)` where
#   each element is a standard normal random variable. In short, it is the true model
#   that we would like to learn.
# - We generate a matrix of features `X` of shape `(n_samples, n_features)` where each
#   column is a standard normal random variable.
# - We compute the linear predictor `Z` as the dot product of the features and the
#   vector of coefficients `true_coef`.
# - The linear predictor `Z` is transformed into class probabilities using the sigmoid
#   function. To create rare positive events, we shift the intercept of the sigmoid
#   function.
# - Finally, we generate a binary target variable `y` where each event is sampled by
#   drawing a sample from a binomial distribution with `n=1` and `p` being the
#   probability of the positive class previously computed.

# %%
import numpy as np
import pandas as pd
from scipy.special import expit

rng = np.random.default_rng(0)
n_samples, n_features = 1_000_000, 5

true_coef = rng.normal(size=n_features)
X = rng.normal(size=(n_samples, n_features))
Z = X @ true_coef
y = rng.binomial(n=1, p=expit(Z - 4))

# create pandas data structures for convenience
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
y = pd.Series(y, name="target")

# %% [markdown]
#
# Let's look at the true target and especially the class frequencies and absolute
# counts.

# %%
print(f"Class frequencies:\n {y.value_counts(normalize=True) * 100}")
print(f"Class counts:\n {y.value_counts()}\n")

# %% [markdown]
#
# Looking at the true target distribution, we therefore observe that the probability
# for a sample to be the positive class with label 1 is rare (~2.5%). When it comes to
# absolute counts, because we generated a 1,000,000 samples, the number of events of
# interest is rather high (25,000).
#
# A particular challenge when dealing real-world imbalanced datasets is that the number
# of available samples of the rare event can be usually low even with a large number
# of samples. Therefore, it is always important to check the absolute counts of the
# rare event and if the dataset contains less than 1,000 samples of the rare event,
# then you are exactly in the same situation as having a dataset with a low number of
# samples with all related challenges (e.g. large variance of the estimator, weak
# signal, etc.).
#
# ## Learning a predictive model
#
# Here, we know that our generative process is a linear relationship between the
# features and the target. Therefore, a linear predictive model is perfectly suited to
# learn the true model. We therefore train a logistic regression model.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)

# %% [markdown]
#
# Let's check if the learned model is able to recover the true model.

# %%
comparison_coef = pd.DataFrame(
    {"Generative model": true_coef, "Learned model": model.coef_.flatten()},
    index=model.feature_names_in_,
)
ax = comparison_coef.plot.barh()
_ = ax.set(
    title="Comparison of the true and learned model coefficients",
    xlabel="Coefficient value",
    ylabel="Feature",
)

# %% [markdown]
#
# We observe that the learned model is able to recover the true model coefficients.
# However, be aware that it is not necessarily the case. Let's do a small exercise that
# illustrates when one of the assumptions to recover the true model is not met.
#
# ### Exercise
#
# Below, write a small function that embed the generative process that we defined above.
# This time only generate 10,000 samples, train a logistic regression model and check
# the learned model coefficients.
#
# Do you recover the true model coefficients? If not, what is the reason?

# %% [markdown]
#
# ### Solution


def generate_imbalanced_dataset(n_samples=10_000, n_features=5, seed=0):
    rng = np.random.default_rng(seed)

    true_coef = rng.normal(size=n_features)
    X = rng.normal(size=(n_samples, n_features))
    Z = X @ true_coef
    y = rng.binomial(n=1, p=expit(Z - 4))

    # create pandas data structures for convenience
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    return X, y


X_exercise, y_exercise = generate_imbalanced_dataset(n_samples=10_000)
model_exercise = LogisticRegression().fit(X_exercise, y_exercise)

comparison_coef_exercise = pd.DataFrame(
    {"Generative model": true_coef, "Learned model": model_exercise.coef_.flatten()},
    index=model_exercise.feature_names_in_,
)
ax = comparison_coef_exercise.plot.barh()
_ = ax.set(
    title="Comparison of the true and learned model coefficients",
    xlabel="Coefficient value",
    ylabel="Feature",
)

# %% [markdown]
#
# We observe that we have a larger difference between the coefficients of the true
# generative process and the learned model. The reason is that the coefficients of the
# generative process can be recovered under the following assumptions:
#
# - An infinite number of samples is available. Therefore, with a larger number of
#   samples, the coefficients of the predictive model will get closer to the true
#   coefficients.
# - The predictive model should be well specified. In other words, if our predictive
#   model is not flexible enough then it will underfit and not recover all the signal
#   of the true model.
#
# There is an additional assumption for which we need to study the the probabilities
# estimated by our predictive model.

# %%
y_proba = model.predict_proba(X)
y_proba = pd.DataFrame(y_proba, columns=["p_hat(y=0)", "p_hat(y=1)"])
y_proba

# %% [markdown]
#
# Our predictive model is capable of estimating the probabilities of the class of
# interest (i.e. `p_hat(y=1)`). However, those numbers are estimated and does not
# necessarily reflect the true probabilities. Here, we can compute the mean of the
# estimated probabilities and check if we are close to the true probability of the
# positive class.

# %%
y_proba.mean() * 100

# %%
_ = y_proba.plot.hist(
    bins=100, figsize=(10, 5), subplots=True, layout=(1, 2), sharey=True
)

# %% [markdown]
#
# For our example, we are indeed close to the true frequency. The reason is that the
# algorithm used `LogisticRegression` minimizes a "strictly proper" scoring rule. If not
# using such loss function, they are also not theoretical guarantees that the estimated
# probabilities will be close to the true
# probabilities.
#
# To conclude, the three above conditions work together: the strictly proper scoring
# rule provides the right objective, the well-specified model ensures the true
# coefficients exist within the model's parameter space, and infinite samples allow
# the optimization to converge to the global optimum that corresponds to these true
# coefficients.
#
# With these information, we would expect our classifier to be well calibrated. We can
# check that by plotting the calibration curve.

# %%
from sklearn.calibration import CalibrationDisplay

display = CalibrationDisplay.from_estimator(model, X, y, n_bins=10, strategy="quantile")
display.plot()

# %% [markdown]
#
# Since we have rare events, than only few samples have high probability and the
# quantile-based strategy will not show a curve on the right side of the plot.
# Let's zoom in on the plot to see the curve.

# %%
display.plot()
axis_lim = (
    min(display.prob_true.min(), display.prob_pred.min()) * 0.9,
    max(display.prob_true.max(), display.prob_pred.max()) * 1.1,
)
display.ax_.set(xlim=axis_lim, ylim=axis_lim)

# %% [markdown]
#
# We observe that the logistic regression is well calibrated as the curve is close to
# the diagonal line. It therefore means that the probabilities estimated by the model
# are close to the true probabilities.
#
# ## The problem behind decision making

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
