# %% [markdown]
#
# # Handling prevalence mismatch in imbalanced classification problems
#
# The purpose of this study is to illustrate how to handle extreme class
# imbalance in the common setting where the data acquisition process does not
# reflect the true class imbalance of the target distribution.
#
# This setting is quite common in practice: for instance, in medical diagnosis,
# the data acquisition process might collect data for all known cases of a rare
# disease (positive class): since the cases are rare, we don't want to waste
# any of them. However, it would be infeasible to collect data for all subjects
# not known to have the disease (negative class) because it would be too costly
# (and might also cause ethical and privacy issues). Instead, we might randomly
# sample subjects from the population to collect data about them (and check
# that they do not have the disease).
#
# As a result, the prevalence of the positive class in the observed data does
# not reflect the true prevalence of the positive class in the target
# population. Still, we want to be able to precisely estimate the performance
# of our computer-aided diagnosis system when deployed in the target
# population, only from the observed data at hand and ensure that its
# probabilistic predictions are as accurate as possible in the deployed
# setting.
#
# The lack of match of prevalence between the collected dataset and the
# deployment setting is a result of cost or computational constraints that
# prevent us from training and evaluating our predictive model on the full
# target population. It is not the result of a modeling choice of the
# data-scientist.
#
# Note that the classes are extremely imbalanced in the target population but
# can also be imbalanced in the collected dataset, although likely less so.
#
# In this study, we will illustrate how to handle such a setting by correcting
# the observed data to better match the target population. We will use
# synthetic data generated from a known data generating process so as to make
# it possible to check that our evaluation results on the observed data can be
# correctly interpreted in the context of the target population.
#
# ## Data generating process
#
# Let's define a "true" data generating process that represent some fundamental
# mechanism about the world. The true data generating process is generally
# unknown, and the goal of machine learning is to approximate it as closely as
# possible from a finite sample of data points.
#
# Here, we want to simulate a binary classification problem where the target
# variable is a binary variable with a positive class that is very rare.
#
# The typical application domain could be medical diagnosis where the feature
# values are physiological measurements and the target variable is a binary
# indicator for the presence of a rare disease of interest. Other application
# domains with extreme imbalance are fraud detection, credit scoring,
# predictive maintenance, etc.
#
# For the sake of simplicity, we start this study by assuming that the data
# generating process is a linear model with a logistic link function: the
# features influence the probability of developing the disease but we expect
# them to provide only a partial information about the true risk as other
# unobserved factors may also influence disease development. We assume the
# unobserved factors to be independent of the observed features and all
# distribution to be stationary over time.
#
# We will later relax the linearity assumption but other assumptions will still
# apply.

# %%
import numpy as np
import pandas as pd
from scipy.special import expit, logit

rng = np.random.default_rng(0)
dtype = np.float32  # use float32 to save memory
n_features = 5
true_coef = rng.normal(size=n_features).astype(dtype)
true_intercept = -6


def sample_from_linear_model(true_coef, true_intercept, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(dtype)
    Z = X @ true_coef
    true_positive_proba = expit(Z + true_intercept)
    y = rng.binomial(n=1, p=true_positive_proba)
    true_proba = np.hstack(
        [1 - true_positive_proba[:, np.newaxis], true_positive_proba[:, np.newaxis]]
    )

    # create pandas data structures for convenience
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")
    return X, y, true_proba


# %% [markdown]
#
# The world can generate a lot of data from a fixed stationary process. Some of that data
# cannot be accessed at the time of the study but we generate it anyway to be able to
# compute metrics on the future population.

# %%
n_samples = 30_000_000
X_past, y_past, true_proba_past = sample_from_linear_model(
    true_coef, true_intercept, n_samples, seed=0
)
X_future, y_future, true_proba_future = sample_from_linear_model(
    true_coef, true_intercept, n_samples, seed=1
)

# %%
X_past.memory_usage().sum() / 1e6  # in MB

# %%
y_past.sum()

# %%
true_positive_rate_past = y_past.mean()
true_positive_rate_past

# %% [markdown]
#
# Let us start this study by cheating and compute the metrics obtained from the
# true probabilities computed from the data generating process for the future
# feature values.

# %%
true_positive_rate_future = y_future.mean()
true_positive_rate_future

# %%
pd.Series(true_proba_future[:, 1]).describe()

# %%
from sklearn.metrics import roc_auc_score, log_loss


roc_auc_score(y_future, true_proba_future[:, 1])

# %%
log_loss(y_future, true_proba_future)

# %%
log_loss(y_past, true_proba_past)

# %% [markdown]
#
# ### Questions:
# - What is the best possible value for the ROC AUC score (in general)?
# - What is the best possible value for the log-loss (in general)?
# - Why cannot we reach these values even when using the true probabilities?
# - What is the name name of the error rate obtained when computed from the
#   true probabilities?

# %%
# # TODO: write your answers here before scrolling down.
#
#
#
#
#
#
#
#
#
#
#
#

# %% [markdown]
#
# ### Answers:
# - The best possible value for the ROC AUC score is 1.0.
# - The best possible value for the log-loss is 0.0.
# - We cannot reach these values because the true probabilities reflect some
#   inherent uncertainty in the data: the value of the target variable is not
#   deterministic given the available features. The outcome can also be
#   influenced by unobserved factors that are not captured by the features.
# - The error rate obtained from the true probabilities is called the "Bayes
#   error rate" or "irreducible error rate".


# %% [markdown]
#
# Because we are curious, let's check that the use of Logistic Regression on
# such a large number of data points would be able to approximately recover the
# true coefficients and intercept.

# %%
from sklearn.linear_model import LogisticRegression

cheating_model = LogisticRegression(penalty=None).fit(X_future, y_future)
cheating_model.coef_, cheating_model.intercept_
# %%
true_coef, true_intercept

# %%
# TODO: redo the bar plot of the previous notebook to compare the coefficients.

# %%
roc_auc_score(y_future, cheating_model.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, cheating_model.predict_proba(X_future))

# %% [markdown]
#
# ## Prevalence shift induced by the data acquisition process
#
# Subsample the dataset based to collect all the positives and a random sample
# of negatives from the past data. Mere mortal data scientists cannot
# pd.read_csv from the future, unfortunately.
#
# We subsample the negatives to reflect the fact that the negative (control)
# data is less interesting than the rare positive cases: as such the negative
# data is in general not stored fully in databases (or even not acquired at all).

# %%
X_positive = X_past[y_past == 1].copy()
y_positive = y_past[y_past == 1].copy()

rng = np.random.default_rng(0)
negative_indices = rng.choice(
    np.arange(len(y_past))[y_past == 0],
    size=3 * len(y_positive),
    replace=False,
)

X_observed = pd.concat([X_positive, X_past.iloc[negative_indices]])
y_observed = pd.concat([y_positive, y_past.iloc[negative_indices]])

X_observed.shape
# %%
observed_positive_rate = y_observed.sum() / len(y_observed)
observed_positive_rate

# %% [markdown]
#
# Now that the data is collected, we can train-test split it to reflect what a data scientist
# would do in practice.
#
# Note: in a real-world setting, we would rather split the data on a temporal
# basis instead. This would allow to assess if that the stationarity of the
# data generating process is preserved or not.

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X_observed, y_observed, test_size=0.5, random_state=0
)

# %%
logreg_uncorrected = LogisticRegression(penalty=None).fit(X_train, y_train)
logreg_uncorrected.coef_, logreg_uncorrected.intercept_

# %%
roc_auc_score(y_future, logreg_uncorrected.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, logreg_uncorrected.predict_proba(X_future))

# %%
class_weight_for_prevalence_correction = {
    0: (1 - true_positive_rate_past) / (1 - y_train.mean()),
    1: true_positive_rate_past / y_train.mean(),
}
logreg_weighted = LogisticRegression(
    penalty=None, class_weight=class_weight_for_prevalence_correction
).fit(X_train, y_train)
logreg_weighted.coef_, logreg_weighted.intercept_

# %%
roc_auc_score(y_future, logreg_weighted.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, logreg_weighted.predict_proba(X_future))

# %% [markdown]
#
# Let's check that we can get the same results using sample_weight instead of
# class_weight.

# %%
sample_weight_for_prevalence_correction = np.where(
    y_train == 0,
    class_weight_for_prevalence_correction[0],
    class_weight_for_prevalence_correction[1],
)
logreg_weighted2 = LogisticRegression(penalty=None).fit(
    X_train, y_train, sample_weight=sample_weight_for_prevalence_correction
)
logreg_weighted2.coef_, logreg_weighted2.intercept_
np.testing.assert_allclose(logreg_weighted.coef_, logreg_weighted2.coef_)
np.testing.assert_allclose(logreg_weighted.intercept_, logreg_weighted2.intercept_)

# %%
# TODO: link to https://stats.stackexchange.com/a/68726/2150
from scipy.special import logit

logreg_intercept_corrected = LogisticRegression(penalty=None).fit(X_train, y_train)
logreg_intercept_corrected.intercept_ += logit(true_positive_rate_past) - logit(
    y_train.mean()
)
logreg_intercept_corrected.coef_, logreg_intercept_corrected.intercept_

# %%
roc_auc_score(y_future, logreg_intercept_corrected.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, logreg_intercept_corrected.predict_proba(X_future))

# %% [markdown]
#
# Let's now consider a more generic post-hoc imbalance correction that does not
# require the base model to be a Logistic Regression model with an explicit
# `intercept_` parameter. Instead, we assume that the prediction of the base
# model to be computed by applying the `expit` (a.k.a. the logistic sigmoid)
# function to some estimate of the log-odds ratio for each data point. This is
# the case for gradient boosting models where the log-odds are estimated by the
# sum of the predictions of the individual trees in the ensemble for instance.

# %%
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class PostHocPrevalenceCorrection(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, target_positive_rate=0.5):
        self.estimator = estimator
        self.target_positive_rate = target_positive_rate

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            estimator = LogisticRegression()
        else:
            estimator = clone(self.estimator)

        self.estimator_ = estimator.fit(X, y, **fit_params)
        self.observed_positive_rate_ = y.mean()
        return self

    def predict_proba(self, X):
        uncorrected_proba = self.estimator_.predict_proba(X)
        uncorrected_log_odds_ratios = np.log(
            (uncorrected_proba[:, 1] / uncorrected_proba[:, 0])
        )
        corrected_log_odds_ratios = (
            uncorrected_log_odds_ratios
            + logit(self.target_positive_rate)
            - logit(self.observed_positive_rate_)
        )
        corrected_proba = np.zeros_like(uncorrected_proba)
        corrected_proba[:, 1] = expit(corrected_log_odds_ratios)
        corrected_proba[:, 0] = 1 - corrected_proba[:, 1]
        return corrected_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)


logreg_post_hoc = PostHocPrevalenceCorrection(
    estimator=LogisticRegression(penalty=None),
    target_positive_rate=true_positive_rate_past,
).fit(X_train, y_train)
logreg_post_hoc.estimator_.coef_, logreg_post_hoc.estimator_.intercept_

# %% [markdown]
#
# For a Logistic Regression model, this generic post-hoc imbalance correction
# should be strictly equivalent to our previously introduced post-hoc
# correction of the intercept parameter:

# %%
np.testing.assert_allclose(
    logreg_post_hoc.predict_proba(X_future),
    logreg_intercept_corrected.predict_proba(X_future),
)

# %% [markdown]
#
# Therefore we should get the same metrics as before:

# %%
roc_auc_score(y_future, logreg_post_hoc.predict_proba(X_future)[:, 1])
# %%
log_loss(y_future, logreg_post_hoc.predict_proba(X_future))


# %%
class ElkanPrevalenceCorrection(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, target_positive_rate=0.5):
        self.estimator = estimator
        self.target_positive_rate = target_positive_rate

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            estimator = LogisticRegression()
        else:
            estimator = clone(self.estimator)

        self.estimator_ = estimator.fit(X, y, **fit_params)
        self.observed_positive_rate_ = y.mean()
        return self

    def predict_proba(self, X):
        uncorrected_proba = self.estimator_.predict_proba(X)

        corrected_proba = np.zeros_like(uncorrected_proba)
        # p - pb
        numerator = uncorrected_proba[:, 1] - (
            uncorrected_proba[:, 1] * self.observed_positive_rate_
        )
        # b - pb + b'p - b'b
        denominator = (
            self.observed_positive_rate_
            - (uncorrected_proba[:, 1] * self.observed_positive_rate_)
            + (self.target_positive_rate * uncorrected_proba[:, 1])
            - (self.target_positive_rate * self.observed_positive_rate_)
        )
        # b' * numerator / denominator
        corrected_proba[:, 1] = self.target_positive_rate * numerator / denominator
        corrected_proba[:, 0] = 1 - corrected_proba[:, 1]
        return corrected_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)


logreg_elkan = ElkanPrevalenceCorrection(
    estimator=LogisticRegression(penalty=None),
    target_positive_rate=true_positive_rate_past,
).fit(X_train, y_train)
logreg_elkan.estimator_.coef_, logreg_elkan.estimator_.intercept_

# %% [markdown]
#
# For a Logistic Regression model, this generic post-hoc imbalance correction
# should be strictly equivalent to our previously introduced post-hoc
# correction of the intercept parameter:

# %%
np.testing.assert_allclose(
    logreg_elkan.predict_proba(X_future),
    logreg_intercept_corrected.predict_proba(X_future),
)

# %% [markdown]
#
# Therefore we should get the same metrics as before:

# %%
roc_auc_score(y_future, logreg_elkan.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, logreg_elkan.predict_proba(X_future))

# %% [markdown]
#
# ### Questions:
#
# Consider the two kinds of post-hoc imbalance correction presented above:
# - what happens when we pass `target_positive_rate=y_train.mean()`?
# - is it possible to get `predict_proba` values that are not in `[0, 1]` by
#   setting extreme values for `target_positive_rate`?
#
# Hint: the `expit` function is takes any real number as input and returns a
# value in [0, 1]:
#
# $$
# \text{expit}(x) = \frac{1}{1 + e^{-x}} \in [0, 1] \text{ for any } x \in \mathbb{R}
# $$
#
# while its inverse function is the logit function:
#
# $$
# \text{logit}(p) = \log\left(\frac{p}{1 - p}\right) \in \mathbb{R} \text{ for any } p \in [0, 1]
# $$

# %% [markdown]
#
# ## Evaluating models on observed test data
#
# As we saw above, we have several ways to correct the predictions of a
# classifier to account for difference of class distribution between the
# observed training data and the target population.
#
# However, data-scientists do not have access to the future population to assess
# the expected performance of their model on the future population.
#
# Instead, they can only evaluate the model on the observed test data. However,
# the observed test data: in our case, the number of negatives cases is much lower
# in the observed data (train or test) than in the target population.
#
# If we naively evaluate the model on the observed test data, we will get
# misleading results: the model will be evaluated on a test set that does not
# %%
log_loss(y_test, logreg_uncorrected.predict_proba(X_test))

# %%
log_loss(y_test, logreg_intercept_corrected.predict_proba(X_test))

# %% [markdown]
#
# Therefore, we need to apply the same class ratio correction to the evaluation set.
# Scikit-learn metric function usually do not provide a `class_weight` parameter but they
# often provide a `sample_weight` parameter.


# %%
sample_weight_test = np.where(
    y_test == 0,
    (1 - true_positive_rate_past) / (1 - y_test.mean()),
    true_positive_rate_past / y_test.mean(),
)

# %%
log_loss(
    y_test, logreg_uncorrected.predict_proba(X_test), sample_weight=sample_weight_test
)

# %%
log_loss(y_future, logreg_uncorrected.predict_proba(X_future))

# %%
log_loss(
    y_test,
    logreg_intercept_corrected.predict_proba(X_test),
    sample_weight=sample_weight_test,
)

# %%
log_loss(y_future, logreg_intercept_corrected.predict_proba(X_future))

# %% [markdown]
#
# ## Fitting non-linear models

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

gbdt_uncorrected = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)

# %%
roc_auc_score(y_future, gbdt_uncorrected.predict_proba(X_future)[:, 1])
# %%
log_loss(y_future, gbdt_uncorrected.predict_proba(X_future))

# %%
gbdt_weighted = HistGradientBoostingClassifier(
    random_state=0,
    class_weight=class_weight_for_prevalence_correction,
).fit(X_train, y_train)
# %%
roc_auc_score(y_future, gbdt_weighted.predict_proba(X_future)[:, 1])

# %%
log_loss(y_future, gbdt_weighted.predict_proba(X_future))

# %%
gbdt_post_hoc = PostHocPrevalenceCorrection(
    estimator=HistGradientBoostingClassifier(random_state=0),
    target_positive_rate=true_positive_rate_past,
).fit(X_train, y_train)

# %%
roc_auc_score(y_future, gbdt_post_hoc.predict_proba(X_future)[:, 1])
# %%
log_loss(y_future, gbdt_post_hoc.predict_proba(X_future))
# %%
