"""Small survival ranking and forward-chaining regression evaluation."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from reliefe import FeatureRanker, compare_rankings

rng = np.random.RandomState(9)
x = rng.normal(size=(80, 5))
event_time = np.exp(1 - x[:, 2])
censor_time = rng.exponential(4, size=len(x))
survival = np.column_stack((event_time <= censor_time, np.minimum(event_time, censor_time)))
ranker = FeatureRanker("survival-concordance").fit(x, survival)
print("Highest concordance association:", ranker.ranking()[0])

# Only past observations enter the two predictors for each forecast.
series = np.sin(np.arange(100) * .15) + .05 * rng.randn(100)
lagged = np.column_stack((series[1:-1], series[:-2]))
target = series[2:]
rankers = {"one lag": FeatureRanker("f", top_k=1, task="regression"), "both lags": None}
results = compare_rankings(lagged, target, rankers, Ridge(), TimeSeriesSplit(4).split(lagged),
                           scoring="neg_mean_absolute_error")
for name, record in results.items():
    print(name, "mean absolute error:", -record["fold_scores"].mean())
