"""Compare feature budgets on identical training folds (run from the repo root)."""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reliefe import FeatureRanker


def benchmark():
    x, y = load_breast_cancer(return_X_y=True)
    # Small demo with a fixed subset. Keep the same rows and folds for every method.
    x, y = x[::4], y[::4]
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    for method in ("all", "random", "relieff", "f", "forest"):
        budgets = (x.shape[1],) if method == "all" else (3, 8)
        for k in budgets:
            steps = [] if method == "all" else [FeatureRanker(method, top_k=k)]
            pipeline = make_pipeline(*steps, StandardScaler(), LogisticRegression(max_iter=1000))
            scores = cross_val_score(pipeline, x, y, cv=splits, scoring="balanced_accuracy",
                                     error_score="raise")
            print(f"{method:8s} k={k:2d} balanced accuracy={scores.mean():.3f}")


if __name__ == "__main__":
    benchmark()
