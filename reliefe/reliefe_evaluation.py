from .imports import LogisticRegression, f1_score, np


def evaluate_importances_with_logistic_regression(x_train, x_test, y_train,
                                                  y_test, importances, k):
    sorted_args = np.argsort(importances)[::-1][0:k]
    clf_lr = LogisticRegression(solver="lbfgs",
                                max_iter=100000).fit(x_train[:, sorted_args],
                                                     y_train.ravel())
    predictions = clf_lr.predict(x_test[:, sorted_args])
    try:
        return f1_score(predictions, y_test.ravel(), average="micro")

    except:
        ## sc
        return f1_score(predictions, y_test)
