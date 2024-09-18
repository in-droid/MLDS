import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from linear import MultinomialLogReg, OrdinalLogReg

MBOG_TRAIN = 30


def multinomial_bad_ordinal_good(n, rand):
    random.seed(rand)
    feature_1 = np.random.randint(1, 6, size=n) + np.random.randn(n)
    feature_2 = np.random.randint(1, 6, size=n) + np.random.randn(n)
    feature_3 = np.random.uniform(1, 6, size=n) + np.random.randn(n)

    y = np.zeros(n, dtype=int)

    for i in range(n):
        avg_rating = (feature_1[i] + feature_2[i] + feature_3[i]) / 3

        if avg_rating <= 2:
            y[i] = 0
        elif avg_rating <= 4:
            y[i] = 1
        else:
            y[i] = 2

    dataset = {
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'y': y
    }

    df = pd.DataFrame(dataset)
    X = df[['feature_1', 'feature_2', 'feature_3']].to_numpy()
    y = df['y'].to_numpy()
    return X, y


def make_classification_multinomal():
    X, y = make_classification(n_samples=100, n_features=20,
                               n_informative=15,
                               n_classes=3,
                               random_state=0)
    l = MultinomialLogReg()
    multi_sklearn = LogisticRegression(multi_class='multinomial', penalty='none', solver='lbfgs')
    c = l.build(X, y)
    prob = c.predict(X)
    print(metrics.accuracy_score(y, np.argmax(prob, axis=1)))

    multi_sklearn.fit(X, y)
    prob_sklearn = multi_sklearn.predict_proba(X)
    print(metrics.accuracy_score(y, np.argmax(prob_sklearn, axis=1)))


def test_ordinal_params():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1],
                  [1, 1]])
    y = np.array([0, 0, 1, 1, 2])
    l = OrdinalLogReg()
    c = l.build(X, y)

    print(c.deltas)




def bootstrap_acc_variance(preds, classes, n=1000):
    accs = []
    n_samples = len(preds)
    for i in range(n):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_preds = preds[bootstrap_indices]
        bootstrap_gt = classes[bootstrap_indices]
        accs.append(np.mean(bootstrap_preds == bootstrap_gt))
    return np.array(accs).std()

def calc_acc_std_multi_band_ord_good():
    X_train, y_train = multinomial_bad_ordinal_good(MBOG_TRAIN(), 42)
    X_test, y_test = multinomial_bad_ordinal_good(1000, 1)

    model_multi = MultinomialLogReg()
    c_multi = model_multi.build(X_train, y_train)
    model_ordinal = OrdinalLogReg()
    c_ordinal = model_ordinal.build(X_train, y_train)

    y_pred_multi = c_multi.predict(X_test)
    y_preds_classes_multi = np.argmax(y_pred_multi, axis=1)

    y_pred_ordinal = c_ordinal.predict(X_test)
    y_preds_classes_ordinal = np.argmax(y_pred_ordinal, axis=1)

    acc_multi = metrics.accuracy_score(y_test, y_preds_classes_multi)
    acc_ordinal = metrics.accuracy_score(y_test, y_preds_classes_ordinal)

    print(f'Accuracy Multi: {acc_multi}')
    print(f'Accuracy Ordinal: {acc_ordinal}')
    print(f'Std Multi: {bootstrap_acc_variance(y_preds_classes_multi, y_test)}')
    print(f'Std Ordinal: {bootstrap_acc_variance(y_preds_classes_ordinal, y_test)}')

if __name__ == '__main__':
    # test_ordinal_params()
    make_classification_multinomal()
    #calc_acc_std_multi_band_ord_good()
