import random
import numpy as np
import pandas as pd

def all_columns(X, rand):
    return range(X.shape[1])


def bootstrap_index(X, rand, n):
    ind = rand.choices(range(len(X)), k=n)
    return ind


def bootstrap_std_error(y_hat, y, n=1000):
    missclass_rate = []
    seed = random.Random(1)
    for _ in range(n):
        ind = bootstrap_index(y_hat, seed, len(y_hat))
        missclass_rate.append(np.mean(y_hat[ind] != y[ind]))
    return np.std(missclass_rate)


def random_sqrt_columns(X, rand):
    _, n = X.shape
    sqrt_n = int(np.sqrt(n))
    col_ind = range(n)
    c = list(rand.sample(col_ind, sqrt_n))
    return c


def tki(tki_resistance_path='./tki-resistance.csv'):
    df = pd.read_csv(tki_resistance_path)
    legend = {'Bcr-abl': 0, 'Wild type': 1}
    df['Class'] = df['Class'].map(legend)
    train = df[:130]
    test = df[130:]
    y_train = train['Class']
    X_train = train.drop('Class', axis=1)
    y_test = test['Class']
    X_test = test.drop('Class', axis=1)
    return ((X_train.to_numpy(), y_train.to_numpy()),
            (X_test.to_numpy(), y_test.to_numpy()),
            legend)


def calculate_missclassification_rate(y_true, y_pred, boostrap=True):
    missclasification_rate = np.mean(y_true != y_pred)
    if boostrap:
        missclassifation_uncertainty = bootstrap_std_error(y_pred, y_true, n=1000)
    else:
        missclassifation_uncertainty = missclasification_rate * (1 - missclasification_rate) / np.sqrt(len(y_true))

    return missclasification_rate, missclassifation_uncertainty
