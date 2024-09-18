import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerically stable softmax
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(x):
    # optimized sigmoid function, so it doesn't overflow
    y = x.copy()
    postive = x >= 0
    negative = ~postive

    y[postive] = 1 / (1 + np.exp(-x[postive]))
    y[negative] = np.exp(x[negative]) / (1 + np.exp(x[negative]))
    return y


def log_loss(y, p):
    return - np.sum(y * np.log(p))


def one_hot_encode(y):
    y_onehot = np.zeros((len(y), len(np.unique(y))))
    y_onehot[list(range(len(y))), y] = 1
    return y_onehot


