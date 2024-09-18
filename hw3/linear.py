import numpy as np
from utils import softmax, log_loss, one_hot_encode, sigmoid
import scipy.optimize as opt

class MultinomalLogRegModel:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.coef = np.random.randn(num_features, num_classes)
        self.intercept = np.random.randn(num_classes)

    def forward(self, X):
        logits = np.dot(X, self.coef) + self.intercept
        return softmax(logits)

    def predict(self, X):
        return self.forward(X)

    def calculate_loss(self, params, X, y):
        W = params[:self.num_classes * self.num_features].reshape(self.num_features,
                                                                  self.num_classes)
        intercept = params[self.num_features * self.num_classes:]
        self.coef = W
        self.intercept = intercept
        y_hat = self.forward(X)
        return log_loss(y, y_hat)


class MultinomialLogReg:
    def __init__(self):
        self.model = None

    def build(self, X, y):
        y_one_hot = one_hot_encode(y)
        num_classes = len(np.unique(y))
        self.model = MultinomalLogRegModel(num_classes, X.shape[1])
        result = opt.minimize(self.model.calculate_loss,
                              np.concatenate([self.model.coef.ravel(),
                                              self.model.intercept]),
                              args=(X, y_one_hot), method='L-BFGS-B',
                            )

        self.model.coef = result.x[:X.shape[1] * num_classes].reshape(X.shape[1], num_classes)
        self.model.intercept = result.x[X.shape[1] * num_classes:]

        return self.model

    def predict(self, X):
        logits = self.model.forward(X)
        return softmax(logits)


class OrdinalLogRegModel:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.coef = np.random.random((num_features, 1))
        self.intercept = np.random.random(1)
        self.deltas = np.random.random(num_classes - 2)

    @property
    def thresholds(self):
        return np.array([-np.inf, 0, *np.cumsum(self.deltas), np.inf])

    def forward(self, X):
        logits = X @ self.coef + self.intercept
        thresholds_matrix = np.tile(self.thresholds, (X.shape[0], 1))
        logits = np.tile(logits.T, (len(self.thresholds), 1))
        logits = thresholds_matrix - logits.T
        return sigmoid(logits)

    def predict(self, X):
        cumulative_probs = self.forward(X)
        shifted_probs = np.roll(cumulative_probs, 1, axis=1)
        shifted_probs = shifted_probs[:, 1:]
        cumulative_probs = cumulative_probs[:, 1:]
        probs = cumulative_probs - shifted_probs
        return probs

    def calculate_loss(self, params, X, y):
        W = params[:self.num_features].reshape(self.num_features, 1)
        deltas = params[self.num_features:self.num_features + self.num_classes - 2]
        intercept = params[-1]

        self.coef = W
        self.intercept = intercept
        self.deltas = deltas
        y_hat = self.predict(X)
        return log_loss(y, y_hat)


class OrdinalLogReg:
    def __init__(self):
        self.model = None

    def forward(self, X):
        return self.model.forward(X)


    @property
    def thresholds(self):
        return self.model.thresholds

    def build(self, X, y):


        y_one_hot = one_hot_encode(y)
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = OrdinalLogRegModel(num_classes, num_features)

        bounds = [(None, None)] * num_features + [(0, None)] * (num_classes - 2) + [(None, None)]
        result = opt.minimize(self.model.calculate_loss,
                              np.concatenate([self.model.coef.ravel(),
                                              self.model.deltas,
                                              self.model.intercept]),
                              args=(X, y_one_hot),
                              bounds=bounds)
        self.model.coef = result.x[:num_features]
        self.model.deltas = result.x[num_features: num_features + num_classes - 2]
        self.model.intercept = result.x[-1]
        return self.model

    def predict(self, X):
        cumulative_probs = self.forward(X)
        shifted_probs = np.roll(cumulative_probs, 1, axis=1)
        shifted_probs = shifted_probs[:, 1:]
        cumulative_probs = cumulative_probs[:, 1:]
        probs = cumulative_probs - shifted_probs
        return probs
