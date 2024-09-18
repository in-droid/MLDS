import numpy as np

class KernelizedRidgeRegressionModel:
    def __init__(self, X, y, alpha, kernel):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.kernel = kernel

    def predict(self, X):
        n = X.shape[0]
        b = np.ones((n, 1))
        X = np.hstack((b, X))
        K = self.kernel(self.X, X)
        return K.T.dot(self.alpha.T)

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.model = None
        self.X = None

    def fit(self, X, y):
        n = X.shape[0]
        b = np.ones((n, 1))

        self.X = np.hstack((b, X))
        K = self.kernel(self.X, self.X)
        I = np.eye(n)
        alpha = np.linalg.solve(K + self.lambda_ * I, y)

        self.model = KernelizedRidgeRegressionModel(self.X, y, alpha, self.kernel)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
