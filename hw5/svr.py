import numpy as np
import cvxopt

class SVRModel:
    def __init__(self, X, y, alpha, b, kernel, alpha_difference):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.b = b
        self.kernel = kernel
        self.alpha_difference = alpha_difference

    def predict(self, X):
        K = self.kernel(self.X, X)
        alpha_difference = self.alpha.dot(self.alpha_difference)
        return np.dot(alpha_difference, K) + self.b

    def get_alpha(self):
        n = self.X.shape[0]
        return self.alpha.reshape(n, 2)

    def get_support_vectors(self, eps=1e-4):
        alpha = self.get_alpha()
        return np.where((np.abs(np.diff(alpha, axis=1)) > eps).flatten())[0]

    def get_b(self):
        return self.b


class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.model = None
        self.X = None

    def fit(self, X, y):
        n = X.shape[0]
        self.X = X
        K = self.kernel(self.X, self.X)
        idx = np.arange(n)
        values = np.where(idx == 0, -1, -1)
        alpha_difference = np.zeros((2 * n, n))
        alpha_difference[::2] = np.eye(n)
        alpha_difference[1::2, np.arange(n)] = np.diag(values)

        P = np.linalg.multi_dot([alpha_difference, K, alpha_difference.T])
        q = self.epsilon * np.ones(2 * n) - np.array([y, -y]).flatten(order='F')
        G = np.vstack((np.eye(2 * n), -np.eye(2 * n)))
        h = np.hstack([np.ones(2 * n) / self.lambda_, np.zeros(2 * n)])
        A = np.ones(2 * n).reshape(1, 2 * n)
        A[:, 1:2 * n:2] = -1
        b = np.zeros(1)

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).flatten()
        b = np.array(sol['y']).flatten()
        self.model = SVRModel(self.X, y, alpha, b, self.kernel, alpha_difference)

        return self.model
