import numpy as np


class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)


class Polynomial:
    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        return (A.dot(B.T) + 1) ** self.M


class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        if np.isscalar(A) and np.isscalar(B):
            return np.exp(-((A - B) ** 2) / (2 * self.sigma ** 2))
        elif len(A.shape) == 1 and len(B.shape) == 1:
            return np.exp(-np.linalg.norm(A - B) ** 2 / (2 * self.sigma ** 2))
        else:
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)
            dist_sq = np.sum(A ** 2, axis=1)[:, np.newaxis] + np.sum(B ** 2, axis=1) - 2 * np.dot(A, B.T)
            result = np.exp(-dist_sq / (2 * self.sigma ** 2))
            if result.shape[0] == 1 or result.shape[1] == 1:
                result = result.flatten()
            return result
