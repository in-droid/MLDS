import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import StandardScaler
import time
import csv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def one_hot_encode(y):
    y_onehot = np.zeros((len(y), len(np.unique(y))))
    y_onehot[list(range(len(y))), y] = 1
    return y_onehot


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y, y_pred):
    y_ = y.flatten()
    y_pred_ = y_pred.flatten()
    loss = np.mean(((y_ - y_pred_) ** 2)) / 2

    return loss

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerically stable softmax
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y, y_pred):
    return -np.sum(y * np.log(y_pred)) / y.shape[0]


class ANNModel:
    def __init__(self, units, lambda_, n_features, y_classes, loss):
        self.units = units
        self.lambda_ = lambda_
        self.weights_ = None
        self.biases = None
        self.activations = None
        self.n_features = n_features
        self.y_classes = y_classes
        self.loss = loss
        self.init_weights(n_features, units, y_classes)
        self.layer_shapes = [layer.shape for layer in self.weights_]
        self.num_weights = sum(np.prod(layer_shape) for layer_shape in self.layer_shapes)
        self.d_weights = None
        self.d_biases = None
        self.threshold_early_stopping = 5
        self.early_stopping = 0
        self.min_val_loss = 1e6

    def numerical_gradient_optimized(self, X, y, lambda_=0.001, epsilon=1e-5):
        d_W_all = []
        d_B_all = []

        for i in range(len(self.weights_)):
            w = self.weights_[i].copy()
            b = self.biases[i].copy()
            d_w = np.zeros_like(w)
            d_b = np.zeros_like(b)

            for param_name, param, d_param in zip(['d_w', 'd_b'], [w, b], [d_w, d_b]):
                for j in range(param.size):
                    param_plus = param.copy()
                    param_minus = param.copy()
                    param_plus.flat[j] += epsilon
                    param_minus.flat[j] -= epsilon
                    if param_name == 'd_w':
                        self.weights_[i] = param_plus
                    else:
                        self.biases[i] = param_plus

                    y_hat = self.predict(X)
                    loss_plus = self.loss(y, y_hat) + (lambda_ / 2 * np.sum([np.sum(w ** 2) for w in self.weights_]))
                    # print("Loss plus: ", loss_plus)
                    if param_name == 'd_w':
                        self.weights_[i] = param_minus
                    else:
                        self.biases[i] = param_minus

                    y_hat = self.predict(X)
                    loss_minus = self.loss(y, y_hat) + (lambda_ / 2 * np.sum([np.sum(w ** 2) for w in self.weights_]))

                    d_param.flat[j] = (loss_plus - loss_minus) / (2 * epsilon)

            self.weights_[i] = w
            self.biases[i] = b

            d_W_all.append(d_w)
            d_B_all.append(d_b)

        return d_W_all, d_B_all


    def weights(self):
        return [np.r_[w, b.reshape(1, -1)] for w, b in zip(self.weights_, self.biases)]
    
    def init_weights(self, n_features, units, y_classes, test_grad=True):
        np.random.seed(0)
        # TODO: simplify this mess
        if not units:
            self.weights_ = [np.random.randn(n_features, y_classes) * 0.01]
            if test_grad:
                self.biases = [np.random.randn(y_classes) * 0.01]
            else:
                self.biases = [np.zeros(y_classes)]
        else:
            self.weights_ = [np.random.randn(n_features, units[0]) * 0.01]
            if test_grad:
                self.biases = [np.random.randn(units[0]) * 0.01]
            else:
                self.biases = [np.zeros(units[0])]

            for i in range(1, len(units)):
                self.weights_.append(np.random.randn(units[i - 1], units[i]))
                if test_grad:
                    self.biases.append(np.random.randn(units[i]) * 0.01)
                else:
                    self.biases.append(np.zeros(units[i]))

            self.weights_.append(np.random.randn(units[-1], y_classes))
            if test_grad:
                self.biases.append(np.random.randn(y_classes) * 0.01)
            else:
                self.biases.append(np.zeros(y_classes))

    def activation_function(self, x):
        raise NotImplementedError("Activation function not implemented in Base class.")

    def forward(self, a):
        Zs = []
        As = [a]

        for i in range(len(self.weights_) - 1):
            z = np.dot(a, self.weights_[i]) + self.biases[i]
            Zs.append(z)
            a = sigmoid(z)
            As.append(a)

        z = np.dot(a, self.weights_[-1]) + self.biases[-1]
        Zs.append(z)
        # TODO: change this so it's custimizable for regression or classification
        # a = softmax(z)
        a = self.activation_function(z)
        As.append(a)
        return a, Zs, As

    def predict(self, X):
        probs, _, __ = self.forward(X)
        return probs

    def backward(self, y, target, Zs, activations):
        d_weights = [np.zeros(w.shape) for w in self.weights_]
        d_biases = [np.zeros(b.shape) for b in self.biases]

        delta = y - target

        d_biases[-1] = np.mean(delta, axis=0, keepdims=True)
        d_weights[-1] = (1 / len(y)) * np.dot(activations[-2].T, delta) + self.lambda_ * self.weights_[-1]

        for layer in range(2, len(self.weights_) + 1):
            delta = np.dot(delta, self.weights_[-layer + 1].T) * sigmoid_prime(Zs[-layer])
            d_biases[-layer] = np.mean(delta, axis=0, keepdims=True)
            d_weights[-layer] = (1 / y.shape[0]) * np.dot(activations[-layer - 1].T, delta) + self.lambda_ * \
                                self.weights_[-layer]

        return d_weights, d_biases

    def cost_grad(self, X, y):
        y_pred, Zs, As = self.forward(X)

        cost = self.loss(y, y_pred) + (self.lambda_ / 2 * np.sum([np.sum(w ** 2) for w in self.weights_]))
        d_weights, d_biases = self.backward(y_pred, y, Zs, As)

        return cost, d_weights, d_biases

    def cost_val_grad(self, X_train, y_train, X_val, y_val):
        y_pred, Zs, As = self.forward(X_train)
        cost_train = self.loss(y_train, y_pred) + (self.lambda_ / 2 * np.sum([np.sum(w ** 2) for w in self.weights_]))
        d_weights, d_biases = self.backward(y_pred, y_train, Zs, As)

        y_pred_val, _, _ = self.forward(X_val)
        cost_val = self.loss(y_val, y_pred_val) + (self.lambda_ / 2 * np.sum([np.sum(w ** 2) for w in self.weights_]))

        if cost_val > self.min_val_loss:
            self.early_stopping += 1

        else:
            self.early_stopping = 0
            self.min_val_loss = cost_val

        if self.early_stopping >= self.threshold_early_stopping:
            return 0, y_pred, d_weights, d_biases

        return cost_train, y_pred, d_weights, d_biases


class ANNClassificationModel(ANNModel):
    def __init__(self, units, lambda_, n_features, y_classes):
        super().__init__(units, lambda_, n_features, y_classes, cross_entropy_loss)

    def activation_function(self, x):
        return softmax(x)


class ANNRegressionModel(ANNModel):
    def __init__(self, units, lambda_, n_features):
        super().__init__(units, lambda_, n_features, 1, mse_loss)

    def activation_function(self, x):
        return x.reshape(1, -1)

    def forward(self, a):
        a, Zs, As = super().forward(a)
        return a.reshape(-1, 1), Zs, As

    def predict(self, X):
        probs, _, __ = self.forward(X)
        return probs.flatten()


class ANN:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_
        self.model = None

    def wrap_parameters(self, weights, biases):
        return np.concatenate([param.flatten() for param in weights + biases])

    def encode_y(self, y):
        raise NotImplementedError("encode_y not implemented in Base class.")

    def unwrap_parameters(self, wrapped_params, return_type='both'):
        unwrapped_weights = []
        unwrapped_biases = []
        start_idx = 0
        if return_type == 'both':
            bias_index = self.model.num_weights
        elif return_type == 'biases':
            bias_index = 0
        else:
            bias_index = None

        for shape in self.model.layer_shapes:
            num_params = np.prod(shape)

            weight_params = wrapped_params[start_idx:start_idx + num_params]
            weight_params = weight_params.reshape(shape)
            unwrapped_weights.append(weight_params)

            if bias_index is not None:
                bias_params = wrapped_params[bias_index: bias_index + shape[1]]
                bias_params = np.array(bias_params)
                unwrapped_biases.append(bias_params)

            start_idx += num_params
            bias_index += shape[1]
        return unwrapped_weights, unwrapped_biases

    def cost_func(self, params, X, y):
        # Reshape the parameters into weights and biases
        self.model.weights_, self.model.biases = self.unwrap_parameters(params)

        cost, d_weights, d_biases = self.model.cost_grad(X, y)

        self.model.d_weights, self.model.d_biases = d_weights, d_biases

        grad = self.wrap_parameters(d_weights, d_biases)

        return cost, grad

    def cost_func_early_stopping(self, params, X_train, y_train, X_val, y_val):
        # Reshape the parameters into weights and biases
        self.model.weights_, self.model.biases = self.unwrap_parameters(params)

        cost, d_weights, d_biases = self.model.cost_val_grad(X_train, y_train, X_val, y_val)

        self.model.d_weights, self.model.d_biases = d_weights, d_biases

        grad = self.wrap_parameters(d_weights, d_biases)

        return cost, grad

    def create_model(self, n_features, y):
        raise NotImplementedError("create_model not implemented in Base class.")

    def fit(self, X, y):
        n_instances, n_features = X.shape
        self.model = self.create_model(n_features, y)
        # self.model = ANNClassificationModel(self.units, self.lambda_, n_features, y_classes)
        y_encoded = self.encode_y(y)
        initial_params = self.wrap_parameters(self.model.weights_, self.model.biases)

        params, _, _ = fmin_l_bfgs_b(self.cost_func,
                                     initial_params,
                                     args=(X, y_encoded),
                                     approx_grad=False
                                     )
        self.model.weights_, self.model.biases = self.unwrap_parameters(params)

        return self.model

    def fit_early_stopping(self, X_train, y_train, X_val, y_val):
        n_instances, n_features = X_train.shape
        self.model = self.create_model(n_features, y_train)
        y_train_encoded = self.encode_y(y_train)
        y_val_encoded = self.encode_y(y_val)
        initial_params = self.wrap_parameters(self.model.weights_, self.model.biases)

        params, _, _ = fmin_l_bfgs_b(self.cost_func_early_stopping,
                                     initial_params,
                                     args=(X_train, y_train_encoded, X_val, y_val_encoded),
                                     approx_grad=False
                                     )
        self.model.weights_, self.model.biases = self.unwrap_parameters(params)

        return self.model

    def forward(self, X):
        return self.model.forward(X)


class ANNClassification(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)

    def encode_y(self, y):
        return one_hot_encode(y)

    def create_model(self, n_features, y):
        y_classes = len(np.unique(y))
        return ANNClassificationModel(self.units, self.lambda_, n_features, y_classes)

    def forward(self, X):
        return self.model.forward(X)


class ANNRegression(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)

    def encode_y(self, y):
        return y.reshape(-1, 1)

    def create_model(self, n_features, y):
        return ANNRegressionModel(self.units, self.lambda_, n_features)


def test_numerical_gradients_classification():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 2, 3])
    hard_y = np.array([0, 1, 1, 0])
    hard_y = y
    lambda_ = 0.01

    model = ANNClassificationModel(units=[10, 20],
                                   lambda_=lambda_,
                                   n_features=2,
                                   y_classes=4,

                                   )
    y = one_hot_encode(hard_y)
    y_pred, Zs, As = model.forward(X)
    d_weights, d_biases = model.backward(y_pred, y, Zs, As)
    d_weights_num, d_biases_num = model.numerical_gradient_optimized(X, y, lambda_=lambda_)
    d_biases_num = [d_bias.reshape(1, -1) for d_bias in d_biases_num]
    for d_weight, d_weight_num in zip(d_weights, d_weights_num):
        np.testing.assert_allclose(d_weight, d_weight_num, atol=1e-4)
    for d_bias, d_bias_num in zip(d_biases, d_biases_num):
        np.testing.assert_allclose(d_bias, d_bias_num, atol=1e-4)


def test_numerical_gradients_regression():
    np.random.seed(0)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 2, 3])
    # hard_y = np.array([0, 1, 1, 0])
    # hard_y = y
    lambda_ = 0

    model = ANNRegressionModel(units=[10, 20],
                               lambda_=lambda_,
                               n_features=2,
                               )
    y = y.reshape(-1, 1)
    y_pred, Zs, As = model.forward(X)
    d_weights, d_biases = model.backward(y_pred, y, Zs, As)
    d_weights_num, d_biases_num = model.numerical_gradient_optimized(X, y, lambda_=lambda_)
    d_biases_num = [d_bias.reshape(1, -1) for d_bias in d_biases_num]

    for d_weight, d_weight_num in zip(d_weights, d_weights_num):
        np.testing.assert_allclose(d_weight, d_weight_num, atol=1e-4)
    for d_bias, d_bias_num in zip(d_biases, d_biases_num):
        np.testing.assert_allclose(d_bias, d_bias_num, atol=1e-4)



if __name__ == '__main__':
    test_numerical_gradients_classification()
    test_numerical_gradients_regression()
