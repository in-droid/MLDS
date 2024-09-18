import numpy as np
from utils import all_columns


class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.root = None
        self.used_features = set()

    def find_split_feature_threshold(self, X, y, selected_features):
        best_threshold = None
        best_gini = 1
        best_feature = None
        _, num_of_cols = X.shape
        num_start = np.array([np.sum(y == i) for i in (set(y))])
        n = len(y)

        for col_idx in selected_features:
            selected_column = X[:, col_idx]
            thresholds_sorted, classes_sorted = zip(*sorted(zip(selected_column, y),
                                                            key=lambda x: x[0]))

            num_of_left_values = np.array([0] * len(set(y)))
            num_right_values = num_start.copy()

            for i in range(1, len(thresholds_sorted)):
                num_of_left_values[classes_sorted[i - 1]] += 1
                num_right_values[classes_sorted[i - 1]] -= 1

                if thresholds_sorted[i] == thresholds_sorted[i - 1]:
                    continue

                gini_left = 1 - np.sum((num_of_left_values / i) ** 2)
                gini_right = 1 - np.sum((num_right_values / (len(y) - i)) ** 2)
                gini = i / n * gini_left + (n - i) / n * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_threshold = (thresholds_sorted[i] + thresholds_sorted[i - 1]) / 2
                    best_feature = col_idx

                    if gini == 0:
                        return best_feature, best_threshold, best_gini

        return best_feature, best_threshold, best_gini

    def build(self, X, y):

        X_train = X[:, ]
        y_train = y.copy()
        selected_features = list(self.get_candidate_columns(X, self.rand))
        gini = 1 - np.sum((np.bincount(y_train) / len(y_train)) ** 2)

        if gini == 0 or len(y) < self.min_samples:
            node = TreeNode(None, None, selected_features, None, None, None)
            node.class_confidences = np.bincount(y_train) / len(y_train)
            self.root = node if self.root is None else self.root
            return node

        feature, threshold, gini = self.find_split_feature_threshold(X_train, y, selected_features)

        if feature is not None:
            self.used_features.add(feature)

        node = TreeNode(feature, threshold, selected_features, None, None, None)

        if self.root is None:
            self.root = node

        if threshold is None:
            node.class_confidences = np.bincount(y_train) / len(y_train)
            return node

        left = y_train[X_train[:, feature] <= threshold]
        right = y_train[X_train[:, feature] > threshold]

        node.left = self.build(X_train[X_train[:, feature] <= threshold], left)
        node.right = self.build(X_train[X_train[:, feature] > threshold], right)

        node.left.parent = node
        node.right.parent = node

        return node

    def predict(self, X):
        return self.root.predict(X)


class TreeNode:
    def __init__(self, feature, threshold, selected_features, left, right, parent=None):
        self.feature = feature
        self.threshold = threshold
        self.selected_features = selected_features
        self.left = left
        self.right = right
        self.parent = parent
        self.class_confidences = None

    def predict(self, X):
        return np.array([self.predict_instance(x) for x in X])

    def predict_instance(self, x):
        if self.class_confidences is not None:
            return np.argmax(self.class_confidences)
        else:
            if x[self.feature] > self.threshold:
                return self.right.predict_instance(x)

            return self.left.predict_instance(x)
