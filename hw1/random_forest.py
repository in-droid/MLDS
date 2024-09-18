import random
import numpy as np
from scipy.stats import mode
from utils import bootstrap_index, random_sqrt_columns
from tree import Tree

class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftrees = [Tree(rand=self.rand,
                             get_candidate_columns=random_sqrt_columns,
                             min_samples=2)
                        for _ in range(n)
                        ]

    def build(self, X, y):
        trained_trees = []
        all_indices = set(range(len(X)))
        oob_indices = []
        for tree in self.rftrees:
            selected_idx = bootstrap_index(X, self.rand, len(X))
            X_boot = X[selected_idx]
            y_boot = y[selected_idx]
            oob_idx = list(all_indices - set(selected_idx))
            oob_indices.append(oob_idx)
            trained_tree = tree.build(X_boot, y_boot)
            trained_trees.append(tree)
        return RFModel(trained_trees, X, y, oob_indices)


class RFModel:

    def __init__(self, trees, X, y, oob_indices, rand=random.Random(1)):
        self.trees = trees
        self.X = X
        self.y = y
        self.oob_by_tree = zip(trees, oob_indices)
        self.feature_importance = np.zeros(X.shape[1])
        self.rand = rand

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return mode(preds, axis=0)[0]

    def importance(self):
        feature_importance = np.zeros(self.X.shape[1])
        num_of_trees_for_feature = np.zeros(self.X.shape[1])
        for tree, oob_ids in self.oob_by_tree:
            for feature in tree.used_features:
                X_oob = self.X[oob_ids, :]
                y_oob = self.y[oob_ids]
                missclassification_rate = np.mean(y_oob != tree.predict(X_oob))
                X_oob_shuffled = X_oob.copy()
                selected_feature = X_oob_shuffled[:, feature]
                self.rand.shuffle(selected_feature)
                missclassification_rate_shuffled = np.mean(y_oob != tree.predict(X_oob_shuffled))

                feature_importance[feature] += missclassification_rate_shuffled - missclassification_rate
                num_of_trees_for_feature[feature] += 1

        return np.divide(feature_importance,
                         num_of_trees_for_feature,
                         out=np.zeros_like(feature_importance),
                         where=num_of_trees_for_feature != 0)


def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]
