import time 
import random
import numpy as np
from tree import Tree
from random_forest import RandomForest
from utils import all_columns, calculate_missclassification_rate, tki

TKI_RESISTANCE_PATH = './tki-resistance.csv'

def hw_tree_full(learn, test, std_bootstrap=True):
    X_learn, y_learn = learn
    X_test, y_test = test
    start = time.time()
    tree = Tree(rand=np.random.RandomState(1),
                get_candidate_columns=all_columns,
                min_samples=2)

    model = tree.build(X_learn, y_learn)
    end = time.time()
    print("Time to build tree:", end - start)

    y_learn_preds = model.predict(X_learn)
    y_test_preds = model.predict(X_test)

    return (calculate_missclassification_rate(y_learn, y_learn_preds, std_bootstrap),
            calculate_missclassification_rate(y_test, y_test_preds, std_bootstrap)
            )


def hw_randomforests(learn, test, std_bootstrap=True):
    X_learn, y_learn = learn
    X_test, y_test = test

    rf = RandomForest(rand=random.Random(10),
                      n=100)
    start = time.time()
    model = rf.build(X_learn, y_learn)
    end = time.time()
    print("Time to build random forest:", end - start)

    y_learn_preds = model.predict(X_learn)
    y_test_preds = model.predict(X_test)
    return (calculate_missclassification_rate(y_learn, y_learn_preds, std_bootstrap),
            calculate_missclassification_rate(y_test, y_test_preds, std_bootstrap)
            )


if __name__ == "__main__":
    learn, test, legend = tki(TKI_RESISTANCE_PATH)
    print("full", hw_tree_full(learn, test, std_bootstrap=True))
    print("random forest", hw_randomforests(learn, test, std_bootstrap=True))
