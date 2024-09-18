For this homework, you will implement classification trees and random forests. Your implementations must support numeric input variables and a binary target variable.

You will implement these methods as classes (Tree, RandomForest) that provide a method build, which returns the model as an object, whose predict method returns the predicted target class of given input samples (see attached code for usage examples):

Tree - a flexible classification tree with the following attributes: (1) rand, a random generator, for reproducibility, of type random.Random; (2) get_candidate_columns, a function that returns a list of column indices considered for a split (needed for the random forests); and (3) min_samples, the minimum number of samples, where a node is still split. Use the Gini impurity to select the best splits.

RandomForest, with attributes: (1) rand, a random generator; (2) n: number of
bootstrap samples. The RandomForest should use an instance of Tree internally. Build full trees (min_samples=2). For each split, consider random (square root of the number of input variables) variables.

Then, implement permutation-based variable importance. Refer to the "Variable Importance" section from The Elements of Statistical Learning, where the algorithm described in section 10 of Breiman (2001) is actually described well; implement it as method importance() of the random forest model.

Apply the developed methods to the tki-resistance.csv FTIR spectral data set. Always use the first 130 rows of data as the training set and the remainder as the testing set. Do the following:

In function hw_tree_full, build a tree with min_samples=2. Return misclassification rates and standard errors when using training and testing data as test sets.

In function hw_randomforest, use random forests with n=100 trees with min_samples=2. Return misclassification rates and standard errors when using training and testing data as test sets.

As a rough guideline, building the full tree on this data set should take less than 10 seconds - more shows inefficiencies in the implementation. Likewise, computing random forest variable importance for all variables should be faster than building the random forest. There is no need to use multithreading/multiprocessing to achieve this.
This assignment requires that you compute standard errors to quantify the uncertainty of the misclassification rates. Here, we only require an estimate of the uncertainty stemming from a particular test set measurement. Therefore, there is no need to rebuild models when computing standard errors for this assignment.
