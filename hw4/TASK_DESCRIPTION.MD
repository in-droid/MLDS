1. Implement multi-layer fully-connected artificial neural networks (ANN) for classification (target variable with a categorical distribution) and regression (normally distributed target variable) with backpropagation: define the function to optimize, compute its gradient, and use numerical optimization to find a solution. Use bias on every layer and take care not to regularize the bias. Implementations of classification and regression ANN should share as much code as possible; report on what is the same and what needs to be different. Your implementation should allow an arbitrary number of hidden layers of any size (sizes of input and output layers are problem defined).

2. Numerically verify that the gradient and cost are compatible for both types of ANN; describe your procedure and results.

3. Apply ANN to the housing2r and housing3 data sets. For each data set, use one method developed in previous HWs for comparison.

4. Apply your ANN to the data set from train.csv.gz, which groups instances in one of the nine groups. Build an accurate model and evaluate it. Also, report your runtimes. Create a file final.txt with predictions for test.csv.gz; all the code for building the final model and making predictions needs to be in a function create_final_predictions().
