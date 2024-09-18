Implement two regression methods, (1) kernelized ridge regression and (2) Support Vector Regression (SVR), and two kernels:

- Polynomial kernel κ(x,x′)=(1+xx′)M
- RBF kernel κ(x,x′)=exp(−||x−x′||22σ2)

Implement SVR by solving the optimization problem in Eq. (10) from (Smola and Scholkopf, 2004) with cvxopt.solvers.qp. Inputs to qp should be represented so that the solution x contains αi
 and α∗i
 in the following order: [α1,α∗1,α2,α∗2,α3,α∗3,…]
. Set C as 1/λ
. To obtain b, use the output y from cvxopt.solvers.qp. b could, in theory, also be obtained from Eq. (16), but it is very sensitive to inaccuracies (therefore, do not use it; the equation also contains an error).

Apply both regression methods and both kernels to the 1-dimensional sine data set. For each method/kernel find kernel and regularization parameters that fit well. For SVR, also take care to produce a sparse solution. This part aims to showcase what kernels can do and introduce the meaning of parameters. No need to do any formal parameter selection (such as with cross-validation) here. Plot the input data, the fit, and mark support vectors on the plot.

Apply both regression methods and both kernels to the housing2r data set. Use the first 80% of data as a training set and the remaining 20% as a validation set. For each method/kernel, plot MSE on the testing set versus a kernel parameter value (for polynomial kernel, M ∈ [1,10]
, for RBF choose interesting values of σ yourself). Take care to set ϵ properly. Plot two curves for each kernel/method, one with regularization parameter λ=1, and the other with λ set with internal cross validation (for each kernel parameter value separately). For SVR, also display the number of support vectors for each score and try to keep it to a minimum while still getting a good fit.

Compare results between kernelized ridge regression and SVR and comment on the differences and similarities. Which learning algorithm would you prefer and why?
