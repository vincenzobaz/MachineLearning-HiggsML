# Machine Learning - Project 1: The Higgs Boson Challenge

## How to create a prediction

To generate a `predictions.csv` file, just execute `run.py`.

## Code Organization

We organized our codebase using several files:

 - `implementations.py` contains the mandatory functions.
 - `preprocessing.py` contains the code executed to clean and prepare the data
    for training and testing.
 - `least_squares.py` contains the `LeastSquares` class which encapsulates the
    lest squares algorithm.
 - `logistic.py` contains the `LogisticRegression` class which encapsulates the
    logistic regression for classification algorithm.
 - `minimizers.py` contains functions used for iterative minimization. In particular
    it contains modularized and documented code for gradient descent and
    Newton's method.
 - `cross_validation.py` contains the function necessary to cross validate a
   given model.
 - `run.py` is the execution script.


### LeastSquares and LogisticRegression
As described in the project report, we tried to combine least squares and
logistic regression. In order to do so and easily test / cross validate our
system, we decided to implement the two algorithms in two separate classes
conforming to the same interface.
The interface is:

```python
class Model:

    def __init__(self, degree, solver, **kwargs):
        pass

    def train(self, y, x):
        pass

    def predict(self, x_test):
        pass

    def predict_labels(self, x_test):
        pass
```

`solver` is the strategy used to minimize/solve the problem. For example
in `LeastSquares` it could be `'psudo'` or `'direct'` to indicate whether to use
the exact matrix inverse or the pseudo-inverse. `kwargs` contains parameters for
the solver. More detailed documentation can be found in the two classes.

This modularization allows to standardize the two model's interfaces which is very
useful to combine them. We can abstract the details and just call `train/predict`
without needing any knowledge of the underlying mathematical model.

This would be more difficult using `implementations.py`

### Minimizers
The iterative minimization code was extracted in a separated module and generalized
to have a very flexible codebase centralized in one file. Such architecture
allows to have different modules calling the minimizers without having to duplicate
the code. Additional documentation can be found in the module.

This module implements gradient descent and Newton's method with and without
regularization.

## Some odd-looking code and explanations

While implementing some mathematical formulas, we encountered some difficulties
which required us to modify the code diverging from the simple and clean
mathematical formulation to a more efficient or simply more feasible one.

### Building the Hessian matrix for logistic regression
In `implementation.py` there is:

```python
def compute_hessian(tx, w):
    ....
    return np.multiply(tx.T, S.T) @ tx
    ....
```

Usually, in numpy, we use `np.diagflat` or `np.diag` to build a diagonal matrix
given a vector. In this project, this often meant creating huge diagonal matrices
which often resulted in `MemoryError`. As these matrix are very sparse, we
simulated the matrix multiplication by a diagonal matrix by:
`np.multiply(tx.T, vec.T)`

Here is an example in the python shell:

```
>>> import numpy as np
>>> mat = np.ones((3, 3))
>>> vec = np.array([2, 3, 4])
>>> mat @ np.diagflat(vec)
array([[ 2.,  3.,  4.],
       [ 2.,  3.,  4.],
       [ 2.,  3.,  4.]])
>>> np.multiply(mat.T, vec.T)
array([[ 2.,  3.,  4.],
       [ 2.,  3.,  4.],
       [ 2.,  3.,  4.]])
```

### The sigmoid function
The naïve implementations of the sigmoid function `np.exp(t) / ( 1 + np.exp(t))`
resulted in overflows for very large `t`. For this reason we applied two different
functions to each element of the `t` vector according to the element's sign:

 - If `t_i` is positive then we apply `1 / (1 + np.exp(-t))`
 - If `t_i` is negative we apply the classical `np.exp(t) / (1 + np.exp(t))`

### The logistic loss function
While the previous cases produce the exact result, in this one we had to approximate.

The `LinearRegression.compute_loss` often resulted in overflows (returning `np.inf`)
due to large values in the vector `txw = tx @ w`. To avoid such behaviour,
which made impossible to analyze the convergence of the iterative minimizers
and the behavior of our system in general, we analyzed the computation:

`np.sum(np.log(1 + np.exp(txw)) - y * txw)`

The overflow is caused by `np.exp(txw)`. To work around this we decided to approximate,
**for large elements of the `txw` vector**:  `np.log(1 + np.exp(txw)) ~= np.log(np.exp(txw)) = txw`
Note that we apply this approximation only for *large* elements of `txw`.

Our definition of *large* in this case was pretty straightforward: a number *n* is
large if `np.exp(n)` overflows. As throughout the project we work with numbers
of type `np.float64`, we found a value close the critical number by:

```
>>> x = np.arange(1000, dtype=np.float64)
>>> y = np.exp(x)
>>> x[np.where(y == np.inf)][0]
710.0
```

