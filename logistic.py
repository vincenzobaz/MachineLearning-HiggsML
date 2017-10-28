import numpy as np
import minimizers
import implementations
import run as run
from preprocessor import EmptyPreprocessor

class LogisticRegression:
    def __init__(self, degree=1, solver='newton', **kwargs):
        """Model constructor. This model implements LogisticRegression for
        classification

        Keyword arguments:
        preprocessor -- Preprocessor used to prepare x_train before training and
                        x_test before predicting
        solver -- The strategy to minimze the loss function.
                  Currently implemented: (newton, gradient, stochastic)
        kwargs -- Additional arguments for the minimizer

        The additional arguments for the minimizer depend on the minimizer
        chosen.
          - newton:
            - w0: initial solution vector, if none is provided, a vector of zeros
                  is used.
            - max_iters: maximum number of iterations. Default is 100.
            - threshold: if the difference of losses of two consecutive iterations
                         of the minimizer is smaller than this value, the
                         minimizer returns. Default is 0.1.
            - gamma: learning rate. If none is provided, a value is automatically.
                     computed to minimize the distance between to successive ws.
            - lambda_: regularizer coefficient. If none is provided,
                       no regularization is used.
          - gradient:
            - w0: initial solution vector, if none is provided, a vector of zeros
                  is used.
            - max_iters: maximum number of iterations. Default is 100.
            - threshold: if the difference of losses of two consecutive iterations
                         of the minimizer is smaller than this value, the
                         minimizer returns. Default is 1.
            - gamma: learning rate. If none is provided, a value is automatically
                     computed to minimize the distance between to successive ws.
            - lambda_: regularizer coefficient. If none is provided,
                       no regularization is used.
          - stochastic:
            - w0: initial solution vector, if none is provided, a vector of zeros
                  is used.
            - gamma: learning rate. If None is provided, the computation does not
                     start.
            - lambda_: regularizer coefficient. If none is provided,
                       no regularization is used.
            - batch_size: the size of a batch.
            - num_batches: the number of batches to divide the data in.
            - shuffle: boolean indicating whether the data should be shuffled
                       before division.
        """
        self.solver = solver
        self.solver_args = kwargs
        self.degree = degree

    def train(self, y, x):
        """Trains the model on the provided x,y data"""
        processed = run.polynomial_enhancement(x, self.degree)
        y = np.reshape(y, (len(y), 1))

        chooser = {
            'newton': self._newton_s,
            'gradient': self._gradient,
            'stochastic': self._sgd
        }
        chooser[self.solver](y, processed)
        return self

    def predict(self, x_test):
        """Predicts y values for the provided test data"""
        ready = run.polynomial_enhancement(x_test, self.degree)
        return LogisticRegression.sigmoid(ready @ self.model)

    def predict_labels(self, x_test):
        """Generates labels (-1, 1) for classification"""
        probs = self.predict(x_test)
        probs[probs >= 0.5] = 1
        #probs[probs < 0.5] = -1
        probs[probs < 0.5] = 0
        return probs

    @staticmethod
    def sigmoid(t):
        """Logistic function"""
        # We used the two different functions below to avoid overflows and
        # underflows
        negative_ids = np.where(t < 0)
        positive_ids = np.where(t >= 0)
        t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
        t[positive_ids] = np.power(np.exp(-t[positive_ids]) + 1, -1)
        return t

    @staticmethod
    def compute_loss(y, tx, w):
        """Computes loss using log-likelihood"""
        txw = tx @ w
        critical_value = np.float64(709.0)
        overf = np.where(txw >= critical_value)
        postives = np.sum(txw[overf] - y[overf] * txw[overf])
        rest_ids = np.where(txw < critical_value)
        rest = np.sum(np.log(1 + np.exp(txw[rest_ids])) - y[rest_ids] * txw[rest_ids])
        return rest + postives
        #return np.sum(np.log(1 + np.exp(txw)) - y * txw)

    @staticmethod
    def compute_gradient(y, tx, w):
        """Computes the gradient of the loss function"""
        return tx.T @ (LogisticRegression.sigmoid(tx @ w) - y)

    @staticmethod
    def compute_hessian(tx, w):
        """Computes the Hessian of the loss function"""
        tmp = LogisticRegression.sigmoid(tx @ w)
        S = tmp * (1 - tmp)
        # diagflat S => memory error. Simulate same behavior with following line
        return np.multiply(tx.T, S.T) @ tx

    def _newton_s(self, y, tx):
        """Minimizes the loss function using Newton's method"""
        # Retrieve parameters from kwargs or initialize defaults
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 10**(-1))
        gamma = self.solver_args.get('gamma')
        lambda_ = self.solver_args.get('lambda_')

        # Invoke minimizer
        niter, losses, w = minimizers.newton(y, tx, w, max_iter, threshold,
                                             LogisticRegression.compute_gradient,
                                             LogisticRegression.compute_hessian,
                                             LogisticRegression.compute_loss,
                                             gamma, lambda_)

        # Store useful statistics
        self.niter = niter
        self.model = w
        self.losses = losses
        self.loss = losses[-1]
        if niter < max_iter:
            self.converged = True

    def _gradient(self, y, tx):
        """Minimizes the loss function using gradient descent"""
        # Retrieve parameters from kwargs or initialize defaults
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 0.1)
        gamma = self.solver_args.get('gamma')
        lambda_ = self.solver_args.get('lambda_')

        # Invoke minimizer
        niter, losses, w = minimizers.gradient_descent(y, tx, w, max_iter, threshold,
                                                       LogisticRegression.compute_gradient,
                                                       LogisticRegression.compute_loss,
                                                       gamma, lambda_)

        # Store useful statistics
        self.niter = niter
        self.model = w
        self.losses = losses
        self.loss = losses[-1]
        if niter < max_iter:
            self.converged = True

    def _sgd(self, y, tx):
        """Minimizes the loss function using gradient descent"""
        # Retrieve parameters from kwargs or initialize defaults
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        gamma = self.solver_args.get('gamma')
        lambda_ = self.solver_args.get('lambda_')
        batch_size = self.solver_args['batch_size']
        num_batches = self.solver_args['num_batches']
        shuffle = self.solver_args.get('shuffle', True)
        losses = []

        # Break data into batches and perform gradient descent on batches
        # instead of on the entire matrix
        for b_y, b_x in implementations.batch_iter(y, tx, batch_size, num_batches):
            w = minimizers.gradient_descent_step(b_y, b_x, w,
                                                 LogisticRegression.compute_gradient,
                                                 gamma,
                                                 lambda_)
            losses.append(LogisticRegression.compute_loss(b_y, b_x, w))

        # Invoke minimizer
        self.model = w
        self.losses = losses
        self.loss = losses[-1]

