import numpy as np
import implementations as imp
import minimizers
from preprocessor import EmptyPreprocessor

class LeastSquares:
    def __init__(self, preprocessor=EmptyPreprocessor(), solver='pseudo', **kwargs):
        """Class constructor

        Keyword arguments:
        preprocess -- function used to preprocess x before training
        solver -- the least squares problem solver. Currently implemented
                  are (pseudo, direct, gradient, stochastic)
        kwargs -- Additional arguments for the solver

        Exemple of usage:
        ls = LeastSquares(prep, 'gradient', max_iters=100, threshold=0.001)
        ls.train(tx)
        w = ls.model
        loss = ls.loss
        losses = ls.losses # only if used iterative solver (gradient, stochastic)
        predictions = ls.predict(x_test)
        """
        self.preprocessor = preprocessor
        self.solver = solver
        self.solver_args = kwargs

    def train(self, y, x):
        processed = self.preprocessor.preprocess_train(x)

        # Switch statement à-la python
        chooser = {
            'pseudo': self._pseudo_s,
            'direct': self._direct_s,
            'gradient': self._gradient_s,
            'stochastic._gradient': self._sgd_s
        }
        chooser[self.solver](y, processed)

    def predict(self, x_test):
        ready = self.preprocessor.preprocess_test(x_test)
        return x_test @ self.model

    def _direct_s(self, y, tx):
        self.model = np.linalg.inv(tx.T @ tx) @ tx.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    def _pseudo_s(self, y, tx):
        U, S, V = np.linalg.svd(tx, full_matrices=False)
        self.model = V.T @ np.diag(1 / S) @ U.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    @staticmethod
    def compute_gradient(y, tx, w):
        y = np.reshape(y, (len(y), 1))
        e = y - (tx @ w)
        return -1 / len(y) * (tx.T @ e)

    def _gradient_s(self, y, tx):
        niter = 0
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 10**(-3))
        self.losses = []
        gamma = self.solver_args['gamma']
        lambda_ = self.solver_args.get('lambda_')

        niter, losses, w = minimizers.gradient_descent(y, tx, w, max_iter, threshold,
                                                       LeastSquares.compute_gradient,
                                                       self.solver_args.get('compute_loss', imp.rmse),
                                                       gamma, lambda_)

        self.niter = niter
        self.model = w
        self.losses = losses

    def _sgd_s(self, y, tx):
        batch_size = self.solver_args['batch_size']
        num_batches = self.solver_args['num_batches']
        shuffle = self.solver_args['shuffle']
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        loss = 0
        self.losses = []

        for batch_y, batch_tx in imp.batch_iter(y, tx, num_batches, shuffle):
            grad = LeastSquares.compute_gradient(y, tx, w)
            w -= self.solver_args['gamma'] * grad
            loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, w)
            self.losses.append(loss)

        self.model = w
        self.loss = loss

