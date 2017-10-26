import numpy as np
import implementations as imp

class LeastSquares:

    def __init__(self, preprocess, solver='pseudo', **kwargs):
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
        self.preprocess_f = preprocess
        self.solver = solver
        self.solver_args = kwargs

    def train(self, y, x):
        processed = self.preprocess_f(x)

        # Switch statement à-la python
        chooser = {
            'pseudo': self._pseudo_s,
            'direct': self._direct_s,
            'gradient': self._gradient_s,
            'stochastic._gradient': self._sgd_s
        }
        chooser[self.solver](y, x)

    def _direct_s(self, y, tx):
        self.model = np.linalg.inv(tx.T @ tx) @ tx.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    def _pseudo_s(self, y, tx):
        U, S, V = np.linalg.svd(tx, full_matrices=False)
        self.model = V.T @ np.diag(1 / S) @ U.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    def _compute_gradient(self, y, tx, w):
        e = y - (tx @ w)
        return -1 / len(y) * (tx.T @ e)

    def _gradient_s(self, y, tx):
        niter = 0
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        prev_loss = 0
        next_loss = np.inf
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 10**(-3))
        self.losses = []

        while niter < max_iter and np.abs(next_loss - prev_loss) > threshold:
            prev_loss = next_loss
            grad = self._compute_gradient(y, tx, w)
            w -= self.solver_args['gamma'] * grad
            next_loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, w)
            self.losses.append(next_loss)
            niter += 1

        self.model = w
        self.loss = next_loss

    def _sgd_s(self, y, tx):
        batch_size = self.solver_args['batch_size']
        num_batches = self.solver_args['num_batches']
        shuffle = self.solver_args['shuffle']
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        loss = 0
        self.losses = []

        for batch_y, batch_tx in imp.batch_iter(y, tx, num_batches, shuffle):
            grad = self._compute_gradient(y, tx, w)
            w -= self.solver_args['gamma'] * grad
            loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, w)
            self.losses.append(loss)

        self.model = w
        self.loss = loss

    def predict(self, x_test):
        ready = self.preprocess_f(x_test)
        return x_test @ self.model

