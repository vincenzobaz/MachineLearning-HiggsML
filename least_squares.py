import numpy as np
import implementations as imp
from preprocessor import EmptyPreprocessor
import run as run

class LeastSquares:
    def __init__(self, degree=1, solver='pseudo', **kwargs):
        """Model constructor. A model provides an interface composed of
        train, predict, loss, model.
        This model implements the least squares computation.

        Keyword arguments:
        preprocessor -- Preprocessor used to prepare x_train before training
                        and x_test before predicting.
        solver -- the least squares problem solver. Currently implemented
                  are (pseudo, direct).
        kwargs -- Additional arguments for the solver, in this case the loss
                  function. If no compute_loss is provided, rmse will be used.

        Exemple of usage:
        ls = LeastSquares(prep, 'direct')
        ls.train(tx)
        w = ls.model
        loss = ls.loss
        predictions = ls.predict(x_test)
        """
        self.solver = solver
        self.solver_args = kwargs
        self.degree = degree

    def train(self, y, x):
        """Trains the model on the provided x,y data"""
        processed = run.polynomial_enhancement(x, self.degree)
        y = np.reshape(y, (len(y), 1))

        # Switch statement à-la python
        chooser = {
            'pseudo': self._pseudo_s,
            'direct': self._direct_s,
        }
        chooser[self.solver](y, processed)

    def predict(self, x_test):
        """Predicts y values for the provided test data"""
        ready = run.polynomial_enhancement(x_test, self.degree)
        return ready @ self.model

    def predict_labels(self, x_test):
        """Generates labels (-1, 1) for classification"""
        raw_labels = self.predict(x_test)
        raw_labels[raw_labels > 0] = 1
        raw_labels[raw_labels <= 0] = -1
        return raw_labels

    def _direct_s(self, y, tx):
        """Directly computes w by inverting tx.T @ tx"""
        self.model = np.linalg.inv(tx.T @ tx) @ tx.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    def _pseudo_s(self, y, tx):
        """Directly computes w by pseudo inverse of tx.T @ tx"""
        self.model = np.linalg.pinv(tx) @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    @staticmethod
    def compute_gradient(y, tx, w):
        """Computes the gradient of f(w) = tx @ w"""
        e = y - (tx @ w)
        return -1 / len(y) * (tx.T @ e)
