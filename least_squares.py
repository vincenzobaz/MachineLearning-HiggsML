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
        ls = LeastSquares(prep, 'direct')
        ls.train(tx)
        w = ls.model
        loss = ls.loss
        predictions = ls.predict(x_test)
        """
        self.preprocessor = preprocessor
        self.solver = solver
        self.solver_args = kwargs

    def train(self, y, x):
        processed = self.preprocessor.preprocess_train(x)
        y = np.reshape(y, (len(y), 1))

        # Switch statement à-la python
        chooser = {
            'pseudo': self._pseudo_s,
            'direct': self._direct_s,
        }
        chooser[self.solver](y, processed)

    def predict(self, x_test):
        ready = self.preprocessor.preprocess_test(x_test)
        return x_test @ self.model

    def predict_labels(self, x_test):
        raw_labels = self.predict(x_test)
        raw_labels[raw_labels > 0] = 1
        raw_labels[raw_labels <= 0] = -1
        return raw_labels

    def _direct_s(self, y, tx):
        self.model = np.linalg.inv(tx.T @ tx) @ tx.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    def _pseudo_s(self, y, tx):
        U, S, V = np.linalg.svd(tx, full_matrices=False)
        self.model = V.T @ np.diag(1 / S) @ U.T @ y
        self.loss = self.solver_args.get('compute_loss', imp.rmse)(y, tx, self.model)

    @staticmethod
    def compute_gradient(y, tx, w):
        e = y - (tx @ w)
        return -1 / len(y) * (tx.T @ e)
