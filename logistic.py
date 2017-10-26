import numpy as np
import minimizers
from preprocessor import EmptyPreprocessor

class LogisticRegression:
    def __init__(self, preprocessor=EmptyPreprocessor(), solver='newton', **kwargs):
        self.preprocessor = preprocessor
        self.solver = solver
        self.solver_args = kwargs

    def train(self, y, x):
        processed = self.preprocessor.preprocess_train(x)
        y = np.reshape(y, (len(y), 1))

        chooser = {
            'newton': self._newton_s,
            'gradient': self._gradient,
            'stochastic': self._sgd
        }
        chooser[self.solver](y, processed)

    def predict(self, x_test):
        ready = self.preprocessor.preprocess_test(x_test)
        return LogisticRegression.sigmoid(ready @ self.model)

    def predict_labels(self, x_test):
        """
        Produces vector of predictions given the weights vector
        """
        probs = self.predict(x_test)
        probs[probs >= 0.5] = 1
        probs[probs < 0.5] = -1
        return probs

    @staticmethod
    def sigmoid(t):
        """Logistic function"""
        negative_ids = np.where(t < 0)
        positive_ids = np.where(t >= 0)
        t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
        t[positive_ids] = np.power(np.exp(-t[positive_ids]) + 1, -1)
        return t

    @staticmethod
    def compute_loss(y, tx, w):
        """Computes loss using log-likelihood"""
        txw = tx @ w
        return np.sum(np.log(1 + np.exp(txw)) - y * txw)

    @staticmethod
    def compute_gradient(y, tx, w):
        return tx.T @ (LogisticRegression.sigmoid(tx @ w) - y)

    @staticmethod
    def compute_hessian(tx, w):
        tmp = LogisticRegression.sigmoid(tx @ w)
        S = tmp * (1 - tmp)
        # diagflat S => memory error. Simulate same behavior with following line
        return np.multiply(tx.T, S.T) @ tx

    def _newton_s(self, y, tx):
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 10**(-1))
        gamma = self.solver_args.get('gamma')
        lambda_ = self.solver_args.get('lambda_')

        niter, losses, w = minimizers.newton(y, tx, w, max_iter, threshold,
                                             LogisticRegression.compute_gradient,
                                             LogisticRegression.compute_hessian,
                                             LogisticRegression.compute_loss,
                                             gamma, lambda_)

        self.niter = niter
        self.model = w
        self.losses = losses

    def _gradient(self, y, tx):
        w = self.solver_args.get('w0', np.zeros((tx.shape[1], 1)))
        max_iter = self.solver_args.get('max_iters', 100)
        threshold = self.solver_args.get('threshold', 0.1)
        gamma = self.solver_args.get('gamma')
        lambda_ = self.solver_args.get('lambda_')

        niter, losses, w = minimizers.gradient_descent(y, tx, w, max_iter, threshold,
                                                       LogisticRegression.compute_gradient,
                                                       LogisticRegression.compute_loss,
                                                       gamma, lambda_)

        self.niter = niter
        self.model = w
        self.losses = losses

    def _sgd(self, y, tx):
        pass
