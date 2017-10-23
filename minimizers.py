import numpy as np


def armijo_step(grad, w, tx, tests=1000):
    """
    Provides best learning step for the current iteration of newton's method
    using Armijo's rule performing linear search to minimize function
    phi(eta) = f(w + eta * d)
    """
    d = grad / np.linalg.norm(grad) # Compute direction vector
    etas = np.linspace(0, 1, tests+1)[1:] # Learning rates to test
    # Compute for each learning rate, the "length" of move, to minimize.
    r = np.linalg.norm(tx @ (np.tile(w, tests) + np.outer(d, etas)), axis=0)
    return etas[np.argmin(r)] # Take etas that minimizes phi


def newton_step(y, tx, w, compute_gradient, compute_hessian, lambda_=None):
    """Performs one iteration of Newton's method"""
    grad = compute_gradient(y, tx, w)
    hess = compute_hessian(tx, w)

    #tt = np.tril(hess)
    #tt = np.linalg.norm(np.identity(tt.shape) - np.linalg(tt) @ hess)
    #print('Iteration matrix norm =', tt)
    # TODO: Not sure that regularizer is compatible with amijo
    regularizer = (lambda_ * np.linalg.norm(w)) if lambda_ is not None else 0

    w = w - armijo_step(grad, w, tx) * np.linalg.pinv(hess) @ grad + regularizer
    return w


def gradient_descent_step(y, tx, w, compute_gradient, gamma):
    return w - gamma * compute_gradient(y, tx, w)

