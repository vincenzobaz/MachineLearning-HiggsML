import numpy as np


def minimize(y, tx, init_arg, max_iter, threshold, step_f, cost_f):
    losses = []
    prev_loss, next_loss = 0, np.inf
    niter = 0
    w = init_arg.copy()

    while niter < max_iter and np.abs(prev_loss - next_loss) > threshold:
        prev_loss = next_loss
        next_loss = cost_f(y, tx, w)
        w = step_f(y, tx, w)
        losses.append(next_loss)
        niter += 1

    return niter, losses, w


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


def newton_step(y, tx, w, compute_gradient, compute_hessian, gamma=None, lambda_=None):
    """Performs one iteration of Newton's method"""
    grad = compute_gradient(y, tx, w)
    hess = compute_hessian(tx, w)

    # TODO: Not sure that regularizer is compatible with armijo
    regularizer = (lambda_ * np.linalg.norm(w)) if lambda_ is not None else 0
    gamma = armijo_step(grad, w, tx) if not gamma else gamma

    w = w - gamma * np.linalg.pinv(hess) @ grad + regularizer
    return w


def gradient_descent_step(y, tx, w, compute_gradient, gamma):
    return w - gamma * compute_gradient(y, tx, w)


def newton(y, tx, w, max_iter, threshold, compute_gradient, compute_hessian,
        loss_f, gamma=None,  lambda_=None):
    step_f = lambda y, tx, w: newton_step(y, tx, w, compute_gradient, compute_hessian, gamma)

    return minimize(y, tx, w, max_iter, threshold, step_f, loss_f)


def gradient_descent(y, tx, w, max_iter, threshold, compute_gradient, cost_f, gamma, lambda_=None):
    step_f = lambda y, tx, w: gradient_descent_step(y, tx, w, compute_gradient, gamma)

    return minimize(y, tx, w, max_iter, threshold, step_f, cost_f)

# stochastic gradient descent
