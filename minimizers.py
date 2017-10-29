import numpy as np


def minimize(y, tx, init_arg, max_iter, threshold, step_f, loss_f):
    """Performs iterative minimization of the loss function loss_f.

    Keyword arguments:
    init_arg -- Initial value for the argument of the loss function.
    max_iter -- The maximal number of iterations to perform.
    threshold -- if the difference of losses of two consecutive iterations
                 of the minimizer is smaller than this value, the
                 minimizer returns.
    step_f -- Function computing the new value of the argument of the loss
              function.
    loss_f -- The loss function to minimize.
    """
    losses = []
    prev_loss, next_loss = 0, np.inf
    niter = 0
    w = init_arg

    while niter < max_iter and np.abs(next_loss - prev_loss) > threshold:
        prev_loss = next_loss
        w = step_f(y, tx, w)
        next_loss = loss_f(y, tx, w)
        losses.append(next_loss)
        niter += 1

    return niter, losses, w


def newton(y, tx, w, max_iter, threshold, compute_gradient, compute_hessian,
           loss_f, gamma=None, lambda_=None):
    """Minimizes the loss function iteratively using Newton's method.

    Keyword arguments:
    w -- Initial value for the argument of the loss function.
    max_iter -- The maximal number of iterations to perform.
    threshold -- if the difference of losses of two consecutive iterations
                 of the minimizer is smaller than this value, the
                 minimizer returns.
    compute_gradient -- function returning the value of the gradient of the loss
                        funtction evaluated at the given w.
    compute_hessian -- function returning the hessian matrix of the loss
                       funtction evaluated at the given w.
    loss_f -- The loss function to minimize.
    gamma -- learning rate. If none is provided, a value is automatically.
             computed to minimize the distance between to successive ws.
    lambda_ -- regularizer coefficient. If none is provided,
               no regularization is used.
    """
    # Create the step function by currying some arguments
    step_f = lambda y, tx, w: newton_step(y, tx, w,
                                          compute_gradient,
                                          compute_hessian,
                                          gamma,
                                          lambda_)

    return minimize(y, tx, w, max_iter, threshold, step_f, loss_f)


def gradient_descent(y, tx, w, max_iter, threshold, compute_gradient, loss_f,
                     gamma, lambda_=None):
    """Minimizes the loss function iteratively using gradient descent.

    Keyword arguments:
    w -- Initial value for the argument of the loss function.
    max_iter -- The maximal number of iterations to perform.
    threshold -- if the difference of losses of two consecutive iterations
                 of the minimizer is smaller than this value, the
                 minimizer returns.
    compute_gradient -- function returning the value of the gradient of the loss
                        funtction evaluated at the given w.
    loss_f -- The loss function to minimize.
    gamma -- learning rate.
    lambda_ -- regularizer coefficient. If none is provided,
               no regularization is used.
    """
    # Create the step function by currying some arguments
    step_f = lambda y, tx, w: gradient_descent_step(y, tx, w,
                                                    compute_gradient,
                                                    gamma,
                                                    lambda_)

    return minimize(y, tx, w, max_iter, threshold, step_f, loss_f)


def newton_step(y, tx, w, compute_gradient, compute_hessian, gamma=None, lambda_=None):
    """Performs one iteration of Newton's method producing a new w"""
    grad = compute_gradient(y, tx, w)
    hess = compute_hessian(tx, w)

    regularizer = (lambda_ * np.linalg.norm(w)) if lambda_ is not None else 0
    gamma = learning_step(grad, w, tx) if not gamma else gamma

    w = w - gamma * np.linalg.pinv(hess) @ grad + regularizer
    return w


def gradient_descent_step(y, tx, w, compute_gradient, gamma, lambda_):
    """Performs one iteration of gradient descent producing a new w"""
    regularizer = lambda_ * np.linalg.norm(w) if lambda_ is not None else 0
    return w - gamma * compute_gradient(y, tx, w) + regularizer


def learning_step(grad, w, tx, tests=100):
    """
    Provides a good learning step eta for the current iteration of Newton's method
    by taking the value in np.linspace(0, 1, tests) which minimzes the norm
    of phi(eta) = f(w + eta * d)
    """
    d = grad / np.linalg.norm(grad) # Compute direction vector
    etas = np.linspace(0, 1, tests+1)[1:] # Learning rates to test
    # Compute for each learning rate, the "length" of move, to minimize.
    r = np.linalg.norm(tx @ (np.tile(w, tests) + np.outer(d, etas)), axis=0)
    return etas[np.argmin(r)] # Take etas that minimizes phi
