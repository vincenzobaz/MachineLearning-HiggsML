import numpy as np
import scripts.proj1_helpers as helper


def mse(y, tx, w):
    """Compute the mean square error"""
    
    e = y - (tx @ w)
    return (e.T @ e) / (2 * len(y))


def rmse(y, tx, w):
    """Compute the root mean square error"""
    
    return np.sqrt(2 * mse(y, tx, w))


def gradient(y, tx, w):
    """Compute the gradient"""
    
    e = y - (tx @ w)
    return -1 / len(y) * (tx.T @ e)


def least_squares_GD(y, tx, initial_w, max_iters, gamma,
                     compute_loss=mse, compute_gradient=gradient):
    """Linear regression using gradient descent"""
    
    w = initial_w
    loss = 0

    for iter_n in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w  - (gamma * gradient)

    return w, compute_loss(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma,
                      compute_loss=mse, compute_gradient=gradient):
    """Linear regression using stochastic gradient descent"""
    
    w = initial_w
    loss = 0
    for batch_y, batch_tx in batch_iter(y, tx, 1, num_batches=max_iters):
        gradient = compute_gradient(batch_y, batch_tx, w)
        w -= gamma * gradient
        loss = compute_loss(batch_y, batch_tx, w)
    return w, loss


def least_squares(y, tx, compute_loss=mse):
    """Least squares regression using normal equations"""
    
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_, compute_loss=rmse):
    """Ridge regression using normal equations"""
    
    lambda_p = lambda_ * 2 * len(y)
    w = np.linalg.inv(tx.T @ tx + lambda_p *
                      np.identity(tx.shape[1])) @ tx.T @ y
    return w, compute_loss(y, tx, w)


def sigmoid(t):
    """Logistic function"""
    
    # Checking where t is positive to avoid overflow
    negative_ids = np.where(t < 0)
    positive_ids = np.where(t >= 0)
    t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
    t[positive_ids] = np.power(np.exp(-t[positive_ids]) + 1, -1)
    return t


def compute_logistic_loss(y, tx, w):
    """Computes loss using log-likelihood"""

    txw = tx @ w
    return np.sum(np.log(1 + np.exp(txw)) - y * txw)


def compute_logistic_gradient(y, tx, w):
    """Compute gradient for logistic regression"""
    
    return tx.T @ (sigmoid(tx @ w) - y)


def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad

    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Regularized logistic regression using gradient descent"""
    
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        regularizer = lambda_ * np.linalg.norm(w)
        w - w - gamma * grad + regularizer

    return w, compute_logistic_loss(y, tx, w)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


