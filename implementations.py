import numpy as np


def mse(y, tx, w):
    e = y - (tx @ w)
    return (e @ e.T) / (2 * len(y))


def rmse(y, tx, w):
    return np.sqrt(2 * mse(y, tx, w))


def gradient(y, tx, w):
    e = y - (tx @ w)
    return -1 / len(y) * (tx.T @ e)

# TODO: Ask if we can add named optional parametrs
# to avoid this and being able to switch functions
def least_squares_GD(y, tx, initial_w, max_iters, gamma,
                     compute_loss=mse, compute_gradient=gradient):
    iter_n = 0
    w = initial_w
    loss = 0

    for iter_n in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient
        loss = compute_loss(y, tx, w)

    return w, compute_loss(y, tx, w)


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


def least_squares_SGD(y, tx, initial_w, max_iters, gamma,
                      compute_loss=mse, compute_gradient=gradient):
    w = initial_w
    loss = 0
    for batch_y, batch_tx in batch_iter(y, tx, 1, num_batches=max_iters):
        gradient = compute_gradient(batch_y, batch_tx, w)
        w -= gamma * gradient
        loss = compute_loss(batch_y, batch_tx, w)
    return w, loss


def least_squares(y, tx, compute_loss=mse):
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_, compute_loss=rmse):
    lambda_p = lambda_ * 2 * len(y)
    w = np.linalg.inv(tx.T @ tx + lambda_p *
                      np.identity(tx.shape[1])) @ tx.T @ y
    return w, compute_loss(y, tx, w)
