import numpy as np
import proj1_helpers as helper


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


def polynomial_enhancement(x, deg):
    stacked_x = np.tile(x, deg + 1)
    power_vec = np.repeat(np.array(range(deg + 1)), x.shape[1])
    return stacked_x ** power_vec


def pseudo_least_squares(y, tx, compute_loss=mse):
    U, S, V = np.linalg.svd(tx, full_matrices=False)
    w = V.T @ np.diag(1 / S) @ U.T @ y
    loss = compute_loss(y, tx, w)
    return w, loss


def cross_validation(y, x, k_fold, regression_f, degree, seed=1, compute_loss=rmse):
    """
    Computes weights, training and testing error

    regression_f is a regressiong function only accepting y and the tx matrix.
    In case of ridge regression (or any other function needing more paremeters),
    the additional ones can be curried.
    e.g. f = lambda y, tx: ridge_regression(y, tx, lambda_, compute_loss=...)
    """

    def build_k_indices():
        """build k indices for k-fold."""
        num_row = y.shape[0]
        interval = int(num_row / k_fold)
        np.random.seed(seed)
        indices = np.random.permutation(num_row)
        k_indices = [indices[k * interval: (k + 1) * interval]
                     for k in range(k_fold)]
        return np.array(k_indices)

    k_indices = build_k_indices()

    def cross_validation_step(k):
        """Computes one iteration of k-fold cross validation"""
        test_x, test_y = x[k_indices[k]], y[k_indices[k]]
        train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_indices = np.ravel(train_indices)
        tx = polynomial_enhancement(x[train_indices], degree)
        w, loss_tr = regression_f(y[train_indices], tx)
        loss_te = compute_loss(
            test_y, polynomial_enhancement(test_x, degree), w)
        return loss_tr, loss_te, w

    loss_tr = 0
    loss_te = 0
    weigths = np.zeros((degree + 1) * x.shape[1])  # if quadratic, three parameters....
    print(weigths.shape)

    for i in range(k_fold):
        tmp_loss_tr, tmp_loss_te, w = cross_validation_step(i)
        loss_tr += tmp_loss_tr
        loss_te += tmp_loss_te
        weigths += w

    return loss_tr / k_fold, loss_te / k_fold, weigths / k_fold

def cross_validation_v2(y, x, k_fold, regression_f, degree, seed=1, compute_loss=rmse):
    """
    Computes weights, training and testing error

    regression_f is a regressiong function only accepting y and the tx matrix.
    In case of ridge regression (or any other function needing more paremeters),
    the additional ones can be curried.
    e.g. f = lambda y, tx: ridge_regression(y, tx, lambda_, compute_loss=...)
    """

    def build_k_indices():
        """build k indices for k-fold."""
        num_row = y.shape[0]
        interval = int(num_row / k_fold)
        np.random.seed(seed)
        indices = np.random.permutation(num_row)
        k_indices = [indices[k * interval: (k + 1) * interval]
                     for k in range(k_fold)]
        return np.array(k_indices)

    k_indices = build_k_indices()

    def cross_validation_step(k):
        """Computes one iteration of k-fold cross validation"""
        test_x, test_y = x[k_indices[k]], y[k_indices[k]]
        train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_indices = np.ravel(train_indices)
        tx = polynomial_enhancement(x[train_indices], degree)
        w, loss_tr = regression_f(y[train_indices], tx)

        empirical_y_test = helper.predict_labels(w, polynomial_enhancement(test_x, degree))
        sum_vector = (empirical_y_test + test_y)
        accuracy = sum_vector[np.where(sum_vector != 0)].size / test_y.size

        loss_te = compute_loss(
            test_y, polynomial_enhancement(test_x, degree), w)
        return loss_tr, loss_te, w, accuracy

    loss_tr = []
    loss_te = []
    weigths = []  # if quadratic, three parameters....
    accuracy = []

    for i in range(k_fold):
        tmp_loss_tr, tmp_loss_te, w, tmp_accuracy = cross_validation_step(i)
        loss_tr.append(tmp_loss_tr)
        loss_te.append(tmp_loss_te)
        accuracy.append(tmp_accuracy)
        weigths.append(w)

    return accuracy, loss_tr, loss_te, weigths

def logistic_regression(y, tx, max_iter, threshold, lambda_=None):
    def sigmoid(t): 
        """Logistic function"""
        return np.exp(t) / (1 + np.exp(t))

    def compute_loss(w):
        """Computes loss using log-likelihood"""
        txw = tx @ w
        return np.sum(np.log(1 + np.exp(txw)) - y @ txw.T)

    def compute_gradient(w):
        return tx.T @ (sigmoid(tx @ w) - y)

    def compute_hessian(w):
        tmp = sigmoid(tx @ w)
        S = np.diagflat(tmp * (1 - tmp))
        return tx.T @ S @ tx

    def armijo_step(grad, w, tests=1000):
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

    def newton_step(w):
        """Performs one iteration of Newton's method"""
        loss = compute_loss(w)
        grad = compute_gradient(w)
        hess = compute_hessian(w)
        # TODO: Not sure that regularizer is compatible with amijo
        regularizer = lambda_ * np.linalg.norm(w) if lambda_ is not None else 0

        w = w - armijo_step(grad, w) * np.linalg.inv(hess) @ grad + regularizer
        return loss, w

    w = np.zeros((tx.shape[1], 1))
    prev_loss = 0
    next_loss = np.inf
    n_iter = 0

    while(n_iter < max_iter and np.abs(prev_loss - next_loss) >= threshold):
        prev_loss = next_loss
        next_loss, w = newton_step(w)
        print("Current iteration={i}, the loss={l}".format(i=n_iter, l=next_loss))
        n_iter += 1

    return next_loss, w
