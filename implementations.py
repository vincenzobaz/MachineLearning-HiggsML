import numpy as np
import scripts.proj1_helpers as helper


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
    stacked_x = np.tile(x, deg)
    power_vec = np.repeat(np.array(range(1, deg + 1)), x.shape[1])
    return np.hstack((np.ones((stacked_x.shape[0], 1)), stacked_x ** power_vec))


def pseudo_least_squares(y, tx, compute_loss=mse):
    U, S, V = np.linalg.svd(tx, full_matrices=False)
    w = V.T @ np.diag(1 / S) @ U.T @ y
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_spec(data):
    for column in data.T:
        temp = 0
        agg = 0
        for elem in column:
            if elem != -999.0:
                temp += elem
                agg += 1
        column[column == -999.0] = temp / agg


def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def train_predict_logistic(y_train, x_train, x_test, max_iter=100, threshold=1):
    """
    Creates the prediction vector for the provided data after
    normalizing using logistic regression.
    """
    stds = np.std(x_train, axis=0)
    deleted_cols_ids = np.where(stds == 0)
    x_train = np.delete(x_train, deleted_cols_ids, axis=1)
    mean_spec(x_train)
    x_train = standardize(x_train)

    loss, w, _ = logistic_regression(y_train,
                                     x_train,
                                     max_iter=max_iter,
                                     threshold=threshold)

    x_test = np.delete(x_test, deleted_cols_ids, axis=1)
    mean_spec(x_test)
    x_test_cat = standardize(x_test)

    x_test = np.hstack((np.ones((x_test_cat.shape[0], 1)), x_test_cat))

    predictions = logistic_predict_labels(w, x_test)
    return predictions


def train_predict_logistic_cat(y_train, x_train, x_test, max_iter=100, threshold=1):
    """
    Creates the prediction vector for the provided data after normalizing using
    logistic regression. The data is split and in different categories according
    to column PRI_jet_nums and the model is trained independently on each category
    """
    cat_col = 22
    PRI_jet_nums = np.unique(x_train[:, cat_col])
    predictions = np.zeros(x_test.shape[0])

    for num in PRI_jet_nums:

        cat_indices_tr = np.where(x_train[:, cat_col] == num)
        x_train_cat = x_train[cat_indices_tr]
        x_train_cat = np.delete(x_train_cat, cat_col, axis=1)

        cat_indices_te = np.where(x_test[:, cat_col] == num)
        x_test_cat = x_test[cat_indices_te]
        x_test_cat = np.delete(x_test_cat, cat_col, axis=1)

        predictions_cat = train_predict_logistic(y_train[cat_indices_tr],
                x_train_cat,
                x_test_cat,
                max_iter=max_iter,
                threshold=threshold)

        predictions[cat_indices_te] = predictions_cat
    return predictions


def logistic_cross_validation(y, x, k_fold, seed=1, train_predict_logistic=train_predict_logistic_cat):
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
        train_x, train_y = x[train_indices], y[train_indices]

        predictions = train_predict_logistic(train_y, train_x, test_x)

        predictions[predictions < 0] = 0
        su = 0
        for i in range(len(predictions)):
            su += abs(predictions[i] - test_y[i])

        return (len(predictions) - su) / len(predictions)
        #return sum([abs(predictions - test_y)]) / len(predictions)

    #loss_tr = []
    #loss_te = []
    #weigths = []  # if quadratic, three parameters....
    accuracy = []

    for i in range(k_fold):
        tmp_accuracy = cross_validation_step(i)
        accuracy.append(tmp_accuracy)
        print('Executed step', i+1, '/', k_fold, 'of cross validation')

    return accuracy


def sigmoid(t):
    """Logistic function"""
    negative_ids = np.where(t < 0)
    positive_ids = np.where(t >= 0)
    t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
    t[positive_ids] = np.power(np.exp(-t[positive_ids]) + 1, -1)
    return t


def logistic_predict_labels(w, x_test):
    probs = sigmoid(x_test @ w)
    probs[probs >= 0.5] = 1
    probs[probs < 0.5] = -1
    return probs


def logistic_regression(y, tx_data, max_iter, threshold, lambda_=None):

    tx = np.hstack((np.ones((tx_data.shape[0], 1)), tx_data))
    y = np.reshape(y, (len(y), 1))

    def compute_loss(y, tx, w):
        """Computes loss using log-likelihood"""
        txw = tx @ w
        return np.sum(np.log(1 + np.exp(txw)) - y * txw)

    def compute_gradient(y, tx, w):
        return tx.T @ (sigmoid(tx @ w) - y)

    def compute_hessian(tx, w):
        tmp = sigmoid(tx @ w)
        S = tmp * (1 - tmp)
        # diagflat S => memory error. Simulate same behavior with following line
        return np.multiply(tx.T, S.T) @ tx

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

    def newton_step(y, tx, w):
        """Performs one iteration of Newton's method"""
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        hess = compute_hessian(tx, w)
        
        #tt = np.tril(hess)
        #tt = np.linalg.norm(np.identity(tt.shape) - np.linalg(tt) @ hess)
        #print('Iteration matrix norm =', tt)
        # TODO: Not sure that regularizer is compatible with amijo
        regularizer = (lambda_ * np.linalg.norm(w)) if lambda_ is not None else 0

        w = w - armijo_step(grad, w, tx) * np.linalg.inv(hess) @ grad + regularizer
        return loss, w


    w = np.zeros((tx.shape[1], 1))
    prev_loss = 0
    next_loss = np.inf
    losses = []

    for i in range(max_iter):
        prev_loss = next_loss
        next_loss, w = newton_step(y, tx, w)
        if np.abs(prev_loss - next_loss) < threshold:
            break
        #losses.append(next_loss)

        #print("Current iteration={i}, the loss={l}".format(i=i, l=next_loss))
    print('Completed logistic regression with loss', next_loss)
    return next_loss, w, losses

