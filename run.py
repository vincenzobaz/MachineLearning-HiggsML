import numpy as np
import scripts.proj1_helpers as helper
import logistic
import minimizers
import implementations as imp

def polynomial_enhancement(x, deg):
    stacked_x = np.tile(x, deg)
    power_vec = np.repeat(np.array(range(1, deg + 1)), x.shape[1])
    return np.hstack((np.ones((stacked_x.shape[0], 1)), stacked_x ** power_vec))


def pseudo_least_squares(y, tx, compute_loss=imp.mse):
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


def train_predict_logistic(y_train, x_train, x_test, max_iter=100, threshold=1, deg=1):
    """
    Creates the prediction vector for the provided data after
    normalizing using logistic regression.
    """
    stds = np.std(x_train, axis=0)
    deleted_cols_ids = np.where(stds == 0)
    x_train = np.delete(x_train, deleted_cols_ids, axis=1)
    mean_spec(x_train)
    x_train = standardize(x_train)
    x_train = polynomial_enhancement(x_train, deg)
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

    loss, w = logistic.logistic_regression(y_train,
                                           x_train,
                                           max_iter,
                                           threshold,
                                           minimizers.newton)

    x_test = np.delete(x_test, deleted_cols_ids, axis=1)
    mean_spec(x_test)
    x_test_cat = standardize(x_test)
    x_test_cat = polynomial_enhancement(x_test_cat, deg)

    x_test = np.hstack((np.ones((x_test_cat.shape[0], 1)), x_test_cat))

    predictions = logistic.predict_labels(w, x_test)
    return predictions


def train_predict_logistic_cat(y_train, x_train, x_test, max_iter=100, threshold=1, deg=1):
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
                threshold=threshold, deg=deg)

        predictions[cat_indices_te] = predictions_cat.reshape(predictions[cat_indices_te].shape)
    return predictions


def logistic_cross_validation(y, x, k_fold, seed=1, train_predict_logistic=train_predict_logistic_cat, deg=1):
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

        predictions = train_predict_logistic_cat(train_y, train_x, test_x, deg=deg)

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

