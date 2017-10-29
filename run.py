import numpy as np
import scripts.proj1_helpers as helper

from logistic import LogisticRegression
from preprocessing import preprocess


def category_iter(y_train, x_train, cat_col, x_test=None):
    values = np.unique(x_train[:, cat_col])

    for val in values:
        cat_indices_tr = np.where(x_train[:, cat_col] == val)
        x_train_cat = x_train[cat_indices_tr]
        x_train_cat = np.delete(x_train_cat, cat_col, axis=1)
        y_train_cat = y_train[cat_indices_tr]

        if x_test is not None:
            cat_indices_te = np.where(x_test[:, cat_col] == val)
            x_test_cat = x_test[cat_indices_te]
            x_test_cat = np.delete(x_test_cat, cat_col, axis=1)
            yield y_train_cat, x_train_cat, x_test_cat, cat_indices_te
        else:
            yield y_train_cat, x_train_cat


def train_predict_categories(y_train, x_train, x_test, *models):
    """
    Creates the prediction vector for the provided data after normalizing using
    logistic regression. The data is split and trained in different categories
    according to column PRI_jet_nums and the model is trained independently on each category
    """
    cat_col = 22
    for idx, col in enumerate(x_train.T):
        if len(col) == 4 and np.allclose(np.arange(0, 4), col):
            cat_col = idx

    predictions = np.zeros(x_test.shape[0])

    for model, cat_data in zip(models, category_iter(y_train, x_train, cat_col, x_test)):
        y_train_cat, x_train_cat, x_test_cat, cat_indices_te = cat_data
        x_train_cat, x_test_cat = preprocess(x_train_cat, x_test_cat)

        predictions_cat = model.train(y_train_cat, x_train_cat)\
                               .predict_labels(x_test_cat)

        predictions[cat_indices_te] = predictions_cat.reshape(predictions[cat_indices_te].shape)
    return predictions


def best_cross_validation(y, x, k_fold, train_predict_f=train_predict_categories, seed=1):
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

        predictions = train_predict_f(train_y, train_x, test_x)

        su = 0
        for i in range(len(predictions)):
            su += abs(predictions[i] - test_y[i])

        return (len(predictions) - su) / len(predictions)

    accuracy = []

    for i in range(k_fold):
        tmp_accuracy = cross_validation_step(i)
        accuracy.append(tmp_accuracy)
        print('Executed step', i+1, '/', k_fold, 'of cross validation')

    return accuracy

