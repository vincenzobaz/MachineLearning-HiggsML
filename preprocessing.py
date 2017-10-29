import numpy as np


def polynomial_enhancement(x, deg):
    stacked_x = np.tile(x, deg)
    power_vec = np.repeat(np.array(range(1, deg + 1)), x.shape[1])
    return np.hstack((np.ones((stacked_x.shape[0], 1)), stacked_x ** power_vec))


def most_frequent(arr):
    _, counts = np.unique(arr, return_counts=True)
    return np.argmax(counts)


def mean_spec(data):
    arr = data.copy()
    not_nines = arr != -999.0
    for indicator, column in zip(not_nines.T, arr.T):
        mean = np.mean(column[indicator])
        column[np.logical_not(indicator)] = mean
    return arr


def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def preprocess(x_tr, x_te=None):
    x_train = x_tr.copy()

    stds = np.std(x_train, axis=0)
    deleted_cols_ids = np.where(stds == 0)

    x_train = np.delete(x_train, deleted_cols_ids, axis=1)
    x_train = mean_spec(x_train)
    x_train = standardize(x_train)


    if x_te is not None:
        x_test = x_te.copy()
        stds = np.std(x_test, axis=0)
        x_test = np.delete(x_test, deleted_cols_ids, axis=1)
        mean_spec(x_test)
        x_test = standardize(x_test)
        return x_train, x_test
    else:
        return x_train
