import numpy as np
import run

def preprocess(x_tr, x_te=None):
    x_train = x_tr.copy()

    stds = np.std(x_train, axis=0)

    deleted_cols_ids = np.where(stds < 0.7)

    x_train = np.delete(x_train, deleted_cols_ids, axis=1)
    run.mean_spec(x_train)
    x_train = run.standardize(x_train)


    if x_te is not None:
        x_test = x_te.copy()
        stds = np.std(x_test, axis=0)
        x_test = np.delete(x_test, deleted_cols_ids, axis=1)
        run.mean_spec(x_test)
        x_test = run.standardize(x_test)
        return x_train, x_test
    else:
        return x_train

