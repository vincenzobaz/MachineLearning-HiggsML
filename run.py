import numpy as np
import scripts.proj1_helpers as helper

from logistic import LogisticRegression
import preprocessing


def category_iter(y_train, x_train, cat_col, x_test=None):
    """Given the index of a column containing category indexes, iterates over
    the different categories in the data after removing the column
    """
    # Distinct values of the category
    values = np.unique(x_train[:, cat_col])

    # For each distinct category index
    for val in values:
        # Retrieve the indices of the rows beloging to the category
        cat_indices_tr = np.where(x_train[:, cat_col] == val)

        # Extract all rows in the category, for x and y
        x_train_cat = x_train[cat_indices_tr]
        y_train_cat = y_train[cat_indices_tr]

        # Delete the category indicator column
        x_train_cat = np.delete(x_train_cat, cat_col, axis=1)

        if x_test is not None:
            # Apply the same procedure to x_test
            cat_indices_te = np.where(x_test[:, cat_col] == val)
            x_test_cat = x_test[cat_indices_te]
            x_test_cat = np.delete(x_test_cat, cat_col, axis=1)
            yield y_train_cat, x_train_cat, x_test_cat, cat_indices_te
        else:
            yield y_train_cat, x_train_cat


def repeater(x):
    """Generator returning x infinitely"""
    while True:
        yield x


def train_predict_categories(y_train, x_train, x_test, *models):
    """
    Creates the prediction vector for the provided data after normalizing using
    the provided model(s). The data is split and trained in different categories
    according to column PRI_jet_nums. If only one model is provided, it will be
    used for all the categories. If more are provided, each category will have
    its own model.
    """
    # Find PRI_jet_nums column index
    cat_col = 22
    for idx, col in enumerate(x_train.T):
        if len(col) == 4 and np.allclose(np.arange(0, 4), col):
            cat_col = idx

    # Prepare vector to store the predictions
    predictions = np.zeros(x_test.shape[0])

    # If only one model is provided, use it across all categories.
    if len(models) == 1:
        models = repeater(models[0])

    for model, cat_data in zip(models, category_iter(y_train, x_train, cat_col, x_test)):
        # Unpack iterator data
        y_train_cat, x_train_cat, x_test_cat, cat_indices_te = cat_data

        # Preprocess
        x_train_cat, x_test_cat = preprocess(x_train_cat, x_test_cat)

        # Train & predict
        predictions_cat = model.train(y_train_cat, x_train_cat)\
                               .predict_labels(x_test_cat)

        # Store predictions at the right positions in the result vector.
        predictions[cat_indices_te] = predictions_cat.reshape(predictions[cat_indices_te].shape)

    return predictions


def best_cross_validation(y, x, k_fold, train_predict_f=train_predict_categories, seed=1):
    """
    Computes the accuracy of the train_predict_f function using k-fold cross
    validation.

    Keyword arguments:
    y -- The known predictions.
    x -- The data points associated to y predictions.
    k-fold -- The number of folds.
    train_predict_f -- The function used to train and predict the data. It
                       trains on y_train, x_train and predicts for  x_test.
    seed -- seed for the random generator used to split the data.
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
        # Split test data
        test_x, test_y = x[k_indices[k]], y[k_indices[k]]

        # Split training data
        train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_indices = np.ravel(train_indices)
        train_x, train_y = x[train_indices], y[train_indices]

        # Predict
        predictions = train_predict_f(train_y, train_x, test_x)

        # Compute and return accuracy
        return np.sum(test_y == predictions) / len(predictions)

    accuracy = []

    for i in range(k_fold):
        tmp_accuracy = cross_validation_step(i)
        accuracy.append(tmp_accuracy)
        print('Executed step', i+1, '/', k_fold, 'of cross validation')

    return accuracy
