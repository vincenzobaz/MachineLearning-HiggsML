import numpy as np
import scripts.proj1_helpers as helper

from logistic import LogisticRegression
from preprocessing import preprocess


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


if __name__ == "__main__":
    # Import data
    y_train, x_train, ids_train = helper.load_csv_data('train.csv')
    y_test, x_test, ids_test = helper.load_csv_data('test.csv')
    y_train[y_train < 0] = 0

    # Define 1 model per category
    models = [
        LogisticRegression(degree=3, gamma=0.1),
        LogisticRegression(degree=6, gamma=0.1),
        LogisticRegression(degree=6, gamma=0.1),
        LogisticRegression(degree=6, gamma=0.1)
    ]

    # Train and predict
    predictions = train_predict_categories(y_train, x_train, x_test, *models)

    # Prepare for export
    predictions[predictions == 0] = -1

    # Export results
    helper.create_csv_submission(ids_test, predictions, 'predictions.csv')

