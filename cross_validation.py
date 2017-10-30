import numpy as np


def cross_validation(y, x, k_fold, train_predict_f, seed=1):
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
        accuracy = np.array([prediction == real for prediction, real in zip(predictions, test_y)])
        accuracy = np.sum(accuracy) / accuracy.size

        return accuracy

    accuracy = []

    for i in range(k_fold):
        tmp_accuracy = cross_validation_step(i)
        accuracy.append(tmp_accuracy)
        #print('Executed step', i+1, '/', k_fold, 'of cross validation')

    return accuracy

