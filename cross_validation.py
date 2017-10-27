import numpy as np
import implementations as imp

def cross_validation(y, x, k_fold, model, seed=1, compute_loss=imp.rmse):
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
        # tx = polynomial_enhancement(x[train_indices], degree)
        model.train(y[train_indices], x[train_indices])

        predicted_raw_labels = model.predict_labels(test_x)

        accuracy = np.array([label == test_y[i] for i, label in enumerate(predicted_raw_labels)])
        accuracy = np.sum(accuracy) / accuracy.size

        #loss_te = compute_loss(test_y, polynomial_enhancement(test_x, degree), w)
        return accuracy

    loss_tr = []
    loss_te = []
    weigths = []  # if quadratic, three parameters....
    accuracy = []

    for i in range(k_fold):
        print('Step', i + 1, '/', k_fold)
        tmp_accuracy = cross_validation_step(i)
        # loss_tr.append(tmp_loss_tr)
        # loss_te.append(tmp_loss_te)
        accuracy.append(tmp_accuracy)
        # weigths.append(w)

    return accuracy
