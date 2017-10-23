import numpy as np


def sigmoid(t):
    """Logistic function"""
    negative_ids = np.where(t < 0)
    positive_ids = np.where(t >= 0)
    t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
    t[positive_ids] = np.power(np.exp(-t[positive_ids]) + 1, -1)
    return t


def predict_labels(w, x_test):
    """
    Produces vector of predictions given the weights vector
    """
    probs = sigmoid(x_test @ w)
    probs[probs >= 0.5] = 1
    probs[probs < 0.5] = -1
    return probs


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


def logistic_regression(y, tx, max_iter, threshold, minimzer_step, lambda_=None):
    y = np.reshape(y, (len(y), 1))

    w = np.zeros((tx.shape[1], 1))
    prev_loss = 0
    next_loss = np.inf
    losses = []

    for i in range(max_iter):
        prev_loss = next_loss
        next_loss = compute_loss(y, tx, w)
        w = minimzer_step(y, tx, w)
        if np.abs(prev_loss - next_loss) < threshold:
            break
        losses.append(next_loss)

        #print("Current iteration={i}, the loss={l}".format(i=i, l=next_loss))
    print('Completed logistic regression with loss', next_loss)
    return next_loss, w, losses

