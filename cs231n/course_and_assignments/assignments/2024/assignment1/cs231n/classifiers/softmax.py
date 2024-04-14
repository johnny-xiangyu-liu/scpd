from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        s = scores[i]
        s -= np.max(s)
        p = np.sum(np.exp(s))
        loss += -np.log(np.exp(s[y[i]]) / p)
        for j in range(num_classes):
            df = np.exp(s[j]) / p
            if j == y[i]:
                df -= 1
            dW[:, j] += df * X[i]
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = X.shape[0]

#    print("scores:", scores)
    maxes = np.max(scores, axis = 1)
    scores -= maxes[:, np.newaxis]  # normalized scores.
    scores = np.exp(scores)
#    print("exp scores:", scores)
    sums = np.sum(scores, axis = 1)
#    print("sums", sums)

    loss = -np.log(scores[np.arange(num_train), y] / sums)
    loss = np.sum(loss) / num_train
    loss += reg * np.sum(W * W)
#    print("loss", loss)

    ds = scores / sums[:, np.newaxis]
#    print("ds", ds)
#    print("Y:", y)
    ds[np.arange(num_train), y] -= 1
#    print("ds adjust y", ds)
#    print("ds shape", ds.shape)
    dW = X.T.dot(ds) / num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW
