from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
#        print("X[i]shape:", X[i].shape)
        scores = X[i].dot(W)
        dWi = np.zeros(W.shape)
#        print("dWi:", dWi.shape)
        correct_class_score = scores[y[i]]
        dw_yi = np.zeros(X[i].shape)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dWi[:, j] = X[i]
                dw_yi += - X[i]
        dWi[:, y[i]] = dw_yi

        dW += dWi

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Modified the code inline
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    print("W shape:", W.shape)
    print("X shape:", X.shape)
    print("Y shape:", y.shape)
    s = X.dot(W)
    print("s:", s.shape)
    correct_scores = s[np.arange(s.shape[0]), y]
#    print("corrected", correct_scores[:, np.newaxis])
    loss = s - correct_scores[:, np.newaxis] + 1 # note delta = 1. This adds 1 for the corrected class too
    loss = np.maximum(0, loss)
#    print("loss:", loss)
    ds = np.zeros(loss.shape)
    ds[loss > 0] = 1
    ds[np.arange(s.shape[0]), y] = 0
#    print("ds:", ds)
    ds[np.arange(s.shape[0]), y] = -np.sum(ds, axis = 1)
#    print("ds:", ds)

#    print("gradmask", grad_mask)
    loss = np.sum(loss) - 1 * X.shape[0] # substract 1 delta to counter the fact that 1 was previously added for the each corrected class
    loss = np.sum(loss) / X.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = X.T.dot(ds) / X.shape[0]
#    print(dW)
#    print("xi stack", X[:, np.newaxis].swapaxes(1,2).shape)
#    grad_mask = grad_mask[:, np.newaxis]
#    print("grad_mask:", grad_mask.shape)
#    grad_tensor = grad_mask * X[:, np.newaxis].swapaxes(1,2)
#    print("first grad_tensor:", grad_tensor.shape, grad_tensor)

#    print("sum:", np.sum(grad_tensor, axis = 2))
#    print(grad_tensor[np.arange(s.shape[0]), :, y])   
#    grad_tensor[np.arange(s.shape[0]),:, y] = -np.sum(grad_tensor, axis = 2)
#    print("second grad_tensor:", grad_tensor)
#    print("sum", np.sum(grad_tensor, axis = 0))
#    dW = np.sum(grad_tensor, axis = 0) / X.shape[0]
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
