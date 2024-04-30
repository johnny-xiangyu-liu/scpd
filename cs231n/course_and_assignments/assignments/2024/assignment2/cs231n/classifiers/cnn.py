from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C, H, W = input_dim
        conv_w = np.random.normal(0, weight_scale, size = (num_filters, C, filter_size, filter_size))
        conv_b = np.zeros(num_filters)
        self.set_w(1, conv_w)
        self.set_b(1, conv_b)

        max_pool_dim = (C, H// 2, W // 2)
        affine_in_dim = num_filters* H//2 * W//2
        affine_w = np.random.normal(0, weight_scale, size = (affine_in_dim, hidden_dim))
        affine_b = np.zeros(hidden_dim)
        self.set_w(2, affine_w)
        self.set_b(2, affine_b)

        print("w2", self.w(2).shape)

        self.set_w(3, np.random.normal(0, weight_scale, size = ( hidden_dim, num_classes)))
        self.set_b(3, np.zeros(num_classes))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def set_w(self, i, val):
        self.params['W' + str(i)] = val
    def set_b(self, i, val):
        self.params['b' + str(i)] = val
    def w(self, i):
        return self.params['W' + str(i)]
    def b(self, i):
        return self.params['b' + str(i)]
    

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N = X.shape[0]
        i = 0
#        print("original", X.shape)
        caches = []
        X, cache = conv_forward_fast(X, W1, b1, conv_param)
        caches.append(cache)
#        print("conv", X.shape)
        X, cache = relu_forward(X)
        caches.append(cache)

        #2x2 max pool
        X, cache = max_pool_forward_fast(X, pool_param)
        caches.append(cache)
#        print("max pool", X.shape)
        # flatten
        caches.append(X.shape)
        X = X.reshape((N, -1))
#        print("flatten", X.shape)

        # affine
        X, cache = affine_forward(X, W2, b2)
        caches.append(cache)
#        print("hidden", X.shape)
        #relu
        X, cache = relu_forward(X)
        caches.append(cache)
        #affine
        X, cache = affine_forward(X, W3, b3)
        caches.append(cache)
#        print("output", X.shape)
        scores = X
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx = softmax_loss(scores, y)
        for i in [1, 2, 3]:
            loss += np.sum( self.reg * 0.5* self.w(i) * self.w(i))

        # affine:
        dx, dw3, db3 = affine_backward(dx, caches.pop())
        # relu
        dx = relu_backward(dx, caches.pop())
        # affine
        dx, dw2, db2 = affine_backward(dx, caches.pop())

        # deflatten
        dx = dx.reshape(caches.pop())
        #max pool
        dx = max_pool_backward_fast(dx, caches.pop())
        # relu
        dx = relu_backward(dx, caches.pop())        
        # conv
        dx, dw1, db1 = conv_backward_fast(dx, caches.pop())

        assert len(caches) == 0, "cache is not empty"
        
        grads['W1'] = dw1 + self.reg * self.w(1)
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg * self.w(2)
        grads['b2'] = db2
        grads['W3'] = dw3 + self.reg * self.w(3)
        grads['b3'] = db3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
