import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import random
import time

def debug(name, x):
    if False:
        print("{}:{}".format(name, x.shape));

def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    result = []
    def smart_exp(x):
        if x > 100:
            return math.inf;
        else:
            return math.exp(x);
    for classes in x:
        sum = 0;
        soft_max = []
        for val in classes:
            exponential = smart_exp(val);
            if exponential == math.inf:
                sum = math.inf
            if sum != math.inf:
                sum += exponential
            soft_max.append(exponential)
        for i in range(len(soft_max)):
            if sum == math.inf:
                if soft_max[i] == math.inf:
                    soft_max[i] = 1;
                else:
                    soft_max[i] = 1*10**-8;
            else:
                soft_max[i] = soft_max[i] / sum

        result.append(soft_max)

    return np.array(result)
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    def sigmoid_i(xi):
        if xi < 0:
            return np.exp(xi) / (1 + np.exp(xi))
        else:
            return 1 / (1 + np.exp(-xi))

    result =  np.array([np.vectorize(sigmoid_i)(xi) for xi in x])
    return result
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    print("input size:{}, num_hidden:{}, num_output:{}".format(
        input_size, num_hidden, num_output
    ))
    def make(row, col):
        result = np.zeros((row, col));
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = np.random.normal(loc=0, scale = 1.0)
        return result;

    result = {'W1': make(num_hidden,input_size),
              'b1': np.zeros((num_hidden,1)),
              'W2': make(num_output,num_hidden),
              'b2': np.zeros((num_output,1)),
              }

    return result
    # *** END CODE HERE ***

def forward_prop(data, one_hot_labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    x = data
    debug("data", data);
    debug("label", one_hot_labels);

    W1 = params['W1']
    b1 = params['b1']
    z1 = sigmoid(W1.dot(x.T) + b1).T

    W2 = params['W2']
    b2 = params['b2']
    z2 = softmax(W2.dot(z1.T) + b2).T

    n = len(x)
    loss = 0;
    for i in range(n):
        loss += one_hot_labels[i].T.dot(np.log(z2[i]))
    loss *= - 1/n
#    print("loss:{}".format(loss))
    return (z1, z2, loss)
    # *** END CODE HERE ***

def backward_prop(data, one_hot_labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API abo ve

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
#    print("backward prop")
    x = data

    W1 = params['W1']
    b1_ = params['b1']

    W2 = params['W2']
    b2_ = params['b2']

    b1 = np.array([b1_.T[0]]).T
    b2 = np.array([b2_.T[0]]).T


#    print("b1:{}".format(b1_.T))
#    print("b2:{}".format(b2_.T))
    debug("W1", W1)
    debug("b1", b1)
    debug("W2", W2)
    debug("b2", b2)

    z1, z2, loss = forward_prop_func(data, one_hot_labels, params)
    debug("z1", z1)
    debug("z2", z2)
    def single(x, ey, z1, z2):
#        print(">>single:")
        debug("x", x)
        debug("ey", ey)
        debug("z1", z1)
        debug("z2", z2)
        debug("W1", W1)
        debug("b1", b1)
        debug("W2", W2)
        debug("b2", b2)
    #  d(y)/d(w2) = d(y) / d(h_hat_theta) * d(h_hat_theta)/d(w2)
        dy_dh_hat_theta = z2 - ey
        debug("dy_dh_hat_theta", dy_dh_hat_theta)
        #  h_hat_theta = W2.T.dot(z1) + b2
        h_hat_theta = W2.dot(z1) + b2
        debug("h_hat_theta", h_hat_theta)
        d_y_d_w2 = np.outer(dy_dh_hat_theta.T, z1.T)
        debug("dy_dw2", d_y_d_w2)

        dh_hat_theta_db2 = np.identity(len(h_hat_theta))
        dy_db2 = dh_hat_theta_db2.T.dot(dy_dh_hat_theta)
        debug("dy_db2", dy_db2)


        dh_hat_theta_dz1 = W2.T
        dy_dz1 = dh_hat_theta_dz1.dot(dy_dh_hat_theta)
        debug("dy_dz1", dy_dz1)
        # dz1_dw1 = sigmoid * (1 - sigmoid) * [x ]

        preactivation = W1.dot(x) + b1
        debug("preact", preactivation)
        sig_preact_value = sigmoid(preactivation.T)
        dz1_dpreact = np.diag([ (s * (1-s)) for s in sig_preact_value[0]])
        debug("dz1_dpreact", dz1_dpreact)
        dy_dpreact = dz1_dpreact.dot(dy_dz1)

        debug("dy_dpreact", dy_dpreact)

        dy_dw1 = np.outer(dy_dpreact.T, x.T)
        debug("dy_dw1", dy_dw1)

        dy_db1 = dy_dpreact
        debug("dy_db1", dy_db1)
        return (d_y_d_w2, dy_db2, dy_dw1, dy_db1)
    dy_dw1_sum = 0
    dy_db1_sum = 0
    dy_dw2_sum = 0
    dy_db2_sum = 0

    for i in range(len(x)):
        xi = np.array([x[i]]).T
        eyi = np.array([one_hot_labels[i]]).T
        z1i = np.array([z1[i]]).T
        z2i = np.array([z2[i]]).T
        dy_dw2, dy_db2, dy_dw1, dy_db1 = single(xi, eyi, z1i, z2i)

        dy_dw1_sum += dy_dw1
        dy_db1_sum += dy_db1
        dy_dw2_sum += dy_dw2
        dy_db2_sum += dy_db2

    return {
        "W2": dy_dw2_sum/len(x),
        "b2": dy_db2_sum/len(x),
        "W1": dy_dw1_sum/len(x),
        "b1": dy_db1_sum/len(x)
    }
    # *** END CODE HERE ***


def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)
p
    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    gradient = backward_prop(data, one_hot_labels, params, forward_prop_func)

    W1 = params['W1']
    W2 = params['W2']
    dW1 = gradient['W1']
    dW2 = gradient['W2']

    dW1 -= reg * 2 * W1
    dW2 -= reg * 2 * W2
    return {
        "W2": dW2,
        "b2": params['b2'],
        "W1": dW1,
        "b1": params['b1']
    }

    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, one_hot_train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        one_hot_train_labels: A numpy array containing the one-hot embeddings of the training labels e_y.
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    batch = []
    ey = []
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i: i + batch_size]
        ey = one_hot_train_labels[i: i+batch_size]
        batch=np.array(batch)
        ey=np.array(ey)

        gradient_map = backward_prop_func(batch, ey, params, forward_prop_func)
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']


        dw1 = gradient_map['W1']
        db1 = gradient_map['b1']
        dw2 = gradient_map['W2']
        db2 = gradient_map['b2']

        lr = learning_rate
        W1 = W1 - lr * dw1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dw2
        b2 = b2 - lr * db2

        params['W1'] = W1
        params['b1'] = b1
        params['W2'] = W2
        params['b2'] = b2
    # *** END CODE HERE ***

    # This function does not return anything
    return
def save_params(params, prefix = ''):
    names = ['W1', 'b1', 'W2', 'b2']
    t = time.time()
    for name in names:
        file_name = "{}-{}-{}.txt".format(prefix,name,t)
        np.savetxt(file_name, params[name])
    return


def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        print("epoch:{}".format(epoch))
        gradient_descent_epoch(train_data, train_labels,
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    save_params(params, name)

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    # convert labels to one-hot embeddings e_y.
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    # convert labels to one-hot embeddings e_y.
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)

    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
