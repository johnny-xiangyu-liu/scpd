import numpy as np
import util
import math

def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Use np.savetxt to save predictions on eval set to save_path
    import pandas as pd
    data = pd.DataFrame({
        "Age": [24, 53, 23, 25, 32, 52, 22, 43, 52, 48],
        "Salary": [40, 52, 25, 77, 48, 110, 38, 44, 27, 65],
        "College": [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    })
    # Create a simple dataset for testing the decision tree
    x_train = np.array(data[["Age", "Salary"]])
    y_train = np.array( data["College"])
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(len(x_train)):
        plt.plot(x_train[i][0], x_train[i][1], 'bx' if y_train[i] == 0 else 'go', linewidth=2)
    plt.savefig("ps3.png")

    clf = LogisticRegression()

    for reg in [True]:
        path = save_path
        clf.l2_reg = reg
        if reg:
            print("with L2 regularization")
            path += "_with_l2";
        clf.fit(x_train, y_train)
        clf.theta = np.array([-1.5, 2])
        pred = clf.predict(x_train)
        for x, p, y in zip(x_train, pred, y_train):
            sign = 1 if p-1 >=0 else -1
            print("x:{}, sign:{} y:{}".format(x, sign,y))


    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=1000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        self.l2_reg = False
        self.lamb = 0.01
        # *** END CODE HERE ***

    def sigmoid(self, val):
        print("val:", val)
        gamma = val
        if gamma < 0:
            return 1 - 1/(1 + math.exp(gamma))
        else:
            return 1/(1 + math.exp(-gamma))

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        print("x")
        print(x)
        print("y")
        print(y)
        self.theta = np.zeros(len(x[0]))
        def h_x_i(x_i):
            return self.sigmoid(self.theta.T.dot(x_i))

        def gradient(x, y):
            size = len(x)
            h_x = np.array([h_x_i(x_i) for x_i in x])
            result = - 1/ size * x.T.dot(y - h_x)
            if self.l2_reg:
                result = result + self.lamb*self.theta
            return result
        def l1_distance(a, b):
            sum = 0;
            for i in range(0, len(a)):
                sum += abs(a[i] - b[i])
            return sum

        for t in range(0, self.max_iter):
            new_theta = self.theta - self.learning_rate * gradient(x,y)
            if l1_distance(new_theta, self.theta) < self.eps:
                break;
            self.theta = new_theta
#            print("t:{} updating:{}".format(t, new_theta));
        print("t:{}, self.theta:{}".format(t, self.theta));
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        score = x.dot(self.theta.T)
        return np.array([self.sigmoid(s) for s in  score])
        # *** END CODE HERE ***

if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a.txt')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')
