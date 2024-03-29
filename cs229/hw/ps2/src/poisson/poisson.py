import numpy as np
import util
import matplotlib.pyplot as plt
import math

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model

    clf = PoissonRegression()
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = clf.predict(x_eval)
    np.savetxt(save_path, y_predict)

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(y_eval, y_predict)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.savefig(save_path +".png")

"""
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(clf.predict(x_train), y_train, marker='x')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.savefig("training.png")
"""
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(len(x[0]))
        print("x")
        print(x)
        print(len(x))
        print(len(x[0]))
        print("y")
        print(y)

        def h(x_i):
            return math.exp(self.theta.T.dot(x_i))

        def gradient_ll_j(X, Y, j):
            sum = 0
            for i in range(len(X)):
                sum += (Y[i] - h(X[i])) * X[i][j]
            return sum

        def gradient(X, Y):
            result = [];
            for j in range(len(X[0])):
                result.append(gradient_ll_j(X,Y,j));
            return np.array(result)

        for i in range(self.max_iter):
            new_theta = self.theta + self.step_size * gradient(x, y)
#            print(self.theta)
            delta = new_theta - self.theta
            if delta.T.dot(delta) <= self.eps:
                break;
            self.theta = new_theta

        print("iteration:{}".format(i))
        print("theta:")
        print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        def h(x_i):
            return math.exp(self.theta.T.dot(x_i))
        return np.array([h(x_i) for x_i in x])
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
