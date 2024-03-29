import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    def mse(x_valid, y_valid, clf):
        sum = 0
        for i in range(0, len(x_valid)):
            sum += (y_valid[i] - clf.predict(x_valid[i]))**2
        return sum / len(x_valid);

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    print("mse", mse(x_valid, y_valid, clf))
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt.figure()
    def plot(X, Y, color):
        x_in = [X[i][1] for i in range(len(X))]
        y_in = Y
        plt.scatter(x_in, y_in, color=color)

    plot(x_train, y_train, 'blue')
    y_prediction = [clf.predict(x) for x in x_valid]
    plot(x_valid, y_prediction, 'red')
    plt.savefig('LWLR_prediction.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """
        # *** START CODE HERE ***
        self.x = x;
        self.y = y;
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def w_i(x_i, x):
            delta = x_i - x
            return np.exp(-delta.T.dot(delta)/(2*self.tau**2))

        W = np.zeros((len(self.x), len(self.x)), dtype = self.x.dtype)
        np.fill_diagonal(W, [w_i(x_i, x) for x_i in self.x])
        theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
        return x.dot(theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
