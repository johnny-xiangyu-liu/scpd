import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # Fit a LWR model with the best tau value

    def mse(x_valid, y_valid, clf):
        sum = 0
        for i in range(0, len(x_valid)):
            sum += (y_valid[i] - clf.predict(x_valid[i]))**2
        return sum / len(x_valid);
    best_mse = None;
    best_clf = None;
    best_tau = None;
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        new_mse = mse(x_valid, y_valid, clf)
        if not best_mse :
            best_mse = new_mse
            best_clf = clf
            best_tau = tau
        elif best_mse > new_mse:
            best_mse = new_mse
            best_clf = clf
            best_tau = tau
    f = open(pred_path, "w")
    f.write("best tau:{}\n".format(best_tau))
    f.write("test mse:{}\n".format(mse(x_test, y_test, best_clf)))
    f.close()
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    def plot_scatter(X, Y, color):
        x_in = [X[i][1] for i in range(len(X))]
        y_in = Y
        plt.scatter(x_in, y_in, color=color)

    def plot(tau, x_train, y_train, x_valid):
      plt.figure()
      plot_scatter(x_train, y_train, 'blue')
      clf = LocallyWeightedLinearRegression(tau)
      clf.fit(x_train, y_train)
      y_prediction = [clf.predict(x) for x in x_valid]
      plot_scatter(x_valid, y_prediction, 'red')
      plt.savefig("LWLR_prediction_for_tau_{}.png".format(tau))

    for tau in tau_values:
        plot(tau, x_train, y_train, x_valid);
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
