import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)

    def train(X, Y, out_path):
        clf = LogisticRegression()
        clf.l2_reg = False
        clf.fit(X, Y)
        y_predict = clf.predict(x_valid)
        print(out_path)
        np.savetxt(out_path, y_predict)

        tp=0
        tn=0
        fp=0
        fn= 0
        for i in range(len(y_valid)):
            predict = y_predict[i] > 0.5
            valid = y_valid[i] == 1
            if predict is valid:
                if predict:
                    tp += 1
                else:
                    tn += 1
            else:
                if predict:
                    fp += 1
                else:
                    fn += 1

        rho = (tp + tn) / len(y_valid)
        A1 = tp / (tp + fn)
        A0 = tn / (tn + fp)
        A_balance = (A0 + A1) /2
        A = (tp + tn) / len(y_valid)

        print("A:{}".format(A));
        print("A_balance:{}".format(A_balance));
        print("A0:{}".format(A0));
        print("A1:{}".format(A1));
        util.plot(x_valid, y_valid, clf.theta, out_path + ".png")

        return clf
    clf = train(x_train, y_train, output_path_naive);
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_train_sampled = []
    y_train_sampled = []
    for i in range(len( x_train)):
        size = 1
        if y_train[i] == 1:
           size = int(1/kappa);
        for j in range(size):
            x_train_sampled.append(x_train[i])
            y_train_sampled.append(y_train[i])
#    print(x_train_sampled)
    clf = train(np.array(x_train_sampled), \
                np.array(y_train_sampled), \
                output_path_upsampling)

    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
