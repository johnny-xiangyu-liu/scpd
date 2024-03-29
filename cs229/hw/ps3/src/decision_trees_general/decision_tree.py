import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fprint(name, val, should_print = None):
    if should_print is None:
        DEBUG = False
        should_print = DEBUG
    if not should_print:
        return
    print("{}:{}".format(name, val))

def nprint(node, idx = 0, name="root"):
    fprint(">>> node ", str(idx) + name)
    fprint("pred_class", node.predicted_class)
    fprint("featuer_index", node.feature_index)
    fprint("threshold", node.threshold)
    if node.is_leaf_node():
        return;
    nprint(node.left, idx +1, name + "_left")
    nprint(node.right, idx + 1, name + "_right")
    if name == "root":
        fprint("==== done===", "")


class DecisionTree:
    def __init__(self, max_depth=None):
        # Initialize the Decision Tree with an optional max depth parameter.
        self.max_depth = max_depth

    def fit(self, X, y):
        # Fit the decision tree model to the training data.
        self.n_classes_ = len(np.unique(y))  # Calculate the number of unique classes.
        self.n_features_ = X.shape[1]  # Number of features.
        self.tree_ = self._grow_tree(X, y)  # Build the decision tree. Return the root node.

    def predict(self, X):
        # Predict class labels for samples in X.
        return [self._predict(inputs) for inputs in X]

    def _misclassification_loss(self, y):
        # TODO: Implement the misclassification loss function.
        loss = None
        # *** START CODE HERE ***
        votes = {}
        for v in y:
            if v not in votes:
                votes[v] = 1;
            else:
                votes[v] = votes[v] + 1
        max_k, max_vote = None, None
        fprint("votes", votes)
        vote_sum = 0
        for k, v in votes.items():
            vote_sum += v
            if max_vote is None or v > max_vote:
                max_vote = v
                max_k = k

        gini_loss = 0;
        for k, v in votes.items():
            pmk = v / vote_sum
            gini_loss += pmk * (1- pmk)
#        miss_vote_count = 0;
#        for k, v in votes.items():
#            if k != max_k:
#                miss_vote_count += v;

#        loss = miss_vote_count / len(y)
        loss = 1- max_vote / vote_sum
        fprint("loss", loss)
        fprint("gini loss", gini_loss)
        # *** END YOUR CODE ***
        # Return the misclassification loss.
#        return gini_loss
        return loss

    def _best_split(self, X, y):
        # TODO: Find the best split for a node.
        # Hint: Iterate through all features and calculate the best split threshold based on the misclassification loss.
        # Hint: You might want to loop through all the unique values in the feature to find the best threshold.
        best_idx, best_thr = None, None
        # *** START CODE HERE ***
        # Calculate the parent's loss to compare with splits

        combined = np.concatenate((X, np.array([y]).T), axis=1)
#        print("combined:{}".format(combined))

        # Return threshold and loss
        def split(index, combined):
            fprint("index", index)
            sorted_arr = combined[combined[:, index].argsort()]
            y_hat = combined[:, -1]
            feature_range, indices = np.unique(sorted_arr[:, index], return_index=True)
            fprint("sorted", sorted_arr)
            fprint("feature_range", feature_range)
            fprint("indeices", indices)
            # all features are the same value
            if len(feature_range) == 1:
                return -1, self._misclassification_loss(y_hat)
            min_loss = None
            min_thr = None
            for i in range(1, len(feature_range)):
                midpoint = (feature_range[i-1] + feature_range[i]) / 2
                split_idx = indices[i]
                newarr = np.vsplit(sorted_arr, np.array([split_idx]))
                loss = self._misclassification_loss(newarr[0][:, -1]) \
                    +  self._misclassification_loss(newarr[1][:, -1])
                fprint("split at", feature_range[i])
                fprint("loss", loss)
                if min_loss is None or min_loss > loss:
                    min_loss = loss
                    min_thr = midpoint
            fprint("min thr", min_thr)
            fprint("min loss", min_loss)
            return min_thr, min_loss


        best_loss = None
        fprint("x shape:", X.shape)
        for i in range(X.shape[1]):
            thr, loss = split(i, combined)
            fprint("i", i)
            fprint("thr", thr)
            fprint("loss", loss)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_thr = thr
                best_idx = i
        # *** END YOUR CODE ***
        # Return the best split with the feature index and threshold.
        fprint(">>> best split with loss{}", best_loss)
        fprint(best_idx, best_thr)
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        # Build a decision tree by recursively finding the best split.
        # param depth is the current depth of the tree.
        # Construct the root node.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        root_node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            # TODO: Find the best split using _best_split and grow the tree recursively.
            # *** START YOUR CODE ***
            if self._misclassification_loss(y) == 0:
                return root_node

            fprint("X", X)
            fprint("y", y)
            best_idx, best_thr = self._best_split(X,y)
            root_node.feature_index = best_idx
            root_node.threshold = best_thr
            fprint("best_idx:", best_idx)
            if best_idx == len(y) -1 or best_idx == -1:
                return root_node

            lx, ly, rx, ry = [],[],[],[]
            for xi, yi in zip(X, y):

                if xi[best_idx] <= best_thr:
                    lx.append(xi)
                    ly.append(yi)
                else:
                    rx.append(xi)
                    ry.append(yi)

            lx = np.array(lx)
            ly = np.array(ly)
            rx = np.array(rx)
            ry = np.array(ry)
            fprint("lx:", lx)
            fprint("ly:", ly)
            fprint("rx:", rx)
            fprint("ry:", ry)
            left_node = self._grow_tree(lx, ly, depth+1)
            right_node = self._grow_tree(rx, ry, depth+1)

            root_node.left = left_node
            root_node.right = right_node
            # *** END YOUR CODE ***
        # Return the root node.
        return root_node

    def _predict(self, inputs):
        # Predict the class of ONE input based on the tree structure.
        fprint("predict inputs", inputs, True)
        node = self.tree_
        nprint(node)
        while node:
            # TODO: Traverse the tree to find the corresponding node and predict the class of the input.
            # Hint: iteratively update the node to be its left or right child until a leaf node is reached.
            # *** START YOUR CODE ***
            if node.is_leaf_node():
                break;
            fprint("feature index:", node.feature_index)
            feature_val = inputs[node.feature_index]
            fprint("feature_val", feature_val)
            fprint("node thre", node.threshold)
            if feature_val < node.threshold:
                node = node.left
            else:
                node = node.right

            # *** END YOUR CODE ***
        return node.predicted_class

class Node:
    def __init__(self, *, predicted_class):
        self.predicted_class = predicted_class  # Class predicted by this node
        self.feature_index = None  # Index of the feature used for splitting
        self.threshold = None  # Threshold value for splitting
        self.left = None  # Left child
        self.right = None  # Right child

    def is_leaf_node(self):
        # Check if this node is a leaf node.
        return self.left is None and self.right is None


if __name__ == "__main__":
    data = pd.DataFrame({
        "Age": [24, 53, 23, 25, 32, 52, 22, 43, 52, 48],
        "Salary": [40, 52, 25, 77, 48, 110, 38, 44, 27, 65],
        "College": [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    })
    # Create a simple dataset for testing the decision tree
    X = np.array(data[["Age", "Salary"]])
    y = np.array(data["College"])
    if False:
        #plot the data
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(len(X)):
            x = X[i]
            if y[i] == 1:
                plt.plot(x[0], x[1], 'bx', linewidth=2)
            else:
                plt.plot(x[0], x[1], 'go', linewidth=2)
            #        plt.plot(x1[y == 0, -2], x2[y == 0, -1], 'go', linewidth=2)
        plt.savefig("college.png")

    # Initialize and fit the decision tree
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    # fprint the classification accuracy
    print(f"Accuracy for college degree dataset: {round(np.mean(clf.predict(X) == y)*100, 2)}%")

    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    print("Multi-class labels", np.unique(y))

    # Split the data into training and testing sets
    # DO NOT MODIFY THE RANDOM STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=229)

    # Train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = tree.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy for iris dataset: {round(accuracy*100, 2)}%")
