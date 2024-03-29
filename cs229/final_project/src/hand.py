import numpy as np
from constants import *

ROTATE_ALONG_X_AXIS=np.array([[1, 0],
                              [0, 1]])
ROTATE_ALONG_X_AXIS=np.array([[1, 0 ],
                              [0, -1]])
class HandData:
    """Hand dataclass
    """

    def __init__(self, left, right):
        """
        Args:
            left: landmark data for left hand (21 rows, len(ALL_DATA_COLUMNS) col)
            right: same as above for right hand
        """
        frame_index = ALL_DATA_COLUMNS.index('frame')
        self.frame = left[0][frame_index]

        x_index = ALL_DATA_COLUMNS.index('x')
        y_index = ALL_DATA_COLUMNS.index('y')


        def get_hand_x_y(landmark):
            return np.array([np.array(
                ROTATE_ALONG_X_AXIS.dot([l[x_index],l[y_index]])) for l in landmark], dtype=np.float64)

        self.left_hand = get_hand_x_y(left)
        self.right_hand = get_hand_x_y(right)

#        print(self.right_hand)


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


class DataItem:
    def __init__(self, hands, label):
        """
        Args:
            hands: a vector of hands representing in each frame of the landmark file
            label: the sign.
        """
        self.hands = hands
        self.label = label
