import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from constants import *
from hand import HandData

DEBUG = False

def dprint(stuff, print_full_np = False):
    if print_full_np:
        np.set_printoptions(threshold=sys.maxsize)

    if DEBUG:
        print(stuff)

    if print_full_np:
        np.set_printoptions(None)

def load_hand_data(pq_path):
    dprint(pq_path)
    data = pd.read_parquet(pq_path, columns=ALL_DATA_COLUMNS).replace(np.nan, 0)
#    print(data['type'].unique())
    dprint(data, True)
    # remove all landmarks that are not hand related
    data = data[data['type'].isin(['left_hand', 'right_hand'])]
#    data = data[data['frame'] == 20]
    dprint("len of data:{}".format(len(data)))
    dprint("data:\n{}".format(data));
    return data

def parse_to_data(landmark):
    data=landmark
    n_frames = int(len(landmark) / ROWS_FOR_HAND)
    data = data.values.reshape(n_frames, 2,21, len(ALL_DATA_COLUMNS))
    dprint(data, True)
    hands= []
    for hand in data:
        left = hand[0]
        right = hand[1]
        hands.append(HandData(left, right))
    return hands

def show(hand_data):
  def show_1_hand(hand, marker = 'b.'):
      def pick(hand, indices):
          result = []
          for i in indices:
              result.append(hand[i])
          return result
      thumb = pick(hand, [0, 1, 2, 3,4])
      index_finger = pick(hand, [0, 5, 6, 7, 8])
      middle_finger = pick(hand, [9, 10, 11, 12])
      ring_finger = pick(hand, [13, 14,15,16])
      little_finger = pick(hand, [0, 17, 18, 19, 20])
      arc = pick(hand, [5,9,13,17])
      for parts in [thumb, index_finger,
                    middle_finger, ring_finger,
                    little_finger, arc]:

          for i in range(len(parts)-1):
              p1 = parts[i]
              p2 = parts[i+1]
              x_y = np.array([p1,p2]).T
              plt.plot(x_y[0], x_y[1], marker, linestyle='-', linewidth=1)

  show_1_hand(hand_data.left_hand, 'b.')
  show_1_hand(hand_data.right_hand, 'r.')

def show_hands(hands_data, path= 'temp.png'):
    figure = plt.figure(figsize=(8, 8))
    i = 1
    cols, rows = 3, len(hands_data) // 3 + 1
    for hand in hands_data:
        figure.add_subplot(rows, cols, i)
        show(hand)
        i+= 1
    plt.savefig(path)
#    plt.show()
    plt.close()

def load_csv(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    print(csv_path)
    csvData = pd.read_csv(csv_path)
#    print(csvData)
    return csvData;

import json

if __name__ == '__main__':
    train = load_csv(DATA_PATH.joinpath("train.csv"))\
        .sort_values(by=['sign'])
#    print(train)

    f = open(INDEX_MAP_DATA_PATH)
    sign_json = json.load(f)
    sign_to_frame_num = {}
    for sign in sign_json:
        break;
#        sign = 'bird'
        sign_df = train.loc[train['sign'] == sign]
        print(sign_df)
        frames= []
        for index, row in sign_df.iterrows():
            pq_path = row['path']
            frame_pd = pd.read_parquet(DATA_PATH.joinpath(pq_path), columns=ALL_DATA_COLUMNS)
            frames.append(len(frame_pd))
        print(frames)
        sign_to_frame_num[sign] = frames
    print(sign_to_frame_num)


#    landmark = load_hand_data(str(DATA_PATH.joinpath("train_landmark_files/25571/1000210073.parquet"))) # sign for "bird"
    #landmark = load_hand_data("../data/train_landmark_files/25571/1000210073.parquet") # sign for "bird"
#    landmark = load_hand_data("../data/train_landmark_files/28656/1000106739.parquet") # sign for "wait"
#    landmark = load_hand_data("../data/train_landmark_files/26734/1000035562.parquet")  # sign language for "blow"
#    hands = parse_to_data(landmark)

    def show_tv(path, name):
        print(">>>> {}".format(name))
        frame_pd = pd.read_parquet(DATA_PATH.joinpath(path), columns=ALL_DATA_COLUMNS)
        print("frame size:{}".format(len(frame_pd)))
        landmark = load_hand_data(str(DATA_PATH.joinpath(path))) # sign for "TV"
        print(landmark)
        hands = parse_to_data(landmark)
        print("hands size:{}".format(len(hands)))
        show_hands(hands, "{}.png".format(name))


    for p, n in [('train_landmark_files/26734/1000035562.parquet', 'blow')]:
        print()
        show_tv(p,n)
