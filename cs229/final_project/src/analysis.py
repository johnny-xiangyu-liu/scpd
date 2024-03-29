#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json


# In[ ]:


ROWS_PER_FRAME = 543  # number of landmarks per frame
ROWS_PER_HAND = 21 # 21 landmarks per hand
ROWS_FOR_HAND = ROWS_PER_HAND * 2 # 2 hands
ALL_DATA_COLUMNS = ['frame', 'row_id', 'type', 'landmark_index','x', 'y',]  # for simplicity, we ignore depth dimension 'z']

#ROOT_DIR=Path(".")
ROOT_DIR=Path(__file__).parent.parent
DATA_PATH=ROOT_DIR.joinpath('data')
INDEX_MAP_DATA_TRIMMED_SIZE = 10 # training on 250 signs takes too long
INDEX_MAP_DATA_PATH=DATA_PATH.joinpath('sign_to_prediction_index_map.json')
TRAINING_DATA=DATA_PATH.joinpath("train.csv")

OUTPUT_PATH=ROOT_DIR.joinpath("output")
PROCESSED_DATA=OUTPUT_PATH.joinpath("processed_train.csv")
MODEL_OUTPUT_PATH=OUTPUT_PATH.joinpath('model')

def DEBUG():
    return os.environ.get("DEBUG") == "True";

def dprint(stuff, print_full_np = False):
    if not DEBUG():
        return
    if print_full_np:
        np.set_printoptions(threshold=sys.maxsize)

    print(stuff)

    if print_full_np:
        np.set_printoptions(None)

def fprint(*args):
    if DEBUG():
        print(*args)

dprint(INDEX_MAP_DATA_PATH)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# hand.py
import numpy as np

class HandData:
    """Hand dataclass
    """

    def __init__(self, left, right):
        """
        Args:
            left: landmark data for left hand (21 rows, len(ALL_DATA_COLUMNS) col)
            right: same as above for right hand
        """
        self.left_hand = left
        self.right_hand = right
        self.both_hands = np.concatenate((self.left_hand, self.right_hand), axis = 0)
        self.raw = np.append(self.left_hand.flatten(), self.right_hand.flatten())


    def to_np(self):
        return np.concatenate((self.left_hand.flatten(), self.right_hand.flatten()), axis = 0)
        return  self.raw


# In[ ]:


#util.py
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

ROTATE_ALONG_X_AXIS=np.array([[1, 0 ],
                              [0, -1]])
def load_hand_data(pq_path):
    dprint(pq_path)
    data = pd.read_parquet(pq_path, columns=ALL_DATA_COLUMNS).replace(np.nan, 0)
#    dprint(data)
    # remove all landmarks that are not hand related
    data = data[data['type'].isin(['left_hand', 'right_hand'])]
#    data = data[data['frame'] == 20]
    dprint("len of data:{}".format(len(data)))
#    dprint("data:\n{}".format(data));
    return data

def parse_to_data(landmark):
    data=landmark
    n_frames = int(len(landmark) / ROWS_FOR_HAND)
    data = data.values.reshape(n_frames, 2,21, len(ALL_DATA_COLUMNS))
#    dprint(data, True)
    hands= []
    for hand in data:
        left = hand[0]
        right = hand[1]
#        print("left:", left)

        x_index = ALL_DATA_COLUMNS.index('x')
        y_index = ALL_DATA_COLUMNS.index('y')

        def get_hand_x_y(landmark):
            return np.array([np.array(
                ROTATE_ALONG_X_AXIS.dot([l[x_index],l[y_index]])) for l in landmark], dtype=np.float32)
        left_hand = get_hand_x_y(left)
        right_hand = get_hand_x_y(right)
        if (np.sum(left_hand) == 0 and np.sum(right_hand) == 0):
            continue;
        hands.append(HandData(left_hand, right_hand))
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
    plt.savefig(OUTPUT_PATH.joinpath(path))
    plt.show()
    plt.close()


# split training data into training and validation set
def split(pd):
    tail = len(pd) // 5 # 20%
    head = len(pd) - tail
    return pd.head(head), pd.tail(tail)

def append_landmark_size(data):
    dprint("data:\n{}".format(data));
    landmark_sizes = [0 for i in range(len(data))]
    for index, row in data.iterrows():
        pq_path = row['path']
        landmark_pd = pd.read_parquet(DATA_PATH.joinpath(pq_path), columns=['frame'])
        landmark_sizes[index]= len(landmark_pd)
    print(landmark_sizes)
    df2 = data.assign(landmark_size=landmark_sizes)
    df2.to_csv(PROCESSED_DATA, index=False)


def load_trimmed_signs():
    f = open(INDEX_MAP_DATA_PATH)
    sign_json = json.load(f)

    trimmed_sign = {}
    i = 0
    for k in sign_json:
        i+=1
        if i > INDEX_MAP_DATA_TRIMMED_SIZE:
            break;
        trimmed_sign[k] = sign_json[k]

    print("trimmed sign size:", len(trimmed_sign))
    return trimmed_sign

def load_csv(csv_path, signs = load_trimmed_signs()):
    """Load dataset from a CSV file.
    """

    # Load headers
    dprint(csv_path)
    csvData = pd.read_csv(csv_path)
    print("data size:", len(csvData))
    print(signs.keys())
    csvData = csvData.loc[csvData['sign'].isin(signs.keys())]
    print("data size after trimming:", len(csvData))

    dprint(csvData)
    return csvData;


# append_landmark_size(load_csv(TRAINING_DATA))



# In[ ]:


print(load_csv(TRAINING_DATA))
#print(load_csv(PROCESSED_DATA))
train = load_csv(TRAINING_DATA)
#landmark = load_hand_data(str(DATA_PATH.joinpath("train_landmark_files/25571/1000210073.parquet"))) # sign for "bird"
#    landmark = load_hand_data(str(DATA_PATH.joinpath("train_landmark_files/28656/1000106739.parquet"))) # sign for "wait"
landmark = load_hand_data(str(DATA_PATH.joinpath("train_landmark_files/26734/1000035562.parquet")))  # sign language for "blow"
hands = parse_to_data(landmark)
# show_hands(hands)

#print(load_sign())

def showFrameNum(csv, sign, pick_same_person = False):
    data = csv.loc[csv['sign'].isin([sign])]
    file_name = sign +"_frame_count.png"
    title = "'{}' signed by different people".format(sign)
    if pick_same_person:
        participants = data.participant_id.unique()
        # pick the first one
        p_id = participants[0]
        data = data.loc[data['participant_id'].isin([p_id])]

        title = "'{}' signed by the same person".format(sign)
        file_name = sign +"_by_"+str(p_id) + "_frame_count.png"

    print(data)
    frame_sizes = [0 for i in range(len(data))]
    i = 0
    for index, row in data.iterrows():
        pq_path = row['path']
        landmark_pd = pd.read_parquet(DATA_PATH.joinpath(pq_path),)# columns=['frame'])
        frames = landmark_pd.frame.unique()
#        print(frames)
#        print(len(frames))
        frame_sizes[i]= len(frames)
        i +=1
    figure = plt.figure()
    # creating the bar plot
    plt.bar([i for i in range(len(frame_sizes))], frame_sizes, color ='maroon',
            width = 0.4)

    plt.xlabel("Video id")
    plt.ylabel("No. of frames per video")
    plt.title(title)
    plt.savefig(OUTPUT_PATH.joinpath(file_name))
    plt.show()



#showFrameNum(train, 'TV')
#showFrameNum(train, 'TV', True)


def plotLR(stats_name, const_lr_name):
    print(stats_name)
    stats = pd.read_csv(OUTPUT_PATH.joinpath(stats_name +"_training_summary.csv"))[:100]
    const_lr_stats = pd.read_csv(OUTPUT_PATH.joinpath(const_lr_name +"_training_summary.csv"))[:100]

    fig, ax = plt.subplots()
    stats[["training_losses"]].plot(xlabel='epoch', ylabel='Training loss', ax =ax)
    const_lr_stats[["training_losses"]].plot(ax = ax)
    second_ax = stats[["training lr"]].plot(secondary_y=True, style='g:', ax = ax, ylabel='lr')
    const_lr_stats[["training lr"]].plot(secondary_y=True, style='b:', ax = ax)
    ax.legend(["StepLR Training Loss", "Const LR Traninng Loss", "Step LR", "Const LR"])
    second_ax.legend(["Step LR", "Const LR"])
#    plt.show()
    plt.savefig(OUTPUT_PATH.joinpath("learning_rate_comparison"))
    plt.close()
    return
    lr  = stats['training lr']

    train_loss = stats['training_losses']
    train_accuracy = stats['training_accuracies (in %)']
    test_loss = stats['test_losses']
    test_accuracy = stats['test_accuracies (in %)']

    print(lr)
    print(train_loss)
    print(train_accuracy)
    print(test_loss)
    print(test_accuracy)


plotLR("lstm", "lstm_const_lr")

def plotTraining(stats_name):
    print(stats_name)
    stats = pd.read_csv(OUTPUT_PATH.joinpath(stats_name +"_training_summary.csv"))

    fig, ax = plt.subplots()
    stats[["training_losses"]].plot(xlabel='epoch', ylabel='Training loss', ax =ax)
    stats[["test_losses"]].plot(ax = ax)
    second_ax = stats[["training_accuracies (in %)"]].plot(secondary_y=True, style='g:', ax = ax, ylabel='Accuracies in %')
    stats[["test_accuracies (in %)"]].plot(secondary_y=True, style='b:', ax = ax)

    ax.legend(["Train Loss", "Test Loss"])
    second_ax.legend(["Train Accuracy", "Test Accuracy"])
#    plt.show()
    plt.savefig(OUTPUT_PATH.joinpath("training_test_loss"))
    plt.close()
    return

plotTraining("lstm_batch_10")
