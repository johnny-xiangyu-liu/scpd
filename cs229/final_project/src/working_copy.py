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
#         frame_index = ALL_DATA_COLUMNS.index('frame')
#         self.frame = left[0][frame_index]

#         x_index = ALL_DATA_COLUMNS.index('x')
#         y_index = ALL_DATA_COLUMNS.index('y')

#         def get_hand_x_y(landmark):
#             return np.array([np.array(
#                 ROTATE_ALONG_X_AXIS.dot([l[x_index],l[y_index]])) for l in landmark], dtype=np.float32)
#        self.left_hand = get_hand_x_y(left)
#        self.right_hand = get_hand_x_y(right)
        self.left_hand = left
        self.right_hand = right
        self.both_hands = np.concatenate((self.left_hand, self.right_hand), axis = 0)
        self.raw = np.append(self.left_hand.flatten(), self.right_hand.flatten())
        # print("left_hand:{}".format(self.left_hand.shape))
        # print("right_hand:{}".format(self.right_hand.shape))
        # print("hand_shape:{}".format(self.both_hands.shape))
        # print("hand:{}".format(self.both_hands))

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


# In[ ]:


# custom dataset for py torch
import torch
from torch.utils.data import Dataset
import json

# Takes in the panda file from train.csv and load data
class HandDataSet(torch.utils.data.Dataset):
    def __init__(self, dataframe=load_csv(TRAINING_DATA)):
        self.pd = dataframe
        self.sign_json = load_trimmed_signs()

    def __len__(self):
        return len(self.pd)

    def __getitem__(self, idx):
        try:
            row = self.pd.iloc[[idx]]
        except Exception:
          print("Error when loading index:{}, pd size:{}".format(idx, len(self.pd)))
          raise
#        dprint("index:{}, row:{}".format(idx, row))
        landmark_path = row['path'].values[0]
        sign_str = row['sign'].values[0]
        sign = self.sign_json[sign_str]
        landmark = load_hand_data(str(DATA_PATH.joinpath(landmark_path)))
        hands = parse_to_data(landmark)
        return torch.tensor(np.array([hand.to_np() for hand in hands])), sign

# ds = HandDataSet()
# t = ds.__getitem__(2)
# print("t shape:{}".format(t.shape))
#print("frames:{}".format(frames))


# In[ ]:


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# In[ ]:


# load data
from torch.utils.data import DataLoader

# pad each item in the batch so that they have the same
# length of frames
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    '''
    batch_size = len(batch)
    hands, y = batch[0]
    max_len = hands.shape[0]
    for hands, y in batch:
        max_len = max(max_len, hands.shape[0])

    padded_batch = []
    def pad_hands(hands, max_len):
        return torch.cat((hands,
                         torch.zeros((max_len - hands.shape[0],
                                      ROWS_FOR_HAND * 2))))
    for hands, y in batch:
        padded = pad_hands(hands, max_len)
        padded_batch.append((padded, y))

    return padded_batch


# In[ ]:


# Define model
class BaseNeuralNetwork(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name =  self.__class__.__name__

    def name():
        return self.name


# In[ ]:


# Define model
# Referenced from https://gist.github.com/Dvelezs94/dc34d1947ba6d3eb77c0d70328bfe03f
class RNN(BaseNeuralNetwork):
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # left hand array followed by right hand
        self.input_size = ROWS_FOR_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_layer_output_sizes = [300] # for now just 1 hidder layer with 300 internal nodes
        self.hidden_size = 300

        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=False, dtype=torch.float32)


    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape

        for i in range(frame_count):
            X = hand_data[i]
            X = self.i2h(X)
            fprint("x shape", X.shape)
            fprint("hidden_state shape", hidden_state.shape)
            hidden_state = self.h2h(hidden_state)
            hidden_state = torch.tanh(X + hidden_state)
            out = self.h2o(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )

class RNN2(BaseNeuralNetwork):
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # left hand array followed by right hand
        self.input_size = ROWS_FOR_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_layer_output_sizes = [300] # for now just 1 hidder layer with 300 internal nodes
        self.hidden_size = 300

        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2o = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            nn.Linear(self.hidden_size, self.output_size, bias=False, dtype=torch.float32),
            nn.Tanh()
        )



    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape

        for i in range(frame_count):
            X = hand_data[i]
            X = self.i2h(X)
            fprint("x shape", X.shape)
            fprint("hidden_state shape", hidden_state.shape)
            hidden_state = self.h2h(hidden_state)
            hidden_state = torch.tanh(X + hidden_state)
            out = self.h2o(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


# In[ ]:


# Representing 1 hand only.
class SingleHandRNN(BaseNeuralNetwork):
    def __init__(self, name, output=400):
        super().__init__()
        self.input_size = ROWS_PER_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_size = 300
        self.name = name
        #Defining the layers
        # RNN Layer
        # first layer:
        self.i2h = nn.Linear(self.input_size,
                             self.hidden_size, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size,
                            dtype=torch.float32)
        self.activation = nn.Tanh()
        self.layer2 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.Linear(self.hidden_size, self.output_size, dtype=torch.float32))


    def forward(self, X, hidden_state = None):
#        fprint(self.name, X.shape)
#        fprint(self.name + "_hidden", hidden_state.shape)
        batch_size = X.size(0)
        # Passing in the input and hidden state into the model and obtaining outputs
        hidden_state = self.i2h(X) + self.h2h(hidden_state)
        hidden_state = self.activation(hidden_state)

#        out = self.layer2(out)
        out = self.layer3(hidden_state)

        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


# In[ ]:


class ASLRNN(BaseNeuralNetwork):
    # output INDEX_MAP_DATA_TRIMMED_SIZE as the sign map to predict.
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # 2 hands. each have `ROWS_PER_HAND`, each row has x, y coordinates
        self.input_size = (2, ROWS_PER_HAND, 2)
        self.output_size = output

        # Defining the layers
        # Hands
        self.left_hand = SingleHandRNN("left")
        self.right_hand = SingleHandRNN("right")

        self.layer1 = nn.Linear(self.left_hand.output_size + self.right_hand.output_size,
                               self.output_size,
                               dtype=torch.float32)

    def forward(self, hand_data, hidden_state = None):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape
        fprint("hand_data", hand_data.shape)
        split = torch.split(hand_data, row_for_hand // 2,
                            # dim should be row_for_hand
                            dim = 2)
#        fprint("split 0:", split[0].shape)
#        fprint("split 1:", split[1].shape)
        # left:torch.Size([105, 6, 42])
        left_split = split[0]
        # right:torch.Size([105, 6, 42])
        right_split = split[1]

        if hidden_state is None:
            left_hidden = None;
            right_hidden = None;
        else:
            hidden_split = torch.split(hidden_state,
                                       hidden_state.shape[0]//2, dim = 0)

            left_hidden = hidden_split[0]
            right_hidden = hidden_split[1]

#        fprint("left_hidden:", left_hidden.shape)
#       fprint("right_hidden:", right_hidden.shape)
        for i in range(frame_count):
            left = left_split[i]
            right = right_split[i]
            # Passing in the input and hidden state into the model and obtaining outputs
            left_out, left_hidden = self.left_hand(left, left_hidden)
            right_out, right_hidden = self.right_hand(right, right_hidden)

        fprint("left_out:", left_out.shape)
        fprint("left_hidden:", left_hidden.shape)
        fprint("right_out:", right_out.shape)
        fprint("right_hidden:", right_hidden.shape)
        out = torch.cat([left_out, right_out], dim = 1)
        fprint("out:", out.shape)
        hidden_state = torch.cat([left_hidden, right_hidden], dim = 1)
        fprint("hidden_state:", hidden_state.shape)
        out = self.layer1(out)

        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.cat([self.left_hand.init_zero_hidden(batch_size),
                          self.right_hand.init_zero_hidden(batch_size)])

#asl_model = ASLRNN().to(device)
#print(asl_model)

# treating all landmarks the same, instead of splitting to hands
class JointRNN(BaseNeuralNetwork):
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # left hand array followed by right hand
        self.input_size = ROWS_FOR_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_layer_output_sizes = [300] # for now just 1 hidder layer with 300 internal nodes
        self.hidden_size = 300

        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float32)

        self.activation = nn.Tanh()
        self.layer2 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)
        )
        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.LazyLinear( self.output_size, dtype=torch.float32))


    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape

        for i in range(frame_count):
            X = hand_data[i]
            X = self.i2h(X)
            fprint("x shape", X.shape)
            fprint("hidden_state shape", hidden_state.shape)
            hidden_state = self.h2h(hidden_state)
            hidden_state = self.activation(X + hidden_state)
            wrapped_hidden = hidden_state.unsqueeze(1)
            fprint("wrapped_hidden,", wrapped_hidden.shape)
            out = self.layer2(wrapped_hidden)
            fprint("layer2:", out.shape)
            out = self.layer3(out)
            out = out.squeeze(1)
            fprint("out", out.shape)
            fprint("hidden", hidden_state.shape)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


# In[ ]:

# Representing 1 hand only.
class SingleHandRNN2(BaseNeuralNetwork):
    def __init__(self, name, output_feature_size=500):
        super().__init__()
        self.output_feature_size = output_feature_size
        self.hidden_size = self.output_feature_size
        self.name = name
        #Defining the layers
        # convelution layers to conver the input hands
        self.conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.layer1 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            self.conv, self.relu, self.max_pool,
            nn.Dropout(p=0.2)
        )
        self.layer2 = nn.LazyLinear(self.hidden_size, dtype=torch.float32)

        self.h2h = nn.Linear(self.hidden_size, self.hidden_size,
                            dtype=torch.float32)
        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.Linear(self.hidden_size, self.output_feature_size, dtype=torch.float32),
            nn.ReLU())


    def forward(self, X, hidden_state):
        fprint(self.name, X.shape)
        fprint(self.name + "_hidden", hidden_state.shape)
        # X shape should be (batch* 2 * hand_landmarks) # s the x, y cordinates
        batch_size = X.size(0)
        # Passing in the input and hidden state into the model and obtaining outputs
        X = self.layer1(X)
        fprint(self.name, "after layer1:", X.shape)

        X = X.view(batch_size, -1)
        fprint(self.name, "full connected:", X.shape)
        hidden = self.layer2(X)
        fprint(self.name, " after layer 2:", hidden.shape)
        fprint(self.name, "hidden_state:", hidden_state.shape)
        hidden_state = hidden + self.h2h(hidden_state)
        out = self.layer3(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


# In[ ]:


class ASLRNN2(BaseNeuralNetwork):
    # output INDEX_MAP_DATA_TRIMMED_SIZE as the sign map to predict.
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # 2 hands. each have `ROWS_PER_HAND`, each row has x, y coordinates
        self.input_size = (2, ROWS_PER_HAND, 2)
        self.output_size = output

        # Defining the layers
        # Hands
        self.hand_feature_size = 300
        self.left_hand = SingleHandRNN2("left_hand", self.hand_feature_size)
        self.right_hand = SingleHandRNN2("right_hand", self.hand_feature_size)
        self.hidden_size = 500
        # hand output to hidden


        self.layer1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=32, \
                                              kernel_size=2, stride = 1, dtype=torch.float32),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride = 1),
                                    nn.Dropout(p=0.2))

        self.layer2 = nn.LazyLinear(
                                out_features=self.hidden_size, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size,
                             dtype=torch.float32)
        self.hidden_out = nn.ReLU()

        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.Linear(self.hidden_size, self.output_size, dtype=torch.float32),
            nn.ReLU())

    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape
        fprint("hand_data", hand_data.shape)
        split = torch.split(hand_data, row_for_hand // 2,
                            # dim should be row_for_hand
                            dim = 2)
        fprint("split 0:", split[0].shape)
        fprint("split 1:", split[1].shape)
        # left:torch.Size([105, 6, 42])
        left_split = split[0]
        # right:torch.Size([105, 6, 42])
        right_split = split[1]

        # hidden state should be a concat of [left_hidden, right_hidden, joint_hidden]
        #hidden_split = torch.split(hidden_state,hidden_state.shape[0]//3,dim = 0)

        #left_hidden = hidden_split[0]
        #right_hidden = hidden_split[1]
        #joint_hidden = hidden_split[2]
        left_hidden = hidden_state[:, 0:self.left_hand.hidden_size]
        right_hidden = hidden_state[:, self.left_hand.hidden_size: self.left_hand.hidden_size + self.right_hand.hidden_size]
        joint_hidden = hidden_state[:, self.left_hand.hidden_size + self.right_hand.hidden_size:]

        fprint("left_hidden:", left_hidden.shape)
        fprint("right_hidden:", right_hidden.shape)
        fprint("joint_hidden:", joint_hidden.shape)
        for i in range(frame_count):
            # view the (batch, landmark* 2) into (batch, landmark, 2)
            left = left_split[i].view(batch_size, -1, 2)
            right = right_split[i].view(batch_size,-1,2)
            fprint("left view:", left.shape)

            # swap the landmark with the x,y coordinate
            left = torch.swapaxes(left, 1, 2)
            fprint("left swap:", left.shape)
            right = torch.swapaxes(right, 1, 2)
            # Passing in the input and hidden state into the model and obtaining outputs

            left_out, left_hidden = self.left_hand(left, left_hidden)
            right_out, right_hidden = self.right_hand(right, right_hidden)
            left_out = torch.unsqueeze(left_out, 1)
            right_out = torch.unsqueeze(right_out, 1)
            fprint("left out:", left_out.shape)
            fprint("right out:", right_out.shape)
            both_hands = torch.cat((left_out, right_out), dim = 1)
            fprint("both:", both_hands.shape)
            conv_out = self.layer1(both_hands)
            fprint("layer1:", conv_out.shape)
            conv_out = conv_out.view(batch_size, -1)
            fprint(self.name, "conv_out:", conv_out.shape)
            i2h_out = self.layer2(conv_out)
            fprint(self.name, "i2h_out", i2h_out.shape)
            joint_hidden = self.h2h(joint_hidden)
            joint_hidden = self.hidden_out(joint_hidden)
            out = self.layer3(i2h_out + joint_hidden)



        fprint("left_out:", left_out.shape)
        fprint("left_hidden:", left_hidden.shape)
        fprint("right_out:", right_out.shape)
        fprint("right_hidden:", right_hidden.shape)

        hidden_state = torch.cat([left_hidden, right_hidden, joint_hidden], dim = 1)
        fprint("out:", out.shape, "hidden:", hidden_state.shape)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        joint_hidden =torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )
        left_hidden = self.left_hand.init_zero_hidden(batch_size)
        right_hidden = self.right_hand.init_zero_hidden(batch_size)

        result= torch.cat([left_hidden, right_hidden, joint_hidden], dim = 1)
        fprint("init hidden:", result.shape)
        return result


# Representing 1 hand only.
class SingleHandRNN3(BaseNeuralNetwork):
    def __init__(self, name, output_feature_size=500):
        super().__init__()
        self.output_feature_size = output_feature_size
        self.hidden_size = self.output_feature_size
        self.name = name
        #Defining the layers
        # convelution layers to conver the input hands
        self.conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.layer1 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            self.conv, self.relu, self.max_pool,
            nn.Dropout(p=0.2)
        )

        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.LazyLinear(self.output_feature_size, dtype=torch.float32),
            nn.ReLU())


    def forward(self, X):
        fprint(self.name, X.shape)
        # X shape should be (batch* 2 * hand_landmarks) # s the x, y cordinates
        batch_size = X.size(0)
        # Passing in the input and hidden state into the model and obtaining outputs
        X = self.layer1(X)
        fprint(self.name, "after layer1:", X.shape)

        X = X.view(batch_size, -1)
        fprint(self.name, "full connected:", X.shape)
        out = self.layer3(X)
        return out

# In[ ]:


class ASLRNN3(BaseNeuralNetwork):
    # output INDEX_MAP_DATA_TRIMMED_SIZE as the sign map to predict.
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # 2 hands. each have `ROWS_PER_HAND`, each row has x, y coordinates
        self.input_size = (2, ROWS_PER_HAND, 2)
        self.output_size = output

        # Defining the layers
        # Hands
        self.hand_feature_size = 300
        self.left_hand = SingleHandRNN3("left_hand", self.hand_feature_size)
        self.right_hand = SingleHandRNN3("right_hand", self.hand_feature_size)
        self.hidden_size = 500
        # hand output to hidden


        self.layer1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=32, \
                                              kernel_size=2, stride = 1, dtype=torch.float32),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride = 1),
                                    nn.Dropout(p=0.2))

        self.layer2 = nn.LazyLinear(
                                out_features=self.hidden_size, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size,
                             dtype=torch.float32)
        self.hidden_out = nn.ReLU()

        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.Linear(self.hidden_size, self.output_size, dtype=torch.float32),
            nn.ReLU())

    def forward(self, hand_data, hidden):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape
        fprint("hand_data", hand_data.shape)
        split = torch.split(hand_data, row_for_hand // 2,
                            # dim should be row_for_hand
                            dim = 2)
        fprint("split 0:", split[0].shape)
        fprint("split 1:", split[1].shape)
        # left:torch.Size([105, 6, 42])
        left_split = split[0]
        # right:torch.Size([105, 6, 42])
        right_split = split[1]

        for i in range(frame_count):
            # view the (batch, landmark* 2) into (batch, landmark, 2)
            left = left_split[i].view(batch_size, -1, 2)
            right = right_split[i].view(batch_size,-1,2)
            fprint("left view:", left.shape)

            # swap the landmark with the x,y coordinate
            left = torch.swapaxes(left, 1, 2)
            fprint("left swap:", left.shape)
            right = torch.swapaxes(right, 1, 2)
            # Passing in the input and hidden state into the model and obtaining outputs

            left_out = self.left_hand(left)
            right_out = self.right_hand(right)
            left_out = torch.unsqueeze(left_out, 1)
            right_out = torch.unsqueeze(right_out, 1)
            fprint("left out:", left_out.shape)
            fprint("right out:", right_out.shape)
            both_hands = torch.cat((left_out, right_out), dim = 1)
            fprint("both:", both_hands.shape)
            conv_out = self.layer1(both_hands)
            fprint("layer1:", conv_out.shape)
            conv_out = conv_out.view(batch_size, -1)
            fprint(self.name, "conv_out:", conv_out.shape)
            i2h_out = self.layer2(conv_out)
            fprint(self.name, "i2h_out", i2h_out.shape)
            hidden = self.h2h(hidden)
            out = self.layer3(i2h_out + hidden)



        fprint("left_out:", left_out.shape)
        fprint("right_out:", right_out.shape)

        fprint("out:", out.shape, "hidden:", hidden.shape)
        return out, hidden

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


class ASLRNN4(BaseNeuralNetwork):
    # output INDEX_MAP_DATA_TRIMMED_SIZE as the sign map to predict.
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # left hand array followed by right hand
        self.input_size = ROWS_FOR_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_layer_output_sizes = [300] # for now just 1 hidder layer with 300 internal nodes
        self.hidden_size = 300

        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False, dtype=torch.float32)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float32)

        self.activation = nn.Tanh()
        self.layer2 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)
        )
        self.layer3 = torch.nn.Sequential(
            # Fully connected layer
            nn.LazyLinear( self.output_size, dtype=torch.float32))


    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape

        for i in range(frame_count):
            X = hand_data[i]
            X = self.i2h(X)
            fprint("x shape", X.shape)
            fprint("hidden_state shape", hidden_state.shape)
            hidden_state = self.h2h(hidden_state)
            hidden_state = self.activation(X + hidden_state)
            wrapped_hidden = hidden_state.unsqueeze(1)
            fprint("wrapped_hidden,", wrapped_hidden.shape)
            out = self.layer2(wrapped_hidden)
            fprint("layer2:", out.shape)
            out = out.view(batch_size, -1)
            out = self.layer3(out)
            out = out.squeeze(1)
            fprint("out", out.shape)
            fprint("hidden", hidden_state.shape)
        return out, hidden_state

        self.hidden_layer_output_sizes = [300] # for now just 1 hidder layer with 300 internal nodes
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,dtype=torch.float32 )


class LSTMASL(BaseNeuralNetwork):
    # output INDEX_MAP_DATA_TRIMMED_SIZE as the sign map to predict.
    def __init__(self, output=INDEX_MAP_DATA_TRIMMED_SIZE):
        super().__init__()
        # left hand array followed by right hand
        self.input_size = ROWS_FOR_HAND * 2 # each hand point has x and y coordinate
        self.output_size = output
        self.hidden_size = 300
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, dropout = 0.2)
        self.layer2 = nn.Sequential(
            # Use kernel_size = 2 to group the (x,y) corrdinates together
            nn.ReLU(),
            nn.LazyLinear( self.output_size, dtype=torch.float32)
        )

    def forward(self, hand_data, hidden_state):
        # X:torch.Size([105, 6, 84])
        frame_count, batch_size, row_for_hand = hand_data.shape
        fprint("input:", hand_data.shape)
        output, (hn, cn) = self.lstm(hand_data)

        fprint("output", output.shape, output)
        # many to 1, so just need the last frame
        output = output[-1]
        fprint("last", output.shape, output)
        output = self.layer2(output)
        fprint("final", output.shape)
        return output, torch.zeros(1)

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        # hidden state is captured by LSTM
        return torch.zeros(1)


# train the model

# based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
import time
import math
def do_train(data, model, name, optimizer, scheduler, loss_fn, debug = None, log_interval = 100) -> None:
    """
    Trains the model for the specified number of epochs
    Inputs
    ------
    model: RNN model to train
    data: Iterable DataLoader
    epochs: Number of epochs to train the model
    optiimizer: Optimizer to use for each epoch
    loss_fn: Function to calculate loss
    """
    model.to(device)
    model.train()
    epoch_losses = list()
    print("data length:", len(data))

    correct = 0
    size = len(data.dataset)
    num_batches = len(data)
    i = 0
    interval_loss = 0.
    start_time = time.time()
    for batch in data:
        i+= 1
        if debug is not None:
            if i > debug:
                break;
        X, Y = [], []
        for item in batch:
            x, y = item
            X.append(x)
            Y.append(y)

        # this results in a shape (batch, frame count, landmarks)
        X = torch.stack(X)
        fprint("X init shape:", X.shape)

        # transform X into a shape (frame count, batch, landmarks)
        X = torch.swapaxes(X, 0, 1)
        fprint("X transformed shape:", X.shape)
        Y = torch.tensor(np.array(Y))

        hidden = model.init_zero_hidden(len(Y))
        # send tensors to device
        X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)

        dprint("X:{}".format(X.shape))
        dprint("Y:{}".format(Y.shape))
        dprint("hidden:{}".format(hidden.shape))
        # 2. clear gradients
        model.zero_grad()
        loss = 0
        out, hidden = model(X, hidden)
        l = loss_fn(out, Y)

#        sm = nn.functional.softmax(out, dim=0)

        correct += (out.argmax(1) == Y).type(torch.float).sum().item()
        loss += l

        # 4. Compte gradients gradients
        loss.backward()
        # 5. Adjust learnable parameters
        # clip as well to avoid vanishing and exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        interval_loss += loss.item()
        if i % log_interval == 0:
            print("#{}.".format(i), end =" ")
            print("out:{}".format(out.shape))
            print("lose:{}".format(l))

            cur_loss = interval_loss / log_interval
            elapsed = time.time() - start_time
            print('batch#{:5d} | lr {:0002.2f} | ms/batch {:5.2f} |  loss {:5.2f}'.format(
                i,  scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss))

            interval_loss = 0
            start_time = time.time()

        epoch_losses.append(loss.detach().item() / X.shape[1])

    print(name, " weight gradient")
    if name == "asl_v4":
        print("i2h weight.grad:", model.i2h.weight.grad)
    elif name == "basic_rnn2":
        print("i2h weight.grad:", model.i2h.weight.grad)
#    elif name == 'lstm':
#        print("i2h weight.grad:", model.lstm.weight.grad)
    scheduler.step()
    training_avg_epoch_losses = torch.tensor(epoch_losses).mean().item()
    correct_percent = correct/size * 100
    dprint(f"Training Error: \n Correct:{correct}, Accuracy: {correct_percent:>0.1f}%, Avg loss: {training_avg_epoch_losses:>8f} \n")
    return correct_percent, training_avg_epoch_losses

def run_test(dataloader, models, name,  debug = None):
    print("Testing:", name)
    model = models[name]
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    i = 0
    with torch.no_grad():
        for batch in dataloader:
            i+= 1
            if debug is not None and i > debug:
                break;
            X, Y = [], []
            for item in batch:
                x, y = item
                X.append(x)
                Y.append(y)

            # this results in a shape (batch, frame count, landmarks)
            X = torch.stack(X)
            fprint("X init shape:", X.shape)

            # transform X into a shape (frame count, batch, landmarks)
            X = torch.swapaxes(X, 0, 1)
            fprint("X transformed shape:", X.shape)
            Y = torch.tensor(np.array(Y))

            hidden = model.init_zero_hidden(len(Y))
            # send tensors to device
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)

            pred, hidden = model(X, hidden)
            fprint("pred.argmax(1):" , pred.argmax(1))
            fprint("Y:", Y)
            losses = loss_fn(pred, Y)
            fprint("losses:", losses)
            test_loss += losses.item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
    print("correct:", correct, ". size:", size)
    test_loss /= num_batches
    correct_percent = correct/size * 100
    print(f"Test Error: \n Correct:{correct}, Accuracy: {correct_percent:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct_percent, test_loss

def run_training(models, name, train_dataloader, test_data,
                 lr = 0.001,
                 epochs = 6, debug = None):
    print("=> Starting training: ", name)
    model = models[name]
    loss_fn = nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_test_loss = None
    train_accuracies = []
    train_losses = []
    train_epoch_time = []
    train_lr = []
    test_accuracies = []
    test_losses = []
    test_epoch_time = []
    for t in range(epochs):
        start_time = time.time()
        print(f"Epoch {t+1}\n-------------------------------")
        train_lr.append(scheduler.get_last_lr()[0])
        train_accuracy, avg_train_loss \
            = do_train(train_dataloader, model, name,optimizer, \
                       scheduler, loss_fn, debug)
        train_end_time = time.time()
        train_epoch_time.append(train_end_time - start_time)
        print(f'=> epoch: {t + 1}, during training: loss: {avg_train_loss}, accurancy: {train_accuracy}')
        print("Calculating Train Statistics")
#        train_accuracy, avg_train_loss = run_test(train_dataloader, models, name, debug)
        train_accuracies.append(train_accuracy)
        train_losses.append(avg_train_loss)
        print("Calculating Test Statistics")
        test_accuracy, avg_test_loss = run_test(test_data, models, name, debug)
        test_accuracies.append(test_accuracy)
        test_losses.append(avg_test_loss)
        test_epoch_time.append(time.time() - train_end_time)
        if best_test_loss is None or best_test_loss > avg_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH.joinpath(name))
        if True: #(t + 1)% 50 == 0 or t == epochs - 1:
            d = {
                "training_losses": train_losses,
                "training_accuracies (in %)": train_accuracies,
                "training epoch time": train_epoch_time,
                "training lr": train_lr,
                "test_losses": test_losses,
                "test_accuracies (in %)":test_accuracies,
                "test epoch time": test_epoch_time,
            }
            df = pd.DataFrame(data=d)
            df.to_csv(OUTPUT_PATH.joinpath("{}_training_summary.csv".format(name)), index=False)
    print("Done!")

def load_model(name, models):
    models[name].load_state_dict(torch.load(MODEL_OUTPUT_PATH.joinpath(name)))


batch = 10
train, test = split(load_csv(TRAINING_DATA))
print("train:", train)
print("test:", test)
train_dataset = HandDataSet(train)
test_dataset = HandDataSet(test)
train_dataloader = DataLoader(train_dataset, batch_size= batch, collate_fn=collate_fn_padd)
test_dataloader = DataLoader(test_dataset, batch_size = batch, collate_fn=collate_fn_padd)
os.environ["DEBUG"] = "False"

# In[ ]:


asl_model = ASLRNN()
basic_rnn_model = RNN()
basic_rnn2 = RNN2()
joint_rnn_model = JointRNN()
asl_v2 = ASLRNN2()
asl_v3 = ASLRNN3()
asl_v4 = ASLRNN4()
lstm = LSTMASL()
models = { "asl_model_trimmed":  asl_model,
           "basic_rnn_model_trimmed": basic_rnn_model,
           "basic_rnn2": basic_rnn2,
           "joint_rnn_trimmed": joint_rnn_model,
           "asl_v2_trimmed": asl_v2,
           "asl_v3": asl_v3,
           "asl_v4": asl_v4,
           "lstm": lstm}

# In[ ]:

epoch = 100
os.environ["DEBUG"] = "False"
train_debug = None;
lr = 5

def load_train_test(models, name, train_data, test_data, epochs, debug):
    print(">>> Start: ", name, ". epochs:", epochs, train_debug, models[name])
#    load_model(name, models); print("loaded model")
    run_training(models, name,train_data, test_data, lr, epochs, debug)
    print(">>>> test error")
    run_test(test_data, models, name, debug)

load_train_test(models, "lstm", train_dataloader, test_dataloader,
                epoch, train_debug)

#load_train_test(models, "asl_v2_trimmed", train_dataloader, test_dataloader,
#                epoch, train_debug)
#load_train_test(models, "basic_rnn2", train_dataloader, test_dataloader,
#                epoch, train_debug)
exit()
load_train_test(models, "joint_rnn_trimmed", train_dataloader, test_dataloader,
                epoch, train_debug)

load_train_test(models, "basic_rnn_model_trimmed", train_dataloader, test_dataloader, epoch, train_debug)

load_train_test(models, "asl_model_trimmed", train_dataloader, test_dataloader, epoch, train_debug)

#load_model("asl_model_trimmed", models)

#load_model("basic_rnn_model_trimmed", models)
#run_training(models, "basic_rnn_model_trimmed",train_dataloader, test_dataloader,#              epochs = epoch,  debug=train_debug)


#load_model("joint_rnn_trimmed", models)
#run_training(models, "joint_rnn_trimmed",train_dataloader, test_dataloader,
#             epochs = epoch,  debug=train_debug)

# In[ ]:

#print("training error")
#run_test( train_dataloader, models, "asl_model_trimmed", debug=train_debug)
#run_test( train_dataloader, models, "basic_rnn_model_trimmed", debug=train_debug)
#run_test( train_dataloader, models, "joint_rnn_trimmed", debug=train_debug)


#run_test( test_dataloader, models, "asl_model_trimmed", debug=train_debug)
#run_test( test_dataloader, models, "basic_rnn_model_trimmed", debug=train_debug)
#run_test(test_dataloader, models, "joint_rnn_trimmed", debug=train_debug)
# In[ ]:
