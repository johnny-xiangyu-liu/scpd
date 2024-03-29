from util import *
import numpy as np
from constants import *
from hand import HandData
import torch
from torch.utils.data import Dataset


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_pd):
        self.pd = csv_pd

    def __len__(self):
        return self.pd.size

    def __getitem__(self, idx):
        row = self.df.iloc[[index]]
        landmark_path = row['path']
        sign = row['sign']
        landmark = load_hand_data(str(DATA_PATH.joinpath(landkark_path)))
        hands = parse_to_data(landmark)

        return hands, sign
