

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from PIL import Image
import os
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import tv_tensors

import fiftyone as fo
import fiftyone.zoo as foz

import constants
import imagedata

import fiftyone.utils.coco as fouc
from PIL import Image
import json 
import pandas as pd
import time
from dataset import VqaCoco

    
class ResumableDataset(VisionDataset):
    
    def __init__(
        self,
        coco : VqaCoco,
        existing_answer_name: str
    ) -> None:
        self.coco = coco
        with open(constants.TEST_OUTPUT.joinpath(existing_answer_name), 'r') as f:
            data = json.load(f)
    #        print("read data:", len(data))
#            data = json.loads(data) 
            print("load json",len(data))
        self.answers = data
        self.answers_pd = pd.json_normalize(data)
        questions_pd = self.coco.questions
        answered_qids = self.answers_pd['question_id'].tolist()
        print("qis len", len(answered_qids))
        self.missing = questions_pd[~questions_pd['question_id'].isin(answered_qids)]
        print(self.missing)
        
    def __getitem__(self, idx):
#        start_time = time.time()
        image_id = self.missing.iloc[idx]['image_id']
        image_id_str = str(image_id)
        image_path = 'COCO_test2015_' + ("000000000000" + image_id_str)[-12:] + '.jpg'
        image_path = constants.VQA_COCO_TEST.joinpath(image_path)

        return self.coco.get_item_from_path(image_path)

    def __len__(self):
        return len(self.missing)