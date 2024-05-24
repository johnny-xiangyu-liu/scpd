

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

def load(file_path:str, key:str, columns: list[str] = None):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return pd.DataFrame(data[key], columns= columns)

    
def answer_to(answers_pd, question_id: int):
    return answers_pd.loc[answers_pd['question_id'] == question_id]["multiple_choice_answer"].values[0]

def max_len(column):
    return column.str.len().max()

class Coco(VisionDataset):
    
    def __init__(
        self,
        dataset_type: str = "train", # or "validation" or "test"
        classes=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=None, transforms = transforms, transform = transform, target_transform = target_transform)
        self.dataset = foz.load_zoo_dataset("coco-2017", split = dataset_type)
        self.dataset.persistent = True
        self.image_paths = self.dataset.values("filepath")

        
        if dataset_type == "train":
            self.captions = load(constants.CAPTION_TRAIN, key = "annotations", columns= ['image_id', 'caption'])
            self.questions = load(constants.VQA_OPEN_ENDED_QUESTION_TRAIN, key = "questions")
            self.answers = load(constants.VQA_OPEN_ENDED_ANSWER_TRAIN, key = "annotations",
                                columns= ['image_id', 'multiple_choice_answer', 'question_id', 'answer_type', 'question_type'])
        elif dataset_type == "val":
            # don't need to load validation captions
#            self.captions = load(constants.CAPTION_VAL, key = "annotations", columns= ['image_id', 'caption'])
            self.questions = load(constants.VQA_OPEN_ENDED_QUESTION_VAL, key = "questions")
            self.answers = load(constants.VQA_OPEN_ENDED_ANSWER_VAL, key = "annotations",
                                columns= ['image_id', 'multiple_choice_answer', 'question_id', 'answer_type', 'question_type'])
        else:
            self.captions = None
            self.questions = None
            self.answers = None
        
        
        
    def get(self, annotations, image_id):
        if annotations is None:
            return None
        return annotations.loc[annotations['image_id'] == image_id]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_id = int(os.path.basename(image_path).removesuffix('.jpg'))

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        annotations = {}
        captions = self.get(self.captions, image_id)
        questions  = self.get(self.questions, image_id)
        answers = self.get(self.answers, image_id)
        
        if captions is not None:
            annotations['captions'] = captions['caption'].tolist()
        if questions is not None:
            qa = []
            for index, row in questions.iterrows():
                q_id = row['question_id']
                q = row['question']
                ans = answer_to(answers, q_id)
                qa.append((q, ans))
            annotations['qa'] = qa
        return imagedata.ImageData(image_id, image_path, image, annotations)

    def __len__(self):
        return len(self.image_paths)
    