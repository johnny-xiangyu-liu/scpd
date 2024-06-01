

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

def load(file_path:str, key:str, columns: list[str] = None):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return pd.DataFrame(data[key], columns= columns)

    
def answer_to(answers_pd, question_id: int):
    return answers_pd.loc[answers_pd['question_id'] == question_id]["multiple_choice_answer"].values[0]

def max_len(column):
    return column.str.len().max()

def annotate_qa(q, a):
    result = constants.QUESTION_TOKEN + ' ' + q.lower() + ' ' + constants.ANSWER_TOKEN 
    if a is not None:
        # should randomize some empty tokens in after the constants.ANSWER_TOKEN 
        result += ' ' + a.lower() + ' ' + constants.END_TOKEN  
    return result
    
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
                                columns= ['image_id', 'multiple_choice_answer', 'question_id'])
        elif dataset_type == "validation":
            # don't need to load validation captions
            self.captions = None
            self.questions = load(constants.VQA_OPEN_ENDED_QUESTION_VAL, key = "questions")
            self.answers = load(constants.VQA_OPEN_ENDED_ANSWER_VAL, key = "annotations",
                                columns= ['image_id', 'multiple_choice_answer', 'question_id'])
        elif dataset_type == "test":
            self.captions = None
            self.questions = load(constants.VQA_OPEN_ENDED_QUESTION_TEST, key = "questions")
            self.answers = None
        else:
            raise Exception(f"Unknown type {dataset_type}")
        
        
        
    def get(self, annotations, image_id):
        if annotations is None:
            return None
        return annotations.loc[annotations['image_id'] == image_id]

    def __getitem__(self, idx):
#        start_time = time.time()

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_id = int(os.path.basename(image_path).removesuffix('.jpg'))

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        annotations = {}
        captions = None if self.captions is None else self.get(self.captions, image_id)
        questions  = None if self.questions is None else self.get(self.questions, image_id)
        answers = None if self.answers is None else self.get(self.answers, image_id)
        
        if captions is not None:
            annotations['captions'] = captions['caption'].tolist()
        if questions is not None:
            qa = []
            qs = []
            q_ids = []
            for index, row in questions.iterrows():
                q_id = row['question_id']
                q = row['question']
                ans = None if answers is None else answer_to(answers, q_id)
                qa.append(annotate_qa(q, ans))
                qs.append(annotate_qa(q, None))
                q_ids.append(q_id)
            annotations['qa'] = qa
            annotations['q_id'] = q_ids
            annotations['qs'] = qs
        result = imagedata.ImageData(image_id, image_path, image, annotations)
#        print("---Loading ", idx , " took: %s seconds ---" % (time.time() - start_time))

        return result

    def __len__(self):
        return len(self.image_paths)
    