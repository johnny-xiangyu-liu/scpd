
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

class ImageData():
    def __init__(
        self,
        image_id: str,
        image,
        sentences,
        annotations,
    ):
        self.image_id = image_id
        self.img = tv_tensors.Image(image)
        self.annotations = annotations
        self.sentences = sentences

    def image(self):
        return self.img
    
    def boxes(self):
        box = []
        for box_id, boxes in self.annotations['boxes'].items():
            box +=boxes
        return tv_tensors.BoundingBoxes(box, format="XYXY", canvas_size=self.image().shape[-2:])
