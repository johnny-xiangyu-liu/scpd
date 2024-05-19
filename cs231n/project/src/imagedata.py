
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from PIL import Image
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import tv_tensors

class ImageData():
    def __init__(
        self,
        image_id: int,
        image_path: str,
        image,
        annotations,
    ):
        self.image_path = image_path
        self.image = tv_tensors.Image(image)
        self.annotations = annotations
        self.image_id = image_id
        
    def image(self):
        return self.image
    