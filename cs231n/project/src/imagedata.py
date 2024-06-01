
from PIL import Image
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import tv_tensors
import torchvision.transforms as T
import torch
import math
    

from constants import IMAGE_SIZE
def scale_and_pad(image):
    C, H, W = image.shape
    if H > W:
        HH = IMAGE_SIZE
        pad_h = 0
        WW =  math.ceil(IMAGE_SIZE / H * W)
        pad_w = IMAGE_SIZE - WW
    else:
        WW = IMAGE_SIZE
        pad_w= 0
        HH =  math.ceil(IMAGE_SIZE / W * H)
        pad_h = IMAGE_SIZE - HH
    resized = T.Resize(size=(HH, WW))(image)
   
    image_640_640 =  T.Pad(padding=[0, 0, pad_w, pad_h])(resized)
    return T.Resize(size=(IMAGE_SIZE//2, IMAGE_SIZE//2))(image_640_640)  # shrink it to 320X320 to save space

def to_dict(image_data):
    return  {
        'image_id': image_data.image_id,
        'image_path': image_data.image_path,
        'image': image_data.image,
        'annotations': image_data.annotations
    }

def to_image_data(dictionary):
    return ImageData(**dictionary)
   
def get_image(path, device):
    image = Image.open(path).convert("RGB")
    return scale_and_pad(tv_tensors.Image(image, requires_grad = False).to(device))

class ImageData():
    def __init__(
        self,
        image_id: int,
        image_path: str,
        annotations,
    ):
        self.image_path = image_path
        self.annotations = annotations
        self.image_id = image_id
    
    def image_tensor(self, device):
        return get_image(self.image_path, device)
    
    def captions(self):
        if "captions" not in self.annotations:
            return None
        return self.annotations["captions"]
    
    def qa(self):
        if "qa" not in self.annotations:
            return None
        return self.annotations["qa"]     
    
    def qids(self):
        if "qa" not in self.annotations:
            return None
        return self.annotations["q_id"]     
