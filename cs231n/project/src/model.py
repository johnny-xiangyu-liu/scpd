import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


from transformers import AutoTokenizer, BertModel

@torch.no_grad()
class TextEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        
    @torch.no_grad()
    def forward(self, x):
        """
        x: List of string (text)
        """
        tokens = tokenizer(x, padding=True , return_tensors="pt")
        _, output = model(**tokens)
        return output
    


@torch.no_grad()
class MaskRNNBackbone(nn.Module):
    
    def __init__(self):
        super().__init__()

        # micmicking the maskrnn behavior in https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/models/detection/generalized_rcnn.py#L46-L103
        # use maskrnn backbone as the pretrained image backbone
        maskrnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        # always set it to eval so that it's frozen.
        self.transform = maskrnn.transform.eval()       
        self.image_backbone = maskrnn.backbone.eval()
        
        
    @torch.no_grad()
    def forward(self, x):
        """
        x: (N, C, H, W) a batch of images.
        """
        images, _ = self.transform(images, None)
        features = self.image_backbone(images.tensors)
        return features


class VQANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_backbone = MaskRNNBackbone()
        self.text_encoder = TextEncoder()
        
        
     
    def forward(self, x):
        """
        x: (N, C, H, W) a batch of images.
        """
        feature_map = self.image_backbone(x)
        return scores