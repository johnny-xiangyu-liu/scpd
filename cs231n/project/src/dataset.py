

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
from flickr30k_util.flickr30k_entities_utils import get_sentence_data, get_annotations
import imagedata

import fiftyone.utils.coco as fouc
from PIL import Image


class Coco(VisionDataset):
    
    def __init__(
        self,
        dataset_type: str = "train", # or "validation" or "test"
        gt_field="ground_truth",
        classes=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=None, transforms = transforms, transform = transform, target_transform = target_transform)
        self.dataset = foz.load_zoo_dataset("coco-2017", split = dataset_type)
        self.dataset.persistent = True
        self.gt_field = gt_field
        self.img_paths = self.dataset.values("filepath")
        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.dataset.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.dataset[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        print(detections)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return imagedata.ImageData(target["image_id"], img, {}, target)

    def __len__(self):
        return len(self.img_paths)
    

class Flickr30k(VisionDataset):
    """
    Based on https://pytorch.org/vision/stable/_modules/torchvision/datasets/flickr.html#Flickr30k

    Args:
        images_dir (str or ``pathlib.Path``): Root directory where images are downloaded to.
        image_ids_file (string): Path to image ids to load images
        annotation_dir (string): Path to load annotations (bounding boxes)
        sentence_dir (string): Path to load sentences
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        image_ids_file: str,
        images_dir: str = constants.FLICKR30k_IMAGE_DIR,  
        annotation_dir: str= constants.FLICKR30k_ANNOTATION_DIR,
        sentence_dir: str= constants.FLICKR30k_SENTENCE_DIR,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=None, transform=transform, target_transform=target_transform)

        self.images_dir = images_dir
        self.image_ids_file = image_ids_file
        self.annotation_dir = annotation_dir
        self.sentence_dir = sentence_dir

        self.image_ids = list()
        with open(self.image_ids_file) as fh:
            for line in fh:
                self.image_ids.append(line.strip())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.image_ids[index]

        # Image
        filename = os.path.join(self.images_dir, img_id + ".jpg")
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Sentences
        sentences = None
        if self.sentence_dir is not None:
            sentence_file = os.path.join(self.sentence_dir, img_id + ".txt")
            sentences = get_sentence_data(sentence_file)

        # Annotation
        annotations = None
        if self.annotation_dir is not None:
            annotation_file = os.path.join(self.annotation_dir, img_id+ ".xml")
            annotations = get_annotations(annotation_file)

        return imagedata.ImageData(img_id, img, sentences, annotations)


    def __len__(self) -> int:
        return len(self.image_ids)
